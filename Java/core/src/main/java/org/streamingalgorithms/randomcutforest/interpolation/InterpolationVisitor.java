/*
 * Copyright 2026 The streamingalgorithms authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License").
 * You may not use this file except in compliance with the License.
 * A copy of the License is located at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * or in the "license" file accompanying this file. This file is distributed
 * on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
 * express or implied. See the License for the specific language governing
 * permissions and limitations under the License.
 */

package org.streamingalgorithms.randomcutforest.interpolation;

import java.util.Arrays;

import org.streamingalgorithms.randomcutforest.IRFVisitor;
import org.streamingalgorithms.randomcutforest.IVisitorFactory;
import org.streamingalgorithms.randomcutforest.RFVisitor;
import org.streamingalgorithms.randomcutforest.Visitor;
import org.streamingalgorithms.randomcutforest.returntypes.DensityOutput;
import org.streamingalgorithms.randomcutforest.returntypes.InterpolationMeasure;
import org.streamingalgorithms.randomcutforest.tree.ArrayBox;
import org.streamingalgorithms.randomcutforest.tree.IBoundingBoxView;
import org.streamingalgorithms.randomcutforest.tree.INodeView;
import org.streamingalgorithms.randomcutforest.tree.ITree;

/**
 * ArrayBox / two-pass port of {@code SimpleInterpolationVisitor}, kept side by
 * side with it for A/B validation.
 *
 * <p>
 * Deliberately standalone rather than a subclass of
 * {@code AbstractScoringVisitor}: interpolation's shadow-path counterfactual
 * uses the actual node box as the "large" side (it weights by real node mass),
 * whereas the scalar/attribution base uses {@code shadow ∪ point}. That
 * semantic difference means it cannot ride the base's {@code final accept}. It
 * does adopt the family flavor: it reads the *conceptual* box through
 * {@link IBoundingBoxView} (an {@link ArrayBox} works; no {@code BoundingBox}
 * data structure required), reuses one {@link ArrayBox} as the shadow
 * accumulator via in-place {@code addBox}, and never materializes the per-node
 * gap / directional-distance scratch vectors — pass 1 reduces, pass 2
 * recomputes and distributes.
 *
 * <p>
 * Two things are preserved verbatim from the original for parity, both worth
 * flagging:
 * <ul>
 * <li>The gap / range math stays in <b>double</b> (the original used
 * {@code Math.max(..., 0.0)} on double-widened box values). This is NOT the
 * float gap path used by scalar/attribution; SIMD/float questions are deferred
 * so the A/B compares like for like.</li>
 * <li>The directional-distance "max side wins" quirk: when both maxGap and
 * minGap are positive (only possible on the shadow path, where large is the
 * node box), only the high half of {@code distances} receives a contribution.
 * Reproduced exactly; flag separately if it should be revisited.</li>
 * </ul>
 *
 * <p>
 * The scoring functions (field / influence / self) are kept as identical
 * protected methods, NOT lifted to functional fields, on purpose: the A/B test
 * should isolate the box/allocation rewrite from any change to the functions.
 * Lift them to the family's functional-interface style once parity is
 * confirmed.
 */
public class InterpolationVisitor extends RFVisitor<InterpolationMeasure> {

    private final double pointMass;
    private final boolean centerOfMass;

    /** Per-tree measure, filled by the walk, cleared each tree. */
    public DensityOutput stored;
    /** Per-query sum across trees. NOT cleared in reset(). */
    private final DensityOutput folded;

    /**
     * (c) Linear score channel — telescopes to sum(measure). NOT the density
     * scalar.
     */
    private double savedScore;
    private double foldedScore; // per-query linear-score sum; NOT cleared in reset()

    private boolean pointEqualsLeaf; // hitDuplicates analog (density-specific meaning)
    private double savedMass;

    private final ArrayBox leafBox; // reused degenerate leaf box
    // shadowBox + shadowBoxActive inherited from RFVisitor

    private double sumOfNewRange;
    private double sumOfDifferenceInRange;
    private boolean coordInsideBox[];

    // ---- reusable constructor: sizes by dimension, unarmed (treeMass filled by
    // prepare) ----
    InterpolationVisitor(int dimension, double pointMass, boolean centerOfMass) {
        this.pointMass = pointMass;
        this.centerOfMass = centerOfMass;
        this.pointToScore = new float[dimension]; // RFVisitor field, no copy
        this.stored = new DensityOutput(dimension, 0); // sampleSize slot set per-tree in foldOut
        this.folded = new DensityOutput(dimension, 0); // accumulates sampleSize via addToLeft
        this.leafBox = new ArrayBox(dimension);
        this.coordInsideBox = new boolean[dimension];
        setDefaults();
    }

    // ---- legacy constructor: copies point in, arms immediately ----
    public InterpolationVisitor(float[] pointToScore, int treeMass, double pointMass, boolean centerOfMass) {
        this(pointToScore.length, pointMass, centerOfMass);
        System.arraycopy(pointToScore, 0, this.pointToScore, 0, pointToScore.length);
        this.treeMass = treeMass;
    }

    private void setDefaults() {
        savedScore = 0.0;
        pointEqualsLeaf = false;
        pointInsideBox = false;
        shadowBoxActive = false;
        Arrays.fill(coordInsideBox, false);
        stored.clear(); // zero all three DiVectors + sampleSize, in place
        // folded, foldedScore deliberately NOT cleared
    }

    @Override
    protected void reset() {
        // super.reset();
        setDefaults();
    }

    @Override
    public void accept(INodeView node, int depthOfNode) {
        if (pointInsideBox) {
            return;
        }

        IBoundingBoxView small;
        IBoundingBoxView large; // null => large = small ∪ pointToScore, computed inline
        boolean pointMode;

        if (pointEqualsLeaf) {
            ArrayBox sib = (ArrayBox) node.getSiblingBoundingBox(pointToScore);
            if (!shadowBoxActive) { // per-tree guard: reuse-correct
                if (shadowBox == null)
                    shadowBox = sib.copy();
                else
                    shadowBox.copyFrom(sib);
                shadowBoxActive = true;
            } else {
                shadowBox.addBox(sib);
            }
            small = shadowBox;
            large = node.getBoundingBox();
            pointMode = false;
        } else {
            small = node.getBoundingBox();
            large = null;
            pointMode = true;
        }

        reduce(small, large, pointMode);

        double probOfCut = (sumOfNewRange == 0.0) ? 0.0 : sumOfDifferenceInRange / sumOfNewRange;
        if (probOfCut <= 0) {
            pointInsideBox = true;
            return;
        }

        double fieldVal = fieldExt(node, centerOfMass, savedMass, pointToScore);
        double influenceVal = influenceExt(node, centerOfMass, savedMass, pointToScore);
        distribute(small, large, pointMode, fieldVal, influenceVal, 1.0 - probOfCut, true);

        // (c) linear score recurrence — telescopes to sum(measure). Same form as
        // attribution's savedScore.
        savedScore = probOfCut * fieldVal + (1.0 - probOfCut) * savedScore;
    }

    @Override
    public void acceptLeaf(INodeView leafNode, int depthOfNode) {
        leafBox.fromPoint(leafNode.getLeafPoint());
        reduce(leafBox, null, true);
        if (sumOfDifferenceInRange <= 0) {
            savedMass = pointMass + leafNode.getMass();
            pointEqualsLeaf = true;
            double selfF = 0.5 * selfField(leafNode, savedMass) / pointToScore.length;
            double selfI = 0.5 * selfInfluence(leafNode, savedMass) / pointToScore.length;
            for (int i = 0; i < pointToScore.length; i++) {
                stored.measure.high[i] = stored.measure.low[i] = selfF;
                stored.probMass.high[i] = stored.probMass.low[i] = selfI;
            }
            Arrays.fill(coordInsideBox, false); // ← ADD THIS — matches legacy; shadow path re-evaluates containment
            savedScore = selfF * 2 * pointToScore.length;
        } else {
            savedMass = pointMass;
            double fieldVal = fieldPoint(leafNode, savedMass, pointToScore);
            double influenceVal = influencePoint(leafNode, savedMass, pointToScore);
            distribute(leafBox, null, true, fieldVal, influenceVal, 0.0, false);
            // (c) linear score seed = fieldVal * probOfCut (first-touch, sum(measure) at
            // leaf)
            savedScore = (sumOfNewRange == 0) ? 0.0 : fieldVal * (sumOfDifferenceInRange / sumOfNewRange);
        }
    }

    // ---- fold: reuse addToLeft (in-place vector + sampleSize sum) — matches
    // legacy collector exactly ----
    public void foldOut() {
        stored.setSampleSize(treeMass); // per-tree normalizer = live mass (matches legacy path)
        InterpolationMeasure.addToLeft(folded, stored); // sums measure/distances/probMass + sampleSize into folded
        foldedScore += savedScore; // (c) linear channel rides along
    }

    // InterpolationVisitor
    @Override
    public InterpolationMeasure getResult() {
        return stored;
    } // per-tree — was returning folded, FIX

    @Override
    public InterpolationMeasure getFoldResult() {
        return folded;
    } // per-query

    /**
     * (c) LINEAR score from the same walk — this is sum(measure), NOT the density
     * estimate.
     */
    public double getLinearScore() {
        return foldedScore;
    }

    private void reduce(IBoundingBoxView small, IBoundingBoxView large, boolean pointMode) {
        sumOfNewRange = 0.0;
        sumOfDifferenceInRange = 0.0;
        for (int i = 0; i < pointToScore.length; i++) {
            double smallMax = small.getMaxValue(i), smallMin = small.getMinValue(i);
            double largeMax, largeMin;
            if (pointMode) {
                double p = pointToScore[i];
                largeMax = Math.max(smallMax, p);
                largeMin = Math.min(smallMin, p);
            } else {
                largeMax = large.getMaxValue(i);
                largeMin = large.getMinValue(i);
            }

            sumOfNewRange += (largeMax - largeMin); // accumulated for ALL coords, even contained
            if (coordInsideBox[i]) {
                continue; // already contained: contributes nothing further
            }

            double maxGap = Math.max(largeMax - smallMax, 0.0);
            double minGap = Math.max(smallMin - largeMin, 0.0);
            if (maxGap + minGap > 0.0) {
                sumOfDifferenceInRange += (maxGap + minGap);
            } else {
                coordInsideBox[i] = true; // newly contained: skip from now on
            }
        }
    }

    private void distribute(IBoundingBoxView small, IBoundingBoxView large, boolean pointMode, double fieldVal,
            double influenceVal, double decay, boolean accumulate) {
        double invSumNew = (sumOfNewRange == 0) ? 0.0 : 1.0 / sumOfNewRange;
        for (int i = 0; i < pointToScore.length; i++) {
            double probHigh, probLow, ddHigh, ddLow;
            if (coordInsideBox[i]) {
                probHigh = probLow = ddHigh = ddLow = 0.0; // contained: zero contribution, but STILL decay the prior
            } else {
                double smallMax = small.getMaxValue(i), smallMin = small.getMinValue(i);
                double largeMax, largeMin;
                if (pointMode) {
                    double p = pointToScore[i];
                    largeMax = Math.max(smallMax, p);
                    largeMin = Math.min(smallMin, p);
                } else {
                    largeMax = large.getMaxValue(i);
                    largeMin = large.getMinValue(i);
                }
                double oldRange = smallMax - smallMin;
                double maxGap = Math.max(largeMax - smallMax, 0.0);
                double minGap = Math.max(smallMin - largeMin, 0.0);
                probHigh = maxGap * invSumNew;
                probLow = minGap * invSumNew;
                if (maxGap > 0.0) {
                    ddHigh = maxGap + oldRange;
                    ddLow = 0.0;
                } else {
                    ddHigh = 0.0;
                    ddLow = minGap + oldRange;
                }
            }

            if (accumulate) {
                stored.probMass.high[i] = probHigh * influenceVal + decay * stored.probMass.high[i];
                stored.measure.high[i] = probHigh * fieldVal + decay * stored.measure.high[i];
                stored.distances.high[i] = probHigh * ddHigh * influenceVal + decay * stored.distances.high[i];
                stored.probMass.low[i] = probLow * influenceVal + decay * stored.probMass.low[i];
                stored.measure.low[i] = probLow * fieldVal + decay * stored.measure.low[i];
                stored.distances.low[i] = probLow * ddLow * influenceVal + decay * stored.distances.low[i];
            } else {
                stored.probMass.high[i] = probHigh * influenceVal;
                stored.measure.high[i] = probHigh * fieldVal;
                stored.distances.high[i] = probHigh * ddHigh * influenceVal;
                stored.probMass.low[i] = probLow * influenceVal;
                stored.measure.low[i] = probLow * fieldVal;
                stored.distances.low[i] = probLow * ddLow * influenceVal;
            }
        }
    }

    // InterpolationVisitor — per-tree result, for unit tests and A/B
    InterpolationMeasure observeResult() {
        return stored;
    }

    double fieldExt(INodeView n, boolean c, double m, float[] loc) {
        return n.getMass() + m;
    }

    double influenceExt(INodeView n, boolean c, double m, float[] loc) {
        return 1.0;
    }

    double fieldPoint(INodeView n, double m, float[] loc) {
        return n.getMass() + m;
    }

    double influencePoint(INodeView n, double m, float[] loc) {
        return 1.0;
    }

    double selfField(INodeView n, double m) {
        return m;
    }

    double selfInfluence(INodeView n, double m) {
        return 1.0;
    }

    public static IVisitorFactory<InterpolationMeasure> reusableFactory(double pointMass, boolean centerOfMass) {
        return new IVisitorFactory<InterpolationMeasure>() {
            @Override
            public boolean isReusable() {
                return true;
            }

            @Override
            public boolean isFoldable() {
                return true;
            }

            @Override
            public IRFVisitor<InterpolationMeasure> newReusableVisitor(float[] point) {
                return new InterpolationVisitor(point.length, pointMass, centerOfMass);
            }

            @Override
            public Visitor<InterpolationMeasure> newVisitor(ITree<?, ?> tree, float[] point) {
                return new InterpolationVisitor(tree.projectToTree(point), tree.getMass(), pointMass, centerOfMass);
            }
        };
    }
}