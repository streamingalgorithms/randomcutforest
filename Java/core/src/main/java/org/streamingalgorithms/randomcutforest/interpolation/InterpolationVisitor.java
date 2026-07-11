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

import org.streamingalgorithms.randomcutforest.Visitor;
import org.streamingalgorithms.randomcutforest.returntypes.DensityOutput;
import org.streamingalgorithms.randomcutforest.returntypes.InterpolationMeasure;
import org.streamingalgorithms.randomcutforest.tree.ArrayBox;
import org.streamingalgorithms.randomcutforest.tree.IBoundingBoxView;
import org.streamingalgorithms.randomcutforest.tree.INodeView;

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
public class InterpolationVisitor implements Visitor<InterpolationMeasure> {

    private final float[] pointToScore;
    private final int sampleSize;
    private final boolean centerOfMass;
    private final double pointMass;

    public InterpolationMeasure stored;

    protected boolean pointInsideBox;
    private boolean pointEqualsLeaf; // the hitDuplicates analog
    private double savedMass;

    /** Persists across nodes; skips coordinates already contained. */
    private final boolean[] coordInsideBox;

    /** Reused shadow accumulator; one allocation, grown in place. */
    private ArrayBox shadowBox;

    /**
     * Reused degenerate box for the leaf point; filled in place, no per-leaf alloc.
     */
    private final ArrayBox leafBox;

    // per-node reduction outputs (pass 1 -> pass 2)
    private double sumOfNewRange;
    private double sumOfDifferenceInRange;

    public InterpolationVisitor(float[] pointToScore, int sampleSize, double pointMass, boolean centerOfMass) {
        this.pointToScore = Arrays.copyOf(pointToScore, pointToScore.length);
        this.sampleSize = sampleSize;
        this.pointMass = pointMass;
        this.centerOfMass = centerOfMass;
        this.stored = new DensityOutput(pointToScore.length, this.sampleSize);
        this.pointInsideBox = false;
        this.pointEqualsLeaf = false;
        this.coordInsideBox = new boolean[pointToScore.length];
        this.shadowBox = null;
        this.leafBox = new ArrayBox(pointToScore.length);
    }

    @Override
    public InterpolationMeasure getResult() {
        return stored;
    }

    @Override
    public void accept(INodeView node, int depthOfNode) {
        if (pointInsideBox) {
            return;
        }

        IBoundingBoxView small;
        IBoundingBoxView large; // null => large is (small ∪ pointToScore), computed inline
        boolean pointMode;

        if (pointEqualsLeaf) {
            // counterfactual: small = merged siblings, large = the actual node box
            ArrayBox sib = (ArrayBox) node.getSiblingBoundingBox(pointToScore);
            if (shadowBox == null) {
                shadowBox = sib.copy();
            } else {
                shadowBox.addBox(sib);
            }
            small = shadowBox;
            large = node.getBoundingBox();
            pointMode = false;
        } else {
            small = node.getBoundingBox();
            large = null; // large = small ∪ pointToScore, never materialized
            pointMode = true;
        }

        reduce(small, large, pointMode);

        double probOfCut = sumOfDifferenceInRange / sumOfNewRange;
        if (probOfCut <= 0) {
            pointInsideBox = true;
            return;
        }

        double fieldVal = fieldExt(node, centerOfMass, savedMass, pointToScore);
        double influenceVal = influenceExt(node, centerOfMass, savedMass, pointToScore);
        distribute(small, large, pointMode, fieldVal, influenceVal, 1.0 - probOfCut, true);
    }

    @Override
    public void acceptLeaf(INodeView leafNode, int depthOfNode) {
        leafBox.fromPoint(leafNode.getLeafPoint()); // degenerate box, no allocation
        reduce(leafBox, null, true);

        if (sumOfDifferenceInRange <= 0) {
            // query equals the leaf point
            savedMass = pointMass + leafNode.getMass();
            pointEqualsLeaf = true;
            double selfF = 0.5 * selfField(leafNode, savedMass) / pointToScore.length;
            double selfI = 0.5 * selfInfluence(leafNode, savedMass) / pointToScore.length;
            for (int i = 0; i < pointToScore.length; i++) {
                stored.measure.high[i] = stored.measure.low[i] = selfF;
                stored.probMass.high[i] = stored.probMass.low[i] = selfI;
                // distances are intentionally left as initialized (0) here
            }
            Arrays.fill(coordInsideBox, false);
        } else {
            savedMass = pointMass;
            double fieldVal = fieldPoint(leafNode, savedMass, pointToScore);
            double influenceVal = influencePoint(leafNode, savedMass, pointToScore);
            // first-touch (decay unused); leafBox is the degenerate small box
            distribute(leafBox, null, true, fieldVal, influenceVal, 0.0, false);
        }
    }

    /**
     * Pass 1: reduce {@link #sumOfNewRange} and {@link #sumOfDifferenceInRange} and
     * mark newly-contained coordinates. No scratch vectors are written.
     */
    private void reduce(IBoundingBoxView small, IBoundingBoxView large, boolean pointMode) {
        sumOfNewRange = 0.0;
        sumOfDifferenceInRange = 0.0;
        for (int i = 0; i < pointToScore.length; i++) {
            double smallMax = small.getMaxValue(i);
            double smallMin = small.getMinValue(i);
            double largeMax, largeMin;
            if (pointMode) {
                double p = pointToScore[i];
                largeMax = Math.max(smallMax, p);
                largeMin = Math.min(smallMin, p);
            } else {
                largeMax = large.getMaxValue(i);
                largeMin = large.getMinValue(i);
            }

            sumOfNewRange += (largeMax - largeMin); // == largeBox.getRange(i)
            if (coordInsideBox[i]) {
                continue;
            }

            double maxGap = Math.max(largeMax - smallMax, 0.0);
            double minGap = Math.max(smallMin - largeMin, 0.0);
            if (maxGap + minGap > 0.0) {
                sumOfDifferenceInRange += (maxGap + minGap);
            } else {
                coordInsideBox[i] = true;
            }
        }
    }

    /**
     * Pass 2: recompute per-coordinate gaps and directional distance and fold into
     * the six accumulators. {@code accumulate} true applies the {@code decay} to
     * the prior value (internal nodes); false is first-touch (leaf).
     */
    private void distribute(IBoundingBoxView small, IBoundingBoxView large, boolean pointMode, double fieldVal,
            double influenceVal, double decay, boolean accumulate) {
        double invSumNew = 1.0 / sumOfNewRange;
        for (int i = 0; i < pointToScore.length; i++) {
            double probHigh, probLow, ddHigh, ddLow;
            if (coordInsideBox[i]) {
                probHigh = probLow = ddHigh = ddLow = 0.0;
            } else {
                double smallMax = small.getMaxValue(i);
                double smallMin = small.getMinValue(i);
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
                    ddHigh = maxGap + oldRange; // == newRange on the high side
                    ddLow = 0.0;
                } else {
                    ddHigh = 0.0;
                    ddLow = minGap + oldRange; // only when maxGap == 0 (quirk preserved)
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

    // ---- pluggable measures (identical defaults to SimpleInterpolationVisitor)
    // ----

    double fieldExt(INodeView node, boolean centerOfMass, double thisMass, float[] thisLocation) {
        return node.getMass() + thisMass;
    }

    double influenceExt(INodeView node, boolean centerOfMass, double thisMass, float[] thisLocation) {
        return 1.0;
    }

    double fieldPoint(INodeView node, double thisMass, float[] thisLocation) {
        return node.getMass() + thisMass;
    }

    double influencePoint(INodeView node, double thisMass, float[] thisLocation) {
        return 1.0;
    }

    double selfField(INodeView leafNode, double mass) {
        return mass;
    }

    double selfInfluence(INodeView leafNode, double mass) {
        return 1.0;
    }

    @Override
    public boolean isConverged() {
        return pointInsideBox;
    }
}
