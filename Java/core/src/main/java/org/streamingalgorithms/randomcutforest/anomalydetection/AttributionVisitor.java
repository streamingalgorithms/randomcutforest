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

package org.streamingalgorithms.randomcutforest.anomalydetection;

import static org.streamingalgorithms.randomcutforest.DefaultScoreFunctions.DEFAULT_IGNORE_LEAF_MASS_THRESHOLD;
import static org.streamingalgorithms.randomcutforest.tree.VectorSupport.multiply;

import java.util.Arrays;

import org.streamingalgorithms.randomcutforest.DefaultScoreFunctions;
import org.streamingalgorithms.randomcutforest.IRFVisitor;
import org.streamingalgorithms.randomcutforest.IVisitorFactory;
import org.streamingalgorithms.randomcutforest.Visitor;
import org.streamingalgorithms.randomcutforest.returntypes.DiVector;
import org.streamingalgorithms.randomcutforest.tree.INodeView;
import org.streamingalgorithms.randomcutforest.tree.ITree;
import org.streamingalgorithms.randomcutforest.tree.VectorSupport;

/**
 * Directional-attribution visitor. Shares all traversal scaffolding with
 * {@code ScoreVisitor} via {@link AbstractScoringVisitor}; adds only the
 * per-half-dimension components buffer and the directional accumulator.
 *
 * <p>
 * Unlike the scalar visitor, this one does NOT converge on a duplicate leaf: it
 * sets {@code hitDuplicates} and keeps climbing with the shadow box so the
 * direction vector picks up real structure instead of a degenerate uniform
 * fill.
 */
public class AttributionVisitor extends AbstractScoringVisitor<DiVector> {

    /** Per-half-dimension gap contributions; filled by probabilityOfCut. */
    protected final float[] probabilityComponents;

    /** The accumulated directional attribution (length 2 * dimension). */
    protected final double[] directionalAttribution;

    private final double[] foldedAttribution; // per-query sum, sized 2*dim, NOT reset per tree

    protected AttributionVisitor(int dimension, int treeMass, int ignoreLeafMassThreshold,
            DefaultScoreFunctions.ScoreFn scoreSeenFn, DefaultScoreFunctions.ScoreFn scoreUnseenFn,
            DefaultScoreFunctions.DampFn dampFn, DefaultScoreFunctions.Normalizer normalizer) {
        super(dimension, treeMass, ignoreLeafMassThreshold, scoreSeenFn, scoreUnseenFn, dampFn, normalizer);
        // twice as long as the point: positive and negative gaps stored separately.
        this.probabilityComponents = new float[2 * dimension];
        this.directionalAttribution = new double[2 * dimension];
        this.foldedAttribution = new double[2 * dimension];
    }

    protected AttributionVisitor(float[] pointToScore, int treeMass, int ignoreLeafMassThreshold,
            DefaultScoreFunctions.ScoreFn scoreSeenFn, DefaultScoreFunctions.ScoreFn scoreUnseenFn,
            DefaultScoreFunctions.DampFn dampFn, DefaultScoreFunctions.Normalizer normalizer) {
        this(pointToScore.length, treeMass, ignoreLeafMassThreshold, scoreSeenFn, scoreUnseenFn, dampFn, normalizer);
        System.arraycopy(pointToScore, 0, this.pointToScore, 0, pointToScore.length);
        VectorSupport.expandInto(this.pointToScore, 0, this.expandedPoint, 0, pointToScore.length);
    }

    public AttributionVisitor(float[] pointToScore, int treeMass, int ignoreLeafMassThreshold) {
        this(pointToScore, treeMass, ignoreLeafMassThreshold, DefaultScoreFunctions.DEFAULT_SCORE_SEEN,
                DefaultScoreFunctions.DEFAULT_SCORE_UNSEEN, DefaultScoreFunctions.DEFAULT_DAMP,
                DefaultScoreFunctions.DEFAULT_NORMALIZER);
    }

    public AttributionVisitor(float[] pointToScore, int treeMass) {
        this(pointToScore, treeMass, DEFAULT_IGNORE_LEAF_MASS_THRESHOLD);
    }

    @Override
    protected float[] contributionTarget() {
        return probabilityComponents;
    }

    @Override
    protected void updateFromNode(double prob, int depth, int mass) {
        // scalar total, tracked alongside the vector; updated only on the ignoreLeaf
        // path. This is NOT the scalar visitor's removed double-update: savedScore and
        // directionalAttribution are distinct accumulators, and getResult rescales the
        // vector to sum to savedScore.
        if (ignoreLeaf) {
            savedScore = prob * scoreUnseenFn.of(depth, mass) + (1 - prob) * savedScore;
        }

        if (prob <= 0) {
            pointInsideBox = true;
        } else {
            double newScore = scoreUnseenFn.of(depth, mass);
            VectorSupport.updateRecurrence(directionalAttribution, probabilityComponents, newScore, 1 - prob);
        }
    }

    @Override
    public void acceptLeaf(INodeView leafNode, int depthOfNode) {
        // NOTE: one ArrayBox per visitor lifetime (a visitor makes exactly one
        // acceptLeaf call). To eliminate it entirely, expose a reused leaf ArrayBox
        // from INodeView the way getSiblingBoundingBox already reuses the sibling box.
        double probability;
        float[] leafPoint = leafNode.getLeafPoint();
        int dimension = probabilityComponents.length / 2;
        if (Arrays.equals(leafPoint, pointToScore)) {
            hitDuplicates = true;
        }

        if (hitDuplicates && ((!ignoreLeaf) || (leafNode.getMass() > ignoreLeafMassThreshold))) {
            savedScore = dampFn.of(leafNode.getMass(), treeMass) * scoreSeenFn.of(depthOfNode, leafNode.getMass());
        } else {
            savedScore = scoreUnseenFn.of(depthOfNode, leafNode.getMass());
        }

        if (hitDuplicates || (ignoreLeaf && (leafNode.getMass() <= ignoreLeafMassThreshold))) {
            // no better option than an equal attribution
            Arrays.fill(directionalAttribution, savedScore / (2 * pointToScore.length));
        } else {
            // AttributionVisitor, else-branch — replaces the whole scalar loop + normalize
            // loop:
            double sum = VectorSupport.signedGapInto(expandedPoint, 0, +1f, leafPoint, 0, probabilityComponents, 0,
                    dimension)
                    + VectorSupport.signedGapInto(expandedPoint, dimension, -1f, leafPoint, 0, probabilityComponents,
                            dimension, dimension);
            double factor = (sum == 0) ? 0.0 : savedScore / sum;
            for (int i = 0; i < directionalAttribution.length; i++)
                directionalAttribution[i] = probabilityComponents[i];
            multiply(directionalAttribution, factor);
        }
    }

    private double idempotentRefactor() {
        double factor = normalizer.scale(treeMass);
        if (savedScore > 0) {
            if (hitDuplicates || ignoreLeaf) {
                factor *= savedScore / VectorSupport.sum(directionalAttribution);
            }
        } else {
            factor = 0;
        }
        return factor;
    }

    @Override
    public DiVector getResult() {
        return observeResult(); // per-tree — restores the original contract
    }

    @Override
    public DiVector getFoldResult() {
        return new DiVector(foldedAttribution);
    } // per-query (fold path)

    @Override
    public double convergingValue() {
        double factor = idempotentRefactor();
        return sum(directionalAttribution) * factor;
    }

    @Override
    public void reset() {
        super.reset();
        Arrays.fill(probabilityComponents, 1f);
        Arrays.fill(directionalAttribution, 0.0);
        // foldedAttribution deliberately NOT cleared — it accumulates across trees
    }

    // following is for testing
    protected DiVector observeResult() {
        double factor = idempotentRefactor();
        double[] out = Arrays.copyOf(directionalAttribution, directionalAttribution.length);
        for (int i = 0; i < out.length; i++) {
            out[i] *= factor;
        }
        return new DiVector(out);
    }

    // after each tree: scale this tree's result, add into the running sum
    public void foldOut() {
        double factor = idempotentRefactor();
        VectorSupport.axpyInto(foldedAttribution, directionalAttribution, factor);
    }

    public static IVisitorFactory<DiVector> reusableFactory(int ignoreLeafMassThreshold,
            DefaultScoreFunctions.ScoreFn scoreSeenFn, DefaultScoreFunctions.ScoreFn scoreUnseenFn,
            DefaultScoreFunctions.DampFn dampFn, DefaultScoreFunctions.Normalizer normalizer) {
        return new IVisitorFactory<DiVector>() {
            @Override
            public boolean isReusable() {
                return true;
            }

            @Override
            public boolean isFoldable() {
                return true;
            }

            // sizes from the (projected) point; mass + copy come from prepare(tree, point)
            @Override
            public IRFVisitor<DiVector> newReusableVisitor(float[] point) {
                return new AttributionVisitor(point.length, 0, ignoreLeafMassThreshold, scoreSeenFn, scoreUnseenFn,
                        dampFn, normalizer); // sized, unarmed; no copy
            }

            // non-reusable path still supported: arm immediately
            @Override
            public Visitor<DiVector> newVisitor(ITree<?, ?> tree, float[] point) {
                return new AttributionVisitor(tree.projectToTree(point), tree.getMass(), ignoreLeafMassThreshold,
                        scoreSeenFn, scoreUnseenFn, dampFn, normalizer);
            }
        };
    }

    public static double sum(double[] a) {
        double s = 0.0;
        for (int i = 0; i < a.length; i++)
            s += a[i];
        return s;
    }

}
