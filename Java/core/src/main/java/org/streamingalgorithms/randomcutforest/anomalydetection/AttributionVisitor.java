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
    protected final float[] gapComponents;

    /** The accumulated directional attribution (length 2 * dimension). */
    protected final double[] directionalAttribution;

    private final double[] foldedAttribution; // per-query sum, sized 2*dim, NOT reset per tree

    protected AttributionVisitor(int dimension, int treeMass, int ignoreLeafMassThreshold,
            DefaultScoreFunctions.ScoreFn scoreSeenFn, DefaultScoreFunctions.ScoreFn scoreUnseenFn,
            DefaultScoreFunctions.DampFn dampFn, DefaultScoreFunctions.Normalizer normalizer) {
        super(dimension, treeMass, ignoreLeafMassThreshold, scoreSeenFn, scoreUnseenFn, dampFn, normalizer);
        // twice as long as the point: positive and negative gaps stored separately.
        this.gapComponents = new float[2 * dimension];
        this.directionalAttribution = new double[2 * dimension];
        this.foldedAttribution = new double[2 * dimension];
    }

    protected AttributionVisitor(float[] pointToScore, int treeMass, int ignoreLeafMassThreshold,
            DefaultScoreFunctions.ScoreFn scoreSeenFn, DefaultScoreFunctions.ScoreFn scoreUnseenFn,
            DefaultScoreFunctions.DampFn dampFn, DefaultScoreFunctions.Normalizer normalizer) {
        this(pointToScore.length, treeMass, ignoreLeafMassThreshold, scoreSeenFn, scoreUnseenFn, dampFn, normalizer);
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
        return gapComponents;
    }

    @Override
    public boolean needsGap() {
        return true;
    }

    @Override
    protected void updateFromNode(double prob, int depth, int mass, float[] gaps) {
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
            double denom = ranges[0] + ranges[1];
            double inv = (denom == 0.0) ? 0.0 : 1.0 / denom;
            VectorSupport.updateRecurrence(directionalAttribution, gaps, newScore * inv, 1 - prob);
        }
    }

    @Override
    public void acceptLeaf(INodeView leafNode, int depthOfNode) {

        float[] leafPoint = leafNode.getLeafPoint(); // what is in the tree
        float[] expanded = leafNode.expanded(); // this query
        int dimension = gapComponents.length / 2;
        // note the vector {v,-v} still starts with v :)
        if (Arrays.equals(leafPoint, 0, dimension, expanded, 0, dimension)) {
            hitDuplicates = true;
        }

        if (hitDuplicates && ((!ignoreLeaf) || (leafNode.getMass() > ignoreLeafMassThreshold))) {
            savedScore = dampFn.of(leafNode.getMass(), treeMass) * scoreSeenFn.of(depthOfNode, leafNode.getMass());
        } else {
            savedScore = scoreUnseenFn.of(depthOfNode, leafNode.getMass());
        }

        if (hitDuplicates || (ignoreLeaf && (leafNode.getMass() <= ignoreLeafMassThreshold))) {
            // no better option than an equal attribution
            Arrays.fill(directionalAttribution, savedScore / (2 * dimension));
        } else {
            // AttributionVisitor, else-branch — replaces the whole scalar loop + normalize
            // loop:
            double sum = VectorSupport.signedGapInto(expanded, 0, +1f, leafPoint, 0, gapComponents, 0, dimension)
                    + VectorSupport.signedGapInto(expanded, dimension, -1f, leafPoint, 0, gapComponents, dimension,
                            dimension);
            double factor = (sum == 0) ? 0.0 : savedScore / sum;
            for (int i = 0; i < directionalAttribution.length; i++)
                directionalAttribution[i] = gapComponents[i] * factor;
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
        return VectorSupport.sum(directionalAttribution) * factor;
    }

    @Override
    public void reset() {
        super.reset();
        Arrays.fill(gapComponents, 1f);
        Arrays.fill(directionalAttribution, 0.0);
        // foldedAttribution deliberately NOT cleared — it accumulates across trees
    }

    // following is for testing
    protected DiVector observeResult() {
        double factor = idempotentRefactor();
        int d = directionalAttribution.length / 2;
        DiVector out = new DiVector(d);
        for (int i = 0; i < d; i++) {
            out.high[i] = directionalAttribution[i] * factor;
            out.low[i] = directionalAttribution[i + d] * factor;
        }
        return out;
    }

    @Override
    public void resetAcrossQueries(float[] point) {
        Arrays.fill(foldedAttribution, 0.0);
    }

    // after each tree: scale this tree's result, add into the running sum
    public void foldOut() {
        double factor = idempotentRefactor();
        VectorSupport.axpyInto(foldedAttribution, directionalAttribution, factor);
    }

    public static final IVisitorFactory<DiVector> DEFAULT_ATTRIBUTION_FACTORY = reusableFactory(true,
            DEFAULT_IGNORE_LEAF_MASS_THRESHOLD, DefaultScoreFunctions.DEFAULT_SCORE_SEEN,
            DefaultScoreFunctions.DEFAULT_SCORE_UNSEEN, DefaultScoreFunctions.DEFAULT_DAMP,
            DefaultScoreFunctions.DEFAULT_NORMALIZER);

    public static IVisitorFactory<DiVector> reusableFactory(boolean acrossQueries, int ignoreLeafMassThreshold,
            DefaultScoreFunctions.ScoreFn scoreSeenFn, DefaultScoreFunctions.ScoreFn scoreUnseenFn,
            DefaultScoreFunctions.DampFn dampFn, DefaultScoreFunctions.Normalizer normalizer) {
        return new IVisitorFactory<DiVector>() {
            @Override
            public boolean isReusable() {
                return true;
            }

            @Override
            public boolean isReusableAcrossQueries() {
                return acrossQueries;
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
}
