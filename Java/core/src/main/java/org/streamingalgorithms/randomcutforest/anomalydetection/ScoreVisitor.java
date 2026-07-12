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

import static org.streamingalgorithms.randomcutforest.DefaultScoreFunctions.*;

import java.util.Arrays;

import org.streamingalgorithms.randomcutforest.DefaultScoreFunctions;
import org.streamingalgorithms.randomcutforest.IRFVisitor;
import org.streamingalgorithms.randomcutforest.IVisitorFactory;
import org.streamingalgorithms.randomcutforest.Visitor;
import org.streamingalgorithms.randomcutforest.tree.INodeView;
import org.streamingalgorithms.randomcutforest.tree.ITree;

/**
 * Scalar anomaly-score visitor. All traversal scaffolding lives in
 * {@link AbstractScoringVisitor}; this class supplies only the scalar-specific
 * pieces:
 *
 * <ul>
 * <li>{@link #contributionTarget()} returns {@code null} — probability only, no
 * per-coordinate attribution.</li>
 * <li>{@link #updateFromNode} does a single weighted update. No separate
 * ignoreLeaf accounting block; that block in the earlier standalone version was
 * a mis-port from the attribution visitor and produced a double update.</li>
 * <li>{@code acceptLeaf} CONVERGES on a duplicate leaf.</li>
 * </ul>
 */
public class ScoreVisitor extends AbstractScoringVisitor<Double> {

    // package-private: reusable factory only. Unarmed (treeMass 0); first
    // reusableTraverse calls prepare(tree, point) before anything reads state.
    protected ScoreVisitor(int dimension) {
        super(dimension, 0, DEFAULT_IGNORE_LEAF_MASS_THRESHOLD, DEFAULT_SCORE_SEEN, DEFAULT_SCORE_UNSEEN, DEFAULT_DAMP,
                DEFAULT_NORMALIZER);
    }

    protected ScoreVisitor(int dimension, int treeMass, int ignoreLeafMassThreshold,
            DefaultScoreFunctions.ScoreFn scoreSeenFn, DefaultScoreFunctions.ScoreFn scoreUnseenFn,
            DefaultScoreFunctions.DampFn dampFn, DefaultScoreFunctions.Normalizer normalizer) {
        super(dimension, treeMass, ignoreLeafMassThreshold, scoreSeenFn, scoreUnseenFn, dampFn, normalizer);
    }

    protected ScoreVisitor(float[] pointToScore, int treeMass, int ignoreLeafMassThreshold,
            DefaultScoreFunctions.ScoreFn scoreSeenFn, DefaultScoreFunctions.ScoreFn scoreUnseenFn,
            DefaultScoreFunctions.DampFn dampFn, DefaultScoreFunctions.Normalizer normalizer) {
        super(pointToScore, treeMass, ignoreLeafMassThreshold, scoreSeenFn, scoreUnseenFn, dampFn, normalizer);
    }

    public ScoreVisitor(float[] pointToScore, int treeMass, int ignoreLeafMassThreshold,
            DefaultScoreFunctions.ScoreFn scoreSeenFn, DefaultScoreFunctions.ScoreFn scoreUnseenFn,
            DefaultScoreFunctions.DampFn dampFn) {
        this(pointToScore, treeMass, ignoreLeafMassThreshold, scoreSeenFn, scoreUnseenFn, dampFn,
                DefaultScoreFunctions.NO_NORMALIZER);
    }

    public ScoreVisitor(float[] pointToScore, int treeMass, int ignoreLeafMassThreshold) {
        this(pointToScore, treeMass, ignoreLeafMassThreshold, DEFAULT_SCORE_SEEN, DEFAULT_SCORE_UNSEEN,
                DefaultScoreFunctions.DEFAULT_DAMP, DefaultScoreFunctions.DEFAULT_NORMALIZER);
    }

    public ScoreVisitor(float[] pointToScore, int treeMass) {
        this(pointToScore, treeMass, DEFAULT_IGNORE_LEAF_MASS_THRESHOLD);
    }

    @Override
    protected float[] contributionTarget() {
        return null;
    }

    @Override
    protected void updateFromNode(double prob, int depth, int mass) {
        savedScore = prob * scoreUnseenFn.of(depth, mass) + (1 - prob) * savedScore;
    }

    // ScoreVisitor — add fields
    private double foldedScore; // per-query sum; NOT reset per tree

    @Override
    public void acceptLeaf(INodeView leafNode, int depthOfNode) {
        if (Arrays.equals(leafNode.getLeafPoint(), pointToScore)
                && (!ignoreLeaf || (leafNode.getMass() > ignoreLeafMassThreshold))) {
            pointInsideBox = true;
            savedScore = dampFn.of(leafNode.getMass(), treeMass) * scoreSeenFn.of(depthOfNode, leafNode.getMass());
        } else {
            savedScore = scoreUnseenFn.of(depthOfNode, leafNode.getMass());
        }
    }

    @Override
    public Double getResult() {
        return savedScore * normalizer.scale(treeMass);
    }

    // fold: per tree, add this tree's normalized score into the running sum
    public void foldOut() {
        foldedScore += savedScore * normalizer.scale(treeMass);
    }

    @Override
    public Double getFoldResult() {
        // per-query — fold path only
        return foldedScore;
    }

    @Override
    public void reset() {
        super.reset(); // savedScore=0, pointInsideBox=false, shadowBoxActive=false,
                       // hitDuplicates=false
        // foldedScore deliberately NOT cleared — accumulates across trees
        // this function is purely informational placeholder
    }

    @Override
    public double convergingValue() {
        return savedScore * normalizer.scale(treeMass);
    }

    public static IVisitorFactory<Double> reusableFactory(int ignoreLeafMassThreshold, ScoreFn scoreSeenFn,
            ScoreFn scoreUnseenFn, DampFn dampFn, Normalizer normalizer) {
        return new IVisitorFactory<Double>() {
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
            public IRFVisitor<Double> newReusableVisitor(float[] point) {
                return new ScoreVisitor(point.length, 0, ignoreLeafMassThreshold, scoreSeenFn, scoreUnseenFn, dampFn,
                        normalizer); // sized, unarmed; no copy
            }

            @Override
            public Visitor<Double> newVisitor(ITree<?, ?> tree, float[] point) {
                return new ScoreVisitor(tree.projectToTree(point), tree.getMass(), ignoreLeafMassThreshold, scoreSeenFn,
                        scoreUnseenFn, dampFn, normalizer);
            }
        };
    }
}
