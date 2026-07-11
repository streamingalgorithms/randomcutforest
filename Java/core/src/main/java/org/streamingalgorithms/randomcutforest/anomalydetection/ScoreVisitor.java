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
import org.streamingalgorithms.randomcutforest.tree.INodeView;

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
        this(pointToScore, treeMass, ignoreLeafMassThreshold, DefaultScoreFunctions.DEFAULT_SCORE_SEEN,
                DefaultScoreFunctions.DEFAULT_SCORE_UNSEEN, DefaultScoreFunctions.DEFAULT_DAMP,
                DefaultScoreFunctions.DEFAULT_NORMALIZER);
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
}
