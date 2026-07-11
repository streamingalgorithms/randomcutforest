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
import org.streamingalgorithms.randomcutforest.returntypes.DiVector;
import org.streamingalgorithms.randomcutforest.tree.ArrayBox;
import org.streamingalgorithms.randomcutforest.tree.INodeView;

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

    protected AttributionVisitor(float[] pointToScore, int treeMass, int ignoreLeafMassThreshold,
            DefaultScoreFunctions.ScoreFn scoreSeenFn, DefaultScoreFunctions.ScoreFn scoreUnseenFn,
            DefaultScoreFunctions.DampFn dampFn, DefaultScoreFunctions.Normalizer normalizer) {
        super(pointToScore, treeMass, ignoreLeafMassThreshold, scoreSeenFn, scoreUnseenFn, dampFn, normalizer);
        int dimension = pointToScore.length;
        // twice as long as the point: positive and negative gaps stored separately.
        this.probabilityComponents = new float[2 * dimension];
        this.directionalAttribution = new double[2 * dimension];
    }

    public AttributionVisitor(float[] pointToScore, int treeMass, int ignoreLeafMassThreshold,
            DefaultScoreFunctions.ScoreFn scoreSeenFn, DefaultScoreFunctions.ScoreFn scoreUnseenFn,
            DefaultScoreFunctions.DampFn dampFn) {
        this(pointToScore, treeMass, ignoreLeafMassThreshold, scoreSeenFn, scoreUnseenFn, dampFn,
                DefaultScoreFunctions.NO_NORMALIZER);
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
            // probabilityComponents already sum to prob (normalized in gapAttribution).
            for (int i = 0; i < directionalAttribution.length; i++) {
                directionalAttribution[i] = probabilityComponents[i] * newScore
                        + (1 - prob) * directionalAttribution[i];
            }
        }
    }

    @Override
    public void acceptLeaf(INodeView leafNode, int depthOfNode) {
        // NOTE: one ArrayBox per visitor lifetime (a visitor makes exactly one
        // acceptLeaf call). To eliminate it entirely, expose a reused leaf ArrayBox
        // from INodeView the way getSiblingBoundingBox already reuses the sibling box.
        ArrayBox leafBox = new ArrayBox(leafNode.getLeafPoint());
        double probability = leafBox.probabilityOfCut(pointToScore, probabilityComponents);
        if (probability <= 0) {
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
            for (int i = 0; i < probabilityComponents.length; i++) {
                directionalAttribution[i] = savedScore * probabilityComponents[i];
            }
        }
    }

    @Override
    public DiVector getResult() {
        double factor = normalizer.scale(treeMass);
        if (savedScore > 0) {
            if (hitDuplicates || ignoreLeaf) {
                double sum = 0;
                for (int i = 0; i < directionalAttribution.length; i++) {
                    sum += directionalAttribution[i];
                }
                factor *= savedScore / sum;
            }
        } else {
            factor = 0;
        }
        for (int i = 0; i < directionalAttribution.length; i++) {
            directionalAttribution[i] *= factor;
        }
        return new DiVector(directionalAttribution);
    }

    public DiVector observeResult() {
        double factor = normalizer.scale(treeMass);
        double[] out = Arrays.copyOf(directionalAttribution, directionalAttribution.length);
        if (hitDuplicates || ignoreLeaf) {
            double sum = 0;
            for (int i = 0; i < out.length; i++) {
                sum += out[i];
            }
            factor *= savedScore / sum;
        }
        for (int i = 0; i < out.length; i++) {
            out[i] *= factor;
        }
        return new DiVector(out);
    }
}
