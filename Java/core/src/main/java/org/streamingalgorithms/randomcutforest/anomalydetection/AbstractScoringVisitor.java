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
import org.streamingalgorithms.randomcutforest.Visitor;
import org.streamingalgorithms.randomcutforest.tree.ArrayBox;
import org.streamingalgorithms.randomcutforest.tree.INodeView;

/**
 * Shared traversal scaffolding for the scalar and attribution scoring visitors.
 * The two concrete visitors differ in only four places, expressed as the four
 * abstract methods below:
 *
 * <ul>
 * <li>{@link #contributionTarget()} — {@code null} for the scalar visitor
 * (probability only), the per-half-dimension components buffer for the
 * attribution visitor. This is the single {@code probabilityOfCut(point, null)}
 * vs {@code probabilityOfCut(point, vector)} seam; both take the identical
 * arithmetic path in {@code ArrayBox.gapAttribution}.</li>
 * <li>{@link #updateFromNode(double, int, int)} — the accumulator update, and
 * on the shadow path the {@code prob <= 0} convergence. Scalar keeps one scalar
 * accumulator; attribution keeps a scalar total (savedScore) and a directional
 * vector, which is why its update looks like the removed scalar "double
 * compute" but is not — the two writes hit different accumulators.</li>
 * <li>{@code acceptLeaf} — scalar CONVERGES at a duplicate leaf; attribution
 * sets {@code hitDuplicates} and keeps climbing. Genuinely different
 * traversals, so it stays per-subclass.</li>
 * <li>{@code getResult} — {@code Double} vs {@code DiVector}; the reason these
 * cannot be parent/child (conflicting {@code Visitor<R>} type arguments) and
 * must instead be siblings under this base.</li>
 * </ul>
 */
public abstract class AbstractScoringVisitor<R> implements Visitor<R> {

    protected final float[] pointToScore;
    protected final int treeMass;
    protected final boolean ignoreLeaf;
    protected final int ignoreLeafMassThreshold;

    protected final DefaultScoreFunctions.ScoreFn scoreSeenFn;
    protected final DefaultScoreFunctions.ScoreFn scoreUnseenFn;
    protected final DefaultScoreFunctions.DampFn dampFn;
    protected final DefaultScoreFunctions.Normalizer normalizer;

    protected double savedScore;
    protected boolean pointInsideBox;

    /**
     * Set true when a duplicate leaf is reached. The scalar policy never sets this
     * (it converges at the duplicate leaf); the attribution policy sets it and then
     * keeps climbing with the shadow box.
     */
    protected boolean hitDuplicates;

    /** Reused counterfactual box; allocated once, only on the shadow path. */
    protected ArrayBox shadowBox;

    protected AbstractScoringVisitor(float[] pointToScore, int treeMass, int ignoreLeafMassThreshold,
            DefaultScoreFunctions.ScoreFn scoreSeenFn, DefaultScoreFunctions.ScoreFn scoreUnseenFn,
            DefaultScoreFunctions.DampFn dampFn, DefaultScoreFunctions.Normalizer normalizer) {
        this.pointToScore = Arrays.copyOf(pointToScore, pointToScore.length);
        this.treeMass = treeMass;
        this.ignoreLeaf = ignoreLeafMassThreshold > DEFAULT_IGNORE_LEAF_MASS_THRESHOLD;
        this.ignoreLeafMassThreshold = ignoreLeafMassThreshold;
        this.scoreSeenFn = scoreSeenFn;
        this.scoreUnseenFn = scoreUnseenFn;
        this.dampFn = dampFn;
        this.normalizer = normalizer;
        this.savedScore = 0.0;
        this.pointInsideBox = false;
        this.hitDuplicates = false;
        this.shadowBox = null;
    }

    /**
     * {@code null} => probability only (scalar); components buffer => attribution.
     */
    protected abstract float[] contributionTarget();

    /**
     * Fold this node's separation probability into the running result. Owns the
     * accumulator math and, on the shadow path, the {@code prob <= 0} convergence.
     * On the non-shadow path {@code prob <= 0} is already handled by
     * {@link #accept}.
     */
    protected abstract void updateFromNode(double probabilityOfSeparation, int depth, int mass);

    @Override
    public final void accept(INodeView node, int depthOfNode) {
        if (pointInsideBox) {
            return;
        }

        double probabilityOfSeparation;
        if (!(hitDuplicates || ignoreLeaf)) {
            probabilityOfSeparation = node.probabilityOfSeparation(pointToScore, contributionTarget());
            if (probabilityOfSeparation <= 0) {
                // point inside this node's box: nothing above can change the result
                pointInsideBox = true;
                return;
            }
        } else {
            // counterfactual "what if the point and its near neighbor were absent"
            probabilityOfSeparation = acceptShadowBox(node);
        }

        updateFromNode(probabilityOfSeparation, depthOfNode, node.getMass());
    }

    protected final double acceptShadowBox(INodeView node) {
        ArrayBox sib = (ArrayBox) node.getSiblingBoundingBox(pointToScore);
        if (shadowBox == null) {
            shadowBox = sib.copy(); // one-time allocation of the visitor's own reused box
        } else {
            shadowBox.addBox(sib); // grow in place, no allocation
        }
        return shadowBox.probabilityOfCut(pointToScore, contributionTarget());
    }

    @Override
    public final boolean isConverged() {
        return pointInsideBox;
    }
}