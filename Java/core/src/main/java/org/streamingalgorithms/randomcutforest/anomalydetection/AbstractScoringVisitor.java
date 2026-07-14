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
import static org.streamingalgorithms.randomcutforest.DefaultScoreFunctions.ScoreFn;

import org.streamingalgorithms.randomcutforest.DefaultScoreFunctions;
import org.streamingalgorithms.randomcutforest.RFVisitor;
import org.streamingalgorithms.randomcutforest.tree.ArrayBox;
import org.streamingalgorithms.randomcutforest.tree.ArrayBoxSimd;
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
public abstract class AbstractScoringVisitor<R> extends RFVisitor<R> {

    protected boolean ignoreLeaf;
    protected int ignoreLeafMassThreshold;

    protected DefaultScoreFunctions.ScoreFn scoreSeenFn;
    protected DefaultScoreFunctions.ScoreFn scoreUnseenFn;
    protected DefaultScoreFunctions.DampFn dampFn;
    protected DefaultScoreFunctions.Normalizer normalizer;

    protected double savedScore;

    /**
     * Set true when a duplicate leaf is reached. The scalar policy never sets this
     * (it converges at the duplicate leaf); the attribution policy sets it and then
     * keeps climbing with the shadow box.
     */
    protected boolean hitDuplicates;

    protected AbstractScoringVisitor(float[] pointToScore, int treeMass, int ignoreLeafMassThreshold,
            DefaultScoreFunctions.ScoreFn scoreSeenFn, DefaultScoreFunctions.ScoreFn scoreUnseenFn,
            DefaultScoreFunctions.DampFn dampFn, DefaultScoreFunctions.Normalizer normalizer) {
        this(pointToScore.length, treeMass, ignoreLeafMassThreshold, scoreSeenFn, scoreUnseenFn, dampFn, normalizer);
        System.arraycopy(pointToScore, 0, this.pointToScore, 0, pointToScore.length);
        ArrayBoxSimd.expandInto(this.pointToScore, this.expandedPoint);
    }

    // reusable path only: allocates pointToScore directly (no copy).
    // prepare(tree, point) fills mass + projection before the first read.
    protected AbstractScoringVisitor(int dimension, int treeMass, int ignoreLeafMassThreshold, ScoreFn scoreSeenFn,
            ScoreFn scoreUnseenFn, DefaultScoreFunctions.DampFn dampFn, DefaultScoreFunctions.Normalizer normalizer) {
        this.pointToScore = new float[dimension];
        this.expandedPoint = new float[2 * dimension];
        this.treeMass = treeMass;
        this.ignoreLeaf = ignoreLeafMassThreshold > DEFAULT_IGNORE_LEAF_MASS_THRESHOLD;
        this.ignoreLeafMassThreshold = ignoreLeafMassThreshold;
        this.scoreSeenFn = scoreSeenFn;
        this.scoreUnseenFn = scoreUnseenFn;
        this.dampFn = dampFn;
        this.normalizer = normalizer;
        this.shadowBox = null;
        setDefaults();
    }

    void setDefaults() {
        savedScore = 0.0;
        pointInsideBox = false;
        shadowBoxActive = false;
        hitDuplicates = false;
    }

    @Override
    public void reset() {
        setDefaults();
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
            probabilityOfSeparation = node.probabilityOfSeparationSimd(expandedPoint, contributionTarget());

            // probabilityOfSeparation = node.probabilityOfSeparation(pointToScore,
            // contributionTarget());
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
        if (shadowBoxActive == false) {
            if (shadowBox == null) {
                shadowBox = sib.copy(); // one-time allocation of the visitor's own reused box
            } else {
                shadowBox.copyFrom(sib);
            }
            shadowBoxActive = true;
        } else {
            shadowBox.addBox(sib); // grow in place, no allocation
        }
        return shadowBox.probabilityOfCutSimd(expandedPoint, contributionTarget());
        // return shadowBox.probabilityOfCut(pointToScore, contributionTarget());
    }
}