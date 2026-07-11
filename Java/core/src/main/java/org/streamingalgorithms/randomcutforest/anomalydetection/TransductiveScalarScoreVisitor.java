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
 *
 * The file is modified substantially from the original which contained this copyright.
 *
 * Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
import java.util.function.BiFunction;
import java.util.function.Function;

import org.streamingalgorithms.randomcutforest.Visitor;
import org.streamingalgorithms.randomcutforest.tree.IBoundingBoxView;
import org.streamingalgorithms.randomcutforest.tree.INodeView;

/**
 * Transductive scalar scoring: score the point as if the tree had been built
 * with knowledge of it, using a pluggable per-dimension separation weighting
 * ({@code vecSep}).
 *
 * <p>
 * <b>Step 1 of the transductive cleanup: a pure flatten.</b> This class is the
 * consolidation of the former three-level chain
 * {@code AbstractScalarScoreVisitor} &rarr; {@code DynamicScoreVisitor} &rarr;
 * {@code TransductiveScalarScoreVisitor}, which had exactly one concrete leaf
 * (this one) after the unified {@code ScoreVisitor} replaced the other users of
 * the dynamic base. The class name, public surface, and numeric behavior are
 * UNCHANGED, so the existing tests ({@code HyperTreeTest} and the transductive
 * {@code getProbabilityOfSeparation} assertion) pass without edits. The two
 * ancestor files can now be deleted.
 *
 * <p>
 * Two members from the old base were genuinely dead in the transductive path
 * and were dropped: the {@code shadowBox} field (only used by the deleted base
 * {@code accept}) and the RCF {@code getProbabilityOfSeparation} (shadowed by
 * the transductive override below). The {@code ignoreLeafEquals} /
 * {@code ignoreLeafMassThreshold} fields are retained and wired exactly as the
 * old chain wired them (threshold hard-coded to 0, so {@code ignoreLeafEquals}
 * is always false) to keep {@code acceptLeaf} byte-for-byte identical; they are
 * slated for removal in step 2, together with the rename to
 * {@code TransductiveScoreVisitor}.
 */
public class TransductiveScalarScoreVisitor implements Visitor<Double> {

    protected final float[] pointToScore;
    protected final int treeMass;

    protected boolean pointInsideBox;
    protected boolean[] coordInsideBox;
    protected double score;

    /** Retained, always false here (threshold is 0); removed in step 2. */
    protected boolean ignoreLeafEquals;
    protected int ignoreLeafMassThreshold;

    protected final BiFunction<Double, Double, Double> scoreSeenFn;
    protected final BiFunction<Double, Double, Double> scoreUnseenFn;
    protected final BiFunction<Double, Double, Double> dampFn;

    protected final Function<IBoundingBoxView, double[]> vecSepScore;

    /**
     * @param pointToScore the point whose anomaly score we are computing
     * @param treeMass     the total mass of the RandomCutTree scoring the point
     * @param scoreSeen    part of the score when the point matches a leaf
     * @param scoreUnseen  part of the score when the point does not match
     * @param damp         dampening of the effect of the seen points
     * @param vecSep       per-dimension separation weighting; the build-side and
     *                     score-side weightings are identical for this visitor
     */
    public TransductiveScalarScoreVisitor(float[] pointToScore, int treeMass,
            BiFunction<Double, Double, Double> scoreSeen, BiFunction<Double, Double, Double> scoreUnseen,
            BiFunction<Double, Double, Double> damp, Function<IBoundingBoxView, double[]> vecSep) {
        this.pointToScore = Arrays.copyOf(pointToScore, pointToScore.length);
        this.treeMass = treeMass;
        this.pointInsideBox = false;
        this.score = 0.0;
        this.ignoreLeafMassThreshold = 0;
        this.ignoreLeafEquals = (0 > DEFAULT_IGNORE_LEAF_MASS_THRESHOLD); // false
        this.coordInsideBox = new boolean[pointToScore.length];
        this.scoreSeenFn = scoreSeen;
        this.scoreUnseenFn = scoreUnseen;
        this.dampFn = damp;
        this.vecSepScore = vecSep;
        // build weighting is the same as scoring weighting for this visitor
    }

    /** Scores are not normalized: the function ranges are unknown. */
    @Override
    public Double getResult() {
        return score;
    }

    @Override
    public void accept(INodeView node, int depthOfNode) {
        if (pointInsideBox) {
            return;
        }
        // note: score is unchanged before the return. This is only reasonable when the
        // scoring uses the same probability function used to build the trees.

        double probabilityOfSeparation = getProbabilityOfSeparation(node.getBoundingBox());
        double weight = getWeight(node.getCutDimension(), vecSepScore, node.getBoundingBox());
        if (probabilityOfSeparation == 0) {
            pointInsideBox = true;
            return;
        }

        score = probabilityOfSeparation * scoreUnseen(depthOfNode, node.getMass()) + weight * score;
    }

    @Override
    public void acceptLeaf(INodeView leafNode, int depthOfNode) {
        if (Arrays.equals(leafNode.getLeafPoint(), pointToScore)
                && (!ignoreLeafEquals || (leafNode.getMass() > ignoreLeafMassThreshold))) {
            pointInsideBox = true;
            score = damp(leafNode.getMass(), treeMass) * scoreSeen(depthOfNode, leafNode.getMass());
        } else {
            score = scoreUnseen(depthOfNode, leafNode.getMass());
        }
    }

    protected double scoreSeen(int depth, int leafMass) {
        return scoreSeenFn.apply((double) depth, (double) leafMass);
    }

    protected double scoreUnseen(int depth, int leafMass) {
        return scoreUnseenFn.apply((double) depth, (double) leafMass);
    }

    protected double damp(int leafMass, int treeMass) {
        return dampFn.apply((double) leafMass, (double) treeMass);
    }

    /**
     * Transductive separation probability: each dimension's contribution is
     * weighted by {@code vecSep} rather than raw range. Throws on a leaf-degenerate
     * box (where the weighting sums to zero), as before.
     */
    protected double getProbabilityOfSeparation(final IBoundingBoxView boundingBox) {
        double sumOfDenominator = 0d;
        double sumOfNumerator = 0d;

        double[] vec = vecSepScore.apply(boundingBox.getMergedBox(pointToScore));

        for (int i = 0; i < pointToScore.length; ++i) {
            double maxVal = boundingBox.getMaxValue(i);
            double minVal = boundingBox.getMinValue(i);
            double oldRange = maxVal - minVal;
            sumOfDenominator += vec[i];
            if (!coordInsideBox[i]) {
                if (maxVal < pointToScore[i]) {
                    maxVal = pointToScore[i];
                } else if (minVal > pointToScore[i]) {
                    minVal = pointToScore[i];
                }

                double newRange = maxVal - minVal;
                if (newRange > oldRange) {
                    sumOfNumerator += vec[i] * (newRange - oldRange) / newRange;
                } else {
                    coordInsideBox[i] = true;
                }
            }
        }

        if (sumOfDenominator <= 0) {
            // range should only sum to 0 at a leaf; non-leaf nodes contain > 1 distinct
            // point
            throw new IllegalStateException("Incorrect State");
        }
        return sumOfNumerator / sumOfDenominator;
        // For RCFs vec[i] == newRange, so this reduces to the L1 side/sum ratio.
    }

    /**
     * Weight applied to the running score. The build-side weighting function is
     * passed in so the simulated variant can supply a different one than the
     * scoring-side {@link #vecSepScore}; here they are the same.
     */
    protected double getWeight(int dim, Function<IBoundingBoxView, double[]> vecSepBuild,
            final IBoundingBoxView boundingBox) {
        double[] vecSmall = vecSepBuild.apply(boundingBox); // the smaller box was built
        IBoundingBoxView largeBox = boundingBox.getMergedBox(pointToScore);
        double[] vecLarge = vecSepScore.apply(largeBox); // the larger box is only scored
        double sumSmall = 0;
        double sumLarge = 0;
        for (int i = 0; i < pointToScore.length; i++) {
            sumSmall += vecSmall[i];
            sumLarge += vecLarge[i];
        }
        return (boundingBox.getRange(dim) / largeBox.getRange(dim)) * (sumSmall / sumLarge)
                * (vecLarge[dim] / vecSmall[dim]);
        // can exceed 1; sumSmall/sumLarge is the probability of non-separation
    }

    @Override
    public boolean isConverged() {
        return pointInsideBox;
    }
}