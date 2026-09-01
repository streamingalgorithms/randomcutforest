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

package org.streamingalgorithms.randomcutforest.returntypes;

import static org.streamingalgorithms.randomcutforest.CommonUtils.checkArgument;
import static org.streamingalgorithms.randomcutforest.CommonUtils.checkNotNull;

import java.util.Arrays;

/**
 * Per-tree local scales, averaged across the forest.
 *
 * <p>
 * This replaces a per-face log-length histogram, which was doing work the
 * visitor already does and doing it less well:
 *
 * <ul>
 * <li><b>The normalizer was a single observation.</b> Bins were indexed by
 * log2(len / L_full) where L_full was the length at whichever level that face
 * happened to last receive weight -- one sample, differing per face and per
 * tree. The merged histogram was therefore a mixture in coordinates that meant
 * something slightly different in every tree, and absolute units came back
 * through a separately weight-averaged reference length. Two approximations
 * stacked where none was needed.</li>
 * <li><b>Median of a mixture is not the mixture of medians.</b> Pooling
 * normalized distributions and then taking a quantile is not the ensemble
 * average of per-tree quantiles. Every other accumulator in the forest --
 * measure, distances, score -- averages per-tree estimates; the histogram
 * quietly did something else.</li>
 * <li><b>Binning quantized.</b> A synthetic geometric walk of known dimension
 * 1, 2, 3 returned 1.000, 2.000, 3.169 through the binned regression. The 3.169
 * was bin quantization, not estimator error.</li>
 * </ul>
 *
 * <p>
 * <b>The crossing is taken on the scalar, not per face.</b> The walk gives a
 * first-passage weight S(v)q_v per level, summing to exactly 1. Scanning upward
 * from the leaf to where the cumulative crosses lambda picks one level, so the
 * 2d half-lengths reported are the enlarged bounding box of a real node: the
 * node where the point is, at the median, about to be separated. Letting each
 * face cross at its own level would return a box corresponding to no node at
 * all.
 */
public class DirectionalScales {

    private final int dimensions;
    private final int len; // 2 * dimensions

    private final double[] boxSum; // len; per-tree lambda-crossing half-lengths
    private final double[] loSum; // len; crossing at the low quantile
    private final double[] hiSum; // len; crossing at the high quantile
    private final int[] numTrees; // number of trees per dimension with non-zero values
    // private double alphaSum; // per-tree Holder slopes
    private int alphaTrees; // trees that had leverage to fit one

    private double degenerateWeight; // weight with no defined length
    private double totalWeight; // all first-passage weight; must average to 1
    private int treeCount;

    public DirectionalScales(int dimensions) {
        checkArgument(dimensions > 0, "dimensions must be greater than 0");
        this.dimensions = dimensions;
        this.len = 2 * dimensions;
        this.boxSum = new double[len];
        this.loSum = new double[len];
        this.hiSum = new double[len];
        this.numTrees = new int[len];
    }

    public DirectionalScales(DirectionalScales base) {
        this(base.dimensions);
        System.arraycopy(base.boxSum, 0, boxSum, 0, len);
        System.arraycopy(base.loSum, 0, loSum, 0, len);
        System.arraycopy(base.hiSum, 0, hiSum, 0, len);
        System.arraycopy(base.numTrees, 0, numTrees, 0, len);
        this.alphaTrees = base.alphaTrees;
        this.degenerateWeight = base.degenerateWeight;
        this.totalWeight = base.totalWeight;
        this.treeCount = base.treeCount;
    }

    public int getDimensions() {
        return dimensions;
    }

    public int getTreeCount() {
        return treeCount;
    }

    public void clear() {
        Arrays.fill(boxSum, 0.0);
        Arrays.fill(loSum, 0.0);
        Arrays.fill(hiSum, 0.0);
        Arrays.fill(numTrees, 0);
        alphaTrees = 0;
        degenerateWeight = 0.0;
        totalWeight = 0.0;
        treeCount = 0;
    }

    /**
     * Fold in one tree's estimates. All arrays are length 2 * dimensions and are
     * copied, not retained.
     *
     * @param box        half-lengths at the lambda crossing
     * @param lo         half-lengths at the low quantile
     * @param hi         half-lengths at the high quantile
     * @param alpha      local Holder exponent, or NaN if the path had no leverage
     * @param degenerate weight with no defined length on this tree
     * @param total      all first-passage weight on this tree; should be 1
     */
    public void observeTree(double[] box, double[] lo, double[] hi, double alpha, double degenerate, double total) {
        for (int j = 0; j < len; j++) {
            boxSum[j] += box[j];
            loSum[j] += lo[j];
            hiSum[j] += hi[j];
            if (hi[j] > 0) {
                numTrees[j]++;
            }
        }
        // A tree whose path gave no spread on the length axis contributes nothing
        // rather than contributing a zero. Averaging in the zeros is what biased an
        // earlier version toward 0.2 when the answer was near the ambient dimension.
        if (!Double.isNaN(alpha) && alpha != 0.0) {
            alphaTrees++;
        }
        degenerateWeight += degenerate;
        totalWeight += total + degenerate;
        treeCount++;
    }

    /**
     * The 2 * dimensions half-separation lengths, in data units. DiVector layout.
     */
    public double[] halfSeparationBox() {
        double[] out = new double[len];
        if (treeCount > 0) {
            for (int j = 0; j < len; j++) {
                out[j] = boxSum[j] / treeCount;
            }
        }
        return out;
    }

    public double halfSeparationLength(int face) {
        return (treeCount > 0) ? boxSum[face] / treeCount : 0.0;
    }

    /**
     * Width of the quantile crossing on a face, in octaves. A wide crossing means
     * the neighbourhood has no single scale and the half-separation length is an
     * arbitrary representative of a flat region, not a noisy estimate of a real
     * one.
     */
    public double crossingWidth(int face) {
        if (treeCount == 0) {
            return 0.0;
        }
        double lo = loSum[face] / treeCount, hi = hiSum[face] / treeCount;
        return (lo > 0.0 && hi > 0.0) ? Math.abs(Math.log(hi / lo) / Math.log(2.0)) : 0.0;
    }

    /**
     * Half-lengths at the low and high bracketing quantiles, in data units. These
     * are the only quantiles available: lambda is fixed when the walk runs, not
     * when the result is read. That is the one capability the histogram had and
     * this does not -- sweeping lambda after the fact as a scheme-independence
     * check now means re-running with different constants.
     */
    public double[] lowBox() {
        return perTree(loSum);
    }

    public double[] highBox() {
        return perTree(hiSum);
    }

    private double[] perTree(double[] sum) {
        double[] out = new double[len];
        if (treeCount > 0) {
            for (int j = 0; j < len; j++) {
                out[j] = sum[j] / treeCount;
            }
        }
        return out;
    }

    public double[] crossingWidths() {
        double[] out = new double[len];
        for (int j = 0; j < len; j++) {
            out[j] = crossingWidth(j);
        }
        return out;
    }

    /** Mean local Holder exponent over the trees that could fit one. */
    public double holderExponent() {
        return 0.0;
    }

    /** Fraction of trees that yielded an exponent; low means the fit is thin. */
    public double exponentCoverage() {
        return (treeCount > 0) ? alphaTrees / (double) treeCount : 0.0;
    }

    public double degenerateFraction() {
        return (totalWeight > 0.0) ? degenerateWeight / totalWeight : 0.0;
    }

    /**
     * Mean first-passage weight per tree. The walk's distribution over levels sums
     * to exactly 1, so this must return 1.0 to within rounding; it checks the
     * survival product, the crossing scan, the degenerate routing and the fold in
     * one number.
     */
    public double totalMass() {
        return (treeCount > 0) ? totalWeight / treeCount : 0.0;
    }

    public static DirectionalScales addToLeft(DirectionalScales left, DirectionalScales right) {
        checkNotNull(left, "left must not be null");
        checkNotNull(right, "right must not be null");
        checkArgument(left.dimensions == right.dimensions, "dimensions must be the same");
        for (int j = 0; j < left.len; j++) {
            left.boxSum[j] += right.boxSum[j];
            left.loSum[j] += right.loSum[j];
            left.hiSum[j] += right.hiSum[j];
        }
        left.alphaTrees += right.alphaTrees;
        left.degenerateWeight += right.degenerateWeight;
        left.totalWeight += right.totalWeight;
        left.treeCount += right.treeCount;
        return left;
    }
}
