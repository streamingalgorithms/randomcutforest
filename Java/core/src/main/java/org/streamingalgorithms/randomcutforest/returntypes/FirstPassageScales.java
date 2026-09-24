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

import static java.lang.Math.log;
import static org.streamingalgorithms.randomcutforest.CommonUtils.checkArgument;
import static org.streamingalgorithms.randomcutforest.CommonUtils.checkNotNull;

import java.util.Arrays;

/**
 * Three recorded boxes, summed across trees. The visitor chooses the crossings;
 * this class only stores and merges outputs. All extents use double[2*d], high
 * followed by low, and all three boxes divide by the same tree count.
 */
public class FirstPassageScales {
    private final int dimensions;
    private final int len;
    private final double[] cutSum;
    private final double[] passageSum;
    private final double[] stopSum;
    private int treeCount;
    private final double[] gapSum; // Σ m over trees with positive width, per half-axis

    public FirstPassageScales(int dimensions) {
        checkArgument(dimensions > 0, "dimensions must be greater than 0");
        this.dimensions = dimensions;
        len = 2 * dimensions;
        cutSum = new double[len];
        passageSum = new double[len];
        stopSum = new double[len];
        gapSum = new double[len];
    }

    public FirstPassageScales(FirstPassageScales base) {
        this(base.dimensions);
        System.arraycopy(base.cutSum, 0, cutSum, 0, len);
        System.arraycopy(base.passageSum, 0, passageSum, 0, len);
        System.arraycopy(base.stopSum, 0, stopSum, 0, len);
        System.arraycopy(base.gapSum, 0, gapSum, 0, len);
        treeCount = base.treeCount;
    }

    public int getDimensions() {
        return dimensions;
    }

    public int getTreeCount() {
        return treeCount;
    }

    public void clear() {
        Arrays.fill(cutSum, 0.0);
        Arrays.fill(passageSum, 0.0);
        Arrays.fill(stopSum, 0.0);
        Arrays.fill(gapSum, 0.0);
        treeCount = 0;
    }

    public void observeTree(double[] cut, double[] passage, double[] stop, double[] gap, double mass) {
        for (int j = 0; j < len; j++) {
            passageSum[j] += passage[j];
            stopSum[j] += stop[j];
            double weight = 1.0 / Math.pow(Math.max(1.0, mass), 1.0 / dimensions);
            // Mass-weighted harmonic on the cut box. Both accumulators are sums, so
            // addToLeft stays a plain add and the division stays at retrieval.
            // A zero width is skipped rather than contributing 1/0: it would send
            // the sum to infinity and zero that axis for the whole forest.
            cutSum[j] += cut[j];
            gapSum[j] += gap[j];
        }
        treeCount++;
    }

    public double[] gapBox() {
        return perTree(gapSum);
    }

    public double[] cutBox() {
        return perTree(cutSum);
    }

    public double[] passageBox() {
        return perTree(passageSum);
    }

    public double[] stopBox() {
        return perTree(stopSum);
    }

    private double[] perTree(double[] sum) {
        double[] out = new double[len];
        for (int j = 0; j < len; j++) {
            out[j] = treeCount > 0 ? sum[j] / treeCount : 0.0;
        }
        return out;
    }

    /** Compatibility name: volume of the averaged box, NOT mean per-tree volume. */
    public double meanVolume(boolean cut) {
        return volume(cut ? cutSum : passageSum);
    }

    public double stopVolume() {
        return volume(stopSum);
    }

    private double volume(double[] sum) {
        if (treeCount == 0) {
            return 0.0;
        }

        double volume = 1.0;
        int active = 0;
        for (int i = 0; i < dimensions; i++) {
            double width = (sum[i] + sum[i + dimensions]) / treeCount;
            if (width > 0.0) {
                volume *= width;
                active++;
            }
        }
        return (active == 0) ? 0.0 : volume;

    }

    public static FirstPassageScales addToLeft(FirstPassageScales left, FirstPassageScales right) {
        checkNotNull(left, "left must not be null");
        checkNotNull(right, "right must not be null");
        checkArgument(left.dimensions == right.dimensions, "dimensions must be the same");
        for (int j = 0; j < left.len; j++) {
            left.cutSum[j] += right.cutSum[j];
            left.passageSum[j] += right.passageSum[j];
            left.stopSum[j] += right.stopSum[j];
            left.gapSum[j] += right.gapSum[j];
        }
        left.treeCount += right.treeCount;
        return left;
    }

    public void addRatios(double[] out) {
        if (out == null) {
            return;
        }
        checkArgument(out.length == 3, "expected three ratios");
        // Per query, the MEAN log ratio over the axes that have one, so the
        // caller's divisor is the query count rather than an axis count it would
        // have to track. treeCount cancels in every ratio, which is why the raw
        // sums are used rather than the per-tree boxes.
        double[] acc = new double[3];
        int[] terms = new int[3];
        for (int i = 0; i < dimensions; i++) {
            double g = gapSum[i] + gapSum[i + dimensions];
            double c = cutSum[i] + cutSum[i + dimensions];
            double p = passageSum[i] + passageSum[i + dimensions];
            double s = stopSum[i] + stopSum[i + dimensions];
            if (c > 0 && g > 0) {
                acc[0] += log(c / g);
                terms[0]++;
            }
            if (p > 0 && c > 0) {
                acc[1] += log(p / c);
                terms[1]++;
            }
            if (s > 0 && p > 0) {
                acc[2] += log(s / p);
                terms[2]++;
            }
        }
        for (int k = 0; k < 3; k++) {
            if (terms[k] > 0) {
                out[k] += acc[k] / terms[k];
            }
        }
    }
}
