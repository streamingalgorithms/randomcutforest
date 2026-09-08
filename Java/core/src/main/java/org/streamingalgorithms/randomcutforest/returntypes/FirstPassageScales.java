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

    public FirstPassageScales(int dimensions) {
        checkArgument(dimensions > 0, "dimensions must be greater than 0");
        this.dimensions = dimensions;
        len = 2 * dimensions;
        cutSum = new double[len];
        passageSum = new double[len];
        stopSum = new double[len];
    }

    public FirstPassageScales(FirstPassageScales base) {
        this(base.dimensions);
        System.arraycopy(base.cutSum, 0, cutSum, 0, len);
        System.arraycopy(base.passageSum, 0, passageSum, 0, len);
        System.arraycopy(base.stopSum, 0, stopSum, 0, len);
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
        treeCount = 0;
    }

    public void observeTree(double[] cut, double[] passage, double[] stop) {
        checkArgument(cut.length == len && passage.length == len && stop.length == len,
                "box lengths must equal 2 * dimensions");
        for (int j = 0; j < len; j++) {
            cutSum[j] += cut[j];
            passageSum[j] += passage[j];
            stopSum[j] += stop[j];
        }
        treeCount++;
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

    public double cutLength(int face) {
        return treeCount > 0 ? cutSum[face] / treeCount : 0.0;
    }

    public double passageLength(int face) {
        return treeCount > 0 ? passageSum[face] / treeCount : 0.0;
    }

    public double stopLength(int face) {
        return treeCount > 0 ? stopSum[face] / treeCount : 0.0;
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
        for (int i = 0; i < dimensions; i++) {
            double width = (sum[i] + sum[i + dimensions]) / treeCount;
            if (!(width > 0.0)) {
                return 0.0;
            }
            volume *= width;
        }
        return volume;
    }

    public static FirstPassageScales addToLeft(FirstPassageScales left, FirstPassageScales right) {
        checkNotNull(left, "left must not be null");
        checkNotNull(right, "right must not be null");
        checkArgument(left.dimensions == right.dimensions, "dimensions must be the same");
        for (int j = 0; j < left.len; j++) {
            left.cutSum[j] += right.cutSum[j];
            left.passageSum[j] += right.passageSum[j];
            left.stopSum[j] += right.stopSum[j];
        }
        left.treeCount += right.treeCount;
        return left;
    }
}
