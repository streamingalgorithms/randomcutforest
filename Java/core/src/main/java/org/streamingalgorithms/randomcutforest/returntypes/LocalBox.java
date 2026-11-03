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

import java.util.Arrays;

/**
 * One box of an {@link AnisotropicLocalGeometry}, averaged over trees.
 *
 * <p>
 * <b>Half dimensions.</b> A point v is written as [v, -v] and a box as the
 * bound b = [max, -min], so that the box is the set of v with [v, -v] &le; b.
 * Half dimension k is coordinate k of that representation: face k of a box is
 * its max face on axis k for k &lt; d, and its min face on axis k - d
 * otherwise. With u = [q, -q] for the query q, at each face k:
 * <ul>
 * <li>{@code reach[k] = max(0, b[k] - u[k])}: how far face k of the box with
 * the query lies beyond the query;</li>
 * <li>{@code gap[k] = max(0, u[k] - b[k])}: how far the query lies beyond face
 * k of the data box.</li>
 * </ul>
 * In a single tree at most one of the two is positive at each face, and
 * {@code reach - gap} is the signed slack b - u. Both are means over trees;
 * mean reach minus mean gap is the mean slack, since the relation is linear.
 * {@link #offsets()} and {@link #gapOffsets()} give the same numbers as
 * distances below and above the query, axis by axis.
 *
 * <p>
 * {@code mass} and {@code cutProbability} are the means over trees of the node
 * mass and of the query's own cut probability at the level where the box was
 * read. {@code cappedTrees} counts, for the cut and passage boxes, the trees in
 * which the cap mass, not the quantile, set the box; for the reading box, the
 * trees whose root held less than the reading mass.
 */
public record LocalBox(double[] reach, double[] gap, double mass, double cutProbability, int cappedTrees) {

    public LocalBox {
        checkArgument(reach != null && gap != null && reach.length == gap.length && reach.length % 2 == 0,
                "reach and gap must be two arrays of the same even length");
        reach = reach.clone();
        gap = gap.clone();
    }

    @Override
    public double[] reach() {
        return reach.clone();
    }

    @Override
    public double[] gap() {
        return gap.clone();
    }

    public int dimensions() {
        return reach.length / 2;
    }

    /**
     * reach - gap at each face: positive where the data box's face lies beyond the
     * query.
     */
    public double[] slack() {
        double[] out = new double[reach.length];
        for (int k = 0; k < reach.length; k++) {
            out[k] = reach[k] - gap[k];
        }
        return out;
    }

    /** Axis widths of the box with the query. */
    public double[] widthWithQuery() {
        int d = dimensions();
        double[] out = new double[d];
        for (int i = 0; i < d; i++) {
            out[i] = reach[i] + reach[d + i];
        }
        return out;
    }

    /** Axis widths of the data box: the widths with the query less both gaps. */
    public double[] dataWidth() {
        int d = dimensions();
        double[] out = new double[d];
        for (int i = 0; i < d; i++) {
            out[i] = reach[i] + reach[d + i] - gap[i] - gap[d + i];
        }
        return out;
    }

    /** Axes on which the box with the query has positive width. */
    public int activeAxes() {
        int active = 0;
        for (double w : widthWithQuery()) {
            if (w > 0.0) {
                active++;
            }
        }
        return active;
    }

    /**
     * Volume of the box with the query over its active axes; 0 when no axis is
     * active, which {@link #activeAxes()} then reports as 0.
     */
    public double volume() {
        double v = 1.0;
        int active = 0;
        for (double w : widthWithQuery()) {
            if (w > 0.0) {
                v *= w;
                active++;
            }
        }
        return (active > 0) ? v : 0.0;
    }

    /**
     * The box with the query as distances from the query: [i] below it on axis i,
     * [d + i] above it. That is reach with its halves exchanged.
     */
    public double[] offsets() {
        int d = dimensions();
        double[] out = new double[reach.length];
        for (int i = 0; i < d; i++) {
            out[i] = reach[d + i];
            out[d + i] = reach[i];
        }
        return out;
    }

    /**
     * The gap box, between the query and the face it lies beyond, as distances from
     * the query: [i] below it on axis i, [d + i] above it. A query beyond the max
     * face on axis i has the void below it, so this is gap as it stands.
     */
    public double[] gapOffsets() {
        return gap.clone();
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (!(o instanceof LocalBox other)) {
            return false;
        }
        return Arrays.equals(reach, other.reach) && Arrays.equals(gap, other.gap)
                && Double.compare(mass, other.mass) == 0 && Double.compare(cutProbability, other.cutProbability) == 0
                && cappedTrees == other.cappedTrees;
    }

    @Override
    public int hashCode() {
        int h = Arrays.hashCode(reach);
        h = 31 * h + Arrays.hashCode(gap);
        h = 31 * h + Double.hashCode(mass);
        h = 31 * h + Double.hashCode(cutProbability);
        return 31 * h + cappedTrees;
    }

    @Override
    public String toString() {
        return "LocalBox[reach=" + Arrays.toString(reach) + ", gap=" + Arrays.toString(gap) + ", mass=" + mass
                + ", cutProbability=" + cutProbability + ", cappedTrees=" + cappedTrees + "]";
    }
}
