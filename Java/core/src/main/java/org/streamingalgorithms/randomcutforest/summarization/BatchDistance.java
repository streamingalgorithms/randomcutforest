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

package org.streamingalgorithms.randomcutforest.summarization;

import java.util.Arrays;
import java.util.function.ToDoubleBiFunction;

import org.streamingalgorithms.randomcutforest.tree.VectorSupport;

/**
 * The L1 and L2 distances, in a form the summarizer can evaluate for a block of
 * points against many representatives at once.
 *
 * <p>
 * As functions they are VectorSupport.L1distance and VectorSupport.L2distance,
 * so passing {@code BatchDistance.L2} where {@code VectorSupport::L2distance}
 * was passed changes no result. What it adds is that
 * Summarizer.iterativeClustering, given MultiCenter clusters, assigns points in
 * blocks: the points are fetched once per call, laid out coordinate by
 * coordinate, and each representative is measured against a whole block of
 * points at once.
 *
 * <h2>Plain Java, on purpose</h2>
 * <p>
 * The Vector API stays behind {@link VectorSupport}, the library's only entry
 * point to the incubator module, and nothing here refers to it. A
 * lane-per-point kernel written with it would widen float lanes to double lanes
 * across vector shapes, a conversion that is not intrinsified everywhere: on
 * NEON the preferred double species has two lanes, so the float side is a
 * 64-bit species, and where C2 does not intrinsify the part conversion the
 * Vector API runs its Java fallback, which allocates on every call. It would
 * also pass vectors between methods, which costs nothing only while C2 inlines
 * them; when it declines (inlining budget, deoptimization, a C1 tier) every
 * such vector is boxed on the heap in the innermost loop.
 *
 * <p>
 * Here every loop runs over the points of a block for one coordinate of one
 * representative: contiguous floats in, a contiguous array of double sums
 * updated in place, no reduction across iterations. That is the shape C2's own
 * superword pass vectorizes where the platform supports the conversion, and on
 * a platform where it does not, the loop stays a scalar loop. Either way no
 * vector value exists and nothing is allocated; the work arrays belong to the
 * caller.
 *
 * <h2>Exactness</h2>
 * <p>
 * Each point's distance performs the scalar operations of VectorSupportLegacy
 * in the same order (float difference, widened to double, squared or made
 * absolute, summed in double into the same four partial sums combined as (s0 +
 * s1) + (s2 + s3), then the tail, square root for L2). The nearest
 * representative is kept with a strict less-than on the distance, so the first
 * of equals wins, as in GenericMultiCenter. For L2 the comparison is first made
 * on sums of squares, and a square root is taken only for a candidate whose sum
 * is smaller: the correctly rounded square root is monotone, so the candidates
 * skipped are exactly those whose distance could not have been smaller. Every
 * distance is therefore bit for bit what applyAsDouble returns whenever
 * VectorSupport itself takes the scalar path. At or above the vector length
 * VectorSupport sums in float lanes and reduces across them, an order a
 * per-point kernel does not reproduce; there the summarizer does not use the
 * blocks and keeps the point-by-point assignment. Without the incubator module
 * VectorSupport is scalar at every length, and the blocks are always used.
 */
public final class BatchDistance implements ToDoubleBiFunction<float[], float[]> {

    public static final BatchDistance L1 = new BatchDistance(false);

    public static final BatchDistance L2 = new BatchDistance(true);

    /** Doubles of work space nearest needs per point of a block. */
    static final int WORK_PER_POINT = 5;

    final boolean l2;

    private BatchDistance(boolean l2) {
        this.l2 = l2;
    }

    @Override
    public double applyAsDouble(float[] a, float[] b) {
        return l2 ? VectorSupport.L2distance(a, b) : VectorSupport.L1distance(a, b);
    }

    /** True when the blocks reproduce applyAsDouble bit for bit in dimension d. */
    boolean exactAt(int d) {
        return !VectorSupport.vectorizes(d);
    }

    /**
     * For the points [from, to) of {@code points}, where coordinate j of point p is
     * points[j * n + p], and the representatives [r0, r1) of {@code reps}, where
     * coordinate j of representative m is reps[m * d + j]: writes at off + p - from
     * the distance to representative r0 into first, the least
     * distance(representative, point) into best, and into arg the first
     * representative, counted from r0, that attains it. Requires r1 > r0 and
     * work.length at least WORK_PER_POINT * (to - from).
     */
    void nearest(int d, float[] points, int n, int from, int to, float[] reps, int r0, int r1, double[] first,
            double[] best, int[] arg, int off, double[] work) {
        final int len = to - from;
        if (len <= 0) {
            return;
        }
        // the partial sums for j mod 4, then the least sum so far (L2)
        final int s1 = len, s2 = 2 * len, s3 = 3 * len, least = 4 * len;
        final int bound = d & ~3;
        for (int m = r0; m < r1; m++) {
            final int rb = m * d;
            int j = 0;
            if (bound > 0) {
                Arrays.fill(work, 0, 4 * len, 0.0);
                for (; j < bound; j += 4) {
                    accumulate(l2, work, 0, reps[rb + j], points, j * n + from, len);
                    accumulate(l2, work, s1, reps[rb + j + 1], points, (j + 1) * n + from, len);
                    accumulate(l2, work, s2, reps[rb + j + 2], points, (j + 2) * n + from, len);
                    accumulate(l2, work, s3, reps[rb + j + 3], points, (j + 3) * n + from, len);
                }
                for (int i = 0; i < len; i++) {
                    work[i] = (work[i] + work[s1 + i]) + (work[s2 + i] + work[s3 + i]);
                }
            } else {
                Arrays.fill(work, 0, len, 0.0);
            }
            for (; j < d; j++) {
                accumulate(l2, work, 0, reps[rb + j], points, j * n + from, len);
            }
            if (l2) {
                selectL2(work, least, len, m - r0, first, best, arg, off);
            } else {
                selectL1(work, len, m - r0, first, best, arg, off);
            }
        }
    }

    /**
     * work[at + i] += the term of coordinate value r against point from + i, for
     * the block: (double) (r - x) squared, or its absolute value.
     */
    private static void accumulate(boolean l2, double[] work, int at, float r, float[] points, int c, int len) {
        if (l2) {
            for (int i = 0; i < len; i++) {
                float x = r - points[c + i];
                work[at + i] += (double) x * x;
            }
        } else {
            for (int i = 0; i < len; i++) {
                work[at + i] += Math.abs(r - points[c + i]);
            }
        }
    }

    /** Sums of squares in work[0, len), representative k counted from r0. */
    private static void selectL2(double[] work, int least, int len, int k, double[] first, double[] best, int[] arg,
            int off) {
        if (k == 0) {
            for (int i = 0; i < len; i++) {
                double t = work[i];
                double root = Math.sqrt(t);
                work[least + i] = t;
                first[off + i] = root;
                best[off + i] = root;
                arg[off + i] = 0;
            }
        } else {
            for (int i = 0; i < len; i++) {
                double t = work[i];
                if (t < work[least + i]) {
                    double root = Math.sqrt(t);
                    if (root < best[off + i]) {
                        work[least + i] = t;
                        best[off + i] = root;
                        arg[off + i] = k;
                    }
                }
            }
        }
    }

    /**
     * Sums of absolute values in work[0, len), representative k counted from r0.
     */
    private static void selectL1(double[] work, int len, int k, double[] first, double[] best, int[] arg, int off) {
        if (k == 0) {
            for (int i = 0; i < len; i++) {
                first[off + i] = work[i];
                best[off + i] = work[i];
                arg[off + i] = 0;
            }
        } else {
            for (int i = 0; i < len; i++) {
                if (work[i] < best[off + i]) {
                    best[off + i] = work[i];
                    arg[off + i] = k;
                }
            }
        }
    }
}