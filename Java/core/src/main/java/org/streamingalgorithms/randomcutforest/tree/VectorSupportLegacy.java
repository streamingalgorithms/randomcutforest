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

package org.streamingalgorithms.randomcutforest.tree;

import static java.lang.Math.fma;
import static java.lang.Math.max;

import java.util.Arrays;

/**
 * The scalar implementation of every {@link VectorSupport} kernel. Plain Java,
 * no {@code --add-modules}, no incubator module.
 *
 * <h2>This class has two jobs</h2>
 * <ol>
 * <li><b>Standalone B.</b> The whole library, for hosts that will not or cannot
 * run the Vector API, and (per the benchmark) the fast path at low dimension
 * where there is simply nothing to vectorize.</li>
 * <li><b>A's tails.</b> Every {@code *Range(from, to)} helper below is called
 * by {@link VectorSupport} to finish the elements past {@code SP.loopBound(n)}.
 * There is exactly ONE scalar implementation of each kernel in this codebase
 * and both classes use it.</li>
 * </ol>
 *
 * <p>
 * Job 2 is the important one architecturally. The body/tail divergences that
 * shipped previously -- the vector body using {@code fma} while the tail did
 * not, the two halves of {@code probAndDistInto} associating differently -- are
 * not bugs that were fixed so much as a bug *class* that cannot recur once
 * there is no second copy of the scalar code to drift.
 *
 * <p>
 * The tail is not a rounding error in the schedule, either. {@code loopBound}
 * truncates to a multiple of {@code LANES}, so on AVX-512 at dim=30 twelve of
 * sixty elements land here; below dim=8 the vector loop never runs at all.
 * Hence the unrolling below applies to the tails as much as to the standalone
 * path.
 *
 * <h2>Design rules</h2>
 * <ul>
 * <li><b>Four accumulators on every reduction.</b> C2 emits a strictly ordered
 * reduction for FP adds and will not reassociate, so one accumulator runs at FP
 * add *latency* (~4 cycles/element) rather than throughput (~2/cycle). This is
 * the single largest scalar win.</li>
 * <li><b>Reductions accumulate in {@code double}</b> from exactly-widened float
 * addends -- strictly more accurate than A, which sums in float within a
 * block.</li>
 * <li><b>Pure elementwise loops stay plain and rolled.</b> SuperWord takes them
 * (including {@code Math.fma} -> FmaVD); hand-unrolling only obstructs it.</li>
 * <li><b>{@code Math.max}, not a ternary.</b> The Vector API specifies lanewise
 * MAX on floats as {@code Math.max} semantics, so this is exact parity, and C2
 * intrinsifies it branch-free. A ternary would need
 * {@code -XX:+UseCMoveUnconditionally}.</li>
 * </ul>
 *
 * <h2>Parity with {@link VectorSupport}</h2>
 * <ul>
 * <li><b>Bit-identical:</b> every array written elementwise; {@code contains};
 * {@code LInfinitydistance} (max is exact under any association); and
 * <em>everything</em> below {@code rcf.vector.minLength}, where A simply calls
 * this class.</li>
 * <li><b>~2-4 ulp apart:</b> returned sums, because A sums in float within a
 * block of {@code 4*LANES} before promoting. Note A's order is host-dependent
 * regardless -- its block size derives from {@code SPECIES_PREFERRED}. There is
 * no fixed order to be bit-compatible with.</li>
 * </ul>
 */
public final class VectorSupportLegacy {

    private VectorSupportLegacy() {
    }

    // =======================================================================
    // Public API -- thin wrappers over the range helpers.
    // =======================================================================

    /** @see VectorSupport#gapAttribution */
    public static double gapAttribution(float[] values, int offset, int dimensions, double rangeSum, float[] newValues,
            int nvOffset, float[] contrib) {
        final int n = 2 * dimensions;
        // NOTE: '>= n', not '== n'. Every element of [0,n) is unconditionally written
        // on the fill path, so a pooled buffer LARGER than n is perfectly safe. The
        // old '== n' silently disabled fill for exactly those callers.
        final boolean fill = contrib != null && contrib.length >= n;

        final double S = gapRange(newValues, nvOffset, values, offset, fill ? contrib : null, 0, 0, n);

        final double probability = (S == 0.0) ? 0.0 : (rangeSum == 0.0 ? 1.0 : S / (S + rangeSum));

        if (fill) {
            final double denom = rangeSum + S;
            if (denom == 0.0 || S == 0.0) {
                Arrays.fill(contrib, 0, n, 0f); // range fill, so pooling stays legal
            } else {
                scaleRange(contrib, (float) (1.0 / denom), 0, n);
            }
        }
        return probability;
    }

    /** @see VectorSupport#addSlice */
    public static double addSlice(float[] values, int offset, int dimensions, float[] other, int otherOffset) {
        return addSliceRange(values, offset, other, otherOffset, 0, 2 * dimensions);
    }

    /** @see VectorSupport#contains */
    public static boolean contains(float[] values, int offset, int dimensions, float[] other, int otherOffset) {
        return containsRange(values, offset, other, otherOffset, 0, 2 * dimensions);
    }

    /** @see VectorSupport#multiply */
    public static void multiply(double[] dir, double decay) {
        multiplyRange(dir, decay, 0, dir.length);
    }

    /**
     * @see VectorSupport#updateRecurrence
     *
     *      Fused into one pass rather than multiply() followed by the fma loop:
     *      exactly equivalent (fma(comp, a, decay*dir) either way), half the
     *      traffic over dir.
     */
    public static void updateRecurrence(double[] dir, float[] comp, double a, double decay) {
        final int n = dir.length;
        for (int i = 0; i < n; i++) {
            dir[i] = fma(comp[i], a, decay * dir[i]);
        }
    }

    /** @see VectorSupport#axpyInto */
    public static void axpyInto(double[] dst, double[] src, double factor) {
        axpyRange(dst, src, factor, 0, dst.length);
    }

    /** @see VectorSupport#sum */
    public static double sum(double[] a) {
        double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
        final int n = a.length, bound = n & ~3;
        int i = 0;
        for (; i < bound; i += 4) {
            s0 += a[i];
            s1 += a[i + 1];
            s2 += a[i + 2];
            s3 += a[i + 3];
        }
        double s = (s0 + s1) + (s2 + s3);
        for (; i < n; i++) {
            s += a[i];
        }
        return s;
    }

    /** @see VectorSupport#signedGapInto */
    public static double signedGapInto(float[] exp, int expOff, float sign, float[] pt, int ptOff, float[] dst,
            int dstOff, int dim) {
        return signedGapRange(exp, expOff, sign, pt, ptOff, dst, dstOff, 0, dim);
    }

    /** @see VectorSupport#addPointInPlace */
    public static double addPointInPlace(float[] values, int offset, int dim, float[] point, int pOff) {
        return addPointRange(values, offset, dim, point, pOff, 0, dim);
    }

    /** @see VectorSupport#gapInto */
    public static double gapInto(float[] nv, int nvOff, float[] v, int vOff, float[] dst, int dstOff, int n) {
        return gapRange(nv, nvOff, v, vOff, dst, dstOff, 0, n);
    }

    /** @see VectorSupport#probAndDistInto */
    public static void probAndDistInto(float[] gap, float[] dist, float[] box, int boxOff, int dim, double invSumNew) {
        final float inv = (float) invSumNew;
        for (int half = 0; half < 2; half++) {
            probAndDistRange(gap, dist, box, boxOff, dim, inv, half * dim, 0, dim);
        }
    }

    /** @see VectorSupport#probOnlyInto */
    public static void probOnlyInto(float[] gap, float[] dist, int len, double invSumNew) {
        probOnlyRange(gap, dist, (float) invSumNew, 0, len);
    }

    /** @see VectorSupport#L1distance */
    public static double L1distance(float[] a, float[] b) {
        return l1Range(a, b, 0, a.length);
    }

    /** @see VectorSupport#L2distance */
    public static double L2distance(float[] a, float[] b) {
        return Math.sqrt(l2SqRange(a, b, 0, a.length));
    }

    /** @see VectorSupport#LInfinitydistance */
    public static double LInfinitydistance(float[] a, float[] b) {
        return lInfRange(a, b, 0, a.length);
    }

    // =======================================================================
    // Range helpers. These ARE the scalar kernels; VectorSupport calls them
    // for its tails. Every one is [from, to).
    // =======================================================================

    /**
     * Sum of max(0, nv - v) over [from, to); if dst != null also stores each gap to
     * dst[dstOff + i]. Serves both gapAttribution (dst = contrib, dstOff = 0) and
     * gapInto.
     */
    public static double gapRange(float[] nv, int nvOff, float[] v, int vOff, float[] dst, int dstOff, int from,
            int to) {
        // hoisted out of the loop rather than tested per element: C2 would usually
        // unswitch this, but not depending on that costs nothing.
        return (dst == null) ? gapSumOnly(nv, nvOff, v, vOff, from, to)
                : gapSumStore(nv, nvOff, v, vOff, dst, dstOff, from, to);
    }

    private static double gapSumOnly(float[] nv, int nvOff, float[] v, int vOff, int from, int to) {
        double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
        int i = from;
        final int bound = from + ((to - from) & ~3);
        for (; i < bound; i += 4) {
            s0 += max(0f, nv[nvOff + i] - v[vOff + i]);
            s1 += max(0f, nv[nvOff + i + 1] - v[vOff + i + 1]);
            s2 += max(0f, nv[nvOff + i + 2] - v[vOff + i + 2]);
            s3 += max(0f, nv[nvOff + i + 3] - v[vOff + i + 3]);
        }
        double s = (s0 + s1) + (s2 + s3);
        for (; i < to; i++) {
            s += max(0f, nv[nvOff + i] - v[vOff + i]);
        }
        return s;
    }

    private static double gapSumStore(float[] nv, int nvOff, float[] v, int vOff, float[] dst, int dstOff, int from,
            int to) {
        double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
        int i = from;
        final int bound = from + ((to - from) & ~3);
        for (; i < bound; i += 4) {
            float g0 = max(0f, nv[nvOff + i] - v[vOff + i]);
            float g1 = max(0f, nv[nvOff + i + 1] - v[vOff + i + 1]);
            float g2 = max(0f, nv[nvOff + i + 2] - v[vOff + i + 2]);
            float g3 = max(0f, nv[nvOff + i + 3] - v[vOff + i + 3]);
            dst[dstOff + i] = g0;
            dst[dstOff + i + 1] = g1;
            dst[dstOff + i + 2] = g2;
            dst[dstOff + i + 3] = g3;
            s0 += g0;
            s1 += g1;
            s2 += g2;
            s3 += g3;
        }
        double s = (s0 + s1) + (s2 + s3);
        for (; i < to; i++) {
            float g = max(0f, nv[nvOff + i] - v[vOff + i]);
            dst[dstOff + i] = g;
            s += g;
        }
        return s;
    }

    /** Elementwise, no reduction -- left rolled for SuperWord. */
    public static void scaleRange(float[] a, float scale, int from, int to) {
        for (int i = from; i < to; i++) {
            a[i] *= scale;
        }
    }

    public static double addSliceRange(float[] values, int offset, float[] other, int otherOffset, int from, int to) {
        double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
        int i = from;
        final int bound = from + ((to - from) & ~3);
        for (; i < bound; i += 4) {
            float m0 = max(values[offset + i], other[otherOffset + i]);
            float m1 = max(values[offset + i + 1], other[otherOffset + i + 1]);
            float m2 = max(values[offset + i + 2], other[otherOffset + i + 2]);
            float m3 = max(values[offset + i + 3], other[otherOffset + i + 3]);
            values[offset + i] = m0;
            values[offset + i + 1] = m1;
            values[offset + i + 2] = m2;
            values[offset + i + 3] = m3;
            s0 += m0;
            s1 += m1;
            s2 += m2;
            s3 += m3;
        }
        double s = (s0 + s1) + (s2 + s3);
        for (; i < to; i++) {
            float mx = max(values[offset + i], other[otherOffset + i]);
            values[offset + i] = mx;
            s += mx;
        }
        return s;
    }

    /**
     * Deliberately NOT unrolled and NOT branchless: a multi-exit loop, which C2
     * will not vectorize anyway, and the early exit is the whole point. If
     * profiling ever shows the predicate is rarely true early, an
     * {@code int bad |= (values[..] < other[..]) ? 1 : 0} reduction over the full
     * range vectorizes -- but always reads all n.
     */
    public static boolean containsRange(float[] values, int offset, float[] other, int otherOffset, int from, int to) {
        for (int i = from; i < to; i++) {
            if (values[offset + i] < other[otherOffset + i]) {
                return false;
            }
        }
        return true;
    }

    /** Elementwise: C2 emits vmulpd. */
    public static void multiplyRange(double[] dir, double decay, int from, int to) {
        for (int i = from; i < to; i++) {
            dir[i] = decay * dir[i];
        }
    }

    /**
     * Elementwise fma. Math.fma is intrinsified to vfmadd and vectorized by
     * SuperWord as FmaVD. Note C2 will NOT contract a plain {@code a*b+c} into an
     * FMA -- Java forbids it -- so the explicit call is required both for the
     * semantics A's vector body has and for the speed.
     */
    public static void axpyRange(double[] dst, double[] src, double factor, int from, int to) {
        for (int i = from; i < to; i++) {
            dst[i] = fma(src[i], factor, dst[i]);
        }
    }

    public static double signedGapRange(float[] exp, int expOff, float sign, float[] pt, int ptOff, float[] dst,
            int dstOff, int from, int to) {
        double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
        int i = from;
        final int bound = from + ((to - from) & ~3);
        for (; i < bound; i += 4) {
            float g0 = max(0f, exp[expOff + i] - sign * pt[ptOff + i]);
            float g1 = max(0f, exp[expOff + i + 1] - sign * pt[ptOff + i + 1]);
            float g2 = max(0f, exp[expOff + i + 2] - sign * pt[ptOff + i + 2]);
            float g3 = max(0f, exp[expOff + i + 3] - sign * pt[ptOff + i + 3]);
            dst[dstOff + i] = g0;
            dst[dstOff + i + 1] = g1;
            dst[dstOff + i + 2] = g2;
            dst[dstOff + i + 3] = g3;
            s0 += g0;
            s1 += g1;
            s2 += g2;
            s3 += g3;
        }
        double s = (s0 + s1) + (s2 + s3);
        for (; i < to; i++) {
            float g = max(0f, exp[expOff + i] - sign * pt[ptOff + i]);
            dst[dstOff + i] = g;
            s += g;
        }
        return s;
    }

    /** [from, to) indexes the HIGH half; the low half is at +dim throughout. */
    public static double addPointRange(float[] values, int offset, int dim, float[] point, int pOff, int from, int to) {
        double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
        int i = from;
        final int bound = from + ((to - from) & ~3);
        for (; i < bound; i += 4) {
            float p0 = point[pOff + i], p1 = point[pOff + i + 1];
            float p2 = point[pOff + i + 2], p3 = point[pOff + i + 3];

            float h0 = max(values[offset + i], p0);
            float h1 = max(values[offset + i + 1], p1);
            float h2 = max(values[offset + i + 2], p2);
            float h3 = max(values[offset + i + 3], p3);

            float l0 = max(values[offset + i + dim], -p0);
            float l1 = max(values[offset + i + dim + 1], -p1);
            float l2 = max(values[offset + i + dim + 2], -p2);
            float l3 = max(values[offset + i + dim + 3], -p3);

            values[offset + i] = h0;
            values[offset + i + 1] = h1;
            values[offset + i + 2] = h2;
            values[offset + i + 3] = h3;
            values[offset + i + dim] = l0;
            values[offset + i + dim + 1] = l1;
            values[offset + i + dim + 2] = l2;
            values[offset + i + dim + 3] = l3;

            s0 += h0;
            s1 += h1;
            s2 += h2;
            s3 += h3;
            s0 += l0;
            s1 += l1;
            s2 += l2;
            s3 += l3;
        }
        double s = (s0 + s1) + (s2 + s3);
        for (; i < to; i++) {
            float mx = max(values[offset + i], point[pOff + i]);
            float negMin = max(values[offset + i + dim], -point[pOff + i]);
            values[offset + i] = mx;
            values[offset + i + dim] = negMin;
            s += mx;
            s += negMin;
        }
        return s;
    }

    /**
     * One half only; {@code off} selects it. Associates as {@code g + (hi + lo)} --
     * matching what A's vector body computes. (The old scalar tail said
     * {@code (g + hi) + lo}; there is now only one answer.)
     */
    public static void probAndDistRange(float[] gap, float[] dist, float[] box, int boxOff, int dim, float inv, int off,
            int from, int to) {
        for (int i = from; i < to; i++) {
            float g = gap[off + i];
            float p = g * inv;
            gap[off + i] = p;
            float oldR = box[boxOff + i] + box[boxOff + i + dim];
            dist[off + i] = p * (g + oldR);
        }
    }

    public static void probOnlyRange(float[] gap, float[] dist, float inv, int from, int to) {
        for (int i = from; i < to; i++) {
            float g = gap[i];
            float p = g * inv;
            gap[i] = p;
            dist[i] = p * g;
        }
    }

    public static double l1Range(float[] a, float[] b, int from, int to) {
        double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
        int i = from;
        final int bound = from + ((to - from) & ~3);
        for (; i < bound; i += 4) {
            s0 += Math.abs(a[i] - b[i]);
            s1 += Math.abs(a[i + 1] - b[i + 1]);
            s2 += Math.abs(a[i + 2] - b[i + 2]);
            s3 += Math.abs(a[i + 3] - b[i + 3]);
        }
        double s = (s0 + s1) + (s2 + s3);
        for (; i < to; i++) {
            s += Math.abs(a[i] - b[i]);
        }
        return s;
    }

    /**
     * Sum of squares, NOT the root -- so A can add its vector part and take one
     * sqrt. {@code (double) d * d} on the float difference is the exact product
     * (24+24 less_equal 53 bits) and costs nothing over squaring in float.
     */
    public static double l2SqRange(float[] a, float[] b, int from, int to) {
        double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
        int i = from;
        final int bound = from + ((to - from) & ~3);
        for (; i < bound; i += 4) {
            float d0 = a[i] - b[i], d1 = a[i + 1] - b[i + 1];
            float d2 = a[i + 2] - b[i + 2], d3 = a[i + 3] - b[i + 3];
            s0 += (double) d0 * d0;
            s1 += (double) d1 * d1;
            s2 += (double) d2 * d2;
            s3 += (double) d3 * d3;
        }
        double s = (s0 + s1) + (s2 + s3);
        for (; i < to; i++) {
            float d = a[i] - b[i];
            s += (double) d * d;
        }
        return s;
    }

    /** Four independent max chains; max is exact, so association is free. */
    public static float lInfRange(float[] a, float[] b, int from, int to) {
        float m0 = 0f, m1 = 0f, m2 = 0f, m3 = 0f;
        int i = from;
        final int bound = from + ((to - from) & ~3);
        for (; i < bound; i += 4) {
            m0 = max(m0, Math.abs(a[i] - b[i]));
            m1 = max(m1, Math.abs(a[i + 1] - b[i + 1]));
            m2 = max(m2, Math.abs(a[i + 2] - b[i + 2]));
            m3 = max(m3, Math.abs(a[i + 3] - b[i + 3]));
        }
        float m = max(max(m0, m1), max(m2, m3));
        for (; i < to; i++) {
            m = max(m, Math.abs(a[i] - b[i]));
        }
        return m;
    }
}
