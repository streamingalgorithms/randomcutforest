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

    public static double updateBoundsAllInterchanged(float[] values, int offset, int dim, float[] store, int[] pOffs,
            int start, int end) {
        for (int j = 0; j < dim; j++) {
            float h = values[offset + j];
            float l = values[offset + dim + j];
            for (int i = start; i < end; i++) {
                float k = store[pOffs[i] + j];
                h = Math.max(h, k);
                l = Math.max(l, -k);
            }
            values[offset + j] = h;
            values[offset + dim + j] = l;
        }
        double sum = 0.0;
        for (int j = 0; j < 2 * dim; j++)
            sum += (double) values[offset + j];
        return sum;
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

    /** out[0] = delta (overhang sum), out[1] = oldRange (pairwise). */
    public static double deltaAndRange(float[] v, int off, int d, float[] p, int pOff, double[] out) {
        double g0 = 0, g1 = 0, g2 = 0, g3 = 0, r0 = 0, r1 = 0, r2 = 0, r3 = 0;
        int i = 0;
        final int bound = d & ~3;
        for (; i < bound; i += 4) {
            float h0 = v[off + i], l0 = v[off + d + i], q0 = p[pOff + i];
            float h1 = v[off + i + 1], l1 = v[off + d + i + 1], q1 = p[pOff + i + 1];
            float h2 = v[off + i + 2], l2 = v[off + d + i + 2], q2 = p[pOff + i + 2];
            float h3 = v[off + i + 3], l3 = v[off + d + i + 3], q3 = p[pOff + i + 3];

            g0 += max(0f, q0 - h0) + max(0f, -q0 - l0);
            g1 += max(0f, q1 - h1) + max(0f, -q1 - l1);
            g2 += max(0f, q2 - h2) + max(0f, -q2 - l2);
            g3 += max(0f, q3 - h3) + max(0f, -q3 - l3);

            r0 += (double) h0 + l0;
            r1 += (double) h1 + l1;
            r2 += (double) h2 + l2;
            r3 += (double) h3 + l3;
        }
        double g = (g0 + g1) + (g2 + g3);
        double r = (r0 + r1) + (r2 + r3);
        for (; i < d; i++) { // tail, same order
            float h = v[off + i], l = v[off + d + i], q = p[pOff + i];
            g += max(0f, q - h) + max(0f, -q - l);
            r += (double) h + l;
        }
        out[0] = g;
        out[1] = r;
        return g; // also return delta.
    }

    /**
     * Fused union + gap + rangeSum, interchanged. NON-STRUCTURAL callers only:
     * {@link #updateBoundsAllInterchanged} stays byte-identical for makeTree.
     *
     * Blocking by 4 over j is the point. The inner loop reads store[pOffs[i]+j]
     * with stride `dimensions` between points, so one j pulls a fresh line per
     * point; four contiguous j share that line.
     *
     * @return S over [from,to); out[0] receives R over [from,to).
     */
    public static double updateBoundsAndGapRange(float[] values, int offset, int dim, float[] store, int[] pOffs,
            int start, int end, float[] exp, int expOff, float[] gapOut, int gapOff, double[] out, int from, int to) {

        final boolean fill = gapOut != null;
        double S = 0.0, R = 0.0;
        int j = from;
        final int bound = from + ((to - from) & ~3);

        for (; j < bound; j += 4) {
            float h0 = values[offset + j], l0 = values[offset + dim + j];
            float h1 = values[offset + j + 1], l1 = values[offset + dim + j + 1];
            float h2 = values[offset + j + 2], l2 = values[offset + dim + j + 2];
            float h3 = values[offset + j + 3], l3 = values[offset + dim + j + 3];

            for (int i = start; i < end; i++) {
                final int b = pOffs[i] + j; // 4 contiguous floats
                float k0 = store[b], k1 = store[b + 1], k2 = store[b + 2], k3 = store[b + 3];
                h0 = max(h0, k0);
                l0 = max(l0, -k0);
                h1 = max(h1, k1);
                l1 = max(l1, -k1);
                h2 = max(h2, k2);
                l2 = max(l2, -k2);
                h3 = max(h3, k3);
                l3 = max(l3, -k3);
            }

            values[offset + j] = h0;
            values[offset + dim + j] = l0;
            values[offset + j + 1] = h1;
            values[offset + dim + j + 1] = l1;
            values[offset + j + 2] = h2;
            values[offset + dim + j + 2] = l2;
            values[offset + j + 3] = h3;
            values[offset + dim + j + 3] = l3;

            float a0 = max(0f, exp[expOff + j] - h0), b0 = max(0f, exp[expOff + dim + j] - l0);
            float a1 = max(0f, exp[expOff + j + 1] - h1), b1 = max(0f, exp[expOff + dim + j + 1] - l1);
            float a2 = max(0f, exp[expOff + j + 2] - h2), b2 = max(0f, exp[expOff + dim + j + 2] - l2);
            float a3 = max(0f, exp[expOff + j + 3] - h3), b3 = max(0f, exp[expOff + dim + j + 3] - l3);

            if (fill) {
                gapOut[gapOff + j] = a0;
                gapOut[gapOff + dim + j] = b0;
                gapOut[gapOff + j + 1] = a1;
                gapOut[gapOff + dim + j + 1] = b1;
                gapOut[gapOff + j + 2] = a2;
                gapOut[gapOff + dim + j + 2] = b2;
                gapOut[gapOff + j + 3] = a3;
                gapOut[gapOff + dim + j + 3] = b3;
            }

            // promote before combining: (double)a + b is exact (at most one is nonzero)
            S += ((double) a0 + b0) + ((double) a1 + b1) + ((double) a2 + b2) + ((double) a3 + b3);
            R += ((double) h0 + l0) + ((double) h1 + l1) + ((double) h2 + l2) + ((double) h3 + l3);
        }

        for (; j < to; j++) {
            float h = values[offset + j], l = values[offset + dim + j];
            for (int i = start; i < end; i++) {
                float k = store[pOffs[i] + j];
                h = max(h, k);
                l = max(l, -k);
            }
            values[offset + j] = h;
            values[offset + dim + j] = l;
            float a = max(0f, exp[expOff + j] - h), b = max(0f, exp[expOff + dim + j] - l);
            if (fill) {
                gapOut[gapOff + j] = a;
                gapOut[gapOff + dim + j] = b;
            }
            S += (double) a + b;
            R += (double) h + l;
        }

        out[0] = R;
        return S;
    }

    /** Whole-range form. */
    public static double updateBoundsAndGapInterchanged(float[] values, int offset, int dim, float[] store, int[] pOffs,
            int start, int end, float[] exp, int expOff, float[] gapOut, int gapOff, double[] out) {
        return updateBoundsAndGapRange(values, offset, dim, store, pOffs, start, end, exp, expOff, gapOut, gapOff, out,
                0, dim);
    }

    /** One block, both halves. Shared with SIMD's partial-block case. */
    public static double blockGap(float[] values, int offset, int dim, float[] exp, int expOff, float[] contrib,
            int from, int to, boolean fill) {
        double s = 0.0;
        for (int j = from; j < to; j++) {
            float gh = max(0f, exp[expOff + j] - values[offset + j]);
            float gl = max(0f, exp[expOff + dim + j] - values[offset + dim + j]);
            if (fill) {
                contrib[j] = gh;
                contrib[dim + j] = gl;
            }
            s += (double) gh + gl; // exact: at most one term is nonzero
        }
        return s;
    }

    /**
     * Pruned gapAttribution. Iterates only the blocks whose bit is set in
     * {@code mask}, and CLEARS the bit of any block whose gaps are all zero --
     * permanent, since the box only grows. Block b covers [b*bs, min((b+1)*bs,
     * dim)) in BOTH halves.
     *
     * mask is caller-owned (NodeView), length ceil(nBlocks/64), mutated in place.
     * All-zero mask => the point is inside the box in every coordinate.
     */
    public static double gapAttributionPruned(float[] values, int offset, int dim, double rangeSum, float[] exp,
            int expOff, float[] contrib, int bs, long[] mask) {

        final boolean fill = contrib != null;
        double S = 0.0;

        for (int w = 0; w < mask.length; w++) {
            long live = mask[w];
            final int base = w << 6;
            for (long m = live; m != 0; m &= m - 1) {
                final int t = Long.numberOfTrailingZeros(m);
                final int from = (base + t) * bs, to = Math.min(from + bs, dim);
                double blockSum = blockGap(values, offset, dim, exp, expOff, contrib, from, to, fill);
                if (blockSum == 0.0)
                    live &= ~(1L << t);
                S += blockSum;
            }
            mask[w] = live;
        }

        if (S == 0.0)
            return 0.0;
        final double probability = (rangeSum == 0.0) ? 1.0 : S / (S + rangeSum);

        if (fill) {
            final float inv = (float) (1.0 / (rangeSum + S));
            for (int w = 0; w < mask.length; w++) {
                final int base = w << 6;
                for (long m = mask[w]; m != 0; m &= m - 1) {
                    final int from = (base + Long.numberOfTrailingZeros(m)) * bs, to = Math.min(from + bs, dim);
                    for (int j = from; j < to; j++) {
                        contrib[j] *= inv;
                        contrib[dim + j] *= inv;
                    }
                }
            }
        }
        return probability;
    }
}
