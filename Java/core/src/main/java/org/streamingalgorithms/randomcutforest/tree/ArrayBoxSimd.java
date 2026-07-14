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

import static java.lang.Math.max;

import java.util.Arrays;

import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

public final class ArrayBoxSimd {

    private static final VectorSpecies<Float> SP = FloatVector.SPECIES_PREFERRED;
    private static final VectorSpecies<Double> DP = DoubleVector.SPECIES_PREFERRED;
    private static final int LANES = SP.length();
    private static final int F_PARTS = SP.length() / DP.length(); // = 2 on NEON/AVX2/AVX-512

    private ArrayBoxSimd() {
    }

    // ---- gap / cut probability + optional attribution fill ----------------
    public static double gapAttribution(float[] values, int offset, int dimensions, double rangeSum, float[] newValues,
            int nvOffset, float[] contrib) {
        final int n = 2 * dimensions;
        final boolean fill = contrib != null;
        final FloatVector ZERO = FloatVector.zero(SP);

        double S = 0.0;
        int i = 0;
        final int bound = SP.loopBound(n);
        for (; i < bound; i += LANES) {
            FloatVector nv = FloatVector.fromArray(SP, newValues, nvOffset + i);
            FloatVector v = FloatVector.fromArray(SP, values, offset + i);
            FloatVector gap = nv.sub(v).max(ZERO);
            if (fill)
                gap.intoArray(contrib, i);
            S += (double) gap.reduceLanes(VectorOperators.ADD); // float horizontal, double across chunks
        }
        for (; i < n; i++) {
            float g = max(0f, newValues[nvOffset + i] - values[offset + i]);
            if (fill)
                contrib[i] = g;
            S += g;
        }

        double probability = (S == 0.0) ? 0.0 : (rangeSum == 0.0 ? 1.0 : S / (S + rangeSum));

        if (fill) {
            double denom = rangeSum + S;
            if (denom == 0.0 || S == 0.0) {
                Arrays.fill(contrib, 0f);
            } else {
                final float scale = (float) (1.0 / denom); // shared ~0.5ulp, ratios exact
                FloatVector sc = FloatVector.broadcast(SP, scale);
                int j = 0;
                final int b2 = SP.loopBound(n);
                for (; j < b2; j += LANES)
                    FloatVector.fromArray(SP, contrib, j).mul(sc).intoArray(contrib, j);
                for (; j < n; j++)
                    contrib[j] *= scale;
            }
        }
        return probability;
    }

    // ---- box U= box (whole slice, single lanewise max; float reduction) -
    public static double addSlice(float[] values, int offset, int dimensions, float[] other, int otherOffset) {
        final int n = 2 * dimensions;
        double sum = 0.0;
        int i = 0;
        final int bound = SP.loopBound(n);
        for (; i < bound; i += LANES) {
            FloatVector m = FloatVector.fromArray(SP, values, offset + i)
                    .max(FloatVector.fromArray(SP, other, otherOffset + i));
            m.intoArray(values, offset + i);
            sum += (double) m.reduceLanes(VectorOperators.ADD);
        }
        for (; i < n; i++) {
            float mx = max(values[offset + i], other[otherOffset + i]);
            values[offset + i] = mx;
            sum += mx;
        }
        return sum;
    }

    public static boolean contains(float[] values, int offset, int dimensions, float[] other, int otherOffset) {
        final int n = 2 * dimensions;
        int i = 0;
        final int bound = SP.loopBound(n);
        for (; i < bound; i += LANES)
            if (FloatVector.fromArray(SP, values, offset + i).lt(FloatVector.fromArray(SP, other, otherOffset + i))
                    .anyTrue())
                return false;
        for (; i < n; i++)
            if (values[offset + i] < other[otherOffset + i])
                return false;
        return true;
    }

    // ---- expanded query [p, -p] -------------------------------------------
    public static float[] expand(float[] point) {
        float[] e = new float[2 * point.length];
        expandInto(point, e);
        return e;
    }

    /** In-place refill (no allocation) for reused visitors. */
    public static void expandInto(float[] point, float[] dest) {
        final int d = point.length;
        System.arraycopy(point, 0, dest, 0, d);
        for (int i = 0; i < d; i++)
            dest[i + d] = -point[i];
    }

    // ---- AttributionVisitor accumulator kernels ---------------------------

    /**
     * PER-NODE recurrence (hot): dir[i] = comp[i]*a + decay*dir[i], comp float ->
     * double. The float->double widen is the one C2 superword handles poorly, so
     * explicit Vector API earns its keep here -- but it's the SAME F2D that was
     * slow on JDK 21. Microbench this in isolation before wiring; on 21 it may be a
     * wash, on 25 a win.
     */
    public static void updateRecurrenceA(double[] dir, float[] comp, double a, double decay) {
        final int n = dir.length;
        final DoubleVector va = DoubleVector.broadcast(DP, a);
        final DoubleVector vd = DoubleVector.broadcast(DP, decay);
        int i = 0;
        final int bound = SP.loopBound(n);
        for (; i < bound; i += LANES) {
            FloatVector cf = FloatVector.fromArray(SP, comp, i);
            for (int part = 0; part < F_PARTS; part++) {
                int base = i + part * DP.length();
                DoubleVector c = (DoubleVector) cf.convertShape(VectorOperators.F2D, DP, part);
                DoubleVector d = DoubleVector.fromArray(DP, dir, base);
                c.fma(va, d.mul(vd)).intoArray(dir, base); // comp*a + decay*dir
            }
        }
        for (; i < n; i++)
            dir[i] = comp[i] * a + decay * dir[i];
    }

    public static void multiply(double[] dir, double decay) {
        final int n = dir.length;
        final DoubleVector vd = DoubleVector.broadcast(DP, decay);
        int i = 0;
        for (; i < DP.loopBound(n); i += DP.length())
            DoubleVector.fromArray(DP, dir, i).mul(vd).intoArray(dir, i);
        for (; i < n; i++)
            dir[i] = decay * dir[i];
    }

    public static void updateRecurrence(double[] dir, float[] comp, double a, double decay) {
        final int n = dir.length;
        multiply(dir, decay);
        for (int i = 0; i < n; i++)
            dir[i] = Math.fma(comp[i], a, dir[i]);
    }

    /**
     * PER-TREE: dst[i] += src[i]*factor (foldOut). All-double, no widen -- C2 very
     * likely ALREADY autovectorizes this. Verify with perfasm before adopting; if
     * C2 handles it, delete this and keep the plain loop (less Vector API surface
     * to maintain).
     */
    public static void axpyInto(double[] dst, double[] src, double factor) {
        final int n = dst.length;
        final DoubleVector vf = DoubleVector.broadcast(DP, factor);
        int i = 0;
        final int bound = DP.loopBound(n);
        for (; i < bound; i += DP.length()) {
            DoubleVector s = DoubleVector.fromArray(DP, src, i);
            DoubleVector d = DoubleVector.fromArray(DP, dst, i);
            s.fma(vf, d).intoArray(dst, i);
        }
        for (; i < n; i++)
            dst[i] += src[i] * factor;
    }

    /**
     * PER-TREE sum reduction (idempotentRefactor / convergingValue). Same
     * autovectorize caveat.
     */
    public static double sumvec(double[] a) {
        final int n = a.length;
        DoubleVector acc = DoubleVector.zero(DP);
        int i = 0;
        final int bound = DP.loopBound(n);
        for (; i < bound; i += DP.length())
            acc = acc.add(DoubleVector.fromArray(DP, a, i));
        double s = acc.reduceLanes(VectorOperators.ADD);
        for (; i < n; i++)
            s += a[i];
        return s;
    }

    public static double sum(double[] a) {
        double s = 0;
        for (int i = 0; i < a.length; i++)
            s += a[i];
        return s;
    }

    /**
     * dst[dstOff+i] = max(0, exp[expOff+i] - sign*pt[ptOff+i]); returns Σ over the
     * half. exp = canonical [p,-p] (read-only); pt = RAW point, length dim. Two
     * calls do the leaf: high: (exp,0, +1f, pt,0, dst,0, dim) -> max(0, p - pt) low
     * : (exp,dim, -1f, pt,0, dst,dim, dim) -> max(0, pt - p) No leaf box, no copy
     * of exp.
     */
    public static double signedGapInto(float[] exp, int expOff, float sign, float[] pt, int ptOff, float[] dst,
            int dstOff, int dim) {
        final FloatVector ZERO = FloatVector.zero(SP), vs = FloatVector.broadcast(SP, sign);
        double s = 0.0;
        int i = 0;
        final int bound = SP.loopBound(dim);
        for (; i < bound; i += LANES) {
            FloatVector g = FloatVector.fromArray(SP, exp, expOff + i)
                    .sub(FloatVector.fromArray(SP, pt, ptOff + i).mul(vs)).max(ZERO);
            g.intoArray(dst, dstOff + i);
            s += (double) g.reduceLanes(VectorOperators.ADD);
        }
        for (; i < dim; i++) {
            float g = max(0f, exp[expOff + i] - sign * pt[ptOff + i]);
            dst[dstOff + i] = g;
            s += g;
        }
        return s;
    }

    /** shadow path: raw box-vs-box gap dst = max(0, nv - v); returns sum. */
    public static double gapInto(float[] nv, int nvOff, float[] v, int vOff, float[] dst, int dstOff, int n) {
        final FloatVector ZERO = FloatVector.zero(SP);
        double s = 0.0;
        int i = 0;
        final int bound = SP.loopBound(n);
        for (; i < bound; i += LANES) {
            FloatVector g = FloatVector.fromArray(SP, nv, nvOff + i).sub(FloatVector.fromArray(SP, v, vOff + i))
                    .max(ZERO);
            g.intoArray(dst, dstOff + i);
            s += (double) g.reduceLanes(VectorOperators.ADD);
        }
        for (; i < n; i++) {
            float g = max(0f, nv[nvOff + i] - v[vOff + i]);
            dst[dstOff + i] = g;
            s += g;
        }
        return s;
    }

    /**
     * gap -> prob in place; dist = prob*(gap + oldRange), oldRange read inline from
     * the SMALL box: oldRange[i] = box[boxOff+i] + box[boxOff+i+dim] (= max_i -
     * min_i), identical across both halves.
     */
    public static void probAndDistInto(float[] gap, float[] dist, float[] box, int boxOff, int dim, double invSumNew) {
        final float inv = (float) invSumNew;
        final FloatVector vInv = FloatVector.broadcast(SP, inv);
        for (int half = 0; half < 2; half++) {
            final int off = half * dim;
            int i = 0;
            final int bound = SP.loopBound(dim);
            for (; i < bound; i += LANES) {
                FloatVector g = FloatVector.fromArray(SP, gap, off + i);
                FloatVector prob = g.mul(vInv);
                prob.intoArray(gap, off + i);
                FloatVector oldR = FloatVector.fromArray(SP, box, boxOff + i)
                        .add(FloatVector.fromArray(SP, box, boxOff + i + dim));
                prob.mul(g.add(oldR)).intoArray(dist, off + i);
            }
            for (; i < dim; i++) {
                float g = gap[off + i], p = g * inv;
                gap[off + i] = p;
                dist[off + i] = p * (g + box[boxOff + i] + box[boxOff + i + dim]);
            }
        }
    }

    /**
     * leaf seed: prob = gap*inv (in place), dist = prob*gap. oldRange ≡ 0
     * (degenerate box).
     */
    public static void probOnlyInto(float[] gap, float[] dist, int len, double invSumNew) {
        final float inv = (float) invSumNew;
        final FloatVector vInv = FloatVector.broadcast(SP, inv);
        int i = 0;
        final int bound = SP.loopBound(len);
        for (; i < bound; i += LANES) {
            FloatVector g = FloatVector.fromArray(SP, gap, i), p = g.mul(vInv);
            p.intoArray(gap, i);
            p.mul(g).intoArray(dist, i);
        }
        for (; i < len; i++) {
            float g = gap[i], p = g * inv;
            gap[i] = p;
            dist[i] = p * g;
        }
    }

}
