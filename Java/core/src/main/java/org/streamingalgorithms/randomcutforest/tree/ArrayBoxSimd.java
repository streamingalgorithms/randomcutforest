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

/**
 * SIMD kernels for ArrayBox + the AttributionVisitor accumulators.
 *
 * PRECISION POLICY: - gaps = max(0, nv - v) -> float (independent per-coord
 * rounding = noise) - S = sum of gaps -> DOUBLE ACROSS CHUNKS: per-chunk float
 * horizontal reduceLanes (less or equal LANES terms) promoted to double,
 * accumulated in double. ~1e-6 relative -- already below the noise floor; the
 * full F2D-widen double accumulation (see JDK25+ block) cost 2x on JDK 21 for
 * ~1e-8 you cannot use. Revisit on JDK 25 where F2D widening intrinsifies. -
 * scale = 1/denom -> float broadcast. Shared ~0.5ulp factor: perturbs magnitude
 * by ~6e-8, leaves all ratios (the directional info) exact.
 */
public final class ArrayBoxSimd {

    private static final VectorSpecies<Float> SP = FloatVector.SPECIES_PREFERRED;
    private static final VectorSpecies<Double> DP = DoubleVector.SPECIES_PREFERRED;
    private static final int LANES = SP.length();
    private static final int F_PARTS = SP.length() / DP.length(); // = 2 on NEON/AVX2/AVX-512
    static final int SIMD_MIN_SLICE = 2 * LANES;

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

    /*
     * ==== JDK 25+ VARIANT: full double accumulation of S ====================
     * Retest on JDK 25 where convertShape(F2D) intrinsifies to fcvtl/fcvtl2. On JDK
     * 21 this scalarized the widen and ran ~2x slower for ~1e-8 you don't need.
     * Swap the reduction loop above for:
     * 
     * DoubleVector sAcc = DoubleVector.zero(DP); for (; i < bound; i += LANES) {
     * FloatVector gap = FloatVector.fromArray(SP, newValues, nvOffset + i)
     * .sub(FloatVector.fromArray(SP, values, offset + i)) .max(ZERO); if (fill)
     * gap.intoArray(contrib, i); for (int part = 0; part < F_PARTS; part++) sAcc =
     * sAcc.add((DoubleVector) gap.convertShape(VectorOperators.F2D, DP, part)); }
     * double S = sAcc.reduceLanes(VectorOperators.ADD); // + scalar tail
     * =======================================================================
     */

    // ---- box U= box (whole slice, single lanewise max; float reduction) -
    public static double addSlice(float[] values, int offset, int dimensions, float[] other, int otherOffset) {
        final int n = 2 * dimensions;
        if (n < SIMD_MIN_SLICE) {
            double s = 0.0;
            for (int k = 0; k < n; k++) {
                float mx = max(values[offset + k], other[otherOffset + k]);
                values[offset + k] = mx;
                s += mx;
            }
            return s;
        }
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
    public static void updateRecurrence(double[] dir, float[] comp, double a, double decay) {
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
    public static double sum(double[] a) {
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
}
