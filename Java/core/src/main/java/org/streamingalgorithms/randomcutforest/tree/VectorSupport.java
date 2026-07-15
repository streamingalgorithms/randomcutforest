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

public final class VectorSupport {

    private static final VectorSpecies<Float> SP = FloatVector.SPECIES_PREFERRED;
    private static final VectorSpecies<Double> DP = DoubleVector.SPECIES_PREFERRED;
    private static final int LANES = SP.length();

    private VectorSupport() {
    }

    // ---- gap / cut probability + optional attribution fill ----------------
    /**
     * a single source of probability computation Returns S / (S + rangeSum) and, if
     * halfDimensionalContribution is non-null (length == 2 * dimensions, indexed
     * from 0), fills it with the per-half- dimension attribution max(0f,
     * newValues[i] - values[offset+i]) / (rangeSum + S), so the entries sum to the
     * returned probability. Per-coordinate gaps stay float; the reduction into S is
     * done in double. Passing null takes the exact same path and just returns the
     * probability. Contrib cannot be pooled currently because of 0'ing
     */
    public static double gapAttribution(float[] values, int offset, int dimensions, double rangeSum, float[] newValues,
            int nvOffset, float[] contrib) {
        final int n = 2 * dimensions;
        final boolean fill = contrib != null && contrib.length == n;
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
    // returns the new rangesum
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
    /** In-place refill (no allocation) for reused visitors. */
    public static void expandInto(float[] point, int pointOffset, float[] dest, int destOffset, int dimensions) {
        final int d = dimensions;
        System.arraycopy(point, pointOffset, dest, destOffset, d);
        for (int i = 0; i < d; i++)
            dest[i + d + destOffset] = -point[i + pointOffset];
    }

    // ---- AttributionVisitor accumulator kernels ---------------------------

    public static void multiply(double[] dir, double decay) {
        final int n = dir.length;
        final DoubleVector vd = DoubleVector.broadcast(DP, decay);
        int i = 0;
        for (; i < DP.loopBound(n); i += DP.length())
            DoubleVector.fromArray(DP, dir, i).mul(vd).intoArray(dir, i);
        for (; i < n; i++)
            dir[i] = decay * dir[i];
    }

    // the following fails the widen and when written in vector form, creates
    // allocation
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

    public static double addPointInPlace(float[] values, int offset, int dim, float[] point, int pOff) {
        double sum = 0.0;
        int i = 0;
        final int bound = SP.loopBound(dim);
        for (; i < bound; i += LANES) {
            FloatVector p = FloatVector.fromArray(SP, point, pOff + i);
            FloatVector hi = FloatVector.fromArray(SP, values, offset + i).max(p);
            hi.intoArray(values, offset + i);
            FloatVector lo = FloatVector.fromArray(SP, values, offset + i + dim).max(p.neg());
            lo.intoArray(values, offset + i + dim);
            sum += (double) hi.reduceLanes(VectorOperators.ADD) + (double) lo.reduceLanes(VectorOperators.ADD);
        }
        for (; i < dim; i++) {
            float mx = max(values[offset + i], point[pOff + i]);
            float negMin = max(values[offset + i + dim], -point[pOff + i]);
            values[offset + i] = mx;
            values[offset + i + dim] = negMin;
            sum += mx + negMin;
        }
        return sum;
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

    public static double L1distance(float[] a, float[] b) {
        double dist = 0.0;
        int i = 0, bound = SP.loopBound(a.length);
        for (; i < bound; i += LANES) {
            FloatVector d = FloatVector.fromArray(SP, a, i).sub(FloatVector.fromArray(SP, b, i)).abs();
            dist += (double) d.reduceLanes(VectorOperators.ADD);
        }
        for (; i < a.length; i++)
            dist += Math.abs(a[i] - b[i]);
        return dist;
    }

    public static double L2distance(float[] a, float[] b) {
        double dist = 0.0;
        int i = 0, bound = SP.loopBound(a.length);
        for (; i < bound; i += LANES) {
            FloatVector d = FloatVector.fromArray(SP, a, i).sub(FloatVector.fromArray(SP, b, i));
            FloatVector sq = d.mul(d); // d*d = |d|^2, no abs needed
            dist += (double) sq.reduceLanes(VectorOperators.ADD);
        }
        for (; i < a.length; i++) {
            double t = a[i] - b[i];
            dist += t * t;
        }
        return Math.sqrt(dist);
    }

    public static double LInfinitydistance(float[] a, float[] b) {
        double dist = 0.0;
        int i = 0, bound = SP.loopBound(a.length);
        for (; i < bound; i += LANES) {
            FloatVector d = FloatVector.fromArray(SP, a, i).sub(FloatVector.fromArray(SP, b, i)).abs();
            dist = Math.max(dist, (double) d.reduceLanes(VectorOperators.MAX)); // reduce MAX
        }
        for (; i < a.length; i++)
            dist = Math.max(dist, Math.abs(a[i] - b[i]));
        return dist;
    }

}
