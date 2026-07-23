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

import java.util.Arrays;

import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

/**
 * Vector API kernels. <b>Loading this class requires
 * {@code --add-modules jdk.incubator.vector}</b>, which is exactly why nothing
 * outside {@link VectorSupport} may name it.
 *
 * <h2>Do not call this class directly</h2>
 * <p>
 * Call {@link VectorSupport}. It decides whether the incubator module is even
 * present and whether {@code n} is large enough to be worth vectorizing; this
 * class holds no policy and is unconditionally vectorized. Referencing it from
 * anywhere else reintroduces the hard dependency the facade exists to remove --
 * the resulting failure is a {@code NoClassDefFoundError} raised from whatever
 * happened to touch it first, ten frames below anything the user wrote.
 *
 * <h2>Structure</h2>
 * <p>
 * Every method here is one of two shapes:
 *
 * <ol>
 * <li><b>Blocked reduction.</b> {@code UNROLL} independent chunks are computed,
 * combined with a pairwise vector tree, and horizontally reduced <em>once</em>,
 * with the block subtotal promoted to double. Gaps stay float; accumulation
 * across blocks is double. This is the same float-in/double-across design as
 * before -- only the block size changed, from {@code LANES} to
 * {@code UNROLL * LANES}.</li>
 * <li><b>Scalar tail.</b> Delegated to {@link VectorSupportLegacy}'s
 * {@code *Range} helpers. There is exactly one scalar implementation of each
 * kernel in this codebase and both classes use it.</li>
 * </ol>
 *
 * <h2>Why blocking</h2>
 * <p>
 * The previous form reduced once per chunk, which made {@code S += ...} a
 * loop-carried FP add of length {@code n / LANES}. Neither C2 nor the JIT may
 * reassociate that, so the loop's floor was one FP-add latency (~4 cycles) per
 * chunk. That cost is worst where lanes are narrowest -- i.e. on 128-bit NEON,
 * which is the platform this code was tuned for. Blocking by 4 shortens the
 * chain 4x, removes 3 of every 4 horizontal reduces, and leaves the four
 * in-block chunks fully independent.
 *
 * <p>
 * At dim=30 (n=60), reduces and chain length go 15 -> 4 on NEON, 7 -> 2 on
 * AVX2, 3 -> 1 on AVX-512. Cost: the float tree before promotion deepens from
 * {@code log2(LANES)} to {@code 2 + log2(LANES)}, i.e. ~1.2e-7 -> ~2.4e-7
 * relative on NEON. All terms are {@code max(0, .)}, hence non-negative, hence
 * no cancellation -- that bound is about as benign as float summation gets.
 *
 * <h2>Op vocabulary</h2>
 * <p>
 * <b>Never hold a Vector in a field</b>, static or otherwise -- not even a
 * constant like ZERO. C2 eliminates vector boxes by folding
 * {@code VectorUnbox(VectorBox(v)) -> v}, i.e. by pattern-matching nodes it
 * produced in the same compilation. A field read is a constant oop, not a
 * VectorBox. Seed a loop-carried accumulator phi with one and the phi's inputs
 * are no longer all VectorBox, so the phi cannot be scalarized and every
 * {@code acc.add(g)} in the loop materializes a real vector object on the heap.
 * This cost ~550 KB/op in ScoringBenchmark before it was caught. Zeroes are
 * declared locally and accumulators are seeded with
 * {@code FloatVector.zero(SP)} for exactly this reason; it looks redundant and
 * is not.
 *
 * <p>
 * Deliberately restricted to load / store / add / sub / mul / max / compare /
 * reduceLanes. These have had universal backend coverage since the first
 * incubator. {@code convertShape} (F2D), {@code rearrange}, {@code slice} and
 * masked variants are <b>not</b> used: C2 special-cases vector instances to
 * avoid boxing rather than relying on general escape analysis, so an op that
 * misses an intrinsic rule does not degrade gracefully -- it allocates on the
 * heap and the loop collapses. Do not add one without a {@code -prof gc} run
 * showing {@code gc.alloc.rate.norm} at 0 B/op on every target arch.
 *
 * <h2>Tuning knobs (all constant-folded by C2)</h2>
 * <ul>
 * <li>{@code -Drcf.vector.minLength=N} -- scalar below n=N. Default
 * {@code LANES}. Set 0 to force the vector path (the benchmark does this to
 * measure the raw crossover).</li>
 * <li>{@code -Drcf.vector.enabled=false} -- everything to Legacy.</li>
 * <li>{@code -XX:MaxVectorSize=32} -- caps SPECIES_PREFERRED at 256-bit on
 * AVX-512 hosts where 512-bit downclocking hurts the surrounding scalar code.
 * The species is not pinned in source precisely so this JVM-level escape hatch
 * keeps working.</li>
 * </ul>
 */
public final class VectorSupportSIMD {

    private static final VectorSpecies<Float> SP = FloatVector.SPECIES_PREFERRED;
    private static final VectorSpecies<Double> DP = DoubleVector.SPECIES_PREFERRED;
    private static final int LANES = SP.length();
    private static final int DLANES = DP.length();

    /** Chunks per double-accumulation block. */
    private static final int UNROLL = 4;
    /** Elements per block. LANES is a power of two, hence so is BLOCK. */
    private static final int BLOCK = UNROLL * LANES;

    private VectorSupportSIMD() {
    }

    /** For the facade's describe(). */
    public static String shape() {
        return "species=" + SP + ", lanes=" + LANES + ", block=" + BLOCK;
    }

    public static int lanes() {
        return LANES;
    }

    // ---- gap / cut probability + optional attribution fill ----------------

    /**
     * Single source of probability computation. Returns S / (S + rangeSum) and, if
     * {@code contrib} is non-null (length == 2 * dimensions, indexed from 0), fills
     * it with the per-half-dimension attribution
     * {@code max(0f, newValues[i] - values[offset+i]) / (rangeSum + S)}, so the
     * entries sum to the returned probability. Per-coordinate gaps stay float; the
     * reduction into S is done in double. Passing null takes the same path and just
     * returns the probability.
     */
    public static double gapAttribution(float[] values, int offset, int dimensions, double rangeSum, float[] newValues,
            int nvOffset, float[] contrib) {
        final int n = 2 * dimensions;
        final boolean fill = contrib != null && contrib.length >= n;

        final FloatVector ZERO = FloatVector.zero(SP);
        double S = 0.0;
        int i = 0;

        final int blockBound = n - (n % BLOCK);
        for (; i < blockBound; i += BLOCK) {
            FloatVector g0 = FloatVector.fromArray(SP, newValues, nvOffset + i)
                    .sub(FloatVector.fromArray(SP, values, offset + i)).max(ZERO);
            FloatVector g1 = FloatVector.fromArray(SP, newValues, nvOffset + i + LANES)
                    .sub(FloatVector.fromArray(SP, values, offset + i + LANES)).max(ZERO);
            FloatVector g2 = FloatVector.fromArray(SP, newValues, nvOffset + i + 2 * LANES)
                    .sub(FloatVector.fromArray(SP, values, offset + i + 2 * LANES)).max(ZERO);
            FloatVector g3 = FloatVector.fromArray(SP, newValues, nvOffset + i + 3 * LANES)
                    .sub(FloatVector.fromArray(SP, values, offset + i + 3 * LANES)).max(ZERO);
            if (fill) {
                g0.intoArray(contrib, i);
                g1.intoArray(contrib, i + LANES);
                g2.intoArray(contrib, i + 2 * LANES);
                g3.intoArray(contrib, i + 3 * LANES);
            }
            S += (double) g0.add(g1).add(g2.add(g3)).reduceLanes(VectorOperators.ADD);
        }

        final int chunkBound = SP.loopBound(n);
        if (i < chunkBound) {
            FloatVector acc = FloatVector.zero(SP);
            for (; i < chunkBound; i += LANES) {
                FloatVector g = FloatVector.fromArray(SP, newValues, nvOffset + i)
                        .sub(FloatVector.fromArray(SP, values, offset + i)).max(ZERO);
                if (fill) {
                    g.intoArray(contrib, i);
                }
                acc = acc.add(g);
            }
            S += (double) acc.reduceLanes(VectorOperators.ADD);
        }

        S += VectorSupportLegacy.gapRange(newValues, nvOffset, values, offset, fill ? contrib : null, 0, i, n);

        final double probability = (S == 0.0) ? 0.0 : (rangeSum == 0.0 ? 1.0 : S / (S + rangeSum));

        if (fill) {
            final double denom = rangeSum + S;
            if (denom == 0.0 || S == 0.0) {
                Arrays.fill(contrib, 0, n, 0f);
            } else {
                scaleInPlace(contrib, n, (float) (1.0 / denom));
            }
        }
        return probability;
    }

    /** Elementwise, no reduction. */
    private static void scaleInPlace(float[] a, int n, float scale) {
        final FloatVector sc = FloatVector.broadcast(SP, scale);
        int j = 0;
        final int b = SP.loopBound(n);
        for (; j < b; j += LANES) {
            FloatVector.fromArray(SP, a, j).mul(sc).intoArray(a, j);
        }
        VectorSupportLegacy.scaleRange(a, scale, j, n);
    }

    // ---- box U= box (returns the new rangeSum) -----------------------------

    public static double addSlice(float[] values, int offset, int dimensions, float[] other, int otherOffset) {
        final int n = 2 * dimensions;

        double sum = 0.0;
        int i = 0;

        final int blockBound = n - (n % BLOCK);
        for (; i < blockBound; i += BLOCK) {
            FloatVector m0 = FloatVector.fromArray(SP, values, offset + i)
                    .max(FloatVector.fromArray(SP, other, otherOffset + i));
            FloatVector m1 = FloatVector.fromArray(SP, values, offset + i + LANES)
                    .max(FloatVector.fromArray(SP, other, otherOffset + i + LANES));
            FloatVector m2 = FloatVector.fromArray(SP, values, offset + i + 2 * LANES)
                    .max(FloatVector.fromArray(SP, other, otherOffset + i + 2 * LANES));
            FloatVector m3 = FloatVector.fromArray(SP, values, offset + i + 3 * LANES)
                    .max(FloatVector.fromArray(SP, other, otherOffset + i + 3 * LANES));
            m0.intoArray(values, offset + i);
            m1.intoArray(values, offset + i + LANES);
            m2.intoArray(values, offset + i + 2 * LANES);
            m3.intoArray(values, offset + i + 3 * LANES);
            sum += (double) m0.add(m1).add(m2.add(m3)).reduceLanes(VectorOperators.ADD);
        }

        final int chunkBound = SP.loopBound(n);
        if (i < chunkBound) {
            FloatVector acc = FloatVector.zero(SP);
            for (; i < chunkBound; i += LANES) {
                FloatVector m = FloatVector.fromArray(SP, values, offset + i)
                        .max(FloatVector.fromArray(SP, other, otherOffset + i));
                m.intoArray(values, offset + i);
                acc = acc.add(m);
            }
            sum += (double) acc.reduceLanes(VectorOperators.ADD);
        }

        sum += VectorSupportLegacy.addSliceRange(values, offset, other, otherOffset, i, n);
        return sum;
    }

    /**
     * No reduction, early exit. C2 would never vectorize the scalar equivalent (a
     * multi-exit loop), so this one keeps its per-chunk shape.
     */
    public static boolean contains(float[] values, int offset, int dimensions, float[] other, int otherOffset) {
        final int n = 2 * dimensions;
        int i = 0;
        final int bound = SP.loopBound(n);
        for (; i < bound; i += LANES) {
            if (FloatVector.fromArray(SP, values, offset + i).lt(FloatVector.fromArray(SP, other, otherOffset + i))
                    .anyTrue()) {
                return false;
            }
        }
        return VectorSupportLegacy.containsRange(values, offset, other, otherOffset, i, n);
    }

    // ---- AttributionVisitor accumulator kernels ---------------------------

    /**
     * Elementwise, no reduction -- C2's SuperWord already emits vmulpd for the
     * scalar form. This exists so the benchmark can prove whether the Vector API
     * adds anything; if it does not, delete it and keep the plain loop.
     */
    public static void multiply(double[] dir, double decay) {
        final int n = dir.length;
        final DoubleVector vd = DoubleVector.broadcast(DP, decay);
        int i = 0;
        final int bound = DP.loopBound(n);
        for (; i < bound; i += DLANES) {
            DoubleVector.fromArray(DP, dir, i).mul(vd).intoArray(dir, i);
        }
        VectorSupportLegacy.multiplyRange(dir, decay, i, n);
    }

    /** Elementwise fma. Same autovectorize caveat as {@link #multiply}. */
    public static void axpyInto(double[] dst, double[] src, double factor) {
        final int n = dst.length;
        final DoubleVector vf = DoubleVector.broadcast(DP, factor);
        int i = 0;
        final int bound = DP.loopBound(n);
        for (; i < bound; i += DLANES) {
            DoubleVector.fromArray(DP, src, i).fma(vf, DoubleVector.fromArray(DP, dst, i)).intoArray(dst, i);
        }
        VectorSupportLegacy.axpyRange(dst, src, factor, i, n);
    }

    /**
     * dst[dstOff+i] = max(0, exp[expOff+i] - sign*pt[ptOff+i]); returns the sum
     * over the half. exp = canonical [p,-p] (read-only); pt = RAW point, length
     * dim. Two calls do the leaf: high: (exp,0,+1f,pt,0,dst,0,dim); low:
     * (exp,dim,-1f,pt,0,dst,dim,dim). No leaf box, no copy of exp.
     */
    public static double signedGapInto(float[] exp, int expOff, float sign, float[] pt, int ptOff, float[] dst,
            int dstOff, int dim) {
        final FloatVector vs = FloatVector.broadcast(SP, sign);
        final FloatVector ZERO = FloatVector.zero(SP);
        double s = 0.0;
        int i = 0;

        final int blockBound = dim - (dim % BLOCK);
        for (; i < blockBound; i += BLOCK) {
            FloatVector g0 = FloatVector.fromArray(SP, exp, expOff + i)
                    .sub(FloatVector.fromArray(SP, pt, ptOff + i).mul(vs)).max(ZERO);
            FloatVector g1 = FloatVector.fromArray(SP, exp, expOff + i + LANES)
                    .sub(FloatVector.fromArray(SP, pt, ptOff + i + LANES).mul(vs)).max(ZERO);
            FloatVector g2 = FloatVector.fromArray(SP, exp, expOff + i + 2 * LANES)
                    .sub(FloatVector.fromArray(SP, pt, ptOff + i + 2 * LANES).mul(vs)).max(ZERO);
            FloatVector g3 = FloatVector.fromArray(SP, exp, expOff + i + 3 * LANES)
                    .sub(FloatVector.fromArray(SP, pt, ptOff + i + 3 * LANES).mul(vs)).max(ZERO);
            g0.intoArray(dst, dstOff + i);
            g1.intoArray(dst, dstOff + i + LANES);
            g2.intoArray(dst, dstOff + i + 2 * LANES);
            g3.intoArray(dst, dstOff + i + 3 * LANES);
            s += (double) g0.add(g1).add(g2.add(g3)).reduceLanes(VectorOperators.ADD);
        }

        final int chunkBound = SP.loopBound(dim);
        if (i < chunkBound) {
            FloatVector acc = FloatVector.zero(SP);
            for (; i < chunkBound; i += LANES) {
                FloatVector g = FloatVector.fromArray(SP, exp, expOff + i)
                        .sub(FloatVector.fromArray(SP, pt, ptOff + i).mul(vs)).max(ZERO);
                g.intoArray(dst, dstOff + i);
                acc = acc.add(g);
            }
            s += (double) acc.reduceLanes(VectorOperators.ADD);
        }

        s += VectorSupportLegacy.signedGapRange(exp, expOff, sign, pt, ptOff, dst, dstOff, i, dim);
        return s;
    }

    /**
     * The old form reduced TWICE per chunk (high half and low half), so this is the
     * kernel the blocking helps most.
     */
    public static double addPointInPlace(float[] values, int offset, int dim, float[] point, int pOff) {
        double sum = 0.0;
        int i = 0;

        final int blockBound = dim - (dim % (2 * LANES));
        for (; i < blockBound; i += 2 * LANES) {
            FloatVector p0 = FloatVector.fromArray(SP, point, pOff + i);
            FloatVector p1 = FloatVector.fromArray(SP, point, pOff + i + LANES);

            FloatVector h0 = FloatVector.fromArray(SP, values, offset + i).max(p0);
            FloatVector h1 = FloatVector.fromArray(SP, values, offset + i + LANES).max(p1);
            FloatVector l0 = FloatVector.fromArray(SP, values, offset + i + dim).max(p0.neg());
            FloatVector l1 = FloatVector.fromArray(SP, values, offset + i + dim + LANES).max(p1.neg());

            h0.intoArray(values, offset + i);
            h1.intoArray(values, offset + i + LANES);
            l0.intoArray(values, offset + i + dim);
            l1.intoArray(values, offset + i + dim + LANES);

            sum += (double) h0.add(h1).add(l0.add(l1)).reduceLanes(VectorOperators.ADD);
        }

        final int chunkBound = SP.loopBound(dim);
        if (i < chunkBound) {
            FloatVector acc = FloatVector.zero(SP);
            for (; i < chunkBound; i += LANES) {
                FloatVector p = FloatVector.fromArray(SP, point, pOff + i);
                FloatVector hi = FloatVector.fromArray(SP, values, offset + i).max(p);
                FloatVector lo = FloatVector.fromArray(SP, values, offset + i + dim).max(p.neg());
                hi.intoArray(values, offset + i);
                lo.intoArray(values, offset + i + dim);
                acc = acc.add(hi).add(lo);
            }
            sum += (double) acc.reduceLanes(VectorOperators.ADD);
        }

        sum += VectorSupportLegacy.addPointRange(values, offset, dim, point, pOff, i, dim);
        return sum;
    }

    /** shadow path: raw box-vs-box gap dst = max(0, nv - v); returns sum. */
    public static double gapInto(float[] nv, int nvOff, float[] v, int vOff, float[] dst, int dstOff, int n) {
        final FloatVector ZERO = FloatVector.zero(SP);
        double s = 0.0;
        int i = 0;

        final int blockBound = n - (n % BLOCK);
        for (; i < blockBound; i += BLOCK) {
            FloatVector g0 = FloatVector.fromArray(SP, nv, nvOff + i).sub(FloatVector.fromArray(SP, v, vOff + i))
                    .max(ZERO);
            FloatVector g1 = FloatVector.fromArray(SP, nv, nvOff + i + LANES)
                    .sub(FloatVector.fromArray(SP, v, vOff + i + LANES)).max(ZERO);
            FloatVector g2 = FloatVector.fromArray(SP, nv, nvOff + i + 2 * LANES)
                    .sub(FloatVector.fromArray(SP, v, vOff + i + 2 * LANES)).max(ZERO);
            FloatVector g3 = FloatVector.fromArray(SP, nv, nvOff + i + 3 * LANES)
                    .sub(FloatVector.fromArray(SP, v, vOff + i + 3 * LANES)).max(ZERO);
            g0.intoArray(dst, dstOff + i);
            g1.intoArray(dst, dstOff + i + LANES);
            g2.intoArray(dst, dstOff + i + 2 * LANES);
            g3.intoArray(dst, dstOff + i + 3 * LANES);
            s += (double) g0.add(g1).add(g2.add(g3)).reduceLanes(VectorOperators.ADD);
        }

        final int chunkBound = SP.loopBound(n);
        if (i < chunkBound) {
            FloatVector acc = FloatVector.zero(SP);
            for (; i < chunkBound; i += LANES) {
                FloatVector g = FloatVector.fromArray(SP, nv, nvOff + i).sub(FloatVector.fromArray(SP, v, vOff + i))
                        .max(ZERO);
                g.intoArray(dst, dstOff + i);
                acc = acc.add(g);
            }
            s += (double) acc.reduceLanes(VectorOperators.ADD);
        }

        s += VectorSupportLegacy.gapRange(nv, nvOff, v, vOff, dst, dstOff, i, n);
        return s;
    }

    /**
     * gap -> prob in place; dist = prob*(gap + oldRange), oldRange read inline from
     * the SMALL box: oldRange[i] = box[boxOff+i] + box[boxOff+i+dim] (= max_i -
     * min_i), identical across both halves. No reduction.
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
            VectorSupportLegacy.probAndDistRange(gap, dist, box, boxOff, dim, inv, off, i, dim);
        }
    }

    /**
     * leaf seed: prob = gap*inv (in place), dist = prob*gap. oldRange == 0
     * (degenerate box). No reduction.
     */
    public static void probOnlyInto(float[] gap, float[] dist, int len, double invSumNew) {
        final float inv = (float) invSumNew;
        final FloatVector vInv = FloatVector.broadcast(SP, inv);
        int i = 0;
        final int bound = SP.loopBound(len);
        for (; i < bound; i += LANES) {
            FloatVector g = FloatVector.fromArray(SP, gap, i);
            FloatVector p = g.mul(vInv);
            p.intoArray(gap, i);
            p.mul(g).intoArray(dist, i);
        }
        VectorSupportLegacy.probOnlyRange(gap, dist, inv, i, len);
    }

    public static double updateBoundsAllInterchanged(float[] values, int offset, int dim, float[] store, int[] pOffs,
            int start, int end) {
        final int vlen = SP.length();
        final int bound = SP.loopBound(dim);

        int j = 0;
        for (; j < bound; j += vlen) {
            FloatVector h0 = FloatVector.fromArray(SP, values, offset + j);
            FloatVector l0 = FloatVector.fromArray(SP, values, offset + dim + j);
            FloatVector seed = FloatVector.broadcast(SP, Float.NEGATIVE_INFINITY); // was: = NEG_INF_V
            FloatVector h1 = seed, l1 = seed;

            int i = start;
            for (; i + 1 < end; i += 2) {
                FloatVector k0 = FloatVector.fromArray(SP, store, pOffs[i] + j);
                FloatVector k1 = FloatVector.fromArray(SP, store, pOffs[i + 1] + j);
                h0 = h0.max(k0);
                h1 = h1.max(k1);
                l0 = l0.max(k0.neg());
                l1 = l1.max(k1.neg());
            }
            if (i < end) {
                FloatVector k = FloatVector.fromArray(SP, store, pOffs[i] + j);
                h0 = h0.max(k);
                l0 = l0.max(k.neg());
            }
            h0.max(h1).intoArray(values, offset + j);
            l0.max(l1).intoArray(values, offset + dim + j);
        }

        for (; j < dim; j++) { // scalar tail over j, same interchange
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

        double sum = 0.0; // scalar, original order — agreed
        for (int j2 = 0; j2 < 2 * dim; j2++)
            sum += (double) values[offset + j2];
        return sum;
    }

    public static double gapAttributionPruned(float[] values, int offset, int dim, double rangeSum, float[] exp,
            int expOff, float[] contrib, int bs, long[] mask) {

        final boolean fill = contrib != null;
        final FloatVector ZERO = FloatVector.zero(SP); // local; never a field
        double S = 0.0;

        for (int w = 0; w < mask.length; w++) {
            long live = mask[w];
            final int base = w << 6;
            for (long m = live; m != 0; m &= m - 1) {
                final int t = Long.numberOfTrailingZeros(m);
                final int from = (base + t) * bs, to = Math.min(from + bs, dim);
                double blockSum;
                if (to - from == LANES) {
                    FloatVector gh = FloatVector.fromArray(SP, exp, expOff + from)
                            .sub(FloatVector.fromArray(SP, values, offset + from)).max(ZERO);
                    FloatVector gl = FloatVector.fromArray(SP, exp, expOff + dim + from)
                            .sub(FloatVector.fromArray(SP, values, offset + dim + from)).max(ZERO);
                    if (fill) {
                        gh.intoArray(contrib, from);
                        gl.intoArray(contrib, dim + from);
                    }
                    blockSum = gh.add(gl).reduceLanes(VectorOperators.ADD);
                } else {
                    blockSum = VectorSupportLegacy.blockGap(values, offset, dim, exp, expOff, contrib, from, to, fill);
                }
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
            final float invF = (float) (1.0 / (rangeSum + S));
            final FloatVector inv = FloatVector.broadcast(SP, invF);
            for (int w = 0; w < mask.length; w++) {
                final int base = w << 6;
                for (long m = mask[w]; m != 0; m &= m - 1) {
                    final int from = (base + Long.numberOfTrailingZeros(m)) * bs, to = Math.min(from + bs, dim);
                    if (to - from == LANES) {
                        FloatVector.fromArray(SP, contrib, from).mul(inv).intoArray(contrib, from);
                        FloatVector.fromArray(SP, contrib, dim + from).mul(inv).intoArray(contrib, dim + from);
                    } else {
                        for (int j = from; j < to; j++) {
                            contrib[j] *= invF;
                            contrib[dim + j] *= invF;
                        }
                    }
                }
            }
        }
        return probability;
    }

    // ---- distances ---------------------------------------------------------

    public static double L1distance(float[] a, float[] b) {
        final int n = a.length;
        double dist = 0.0;
        int i = 0;

        final int blockBound = n - (n % BLOCK);
        for (; i < blockBound; i += BLOCK) {
            FloatVector d0 = FloatVector.fromArray(SP, a, i).sub(FloatVector.fromArray(SP, b, i)).abs();
            FloatVector d1 = FloatVector.fromArray(SP, a, i + LANES).sub(FloatVector.fromArray(SP, b, i + LANES)).abs();
            FloatVector d2 = FloatVector.fromArray(SP, a, i + 2 * LANES)
                    .sub(FloatVector.fromArray(SP, b, i + 2 * LANES)).abs();
            FloatVector d3 = FloatVector.fromArray(SP, a, i + 3 * LANES)
                    .sub(FloatVector.fromArray(SP, b, i + 3 * LANES)).abs();
            dist += (double) d0.add(d1).add(d2.add(d3)).reduceLanes(VectorOperators.ADD);
        }

        final int chunkBound = SP.loopBound(n);
        if (i < chunkBound) {
            FloatVector acc = FloatVector.zero(SP);
            for (; i < chunkBound; i += LANES) {
                acc = acc.add(FloatVector.fromArray(SP, a, i).sub(FloatVector.fromArray(SP, b, i)).abs());
            }
            dist += (double) acc.reduceLanes(VectorOperators.ADD);
        }

        dist += VectorSupportLegacy.l1Range(a, b, i, n);
        return dist;
    }

    public static double L2distance(float[] a, float[] b) {
        final int n = a.length;
        double sq = 0.0;
        int i = 0;

        final int blockBound = n - (n % BLOCK);
        for (; i < blockBound; i += BLOCK) {
            FloatVector d0 = FloatVector.fromArray(SP, a, i).sub(FloatVector.fromArray(SP, b, i));
            FloatVector d1 = FloatVector.fromArray(SP, a, i + LANES).sub(FloatVector.fromArray(SP, b, i + LANES));
            FloatVector d2 = FloatVector.fromArray(SP, a, i + 2 * LANES)
                    .sub(FloatVector.fromArray(SP, b, i + 2 * LANES));
            FloatVector d3 = FloatVector.fromArray(SP, a, i + 3 * LANES)
                    .sub(FloatVector.fromArray(SP, b, i + 3 * LANES));
            FloatVector s0 = d0.mul(d0), s1 = d1.mul(d1), s2 = d2.mul(d2), s3 = d3.mul(d3);
            sq += (double) s0.add(s1).add(s2.add(s3)).reduceLanes(VectorOperators.ADD);
        }

        final int chunkBound = SP.loopBound(n);
        if (i < chunkBound) {
            FloatVector acc = FloatVector.zero(SP);
            for (; i < chunkBound; i += LANES) {
                FloatVector d = FloatVector.fromArray(SP, a, i).sub(FloatVector.fromArray(SP, b, i));
                acc = acc.add(d.mul(d));
            }
            sq += (double) acc.reduceLanes(VectorOperators.ADD);
        }

        sq += VectorSupportLegacy.l2SqRange(a, b, i, n);
        return Math.sqrt(sq);
    }

    /**
     * Bit-identical to the per-chunk form: max is exact and associative, so no
     * numerics argument applies here at all. Four independent max chains.
     */
    public static double LInfinitydistance(float[] a, float[] b) {
        final int n = a.length;
        FloatVector m0 = FloatVector.zero(SP), m1 = FloatVector.zero(SP);
        FloatVector m2 = FloatVector.zero(SP), m3 = FloatVector.zero(SP);
        int i = 0;

        final int blockBound = n - (n % BLOCK);
        for (; i < blockBound; i += BLOCK) {
            m0 = m0.max(FloatVector.fromArray(SP, a, i).sub(FloatVector.fromArray(SP, b, i)).abs());
            m1 = m1.max(FloatVector.fromArray(SP, a, i + LANES).sub(FloatVector.fromArray(SP, b, i + LANES)).abs());
            m2 = m2.max(
                    FloatVector.fromArray(SP, a, i + 2 * LANES).sub(FloatVector.fromArray(SP, b, i + 2 * LANES)).abs());
            m3 = m3.max(
                    FloatVector.fromArray(SP, a, i + 3 * LANES).sub(FloatVector.fromArray(SP, b, i + 3 * LANES)).abs());
        }
        final int chunkBound = SP.loopBound(n);
        for (; i < chunkBound; i += LANES) {
            m0 = m0.max(FloatVector.fromArray(SP, a, i).sub(FloatVector.fromArray(SP, b, i)).abs());
        }

        double dist = m0.max(m1).max(m2.max(m3)).reduceLanes(VectorOperators.MAX);
        return Math.max(dist, VectorSupportLegacy.lInfRange(a, b, i, n));
    }

    public static double updateBoundsAndGapInterchanged(float[] values, int offset, int dim, float[] store, int[] pOffs,
            int start, int end, float[] exp, int expOff, float[] gapOut, int gapOff, double[] out) {

        final boolean fill = gapOut != null;
        final FloatVector ZERO = FloatVector.zero(SP); // local; never a field
        double S = 0.0, R = 0.0;

        int j = 0;
        final int bound = SP.loopBound(dim);
        for (; j < bound; j += LANES) {
            FloatVector h0 = FloatVector.fromArray(SP, values, offset + j);
            FloatVector l0 = FloatVector.fromArray(SP, values, offset + dim + j);
            FloatVector seed = FloatVector.broadcast(SP, Float.NEGATIVE_INFINITY);
            FloatVector h1 = seed, l1 = seed;

            int i = start;
            for (; i + 1 < end; i += 2) {
                FloatVector k0 = FloatVector.fromArray(SP, store, pOffs[i] + j);
                FloatVector k1 = FloatVector.fromArray(SP, store, pOffs[i + 1] + j);
                h0 = h0.max(k0);
                h1 = h1.max(k1);
                l0 = l0.max(k0.neg());
                l1 = l1.max(k1.neg());
            }
            if (i < end) {
                FloatVector k = FloatVector.fromArray(SP, store, pOffs[i] + j);
                h0 = h0.max(k);
                l0 = l0.max(k.neg());
            }

            FloatVector h = h0.max(h1);
            FloatVector l = l0.max(l1);
            h.intoArray(values, offset + j);
            l.intoArray(values, offset + dim + j);

            FloatVector gh = FloatVector.fromArray(SP, exp, expOff + j).sub(h).max(ZERO);
            FloatVector gl = FloatVector.fromArray(SP, exp, expOff + dim + j).sub(l).max(ZERO);
            if (fill) {
                gh.intoArray(gapOut, gapOff + j);
                gl.intoArray(gapOut, gapOff + dim + j);
            }
            S += (double) gh.add(gl).reduceLanes(VectorOperators.ADD);
            R += (double) h.add(l).reduceLanes(VectorOperators.ADD);
        }

        // scalar tail: same kernel, one implementation
        S += VectorSupportLegacy.updateBoundsAndGapRange(values, offset, dim, store, pOffs, start, end, exp, expOff,
                gapOut, gapOff, out, j, dim);
        R += out[0]; // read the tail's partial before overwriting
        out[0] = R;
        return S;
    }
}
