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

package org.streamingalgorithms.randomcutforest.benchmark;

import java.util.Random;
import java.util.concurrent.TimeUnit;

import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Level;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.Warmup;
import org.openjdk.jmh.runner.Runner;
import org.openjdk.jmh.runner.RunnerException;
import org.openjdk.jmh.runner.options.OptionsBuilder;
import org.streamingalgorithms.randomcutforest.tree.VectorSupport;
import org.streamingalgorithms.randomcutforest.tree.VectorSupportLegacy;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

/**
 * A/B/(C) kernel benchmark.
 *
 * <pre>
 *   A = VectorSupport         (Vector API, blocked reduce)   suffix _A_vector
 *   B = VectorSupportLegacy   (plain Java, C2 only)          suffix _B_scalar
 *   C = Baseline              (Vector API, per-chunk reduce) suffix _C_perChunk
 * </pre>
 *
 * <h2>Run</h2>
 *
 * <pre>
 *   java --add-modules jdk.incubator.vector -jar target/benchmarks.jar VectorSupportBench -prof gc
 * </pre>
 *
 * <h2>The three questions this answers</h2>
 * <ol>
 * <li><b>Where is the A/B crossover?</b> That number sets the default for
 * {@code -Drcf.vector.minLength}. Below it, ship the scalar path -- not as a
 * fallback but as the fast path. Expect the crossover to move right as LANES
 * grows: on AVX-512 at dim=4 the vector loop does not execute a single
 * iteration.</li>
 * <li><b>Did the blocking pay?</b> A vs C, on the two kernels where C is
 * implemented. If A is not clearly ahead on 128-bit NEON at dim=30, the whole
 * premise was wrong and I would want to know.</li>
 * <li><b>Are multiply/axpyInto/updateRecurrence worth any Vector API at
 * all?</b> These are elementwise; C2's SuperWord already handles them. If A
 * does not beat B, delete them from A. ({@code updateRecurrence} is already
 * unconditionally delegated, so its two arms should tie -- that arm is a
 * control, and if it does NOT tie, something is wrong with the harness.)</li>
 * </ol>
 *
 * <h2>Reading it</h2>
 * <ul>
 * <li><b>{@code -prof gc} is not optional.</b> If any {@code _A_} or
 * {@code _C_} arm reports {@code gc.alloc.rate.norm} meaningfully above 0 B/op,
 * a vector box escaped and you are timing an allocation benchmark, not a
 * kernel. C2 special-cases vector instances to avoid boxing rather than relying
 * on general escape analysis, so this fails cliff-edged, not gradually.</li>
 * <li><b>This measures the L1-resident, perfectly-predicted case</b> -- one
 * fixed box, one fixed point, no tree traversal, no cache misses. It is an
 * upper bound on how much the kernel can matter. A 2x kernel win can be 3%
 * end-to-end; only a forest-level benchmark settles that.</li>
 * <li>On AVX-512 hosts, re-run with
 * {@code -jvmArgsAppend -XX:MaxVectorSize=32}. SPECIES_PREFERRED takes 512-bit
 * lanes there, and the downclocking that can cause will not show up in a
 * microbenchmark but will in your production mix.</li>
 * </ul>
 *
 * <h2>Harness notes</h2>
 * <ul>
 * <li>{@code -Drcf.vector.minLength=0} is set at the class level, which forces
 * A onto the vector path at every dimension. Without it, A would silently
 * delegate to B below LANES and the two arms would measure identical code.</li>
 * <li>{@code decay = invSumNew = 1.0} is deliberate. These kernels are
 * multiplicative and in-place; any other value walks the arrays into the
 * subnormal range over millions of invocations, and subnormal arithmetic is
 * ~100x slower on x86. Identical instruction sequence, different fixed
 * point.</li>
 * <li>{@code addSlice}/{@code addPointInPlace} saturate their box to the
 * elementwise max within an iteration. Harmless: {@code Math.max} and lanewise
 * MAX are both branch-free, so cost is data-independent. Restored at
 * {@code Level.Iteration} regardless -- not {@code Level.Invocation}, which at
 * tens of ns per op would dominate.</li>
 * <li>{@code contains} is set up to return true, i.e. the full-scan worst case.
 * If production data fails the predicate early, B looks better than this
 * reports.</li>
 * </ul>
 *
 * java --add-modules jdk.incubator.vector -jar benchmark/target/benchmarks.jar
 * \ VectorSupportBench -prof gc -p dimensions=100
 *
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.NANOSECONDS)
@Warmup(iterations = 5, time = 1, timeUnit = TimeUnit.SECONDS)
@Measurement(iterations = 8, time = 1, timeUnit = TimeUnit.SECONDS)
@Fork(value = 3, jvmArgsAppend = { "--add-modules=jdk.incubator.vector", "-XX:+UseFMA", "-Drcf.vector.minLength=0" })
public class VectorSupportBench {

    @State(Scope.Thread)
    public static class Data {

        /**
         * 2..8 is where users actually live; 24 = 3 base-dim x shingle 8; 100 = D1/D2
         * effective. The bottom half decides whether the low-n dispatch is worth
         * having, and is the part with no prior data.
         */
        @Param({ "2", "3", "5", "8", "16", "24", "64", "100", "128", "256" })
        public int dimensions;

        public int n;
        public float[] values, other, newValues, expanded, contrib, gap, dist, point, va, vb, comp;
        public double[] dir, src, dst;

        private float[] valuesP, gapP;
        private double[] dirP, dstP;

        public double rangeSum;
        public final double decay = 1.0;
        public final double aCoef = 0.5;
        public final double factor = 1.0e-3;
        public final double invSumNew = 1.0;

        @Setup(Level.Trial)
        public void setupTrial() {
            System.out.println("  " + VectorSupport.describe());
            final Random r = new Random(42);
            final int d = dimensions;
            n = 2 * d;

            values = new float[n];
            other = new float[n];
            newValues = new float[n];
            expanded = new float[n];
            contrib = new float[n];
            gap = new float[n];
            dist = new float[n];
            comp = new float[n];
            point = new float[d];
            va = new float[d];
            vb = new float[d];
            dir = new double[n];
            src = new double[n];
            dst = new double[n];

            for (int i = 0; i < d; i++) {
                values[i] = 1f + Math.abs((float) r.nextGaussian());
                values[i + d] = 1f + Math.abs((float) r.nextGaussian());
                other[i] = values[i] * 0.5f;
                other[i + d] = values[i + d] * 0.5f;
                newValues[i] = values[i] + (float) r.nextGaussian();
                newValues[i + d] = values[i + d] + (float) r.nextGaussian();
                point[i] = (float) (2.0 * r.nextGaussian());
                va[i] = (float) r.nextGaussian();
                vb[i] = (float) r.nextGaussian();
            }
            for (int i = 0; i < n; i++) {
                gap[i] = Math.abs((float) r.nextGaussian());
                comp[i] = (float) r.nextGaussian();
                dir[i] = r.nextGaussian();
                src[i] = r.nextGaussian();
                dst[i] = r.nextGaussian();
            }
            VectorSupport.expandInto(point, 0, expanded, 0, d);

            double s = 0;
            for (float v : values) {
                s += v;
            }
            rangeSum = s;

            valuesP = values.clone();
            gapP = gap.clone();
            dirP = dir.clone();
            dstP = dst.clone();
        }

        @Setup(Level.Iteration)
        public void reset() {
            System.arraycopy(valuesP, 0, values, 0, n);
            System.arraycopy(gapP, 0, gap, 0, n);
            System.arraycopy(dirP, 0, dir, 0, n);
            System.arraycopy(dstP, 0, dst, 0, n);
        }
    }

    // ---- gapAttribution, probability only -----------------------------------

    @Benchmark
    public double gapAttributionProb_A_vector(Data s) {
        return VectorSupport.gapAttribution(s.values, 0, s.dimensions, s.rangeSum, s.newValues, 0, null, null);
    }

    @Benchmark
    public double gapAttributionProb_B_scalar(Data s) {
        return VectorSupportLegacy.gapAttribution(s.values, 0, s.dimensions, s.rangeSum, s.newValues, 0, null, null);
    }

    @Benchmark
    public double gapAttributionProb_C_perChunk(Data s) {
        return Baseline.gapAttribution(s.values, 0, s.dimensions, s.rangeSum, s.newValues, 0, null);
    }

    // ---- gapAttribution, with attribution fill -------------------------------

    @Benchmark
    public double gapAttributionFill_A_vector(Data s) {
        return VectorSupport.gapAttribution(s.values, 0, s.dimensions, s.rangeSum, s.newValues, 0, s.contrib, null);
    }

    @Benchmark
    public double gapAttributionFill_B_scalar(Data s) {
        return VectorSupportLegacy.gapAttribution(s.values, 0, s.dimensions, s.rangeSum, s.newValues, 0, s.contrib,
                null);
    }

    // ---- addSlice ------------------------------------------------------------

    @Benchmark
    public double addSlice_A_vector(Data s) {
        return VectorSupport.addSlice(s.values, 0, s.dimensions, s.other, 0);
    }

    @Benchmark
    public double addSlice_B_scalar(Data s) {
        return VectorSupportLegacy.addSlice(s.values, 0, s.dimensions, s.other, 0);
    }

    // ---- contains ------------------------------------------------------------

    @Benchmark
    public boolean contains_A_vector(Data s) {
        return VectorSupport.contains(s.values, 0, s.dimensions, s.other, 0);
    }

    @Benchmark
    public boolean contains_B_scalar(Data s) {
        return VectorSupportLegacy.contains(s.values, 0, s.dimensions, s.other, 0);
    }

    // ---- multiply (C2 autovectorizes the scalar form -- does A add anything?) -

    @Benchmark
    public void multiply_A_vector(Data s) {
        VectorSupport.multiply(s.dir, s.decay);
    }

    @Benchmark
    public void multiply_B_scalar(Data s) {
        VectorSupportLegacy.multiply(s.dir, s.decay);
    }

    // ---- updateRecurrence (control: A delegates, so these MUST tie) ----------

    @Benchmark
    public void updateRecurrence_A_vector(Data s) {
        VectorSupport.updateRecurrence(s.dir, s.comp, s.aCoef, s.decay);
    }

    @Benchmark
    public void updateRecurrence_B_scalar(Data s) {
        VectorSupportLegacy.updateRecurrence(s.dir, s.comp, s.aCoef, s.decay);
    }

    // ---- axpyInto ------------------------------------------------------------

    @Benchmark
    public void axpyInto_A_vector(Data s) {
        VectorSupport.axpyInto(s.dst, s.src, s.factor);
    }

    @Benchmark
    public void axpyInto_B_scalar(Data s) {
        VectorSupportLegacy.axpyInto(s.dst, s.src, s.factor);
    }

    // ---- signedGapInto (both halves = one leaf) ------------------------------

    @Benchmark
    public double signedGapInto_A_vector(Data s) {
        double hi = VectorSupport.signedGapInto(s.expanded, 0, +1f, s.point, 0, s.gap, 0, s.dimensions);
        double lo = VectorSupport.signedGapInto(s.expanded, s.dimensions, -1f, s.point, 0, s.gap, s.dimensions,
                s.dimensions);
        return hi + lo;
    }

    @Benchmark
    public double signedGapInto_B_scalar(Data s) {
        double hi = VectorSupportLegacy.signedGapInto(s.expanded, 0, +1f, s.point, 0, s.gap, 0, s.dimensions);
        double lo = VectorSupportLegacy.signedGapInto(s.expanded, s.dimensions, -1f, s.point, 0, s.gap, s.dimensions,
                s.dimensions);
        return hi + lo;
    }

    // ---- addPointInPlace (reduced TWICE per chunk before -- biggest win) -----

    @Benchmark
    public double addPointInPlace_A_vector(Data s) {
        return VectorSupport.addPointInPlace(s.values, 0, s.dimensions, s.point, 0);
    }

    @Benchmark
    public double addPointInPlace_B_scalar(Data s) {
        return VectorSupportLegacy.addPointInPlace(s.values, 0, s.dimensions, s.point, 0);
    }

    @Benchmark
    public double addPointInPlace_C_perChunk(Data s) {
        return Baseline.addPointInPlace(s.values, 0, s.dimensions, s.point, 0);
    }

    // ---- gapInto -------------------------------------------------------------

    @Benchmark
    public double gapInto_A_vector(Data s) {
        return VectorSupport.gapInto(s.newValues, 0, s.values, 0, s.gap, 0, s.n);
    }

    @Benchmark
    public double gapInto_B_scalar(Data s) {
        return VectorSupportLegacy.gapInto(s.newValues, 0, s.values, 0, s.gap, 0, s.n);
    }

    // ---- probAndDistInto -----------------------------------------------------

    @Benchmark
    public void probAndDistInto_A_vector(Data s) {
        VectorSupport.probAndDistInto(s.gap, s.dist, s.values, 0, s.dimensions, s.invSumNew);
    }

    @Benchmark
    public void probAndDistInto_B_scalar(Data s) {
        VectorSupportLegacy.probAndDistInto(s.gap, s.dist, s.values, 0, s.dimensions, s.invSumNew);
    }

    // ---- probOnlyInto --------------------------------------------------------

    @Benchmark
    public void probOnlyInto_A_vector(Data s) {
        VectorSupport.probOnlyInto(s.gap, s.dist, s.n, s.invSumNew);
    }

    @Benchmark
    public void probOnlyInto_B_scalar(Data s) {
        VectorSupportLegacy.probOnlyInto(s.gap, s.dist, s.n, s.invSumNew);
    }

    // ---- distances -----------------------------------------------------------

    @Benchmark
    public double l1_A_vector(Data s) {
        return VectorSupport.L1distance(s.va, s.vb);
    }

    @Benchmark
    public double l1_B_scalar(Data s) {
        return VectorSupportLegacy.L1distance(s.va, s.vb);
    }

    @Benchmark
    public double l2_A_vector(Data s) {
        return VectorSupport.L2distance(s.va, s.vb);
    }

    @Benchmark
    public double l2_B_scalar(Data s) {
        return VectorSupportLegacy.L2distance(s.va, s.vb);
    }

    @Benchmark
    public double lInf_A_vector(Data s) {
        return VectorSupport.LInfinitydistance(s.va, s.vb);
    }

    @Benchmark
    public double lInf_B_scalar(Data s) {
        return VectorSupportLegacy.LInfinitydistance(s.va, s.vb);
    }

    public static void main(String[] args) throws RunnerException {
        new Runner(new OptionsBuilder().include(VectorSupportBench.class.getSimpleName()).build()).run();
    }

    // =======================================================================
    // C: the shipped-today kernels, verbatim -- one horizontal reduce per
    // chunk, float within the chunk, double across chunks. Kept ONLY to prove
    // the blocking was worth doing. Delete this class once you have the number.
    // =======================================================================
    static final class Baseline {
        private static final VectorSpecies<Float> SP = FloatVector.SPECIES_PREFERRED;
        private static final int LANES = SP.length();

        private Baseline() {
        }

        static double gapAttribution(float[] values, int offset, int dimensions, double rangeSum, float[] newValues,
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
                if (fill) {
                    gap.intoArray(contrib, i);
                }
                S += (double) gap.reduceLanes(VectorOperators.ADD);
            }
            for (; i < n; i++) {
                float g = Math.max(0f, newValues[nvOffset + i] - values[offset + i]);
                if (fill) {
                    contrib[i] = g;
                }
                S += g;
            }

            double probability = (S == 0.0) ? 0.0 : (rangeSum == 0.0 ? 1.0 : S / (S + rangeSum));

            if (fill) {
                double denom = rangeSum + S;
                if (denom == 0.0 || S == 0.0) {
                    java.util.Arrays.fill(contrib, 0f);
                } else {
                    final float scale = (float) (1.0 / denom);
                    FloatVector sc = FloatVector.broadcast(SP, scale);
                    int j = 0;
                    final int b2 = SP.loopBound(n);
                    for (; j < b2; j += LANES) {
                        FloatVector.fromArray(SP, contrib, j).mul(sc).intoArray(contrib, j);
                    }
                    for (; j < n; j++) {
                        contrib[j] *= scale;
                    }
                }
            }
            return probability;
        }

        static double addPointInPlace(float[] values, int offset, int dim, float[] point, int pOff) {
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
                float mx = Math.max(values[offset + i], point[pOff + i]);
                float negMin = Math.max(values[offset + i + dim], -point[pOff + i]);
                values[offset + i] = mx;
                values[offset + i + dim] = negMin;
                sum += mx + negMin;
            }
            return sum;
        }
    }
}
