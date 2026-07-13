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
import org.openjdk.jmh.annotations.OperationsPerInvocation;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.Warmup;
import org.openjdk.jmh.infra.Blackhole;
import org.streamingalgorithms.randomcutforest.tree.ArrayBox;
import org.streamingalgorithms.randomcutforest.tree.ArrayBoxSimd;

/**
 * ===========================================================================
 * MICROBENCH -- ArrayBox.gapAttribution (scalar) vs ArrayBoxSimd.gapAttribution
 * (Vector API). This is the KERNEL question, deliberately NOT the end-to-end
 * question ScoringBenchmark answers: there the box arithmetic is a slice of
 * per-node cost next to traversal, the box cache, and DiVector allocation, and
 * a 4-8x kernel win shows up as single-digit-percent throughput lost in fork
 * variance. Here the kernel is the ONLY thing in the timed loop, so the win (or
 * its absence) is legible, and -prof perfasm shows the exact instructions.
 *
 * WHAT'S FAIR ABOUT THE COMPARISON. Both paths read the SAME packed float[]
 * buffer (box slice j at offset j*2*dim); the scalar path enters through the
 * production instance method ArrayBox.probabilityOfCut, the SIMD path through
 * the static kernel on the raw slice. The expanded query [p,-p] is built ONCE
 * in @Setup, not per call -- that mirrors the visitor integration, where the
 * point is fixed for a whole traversal and expansion is amortized across every
 * node. Timing per-call expansion would measure a cost the real hot path never
 * pays.
 *
 * READING THE SWEEP (the crossover is the point). - SPECIES_PREFERRED length is
 * 4 on NEON-128, 8 on AVX2-256, 16 on AVX-512. - ArrayBoxSimd falls back to
 * scalar when the slice (2*dim) < 2*LANES, so for dim below ~LANES the "simd*"
 * rows will MATCH the "scalar*" rows by construction -- that equality is the
 * guard working, not a null result. - The interesting rows are dim >= 8 on NEON
 * and dim >= 8..16 on AVX2, where the slice is several full vectors and the gap
 * kernel is memory-light. - Working set = 256 * 2*dim floats, so it grows with
 * dim (dim=8 ~16KB / L1, dim=128 ~256KB / L2). Cross-dim numbers are
 * throughput-comparable but the largest dims start paying L2 latency -- read
 * the SIMD/scalar RATIO within a dim, not raw ns across dims.
 *
 * CORRECTNESS GATE. @Setup runs both kernels over every box and aborts the
 * trial if they disagree (prob relative err or per-component abs err > 1e-4).
 * Tiny diffs are expected -- float reduction is reordered -- but a real port
 * bug or a silent scalar-fallback-with-wrong-result trips this before any
 * number is trusted.
 *
 * REQUIRES: ArrayBoxSimd must be `public` and live in
 * org.streamingalgorithms.randomcutforest.tree (so both it and ArrayBox are
 * importable here and it can reach ArrayBox internals when you wire the shadow
 * paths). If you kept it package-private in the default package, move it.
 * ===========================================================================
 * run: ts=$(date +%Y%m%d-%H%M%S) java --add-modules jdk.incubator.vector \ -jar
 * benchmark/target/benchmarks.jar ArrayBoxScoringBench \ -rf json -rff
 * "boxkernel-$ts.json" | tee "boxkernel-$ts.txt"
 *
 * verify codegen (do this once, don't trust the ns otherwise): java
 * --add-modules jdk.incubator.vector \ -jar benchmark/target/benchmarks.jar
 * "ArrayBoxScoringBench.simdProbability" \ -p dim=32 -prof
 * "perfasm:intelSyntax=true" 2>&1 | grep -E 'maxps|fmax' # AVX2 -> vmaxps %ymm*
 * ; NEON -> fmax v*.4s ; if you see scalar maxss, it # didn't intrinsify --
 * check warmup and that the box arrives as ArrayBox.
 * ===========================================================================
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.NANOSECONDS)
@Warmup(iterations = 5, time = 2)
@Measurement(iterations = 5, time = 2)
@Fork(value = 3, jvmArgsAppend = { "--add-modules=jdk.incubator.vector" })
@State(Scope.Thread)
public class ArrayBoxScoringBench {

    /**
     * Fixed op count so @OperationsPerInvocation is a literal; ns/op is per single
     * box-kernel.
     */
    static final int NBOXES = 256;

    @Param({ "1", "2", "3", "4", "8", "16", "32", "64", "128" })
    int dim;

    private float[] buffer; // NBOXES boxes packed contiguously, slice j at offsets[j]
    private int[] offsets; // offsets[j] = j * 2 * dim
    private double[] rangeSums; // per-box encoded range sum (= sum of the slice)
    private ArrayBox[] boxes; // views over `buffer` -- the scalar entry point
    private float[] point; // fixed query, length dim
    private float[] expanded; // [p, -p], length 2*dim -- built ONCE (visitor amortization)
    private float[] contrib; // reused attribution buffer, length 2*dim

    @Setup(Level.Trial)
    public void setUp() {
        final int slice = 2 * dim;
        final Random rng = new Random(0x9E3779B97F4A7C15L ^ dim);

        buffer = new float[NBOXES * slice];
        offsets = new int[NBOXES];
        rangeSums = new double[NBOXES];
        boxes = new ArrayBox[NBOXES];
        contrib = new float[slice];

        // Query spread wider than the boxes so some coords land outside (positive
        // gap) and some inside (zero gap): a realistic mixed S, never all-zero.
        point = new float[dim];
        for (int i = 0; i < dim; i++) {
            point[i] = (float) (rng.nextGaussian() * 1.5);
        }
        expanded = ArrayBoxSimd.expand(point);

        for (int j = 0; j < NBOXES; j++) {
            final int off = j * slice;
            offsets[j] = off;
            double sum = 0.0;
            for (int i = 0; i < dim; i++) {
                float center = (float) (rng.nextGaussian() * 0.5);
                float half = 0.5f + (float) rng.nextDouble(); // (0.5, 1.5)
                float mx = center + half;
                float mn = center - half;
                buffer[off + i] = mx; // max half
                buffer[off + i + dim] = -mn; // negated min half (the encoding)
                sum += (mx - mn);
            }
            rangeSums[j] = sum;
            boxes[j] = new ArrayBox(buffer, off, dim, sum); // stores by reference
        }

        verifyAgreement(); // fail fast before timing a wrong path
    }

    /**
     * Both kernels, every box, must agree. Guards against silent fallback / port
     * bugs.
     */
    private void verifyAgreement() {
        final int slice = 2 * dim;
        float[] cA = new float[slice];
        float[] cB = new float[slice];
        double worst = 0.0;
        for (int j = 0; j < NBOXES; j++) {
            double pa = boxes[j].probabilityOfCut(point, cA);
            double pb = ArrayBoxSimd.gapAttribution(buffer, offsets[j], dim, rangeSums[j], expanded, 0, cB);
            worst = Math.max(worst, Math.abs(pa - pb) / Math.max(1e-12, Math.abs(pa)));
            for (int k = 0; k < slice; k++) {
                worst = Math.max(worst, Math.abs(cA[k] - cB[k]));
            }
        }
        if (worst > 1e-4) {
            throw new AssertionError("scalar vs SIMD disagree at dim=" + dim + " worst=" + worst
                    + " -- do NOT trust timings until this is resolved");
        }
    }

    // ---- probability only (ScoreVisitor path: contrib == null) ------------

    @Benchmark
    @OperationsPerInvocation(NBOXES)
    public double scalarProbability() {
        final ArrayBox[] b = boxes;
        final float[] p = point;
        double checksum = 0.0;
        for (int j = 0; j < NBOXES; j++) {
            checksum += b[j].probabilityOfCut(p, null);
        }
        return checksum;
    }

    @Benchmark
    @OperationsPerInvocation(NBOXES)
    public double simdProbability() {
        final float[] buf = buffer;
        final int[] offs = offsets;
        final double[] rs = rangeSums;
        final float[] ex = expanded;
        final int d = dim;
        double checksum = 0.0;
        for (int j = 0; j < NBOXES; j++) {
            checksum += ArrayBoxSimd.gapAttribution(buf, offs[j], d, rs[j], ex, 0, null);
        }
        return checksum;
    }

    // ---- probability + attribution fill (AttributionVisitor path) ---------

    @Benchmark
    @OperationsPerInvocation(NBOXES)
    public double scalarAttribution(Blackhole bh) {
        final ArrayBox[] b = boxes;
        final float[] p = point;
        final float[] c = contrib;
        double checksum = 0.0;
        for (int j = 0; j < NBOXES; j++) {
            checksum += b[j].probabilityOfCut(p, c);
            bh.consume(c); // keep the fill alive (c escapes -> stores retained)
        }
        return checksum;
    }

    @Benchmark
    @OperationsPerInvocation(NBOXES)
    public double simdAttribution(Blackhole bh) {
        final float[] buf = buffer;
        final int[] offs = offsets;
        final double[] rs = rangeSums;
        final float[] ex = expanded;
        final float[] c = contrib;
        final int d = dim;
        double checksum = 0.0;
        for (int j = 0; j < NBOXES; j++) {
            checksum += ArrayBoxSimd.gapAttribution(buf, offs[j], d, rs[j], ex, 0, c);
            bh.consume(c);
        }
        return checksum;
    }
}
