package org.streamingalgorithms.randomcutforest.benchmark;

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

import java.time.Duration;
import java.time.Instant;
import java.util.Random;

import org.openjdk.jol.info.GraphLayout;
import org.streamingalgorithms.randomcutforest.RandomCutForest;
import org.streamingalgorithms.randomcutforest.benchmark.operations.Datasets;
import org.streamingalgorithms.randomcutforest.benchmark.operations.ScoringOperations;
import org.streamingalgorithms.randomcutforest.benchmark.operations.ScoringOperations.Func;
import org.streamingalgorithms.randomcutforest.benchmark.operations.ScoringOperations.Kind;
import org.streamingalgorithms.randomcutforest.benchmark.operations.ScoringOperations.Mode;

/**
 * ===========================================================================
 * OP 1 -- RCF scoring. COLD driver (wall-clock, first-invocation, no warmup).
 *
 * Same operation as ScoringBenchmark (both call ScoringOperations.scoreOne), so
 * the two can never drift. This driver owns the regime JMH structurally cannot
 * show: the NON-stationary single pass production actually experiences. It
 * makes no steady-state claim, so: - it may sweep the update probability (score
 * each point, update ~prob of them) -- a coherent single pass, not a stationary
 * loop; - it is the natural home for GC profiling and JFR on the FIRST
 * invocation (run under -XX:StartFlightRecording=... or async-profiler;
 * deserialization cold cost, cold cache, cold JIT are all visible here and
 * nowhere in JMH).
 *
 * It is not a functional test, but it carries the cheap finiteness /
 * non-degeneracy ORACLE that the JMH region must not (asserting inside a warmed
 * hot loop would distort it). If a cell is non-finite or all-zero it throws.
 *
 * Intentionally longer than the JMH run. Every cell rebuilds and re-saturates a
 * forest (25k updates), so the sweep below is minutes, by design. Subset via
 * args if needed.
 * ===========================================================================
 * run as java -Djdk.attach.allowAttachSelf=true -Djol.magicFieldOffset=true \
 * -XX:StartFlightRecording=filename=cold-scoring.jfr,settings=profile \ -cp
 * benchmark/target/benchmarks.jar \
 * org.streamingalgorithms.randomcutforest.benchmark.ScoringColdMain
 */
public final class ScoringColdMain {

    private static final double[] CACHE = { 0.0, 0.2, 0.5, 1.0 };
    private static final double[] PROB = { 0.0, 0.1, 1.0 }; // floor / trickle / every-point
    private static final long SEED = 99L;

    private ScoringColdMain() {
    }

    public static void main(String[] args) {
        // Pass A -- coverage / drift tripwire: run EVERY (mode,kind,func) cell through
        // the shared scoreOne at one representative (dataset,cache), prob=0. Proves the
        // whole surface is finite and exercised via one code path.
        System.out.printf("%n==== coverage pass | D2 | cache=0.20 | prob=0.00 | %d dims ====%n",
                Datasets.Id.D2.effectiveDims());
        Datasets.Prepared cov = Datasets.prepare(Datasets.Id.D2, 0.20, SEED);
        for (Mode m : Mode.values()) {
            for (Kind k : Kind.values()) {
                for (Func fn : Func.values()) {
                    runCell(cov, Datasets.Id.D2, 0.20, m, k, fn, 0.0, /* rebuildAllowed */ false);
                }
            }
        }

        // Pass B -- throughput sweep on representative kinds across dataset x cache x
        // prob. SCALAR/DEFAULT/EXACT is the headline; DENSITY exposes the
        // InterpolationMeasure allocation cost. prob NE 0,1 can mutate the forest
        for (Datasets.Id id : Datasets.Id.values()) {
            for (double cache : CACHE) {
                for (double prob : PROB) {
                    System.out.printf("%n==== %s | cache=%.2f | prob=%.2f | %d dims ====%n", id, cache, prob,
                            id.effectiveDims());
                    runCell(Datasets.prepare(id, cache, SEED), id, cache, Mode.EXACT, Kind.SCALAR, Func.DEFAULT, prob,
                            true);
                    runCell(Datasets.prepare(id, cache, SEED), id, cache, Mode.EXACT, Kind.DENSITY, Func.DEFAULT, prob,
                            true);
                }
            }
        }
    }

    /**
     * One cold cell: single wall-clock pass over the hold-out, scoring every point
     * and updating ~prob of them, then JOL footprint + oracle + one row. The update
     * set is precomputed OUTSIDE the timed region (fixed indices, so cells are
     * comparable and there is no RNG on the hot path).
     */
    private static void runCell(Datasets.Prepared p, Datasets.Id id, double cache, Mode mode, Kind kind, Func func,
            double prob, boolean rebuildAllowed) {
        final RandomCutForest f = p.forest;
        final double[][] dD = p.scoreD;
        final float[][] dF = p.scoreF;

        boolean[] doUpdate = new boolean[Datasets.SCORED];
        if (prob > 0.0) {
            Random rnd = new Random(0);
            for (int i = 0; i < Datasets.SCORED; i++) {
                doUpdate[i] = rnd.nextDouble() < prob;
            }
        }

        double checksum = 0.0;
        Instant start = Instant.now();
        for (int i = 0; i < Datasets.SCORED; i++) {
            checksum += ScoringOperations.scoreOne(f, dD[i], dF[i], mode, kind, func);
            // doUpdate = 0 allows for pure scoring measurements
            // doUpdate = 1 is the intended use in RCF
            if (doUpdate[i]) {
                f.update(dD[i]);
            }
        }
        double seconds = Duration.between(start, Instant.now()).toNanos() / 1_000_000_000.0;

        double forestMb = GraphLayout.parseInstance(f).totalSize() / 1048576.0;
        double tuplesPerSec = Datasets.SCORED / seconds;

        String label = String.format("%-5s %-6s %-11s %-7s prob=%.2f", id, mode, kind, func, prob);
        System.out.printf("%s | %7.3f s | %10.1f tuples/s | forest %6.3f MB | %d dims | chk=%.4g%n", label, seconds,
                tuplesPerSec, forestMb, p.dims, checksum);

        if (!Double.isFinite(checksum)) {
            throw new AssertionError(label + ": non-finite checksum");
        }
        if (checksum == 0.0) {
            throw new AssertionError(label + ": degenerate (all-zero) checksum");
        }
    }
}
