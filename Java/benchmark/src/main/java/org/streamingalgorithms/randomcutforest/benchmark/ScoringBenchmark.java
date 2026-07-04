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
import org.openjdk.jmh.annotations.TearDown;
import org.openjdk.jmh.annotations.Warmup;
import org.openjdk.jol.info.GraphLayout;
import org.streamingalgorithms.randomcutforest.RandomCutForest;
import org.streamingalgorithms.randomcutforest.benchmark.operations.Datasets;
import org.streamingalgorithms.randomcutforest.benchmark.operations.ScoringOperations;

/**
 * ===========================================================================
 * OP 1 -- RCF scoring throughput. JMH driver (warmed, steady-state).
 *
 * STATIONARITY ASSUMPTION (read before trusting a number). RCF is never
 * stationary in production -- it is continuously-updated ML: the forest fills
 * up, the box cache warms, the decay tail runs. JMH's validity REQUIRES a
 * steady state, so this driver measures a useful FICTION: the forest is frozen
 * (update probability = 0, no mutation in the measured region) and
 * pre-saturated (Datasets.prepare drove 25k points through it, so the sampler
 * sits in dynamic equilibrium -- size stationary, per-call cost stationary).
 * What we get is a clean, controlled, apples-to-apples comparison ACROSS
 * scoring entry points. The non-stationary trajectory that production actually
 * experiences is the cold driver's job, not this one.
 *
 * This is a MICROBENCHMARK. It is (optionally) evaluated in parallel two ways,
 * BOTH off in this baseline: - inside a forest, across trees:
 * parallelExecutionEnabled(true) + @Threads(1) - across independent models:
 * parallelExecutionEnabled(false) + @Threads(N) + @State(Scope.Thread) forest
 * See the SYNC/ASYNC note in the module README before enabling either; the
 * across-models path must keep the data at Scope.Benchmark (shared, read-only)
 * or it OOMs on N copies of the dataset.
 *
 * PROFILING. Allocation flow is JMH-native here: run with `-prof gc` and read
 * gc.alloc.rate.norm (B/op) -- SCALAR should read ~0, ATTRIBUTION and DENSITY
 * nonzero (DiVector / InterpolationMeasure). Retained footprint is the JOL line
 * in @TearDown (a side channel, never a JMH score). `-prof jfr` also works.
 *
 * DYNAMIC-CACHE FORWARD NOTE. Today boundingBoxCacheFraction residency is
 * static at 0.0 and 1.0. For dynamic residency add flags for freezing. The cold
 * driver needs no such flag -- it is allowed to be non-stationary.
 * ===========================================================================
 * run as ts=$(date +%Y%m%d-%H%M%S) java -jar benchmark/target/benchmarks.jar
 * ScoringBenchmark -prof gc \ -rf json -rff "scoring-$ts.json" | tee
 * "scoring-$ts.txt"
 */
@BenchmarkMode(Mode.Throughput)
@OutputTimeUnit(TimeUnit.SECONDS)
@Warmup(iterations = 3, time = 2)
@Measurement(iterations = 5, time = 2)
@Fork(value = 1, jvmArgsAppend = { "-Djdk.attach.allowAttachSelf=true", "-Djol.magicFieldOffset=true" })
@State(Scope.Benchmark)
public class ScoringBenchmark {

    @Param({ "D1", "D2" })
    Datasets.Id dataset;

    @Param({ "0.0", "0.2", "0.5", "1.0" })
    double cacheFraction;

    // No explicit value list on an enum @Param -> JMH enumerates ALL constants.
    // The full (mode x kind x func) product is the matrix; the label is derived,
    // so it cannot drift from what ran. (APPROX x DENSITY is a reserved cell that
    // currently mirrors EXACT x DENSITY -- see ScoringOperations.)
    @Param
    ScoringOperations.Mode mode;

    @Param
    ScoringOperations.Kind kind;

    @Param
    ScoringOperations.Func func;

    private RandomCutForest forest;
    private double[][] scoreD;
    private float[][] scoreF;

    @Setup(Level.Trial)
    public void setUp() {
        Datasets.Prepared p = Datasets.prepare(dataset, cacheFraction, 99L);
        forest = p.forest;
        scoreD = p.scoreD;
        scoreF = p.scoreF;
        // Frozen from here: prob = 0, no update in the measured region.
    }

    @Benchmark
    @OperationsPerInvocation(Datasets.SCORED)
    public double score() {
        // Hoist fields to locals so the hot loop is clean; return the checksum so
        // JMH consumes it (no Blackhole needed, no dead-code elimination).
        final RandomCutForest f = forest;
        final ScoringOperations.Mode m = mode;
        final ScoringOperations.Kind k = kind;
        final ScoringOperations.Func fn = func;
        final double[][] dD = scoreD;
        final float[][] dF = scoreF;
        double checksum = 0.0;
        for (int i = 0; i < Datasets.SCORED; i++) {
            checksum += ScoringOperations.scoreOne(f, dD[i], dF[i], m, k, fn);
        }
        return checksum;
    }

    @TearDown(Level.Trial)
    public void footprint() {
        long bytes = GraphLayout.parseInstance(forest).totalSize();
        System.err.printf("JOL forest[%s] cache=%.2f : %.3f MB | %d dims%n", dataset, cacheFraction, bytes / 1048576.0,
                dataset.effectiveDims());
    }
}
