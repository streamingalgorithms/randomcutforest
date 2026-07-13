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

import java.util.concurrent.TimeUnit;

import org.openjdk.jmh.annotations.*;
import org.openjdk.jol.info.GraphLayout;
import org.streamingalgorithms.randomcutforest.RandomCutForest;
import org.streamingalgorithms.randomcutforest.benchmark.operations.Datasets;

/**
 * ===========================================================================
 * OP 2 -- RCF MULTI-VISITOR throughput (impute + extrapolate). Sibling of
 * ScoringBenchmark; SAME frozen/pre-saturated fiction (see that file's header).
 *
 * WHY A SEPARATE DRIVER. The reuse+fold refactor never touched traverseMulti.
 * ImputeVisitor still allocates: - one `new ImputeVisitor` per tree
 * (non-reusable factory) - one `new ImputeVisitor(this)` per FORK (==
 * #missing-dim cuts on path) - one ConditionalTreeSample per tree - two
 * Arrays.copyOf(queryPoint) per visitor (ctor + copy-ctor) - at cache=0: one
 * getArrayBox() per internal node per tree (box rebuild) This driver exists to
 * size those BEFORE any ReusableMultiVisitor work.
 *
 * READING IT (-prof gc, gc.alloc.rate.norm B/op): cache=1.0 B/op = fork +
 * per-tree visitor + sample + queryPoint copies (box-free; cached slices, no
 * getArrayBox) cache=0.0 - cache=1.0 = box reconstruction allocations
 * (per-node, per-tree) d(B/op)/d(missing) = the fork cost; slope isolates new
 * ImputeVisitor(this)
 *
 * LAYERING. This is the RAW RCF (visitor) floor. TRCF/Caster forecast stacked
 * on top add the (double-precision) preprocessor allocation; measure those in a
 * sibling driver and read the DIFFERENCE as preprocessor cost -- don't
 * conflate.
 *
 * run: java -jar benchmark/target/benchmarks.jar ImputeForecastBenchmark \
 * -prof gc -rf json -rff "impute-$(date +%Y%m%d-%H%M%S).json"
 *
 * or java -jar benchmark/target/benchmarks.jar \
 * "ImputeForecastBenchmark.forecast" \ -p dataset=D2 -p missing=5 -p
 * cacheFraction=0.0,1.0 -prof gc
 *
 * or java -jar benchmark/target/benchmarks.jar \
 * "ImputeForecastBenchmark.impute" \ -p dataset=D1 -p cacheFraction=1.0 -p
 * missing=1,5,10 -prof gc
 * ===========================================================================
 */
@BenchmarkMode(Mode.Throughput)
@OutputTimeUnit(TimeUnit.SECONDS)
@Warmup(iterations = 3, time = 2)
@Measurement(iterations = 5, time = 2)
@Fork(value = 1, jvmArgsAppend = { "-Djdk.attach.allowAttachSelf=true", "-Djol.magicFieldOffset=true" })
@State(Scope.Benchmark)
public class ImputeForecastBenchmark {

    // impute is ~50 trees x multi-traverse x forks per point -> keep the op count
    // modest so an iteration stays inside the 2s window. forecast is HORIZON x
    // (summary over 50 trees) per call, heavier still, so it gets fewer ops.
    static final int IMPUTE_OPS = 2000;
    static final int FORECAST_OPS = 200;
    static final int HORIZON = 3; // matches Models.FORECAST_HORIZON

    @Param({ "D1", "D2" })
    Datasets.Id dataset;

    // 0.0 vs 1.0 is the decomposition lever (box vs box-free). 0.5 optional.
    @Param({ "0.0", "1.0" })
    double cacheFraction;

    // fork-count knob: forks == triggers == missing-dim cuts on the path.
    // forecast IGNORES this (its missing block is fixed by shingle) -> when
    // reading forecast rows, filter to a single `missing` value.
    @Param({ "1", "5", "10" })
    int missing;

    private RandomCutForest forest;
    private float[][] scoreF;
    private int[] missingIdx;
    private int shingleSize;

    @Setup(Level.Trial)
    public void setUp() {
        Datasets.Prepared p = Datasets.prepare(dataset, cacheFraction, 99L);
        forest = p.forest;
        scoreF = p.scoreF;
        shingleSize = forest.getShingleSize();

        // setUp: build missing indices in INPUT space, not shingled space
        int inputDims = forest.getDimensions() / Math.max(1, forest.getShingleSize()); // 100/10 = 10 for D2
// clamp so missing < inputDims (can't impute the entire input vector)
        int m = Math.min(missing, inputDims - 1);
        missingIdx = new int[m];
        for (int i = 0; i < m; i++) {
            missingIdx[i] = inputDims - m + i; // last m input coords
        }
        // Frozen from here: prob = 0, no update in the measured region.
    }

    /**
     * Direct impute path: exercises ImputeVisitor fork/combine +
     * ConditionalTreeSample + ConditionalSampleSummarizer, WITHOUT the horizon
     * loop. This is the unit the ReusableMultiVisitor refactor targets, so it's the
     * number to trust first.
     */
    @Benchmark
    @OperationsPerInvocation(IMPUTE_OPS)
    public double impute() {
        final RandomCutForest f = forest;
        final float[][] d = scoreF;
        final int[] miss = missingIdx;
        double checksum = 0.0;
        for (int i = 0; i < IMPUTE_OPS; i++) {
            float[] imputed = f.imputeMissingValues(d[i % d.length], miss);
            for (int j = 0; j < miss.length; j++) {
                checksum += imputed[miss[j]]; // consume freshly-imputed coords (no DCE)
            }
        }
        return checksum;
    }

    /**
     * End-to-end extrapolate: HORIZON blocks, each a getConditionalFieldSummary
     * over the whole forest. Forecasts from the forest's frozen internal shingle,
     * so every call is identical work on identical state (stationary, ideal for
     * JMH); the freshly allocated return array each call keeps it un-hoistable.
     *
     * D1 (shingleSize==1) has NO valid forecast (block==dims); that cell returns
     * NaN -- a reserved/empty cell, filter it out in analysis (same convention as
     * ScoringOps' APPROX x DENSITY mirror). Run forecast with -p dataset=D2.
     */
    @Benchmark
    @OperationsPerInvocation(FORECAST_OPS)
    public double forecast() {
        if (shingleSize <= 1) {
            return Double.NaN; // reserved cell: forecast undefined without a shingle
        }
        final RandomCutForest f = forest;
        double checksum = 0.0;
        for (int i = 0; i < FORECAST_OPS; i++) {
            double[] out = f.extrapolate(HORIZON); // fresh array each call
            for (int j = 0; j < out.length; j++) {
                checksum += out[j];
            }
        }
        return checksum;
    }

    @TearDown(Level.Trial)
    public void footprint() {
        long bytes = GraphLayout.parseInstance(forest).totalSize();
        System.err.printf("JOL forest[%s] cache=%.2f miss=%d : %.3f MB | %d dims%n", dataset, cacheFraction, missing,
                bytes / 1048576.0, dataset.effectiveDims());
    }
}
