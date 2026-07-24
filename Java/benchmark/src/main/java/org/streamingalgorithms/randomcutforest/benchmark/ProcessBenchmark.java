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
import org.streamingalgorithms.randomcutforest.benchmark.operations.Datasets;
import org.streamingalgorithms.randomcutforest.benchmark.operations.JolBreakdown;
import org.streamingalgorithms.randomcutforest.benchmark.operations.Models;

/**
 * ===========================================================================
 * OP 2 &amp; 3 -- streaming throughput: one process() (or score+update) per
 * tuple. JMH driver (warmed, steady-state). No codec axis -- that lives in the
 * Serialization suite.
 *
 * MUTATION + STATIONARITY. process() mutates the model, so unlike scoring there
 * is no prob=0 frozen state to fall back on. We pre-saturate in @Setup (INITIAL
 * points through the model) so the sampler sits in dynamic equilibrium; from
 * there each measured tuple keeps it in equilibrium (size stationary, per-tuple
 * cost stationary) even as contents churn. The `clock` field advances
 * MONOTONICALLY across invocations so timestamps never travel backwards between
 * one invocation's end and the next one's start -- reusing per-invocation
 * timestamps would feed the model out-of-order time. Data cycles; time does
 * not.
 *
 * BASELINE. Kind.RCF (score+update) is the floor; TRCF and CASTER are the same
 * shape plus their internal thresholder / forecast, so the deltas price that
 * overhead per tuple. Allocation flow is JMH-native: run with `-prof gc`.
 * Retained footprint is the JOL breakdown in @TearDown (side channel).
 * ===========================================================================
 * run as ts=$(date +%Y%m%d-%H%M%S) java --add-modules jdk.incubator.vector -jar
 * benchmark/target/benchmarks.jar ProcessBenchmark -prof gc \ -rf json -rff
 * "process-$ts.json" | tee "process-$ts.txt"
 */
@BenchmarkMode(Mode.Throughput)
@OutputTimeUnit(TimeUnit.SECONDS)
@Warmup(iterations = 3, time = 2)
@Measurement(iterations = 5, time = 2)
@Fork(value = 1, jvmArgsAppend = { "-Djdk.attach.allowAttachSelf=true", "-Djol.magicFieldOffset=true" })
@State(Scope.Benchmark)
public class ProcessBenchmark {

    @Param({ "RCF", "TRCF", "CASTER" })
    Models.Kind model;

    @Param({ "D1", "D2" })
    Datasets.Id dataset;

    @Param({ "0.0", "0.001", "0.2", "0.5", "1.0" })
    double cacheFraction;

    private Models.Kind kind;
    private Object modelObj;
    private float[][] stream;
    private long clock;

    @Setup(Level.Trial)
    public void setUp() {
        Models.Prepared p = Models.prepare(model, dataset, cacheFraction, 99L);
        kind = p.kind;
        modelObj = p.model;
        stream = p.stream;
        clock = p.clock0; // continue timestamps after saturation, monotonic across invocations
    }

    @Benchmark
    @OperationsPerInvocation(Datasets.SCORED)
    public double process() {
        final Models.Kind k = kind;
        final Object m = modelObj;
        final float[][] s = stream;
        long c = clock;
        double checksum = 0.0;
        for (int i = 0; i < Datasets.SCORED; i++) {
            checksum += Models.processOne(k, m, s[i], c++);
        }
        clock = c;
        return checksum;
    }

    @TearDown(Level.Trial)
    public void footprint() {
        JolBreakdown b = JolBreakdown.of(kind, modelObj);
        System.err.printf(
                "JOL %-6s[%s] cache=%.2f : whole %6.3f | rcf %6.3f | wrapper %6.3f | store %6.3f | tree %6.3f | sampler %6.3f MB | %d dims%n",
                model, dataset, cacheFraction, b.wholeMb, b.rcfMb, b.wrapperMb(), b.storeMb, b.treeMb, b.samplerMb,
                dataset.effectiveDims());
    }
}
