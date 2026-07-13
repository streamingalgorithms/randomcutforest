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
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.TearDown;
import org.openjdk.jmh.annotations.Warmup;
import org.openjdk.jmh.infra.Blackhole;
import org.streamingalgorithms.randomcutforest.benchmark.operations.Codec;
import org.streamingalgorithms.randomcutforest.benchmark.operations.Datasets;
import org.streamingalgorithms.randomcutforest.benchmark.operations.JolBreakdown;
import org.streamingalgorithms.randomcutforest.benchmark.operations.Models;

/**
 * ===========================================================================
 * OP 4 -- serialization round-trip. JMH driver (warmed, steady-state).
 *
 * MEASURED REGION IS THE ROUND-TRIP AND NOTHING ELSE: wire codecs : decode ->
 * toModel -> toState -> encode controls : toState -> toModel (no wire) The
 * source snapshot is fixed once in @Setup and never advanced inside the loop --
 * no process(), no scoring, no fidelity compare in the timed path. The fidelity
 * check (does the codec preserve the model) lives in SerializationFidelityTest,
 * where running process() on both models to assert equality is the point rather
 * than a perturbation. Its per-tuple cost, if you want it, is a
 * separately-bucketed column in SerializationColdMain -- measured, never
 * assumed.
 *
 * One @Benchmark == one full round-trip (toModel rebuilds the whole forest, so
 * this is ms-scale, not nano). Read allocation with `-prof gc`.
 * ===========================================================================
 * run as ts=$(date +%Y%m%d-%H%M%S) java -jar benchmark/target/benchmarks.jar
 * SerializationBenchmark -prof gc \ -p codecId=FORY,PROTOSTUFF -p model=TRCF -p
 * dataset=D2 \ -rf json -rff "serialization-$ts.json" | tee
 * "serialization-$ts.txt"
 */
@BenchmarkMode(Mode.Throughput)
@OutputTimeUnit(TimeUnit.SECONDS)
@Warmup(iterations = 8, time = 2)
@Measurement(iterations = 6, time = 2)
@Fork(value = 1, jvmArgsAppend = { "-Djdk.attach.allowAttachSelf=true", "-Djol.magicFieldOffset=true" })
@State(Scope.Benchmark)
public class SerializationBenchmark {

    @Param({ "RCF", "TRCF", "CASTER" })
    Models.Kind model;

    @Param({ "D1", "D2" })
    Datasets.Id dataset;

    @Param({ "0.0", "0.2", "0.5", "1.0" })
    double cacheFraction;

    @Param
    Codec.Id codecId;

    private Models.Kind kind;
    private Codec codec;
    private Class<?> stateClass;
    private Object snapshotModel; // fixed pre-trained model (for controls / re-serialize source)
    private Object snapshotState; // toState(snapshotModel), fixed
    private byte[] snapshotWire; // encode(snapshotState), fixed (wire codecs only)

    private boolean skip;

    @Setup(Level.Trial)
    public void setUp() {
        Models.Prepared p = Models.prepare(model, dataset, cacheFraction, 99L);
        kind = p.kind;
        codec = Codec.of(codecId);
        skip = (model == Models.Kind.CASTER && dataset == Datasets.Id.D1)
                || (codec.treeMode() == Models.TreeMode.REBUILD && model != Models.Kind.RCF);
        if (skip) {
            return; // don't build anything
        }
        stateClass = Models.stateClass(kind);
        snapshotModel = p.model;
        snapshotState = Models.toState(kind, snapshotModel, codec.treeMode());
        if (!codec.isControl()) {
            snapshotWire = codec.encode(snapshotState);
        }
    }

    @Benchmark
    public void roundTrip(Blackhole bh) {
        if (skip) {
            return;
        }
        if (codec.isControl()) {
            Object m = Models.toModel(kind, snapshotState, codec.treeMode()); // decode-free control: mapper only
            Object st = Models.toState(kind, m, codec.treeMode());
            bh.consume(m);
            bh.consume(st);
        } else {
            Object st = codec.decode(snapshotWire, stateClass);
            Object m = Models.toModel(kind, st, codec.treeMode());
            Object st2 = Models.toState(kind, m, codec.treeMode());
            byte[] wire = codec.encode(st2);
            bh.consume(m);
            bh.consume(wire);
        }
    }

    @TearDown(Level.Trial)
    public void footprint() {
        JolBreakdown b = JolBreakdown.of(kind, snapshotModel);
        int wire = (snapshotWire == null) ? -1 : snapshotWire.length;
        System.err.printf("JOL %-6s[%s] cache=%.2f codec=%-10s : forest %6.3f MB | wire %8d B | %d dims%n", model,
                dataset, cacheFraction, codec.name(), b.wholeMb, wire, dataset.effectiveDims());
    }
}
