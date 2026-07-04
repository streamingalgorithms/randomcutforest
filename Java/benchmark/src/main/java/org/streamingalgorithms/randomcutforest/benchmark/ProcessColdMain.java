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

import java.lang.management.ManagementFactory;
import java.time.Duration;
import java.time.Instant;

import org.streamingalgorithms.randomcutforest.benchmark.operations.Datasets;
import org.streamingalgorithms.randomcutforest.benchmark.operations.JolBreakdown;
import org.streamingalgorithms.randomcutforest.benchmark.operations.Models;

/**
 * ===========================================================================
 * OP 2 &amp; 3 -- streaming throughput. COLD driver (wall-clock, no warmup).
 *
 * Same processOne as ProcessBenchmark, so the two can't drift. A single pass
 * over the hold-out on a model saturated (untimed) in setup: gives the cold-JIT
 * steady per-tuple cost, per-tuple allocation (ThreadMXBean, TLAB-accounted),
 * and the JOL breakdown. JFR-friendly (-XX:StartFlightRecording). To capture
 * the fill-up transient instead of steady cost, time the saturation loop too --
 * one-line change, noted below.
 * ===========================================================================
 * java -Djdk.attach.allowAttachSelf=true -Djol.magicFieldOffset=true \
 * -XX:StartFlightRecording=filename=proc.jfr,settings=profile \ -cp
 * benchmark/target/benchmarks.jar \
 * org.streamingalgorithms.randomcutforest.benchmark.ProcessColdMain
 * 
 */
public final class ProcessColdMain {

    private static final double[] CACHE = { 0.0, 0.2, 0.5, 1.0 };
    private static final long SEED = 99L;

    private ProcessColdMain() {
    }

    public static void main(String[] args) {
        com.sun.management.ThreadMXBean mem = (com.sun.management.ThreadMXBean) ManagementFactory.getThreadMXBean();
        mem.setThreadAllocatedMemoryEnabled(true);
        final long tid = Thread.currentThread().getId();

        for (Models.Kind kind : Models.Kind.values()) {
            for (Datasets.Id dataset : Datasets.Id.values()) {
                if (kind == Models.Kind.CASTER && dataset == Datasets.Id.D1) {
                    System.out.printf("%n==== CASTER | D1 : SKIPPED (needs shingleSize>1) ====%n");
                    continue; // skip all four caches at once
                }
                System.out.printf("%n==== %-6s | %s | %d dims ====%n", kind, dataset, dataset.effectiveDims());
                for (double cache : CACHE) {
                    // saturation is untimed (to time the fill-up transient instead, move the
                    // timing brackets to enclose Models.prepare)
                    Models.Prepared p = Models.prepare(kind, dataset, cache, SEED);
                    final Object model = p.model;
                    final double[][] s = p.stream;
                    long clock = p.clock0;

                    double checksum = 0.0;
                    long b0 = mem.getThreadAllocatedBytes(tid);
                    Instant start = Instant.now();
                    for (int i = 0; i < Datasets.SCORED; i++) {
                        checksum += Models.processOne(kind, model, s[i], clock++);
                    }
                    double seconds = Duration.between(start, Instant.now()).toNanos() / 1e9;
                    long allocBytes = mem.getThreadAllocatedBytes(tid) - b0;

                    JolBreakdown jb = JolBreakdown.of(kind, model);
                    double tuplesPerSec = Datasets.SCORED / seconds;
                    double bytesPerTuple = (double) allocBytes / Datasets.SCORED;

                    System.out.printf(
                            "cache=%.2f : %7.3f s | %10.1f tuples/s | %8.1f B/tuple | whole %6.3f | tree %6.3f | sampler %6.3f | store %6.3f MB | chk=%.4g%n",
                            cache, seconds, tuplesPerSec, bytesPerTuple, jb.wholeMb, jb.treeMb, jb.samplerMb,
                            jb.storeMb, checksum);

                    if (!Double.isFinite(checksum) || checksum == 0.0) {
                        throw new AssertionError(kind + "/" + dataset + "/cache=" + cache + ": degenerate checksum");
                    }
                }
            }
        }
    }
}
