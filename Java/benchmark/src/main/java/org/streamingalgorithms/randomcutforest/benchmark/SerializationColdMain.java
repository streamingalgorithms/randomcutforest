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

import org.apache.fory.logging.LoggerFactory;
import org.streamingalgorithms.randomcutforest.benchmark.operations.Codec;
import org.streamingalgorithms.randomcutforest.benchmark.operations.Datasets;
import org.streamingalgorithms.randomcutforest.benchmark.operations.JolBreakdown;
import org.streamingalgorithms.randomcutforest.benchmark.operations.Models;

/**
 * ===========================================================================
 * OP 4 -- serialization round-trip. COLD driver (wall-clock, no warmup).
 *
 * The detailed table: per-STAGE timing and allocation buckets (decode / toModel
 * / toState / encode), so you see where the cost and the churn actually land --
 * this is the "which codec, and why" table, lifted from your bakeoff and made
 * model-agnostic. Each stage brackets its ThreadMXBean allocation reads OUTSIDE
 * its nanoTime reads so the meter never lands in the timing bucket.
 *
 * Cold matters most for deserialization, so iteration 0 (cold: class loading,
 * cold JIT) is reported SEPARATELY from the steady average over [WARM, TEST).
 *
 * FIDELITY is out of the measured path by default. Set MEASURE_FIDELITY=true to
 * price the in-memory check as its OWN column (run processOne on the rebuilt
 * model) -- never summed into decode/toModel/toState/encode. Clean by default,
 * quantified on demand. The equality ASSERTION belongs in
 * SerializationFidelityTest, not here.
 * ===========================================================================
 * java -Djdk.attach.allowAttachSelf=true -Djol.magicFieldOffset=true \
 * -XX:+EnableDynamicAgentLoading
 * -XX:StartFlightRecording=filename=proc.jfr,settings=profile \ -cp
 * benchmark/target/benchmarks.jar \
 * org.streamingalgorithms.randomcutforest.benchmark.SerializationColdMain
 */
public final class SerializationColdMain {

    private static final double[] CACHE = { 0.0, 0.2, 0.5, 1.0 };
    private static final long SEED = 99L;
    private static final int TEST = 500; // round-trips per cell (each rebuilds a forest)
    private static final int WARM = TEST / 2; // steady window = [WARM, TEST)
    private static final boolean MEASURE_FIDELITY = false;

    private final com.sun.management.ThreadMXBean mem = (com.sun.management.ThreadMXBean) ManagementFactory
            .getThreadMXBean();
    private final long tid = Thread.currentThread().getId();

    private SerializationColdMain() {
        mem.setThreadAllocatedMemoryEnabled(true);
    }

    public static void main(String[] args) {
        SerializationColdMain m = new SerializationColdMain();
        LoggerFactory.disableLogging(); // FORY produces a lot of info
        for (Models.Kind kind : Models.Kind.values()) {
            for (Datasets.Id dataset : Datasets.Id.values()) {
                for (double cache : CACHE) {
                    if (kind == Models.Kind.CASTER && dataset == Datasets.Id.D1) {
                        System.out.printf("%n==== CASTER | D1 : SKIPPED (needs shingleSize>1) ====%n");
                        continue; // skip all four caches at once
                    }

                    System.out.printf("%n==== %-6s | %s | cache=%.2f | %d dims ====%n", kind, dataset, cache,
                            dataset.effectiveDims());
                    Models.Prepared p = Models.prepare(kind, dataset, cache, SEED);
                    // note model is shared across codecs -- DO NOT MUTATE
                    double forestMb = JolBreakdown.of(kind, p.model).wholeMb;
                    for (Codec.Id id : Codec.Id.values()) {
                        if (id == Codec.Id.REBUILT && kind != Models.Kind.RCF) {
                            continue; // REBUILD is RCF-only
                        }
                        try {
                            m.runCell(kind, dataset, cache, p, forestMb, Codec.of(id));
                        } catch (RuntimeException e) {
                            Throwable root = e;
                            while (root.getCause() != null)
                                root = root.getCause();
                            System.out.printf("%-6s %-10s : FAILED round-trip (%s: %s)%n", kind, id.name(),
                                    root.getClass().getSimpleName(), root.getMessage());
                            continue;
                        }
                    }
                }
            }
        }
    }

    private void runCell(Models.Kind kind, Datasets.Id dataset, double cache, Models.Prepared p, double forestMb,
            Codec codec) {
        Class<?> stateClass = Models.stateClass(kind);
        Models.TreeMode tm = codec.treeMode();
        Object snapshotState = Models.toState(kind, p.model, tm);
        boolean control = codec.isControl();
        byte[] wire = control ? null : codec.encode(snapshotState);

        long decodeNs = 0, toModelNs = 0, toStateNs = 0, encodeNs = 0, fidelityNs = 0;
        long decodeB = 0, toModelB = 0, toStateB = 0, encodeB = 0;
        long coldRoundTripNs = 0;
        long size = 0;
        long clock = p.clock0;

        for (int j = 0; j < TEST; j++) {
            long rt0 = System.nanoTime();

            // --- decode ---
            Object state;
            if (control) {
                state = snapshotState;
            } else {
                long b0 = mem.getThreadAllocatedBytes(tid);
                long t0 = System.nanoTime();
                state = codec.decode(wire, stateClass);
                long t1 = System.nanoTime();
                long b1 = mem.getThreadAllocatedBytes(tid);
                if (j >= WARM) {
                    decodeNs += t1 - t0;
                    decodeB += b1 - b0;
                }
            }

            // --- toModel ---
            long mb0 = mem.getThreadAllocatedBytes(tid);
            long mt0 = System.nanoTime();
            Object model = Models.toModel(kind, state, tm);
            long mt1 = System.nanoTime();
            long mb1 = mem.getThreadAllocatedBytes(tid);
            if (j >= WARM) {
                toModelNs += mt1 - mt0;
                toModelB += mb1 - mb0;
            }

            // --- fidelity (own bucket, never summed elsewhere; off by default) ---
            if (MEASURE_FIDELITY) {
                long ft0 = System.nanoTime();
                double s = Models.processOne(kind, model, p.stream[j % Datasets.SCORED], clock++);
                fidelityNs += System.nanoTime() - ft0;
                if (!Double.isFinite(s)) {
                    throw new AssertionError("non-finite fidelity score");
                }
            }

            // --- toState ---
            long sb0 = mem.getThreadAllocatedBytes(tid);
            long st0 = System.nanoTime();
            Object state2 = Models.toState(kind, model, tm);
            long st1 = System.nanoTime();
            long sb1 = mem.getThreadAllocatedBytes(tid);
            if (j >= WARM) {
                toStateNs += st1 - st0;
                toStateB += sb1 - sb0;
            }

            // --- encode ---
            if (!control) {
                long eb0 = mem.getThreadAllocatedBytes(tid);
                long et0 = System.nanoTime();
                byte[] w = codec.encode(state2);
                long et1 = System.nanoTime();
                long eb1 = mem.getThreadAllocatedBytes(tid);
                size = w.length;
                if (j >= WARM) {
                    encodeNs += et1 - et0;
                    encodeB += eb1 - eb0;
                }
            } else if (codec.probesSize() && j == 0) {
                size = GraphLayout(snapshotState); // STATE control: retained state size, informational
            }

            if (j == 0) {
                coldRoundTripNs = System.nanoTime() - rt0; // the cold first round-trip
            }
        }

        long n = TEST - WARM;
        double codecUs = (decodeNs + encodeNs) / 1e3 / n;
        double codecKb = (decodeB + encodeB) / 1024.0 / n;
        String fid = MEASURE_FIDELITY ? String.format(" | fid %6.1f us", fidelityNs / 1e3 / n) : "";
        System.out.printf(
                "%-6s %-10s : toModel %7.1f us %8.1f KB | toState %7.1f us %8.1f KB | codec %7.1f us %8.1f KB%s | cold-rt %8.1f us | forest %6.2f MB | size %9d%n",
                kind, codec.name(), toModelNs / 1e3 / n, toModelB / 1024.0 / n, toStateNs / 1e3 / n,
                toStateB / 1024.0 / n, codecUs, codecKb, fid, coldRoundTripNs / 1e3, forestMb, size);
    }

    /** Local JOL size (bytes) with the same flag requirements as JolBreakdown. */
    private static long GraphLayout(Object o) {
        return org.openjdk.jol.info.GraphLayout.parseInstance(o).totalSize();
    }
}
