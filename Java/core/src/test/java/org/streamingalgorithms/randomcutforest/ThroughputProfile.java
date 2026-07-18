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

package org.streamingalgorithms.randomcutforest;

import java.lang.management.ManagementFactory;
import java.util.Arrays;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.CyclicBarrier;

import org.junit.jupiter.api.Test;
import org.streamingalgorithms.randomcutforest.testutils.StreamingMultiDimData;

import com.sun.management.ThreadMXBean;

/**
 * Throughput profile. One forest + one producer + one buffer per thread;
 * nothing shared, no barrier between points, no materialized dataset.
 *
 * <h2>What the previous shape measured</h2>
 * <ul>
 * <li><b>A fork-join round trip per data point.</b> submit().join() inside the
 * j-loop is a full barrier 100,000 times over, to dispatch ~11 tasks each. The
 * forests are independent and never needed to be in lockstep.</li>
 * <li><b>A P/E core average.</b> availableProcessors() on an 11-core M3 Pro
 * returns 5P + 6E. Wall clock is set by the slowest thread, so six E-cores at
 * ~1/3 speed drag the aggregate and it reads as "we hit a wall". Hence the
 * per-thread report below: the min/max spread IS the P/E split, and there is no
 * affinity API on macOS to avoid it.</li>
 * <li><b>Cold everything.</b> start was before construction, forests began
 * empty. JIT warmup plus sampler fill-up, where boxes are tiny and the kernel
 * mix is nothing like steady state. Legitimate as a "first 100k points" number,
 * not as a ceiling.</li>
 * <li><b>Shared data rows.</b> Every forest read the same double[][], so the
 * threads shared one working set. Now each has its own stream.</li>
 * </ul>
 *
 * <p>
 * Also: score[] was false-shared (N doubles on one cache line, incremented per
 * point per thread), the timer stopped after the print loop, and mse/mseCount
 * were dead. All gone.
 *
 * <h2>Knobs</h2>
 *
 * <pre>
 *   -Drcf.threads=N    default availableProcessors(); sweep 1..5 on an M3 Pro to
 *                      get the per-P-core number without E-core contamination
 *   -Drcf.warmup=N     untimed points per thread (JIT + sampler fill-up)
 *   -Drcf.measured=N   timed points per thread
 * </pre>
 */
public class ThroughputProfile {

    private int numberOfTrees = 50;
    private int sampleSize = 256;
    private int numberOfAttributes = 3;
    private int shingleSize = 7;
    private int dimensions = numberOfAttributes * shingleSize;
    private double boundingBoxCacheFraction = 1.0;

    private static final int THREADS = Integer.getInteger("rcf.threads", Runtime.getRuntime().availableProcessors());
    private static final int WARMUP = Integer.getInteger("rcf.warmup", 50_000);
    private static final int MEASURED = Integer.getInteger("rcf.measured", 100_000);

    @Test
    public void profileThroughput() throws Exception {

        final RandomCutForest[] forests = new RandomCutForest[THREADS];
        final StreamingMultiDimData[] producers = new StreamingMultiDimData[THREADS];
        for (int k = 0; k < THREADS; k++) {
            forests[k] = RandomCutForest.builder().numberOfTrees(numberOfTrees).dimensions(dimensions)
                    .shingleSize(shingleSize).boundingBoxCacheFraction(boundingBoxCacheFraction).randomSeed(99 + k)
                    .outputAfter(10).parallelExecutionEnabled(false).internalShinglingEnabled(true)
                    .initialAcceptFraction(0.1).sampleSize(sampleSize).build();
            // own seed per forest => own stream => threads share nothing
            producers[k] = new StreamingMultiDimData(60, 100, 5, 99 + k, numberOfAttributes);
        }

        final double[] score = new double[THREADS];
        final long[] nanos = new long[THREADS];
        final long[] alloc = new long[THREADS];

        // gate: every thread warms up, THEN they all start timing together. Otherwise
        // early finishers idle while the last thread is still JIT-ing and the aggregate
        // is meaningless.
        final CyclicBarrier gate = new CyclicBarrier(THREADS + 1);
        final CountDownLatch done = new CountDownLatch(THREADS);
        final ThreadMXBean tmx = (ThreadMXBean) ManagementFactory.getThreadMXBean();

        for (int k = 0; k < THREADS; k++) {
            final int kk = k;
            Thread t = new Thread(() -> {
                final RandomCutForest f = forests[kk];
                final StreamingMultiDimData p = producers[kk];

                // untimed: JIT + drive the sampler past fill-up into equilibrium
                for (int j = 0; j < WARMUP; j++) {
                    double[] pt = p.next();
                    f.getAnomalyScore(pt);
                    f.update(pt);
                }

                await(gate);

                final long tid = Thread.currentThread().threadId();
                final long a0 = tmx.getThreadAllocatedBytes(tid);
                final long t0 = System.nanoTime();
                double s = 0;
                for (int j = 0; j < MEASURED; j++) {
                    double[] pt = p.next();
                    s += f.getAnomalyScore(pt);
                    f.update(pt);
                }
                final long t1 = System.nanoTime();
                nanos[kk] = t1 - t0;
                alloc[kk] = tmx.getThreadAllocatedBytes(tid) - a0;
                score[kk] = s / MEASURED; // single write: no false sharing in the loop
                done.countDown();
            }, "rcf-" + k);
            t.start();
        }

        await(gate); // release everyone at once
        final long start = System.nanoTime();
        done.await();
        final long end = System.nanoTime();
        final double wall = (end - start) * 1e-9;

        // ---- report ------------------------------------------------------

        final long[] perThread = new long[THREADS];
        for (int k = 0; k < THREADS; k++) {
            perThread[k] = (long) (MEASURED * 1e9 / nanos[k]);
        }
        long[] sorted = perThread.clone();
        Arrays.sort(sorted);

        long totalAlloc = 0;
        for (long a : alloc) {
            totalAlloc += a;
        }

        // box cache dominates the working set: trees * samples * 2 * dims * 4 bytes
        final double boxBytesPerForest = (double) numberOfTrees * sampleSize * 2 * dimensions * 4;
        // per point: every node visited reads a box; ~2*ln(sampleSize) levels deep,
        // scored AND updated. Order-of-magnitude, not gospel -- see the notes.
        final double levels = 2 * Math.log(sampleSize);
        final double bytesPerPoint = 2 * numberOfTrees * levels * (2.0 * dimensions * 4);

        System.out.printf("%n=== config ===%n");
        System.out.printf("threads=%d (availableProcessors=%d)  dims=%d  trees=%d  sample=%d  cacheFraction=%.2f%n",
                THREADS, Runtime.getRuntime().availableProcessors(), dimensions, numberOfTrees, sampleSize,
                boundingBoxCacheFraction);
        System.out.printf("warmup=%d/thread  measured=%d/thread%n", WARMUP, MEASURED);

        System.out.printf("%n=== throughput ===%n");
        System.out.printf("wall %.3f s   aggregate %,.0f points/s%n", wall, THREADS * MEASURED / wall);
        System.out.printf("per-thread points/s: min %,d  median %,d  max %,d   (spread %.2fx)%n", sorted[0],
                sorted[THREADS / 2], sorted[THREADS - 1], sorted[THREADS - 1] / (double) sorted[0]);
        System.out.printf("  -> a spread near 3x on Apple silicon is the P/E split, not a scaling limit.%n");
        System.out.printf("  -> re-run with -Drcf.threads=<P-core count> for the per-core ceiling.%n");
        System.out.print("  all: ");
        for (long v : perThread) {
            System.out.printf("%,d ", v);
        }
        System.out.println();

        System.out.printf("%n=== allocation ===%n");
        System.out.printf("%,.0f B/point   %.2f GB/s%n", totalAlloc / (double) (THREADS * MEASURED),
                totalAlloc / wall / 1e9);
        System.out.printf("  -> compare against ScoringBenchmark's gc.alloc.rate.norm. A large gap means the%n");
        System.out.printf("     update path allocates where the scoring path does not.%n");

        System.out.printf("%n=== memory, the number that actually caps this ===%n");
        System.out.printf("box cache: %.1f MB/forest, %.1f MB total working set%n", boxBytesPerForest / 1e6,
                THREADS * boxBytesPerForest / 1e6);
        System.out.printf("est. traffic: ~%,.0f KB/point  ->  ~%.0f GB/s at the aggregate above%n", bytesPerPoint / 1e3,
                bytesPerPoint * THREADS * MEASURED / wall / 1e9);
        System.out.printf("  -> compare to your machine's DRAM bandwidth. If this is >50%% of peak, the kernels%n");
        System.out.printf("     are not the bottleneck and bytes/point is the only lever.%n");

        System.out.printf("%n=== scores (sanity: should be ~1 for calibrated data) ===%n");
        for (int k = 0; k < THREADS; k++) {
            System.out.printf("  forest %2d  mean score %.4f%n", k, score[k]);
        }
    }

    private static void await(CyclicBarrier b) {
        try {
            b.await();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
