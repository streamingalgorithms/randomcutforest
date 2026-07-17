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

package org.streamingalgorithms.randomcutforest.tree;

import org.junit.jupiter.api.Test;
import org.openjdk.jol.info.GraphLayout;
import org.streamingalgorithms.randomcutforest.RandomCutForest;
import org.streamingalgorithms.randomcutforest.executor.SamplerPlusTree;
import org.streamingalgorithms.randomcutforest.testutils.StreamingMultiDimData;

/**
 * <p>
 * Throughput here is single-threaded on purpose: per-entity cost is what the
 * formula needs, and one thread dodges the P/E scheduling noise.
 *
 * <h2>Running</h2>
 *
 * <pre>
 *   -Djol.magicFieldOffset=true
 *   --add-opens=java.base/java.lang=ALL-UNNAMED
 *   --add-modules=jdk.incubator.vector
 *   -Drcf.baseDim=10 -Drcf.shingle=10 -Drcf.trees=50 -Drcf.sample=256
 * </pre>
 */
public class ModelFootprintProfile {

    private static final int BASE_DIM = Integer.getInteger("rcf.baseDim", 10);
    private static final int SHINGLE = Integer.getInteger("rcf.shingle", 10);
    private static final int TREES = Integer.getInteger("rcf.trees", 50);
    private static final int SAMPLE = Integer.getInteger("rcf.sample", 256);

    private static final int WARMUP = Integer.getInteger("rcf.warmup", 50_000);
    private static final int MEASURED = Integer.getInteger("rcf.measured", 50_000);

    private static final double[] CACHE_FRACTIONS = { 0.0, 0.1, 0.25, 0.5, 1.0 };

    /** 8 GB heap x plugins.anomaly_detection.model_max_size_percent (10%). */
    private static final double AD_BYTES_PER_NODE = 8.096e9 * 0.10;
    private static final int CORES_PER_NODE = Runtime.getRuntime().availableProcessors();
    private static final double TARGET_ENTITIES = 1e6;

    /**
     * Detector interval. AD defaults to minutes; Data Prepper's per-event processor
     * is effectively the left edge of this sweep.
     */
    private static final double[] INTERVALS_SECONDS = { 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 10, 60, 300 };

    /**
     * Fraction of single-thread throughput retained per core under concurrency.
     * Default 1.0 (perfect) is a LIE and the CPU column is optimistic by however
     * much. Measure it: ThroughputProfile -Drcf.threads=1,..,N, divide per-thread
     * throughput at N by throughput at 1, and pass it here. It is cf-dependent -- a
     * fat model thrashes cache when replicated per core and a thin one may not --
     * so strictly this should be a per-cf array. One number is already a better
     * approximation than pretending it is 1.0.
     */
    private static final double SCALING = Double.parseDouble(System.getProperty("rcf.scaling", "1.0"));

    /**
     * Projected machine: -Drcf.proj.cores=192 -Drcf.proj.adBytes=37.2e9
     * -Drcf.proj.ipc=0.75
     */
    private static final int PROJ_CORES = Integer.getInteger("rcf.proj.cores", 0);
    private static final double PROJ_AD_BYTES = Double.parseDouble(System.getProperty("rcf.proj.adBytes", "37.2e9"));
    private static final double PROJ_IPC = Double.parseDouble(System.getProperty("rcf.proj.ipc", "1.0"));

    @Test
    public void footprintVersusThroughput() {
        final int dims = BASE_DIM * SHINGLE;
        final double predictedBoxBytes = (double) TREES * (SAMPLE - 1) * 2 * dims * 4;

        System.out.printf("%n=== config ===%n");
        System.out.printf("baseDim=%d x shingle=%d -> dims=%d   trees=%d  sample=%d%n", BASE_DIM, SHINGLE, dims, TREES,
                SAMPLE);
        System.out.printf("predicted box cache at cf=1.0: trees x (sample-1) x 2 x dims x 4B = %.2f MB%n",
                predictedBoxBytes / 1e6);
        System.out.printf("scaling efficiency assumed: %.2f%s%n", SCALING,
                SCALING == 1.0 ? "  <-- PERFECT, i.e. optimistic. See the javadoc; measure it." : "");

        // ---- measure once per cf; everything below is arithmetic on these ----
        final Result[] results = new Result[CACHE_FRACTIONS.length];
        System.out.printf("%n=== measured (single thread, saturated) ===%n");
        System.out.printf("%-6s %10s %10s %12s%n", "cf", "MB/model", "pts/s", "us/point");
        System.out.printf("%s%n", "-".repeat(42));
        for (int i = 0; i < CACHE_FRACTIONS.length; i++) {
            results[i] = measure(dims, CACHE_FRACTIONS[i]);
            System.out.printf("%-6.2f %10.2f %10.0f %12.1f%n", CACHE_FRACTIONS[i], results[i].bytes / 1e6,
                    results[i].pointsPerSecond, 1e6 / results[i].pointsPerSecond);
        }
        double boxMeasured = results[results.length - 1].bytes - results[0].bytes;
        System.out.printf(
                "box cache: measured %.2f MB vs predicted %.2f MB (%+.0f%%)   |   cf=1 is %.1fx the size, %.1fx the speed%n",
                boxMeasured / 1e6, predictedBoxBytes / 1e6, 100 * (boxMeasured / predictedBoxBytes - 1),
                results[results.length - 1].bytes / results[0].bytes,
                results[results.length - 1].pointsPerSecond / results[0].pointsPerSecond);

        report("THIS MACHINE", results, CORES_PER_NODE, AD_BYTES_PER_NODE, 1.0);
        if (PROJ_CORES > 0) {
            report("PROJECTED (-Drcf.proj.*)", results, PROJ_CORES, PROJ_AD_BYTES, PROJ_IPC);
        }
    }

    private void report(String label, Result[] r, int cores, double adBytes, double ipc) {

        System.out.printf("%n=== %s: %d cores, %.2f GB AD budget/node, ipc x%.2f ===%n", label, cores, adBytes / 1e9,
                ipc);

        System.out
                .printf("%ncrossover interval per cf -- longer: MEMORY binds (shrink); shorter: CPU binds (fatten)%n");
        System.out.printf("  interval* = (AD_bytes / model_bytes) / (pts_per_sec * cores * scaling * ipc)%n");
        for (int i = 0; i < r.length; i++) {
            double entMem = adBytes / r[i].bytes;
            double star = entMem / (r[i].pointsPerSecond * ipc * SCALING * cores);
            System.out.printf("  cf=%-5.2f %8.2f MB  ->  %10.2f ms%n", CACHE_FRACTIONS[i], r[i].bytes / 1e6,
                    star * 1000);
        }

        System.out.printf("%n%-12s %8s %10s %14s %16s %8s%n", "interval", "best cf", "MB/model", "entities/node",
                "machines for 1M", "binds");
        System.out.printf("%s%n", "-".repeat(74));
        for (double interval : INTERVALS_SECONDS) {
            int best = -1;
            double bestEnt = -1;
            boolean bestMemBinds = false;
            for (int i = 0; i < r.length; i++) {
                double entMem = adBytes / r[i].bytes;
                double entCpu = r[i].pointsPerSecond * ipc * SCALING * interval * cores;
                double ent = Math.min(entMem, entCpu);
                if (ent > bestEnt) {
                    bestEnt = ent;
                    best = i;
                    bestMemBinds = entMem < entCpu;
                }
            }
            System.out.printf("%-12s %8.2f %10.2f %,14.0f %,16.0f %8s%n", fmt(interval), CACHE_FRACTIONS[best],
                    r[best].bytes / 1e6, bestEnt, Math.ceil(TARGET_ENTITIES / bestEnt),
                    bestMemBinds ? "MEMORY" : "cpu");
        }
        System.out.printf("%n  best cf = argmax min(entities by memory, entities by CPU). There is no single right%n");
        System.out.printf("  cacheFraction: AD at 60s and a per-event pipeline at 1ms want opposite configs.%n");
    }

    private static String fmt(double seconds) {
        return seconds < 1 ? String.format("%.0f ms", seconds * 1000) : String.format("%.0f s", seconds);
    }

    private static final class Result {
        final double bytes;
        final double pointsPerSecond;

        Result(double bytes, double pointsPerSecond) {
            this.bytes = bytes;
            this.pointsPerSecond = pointsPerSecond;
        }
    }

    private Result measure(int dims, double cacheFraction) {
        RandomCutForest forest = RandomCutForest.builder().numberOfTrees(TREES).dimensions(dims).shingleSize(SHINGLE)
                .boundingBoxCacheFraction(cacheFraction).randomSeed(99).outputAfter(10).parallelExecutionEnabled(false)
                .internalShinglingEnabled(true).initialAcceptFraction(0.1).sampleSize(SAMPLE).build();

        StreamingMultiDimData p = new StreamingMultiDimData(60, 100, 5, 99, BASE_DIM);

        // saturate past fill-up: an empty forest has no boxes to cache, so measuring
        // footprint before equilibrium measures nothing.
        for (int j = 0; j < WARMUP; j++) {
            double[] pt = p.next();
            forest.getAnomalyScore(pt);
            forest.update(pt);
        }

        // retained size AFTER saturation, BEFORE timing
        long bytes = GraphLayout.parseInstance(forest).totalSize();

        long storeSize = GraphLayout.parseInstance(forest.getUpdateCoordinator().getStore()).totalSize();
        long treeSize = GraphLayout.parseInstance(firstTree(forest)).totalSize() - storeSize;
        long cache = GraphLayout.parseInstance(firstTree(forest).boundingBoxData).totalSize();

        long t0 = System.nanoTime();
        double sink = 0;
        for (int j = 0; j < MEASURED; j++) {
            double[] pt = p.next();
            sink += forest.getAnomalyScore(pt);
            forest.update(pt);
        }
        long t1 = System.nanoTime();
        if (sink == Double.MIN_VALUE) {
            System.out.print(""); // keep the scores alive
        }

        return new Result(bytes, MEASURED * 1e9 / (t1 - t0));
    }

    private RandomCutTree firstTree(RandomCutForest forest) {
        Object component = forest.getComponents().iterator().next(); // ComponentList is Iterable
        SamplerPlusTree<?, float[]> spt = (SamplerPlusTree<?, float[]>) component;
        return (RandomCutTree) spt.getTree();
    }
}