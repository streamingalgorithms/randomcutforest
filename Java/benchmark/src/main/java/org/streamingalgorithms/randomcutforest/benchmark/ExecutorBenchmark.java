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

import java.util.Arrays;
import java.util.List;
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
import org.streamingalgorithms.randomcutforest.IVisitorFactory;
import org.streamingalgorithms.randomcutforest.RandomCutForest;
import org.streamingalgorithms.randomcutforest.benchmark.operations.Datasets;
import org.streamingalgorithms.randomcutforest.interpolation.AnisotropicLocalVisitor;
import org.streamingalgorithms.randomcutforest.returntypes.AnisotropicLocalGeometry;
import org.streamingalgorithms.randomcutforest.returntypes.LocalGeometryConfig;

/**
 * The parallel traversal executor against the sequential one, on a frozen,
 * saturated forest built by {@link Datasets#prepare}: the anomaly score, and
 * the anisotropic local geometry, which the scoring matrix does not time. Both
 * datasets are read in 100 dimensions: D1 as given, D2 as 10 values shingled 10
 * times inside the forest.
 *
 * <p>
 * batch is the number of points handed to the forest at once. At 1 (the
 * default) every point is its own query, as in {@link ScoringBenchmark}. Above
 * 1 the points go in batches through the batch traversal, which walks each tree
 * for the whole batch before the next and lends each tree a full box cache for
 * its turn; the score is per point either way, so the columns compare directly.
 *
 * <p>
 * threads is the pool of the parallel executor (0 leaves the forest's default,
 * the processors less one) and is ignored by the sequential one. Run with -prof
 * gc for the allocation per point, gc.alloc.rate.norm.
 *
 * <pre>
 * java --add-modules jdk.incubator.vector -jar benchmark/target/benchmarks.jar ExecutorBenchmark -p batch=1,64,1024
 * </pre>
 */
@BenchmarkMode(Mode.Throughput)
@OutputTimeUnit(TimeUnit.SECONDS)
@Warmup(iterations = 2, time = 2)
@Measurement(iterations = 3, time = 2)
@Fork(1)
@State(Scope.Benchmark)
public class ExecutorBenchmark {

    public enum Executor {
        SEQUENTIAL, PARALLEL
    }

    public enum Op {
        SCORE, GEOMETRY
    }

    @Param({ "D1", "D2" })
    Datasets.Id dataset;

    @Param({ "0.0", "0.001", "0.5", "1.0" })
    double cacheFraction;

    @Param
    Executor executor;

    @Param
    Op op;

    @Param({ "0" })
    int threads;

    @Param({ "1" })
    int batch;

    private RandomCutForest forest;
    private float[][] points; // as given: D2's are 10 wide, the forest shingles them
    private float[][][] batches; // for the score, as given; for the geometry, as the trees read them
    private IVisitorFactory<AnisotropicLocalGeometry> geometryFactory;

    @Setup(Level.Trial)
    public void setUp() {
        Datasets.Prepared p = Datasets.prepare(dataset, cacheFraction, 99L, Datasets.INITIAL,
                executor == Executor.PARALLEL, threads);
        forest = p.forest;
        points = p.scoreF;
        geometryFactory = AnisotropicLocalVisitor.factory(LocalGeometryConfig.DEFAULT);
        // frozen from here: no update in the measured region
        if (batch > 1) {
            int count = (Datasets.SCORED + batch - 1) / batch;
            batches = new float[count][][];
            for (int b = 0; b < count; b++) {
                float[][] chunk = Arrays.copyOfRange(points, b * batch, Math.min(Datasets.SCORED, (b + 1) * batch));
                if (op == Op.GEOMETRY) {
                    for (int i = 0; i < chunk.length; i++) {
                        chunk[i] = forest.transformToShingledPoint(chunk[i]);
                    }
                }
                batches[b] = chunk;
            }
        }
    }

    @Benchmark
    @OperationsPerInvocation(Datasets.SCORED)
    public double query() {
        final RandomCutForest f = forest;
        double checksum = 0.0;
        if (batch <= 1) {
            final float[][] p = points;
            if (op == Op.SCORE) {
                for (int i = 0; i < Datasets.SCORED; i++) {
                    checksum += f.getAnomalyScore(p[i]);
                }
            } else {
                for (int i = 0; i < Datasets.SCORED; i++) {
                    checksum += f.getAnisotropicGeometry(p[i]).getTreeMass();
                }
            }
            return checksum;
        }
        if (op == Op.SCORE) {
            for (float[][] chunk : batches) {
                for (double score : f.getAnomalyScore(chunk)) {
                    checksum += score;
                }
            }
        } else {
            for (float[][] chunk : batches) {
                List<AnisotropicLocalGeometry> geometry = f.traverseForest(chunk, geometryFactory,
                        AnisotropicLocalGeometry::merge, x -> x);
                for (AnisotropicLocalGeometry g : geometry) {
                    checksum += g.getTreeMass();
                }
            }
        }
        return checksum;
    }
}
