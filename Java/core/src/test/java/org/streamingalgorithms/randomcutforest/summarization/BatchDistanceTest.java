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

package org.streamingalgorithms.randomcutforest.summarization;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.function.BiFunction;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.streamingalgorithms.randomcutforest.tree.VectorSupport;
import org.streamingalgorithms.randomcutforest.tree.VectorSupportLegacy;
import org.streamingalgorithms.randomcutforest.util.Weighted;

/**
 * The block kernel against VectorSupportLegacy bit for bit, and the block
 * assignment of iterativeClustering against the point-by-point one.
 */
public class BatchDistanceTest {

    @ParameterizedTest
    @ValueSource(ints = { 1, 2, 3, 4, 5, 7, 8, 9, 12, 15, 16, 17, 24, 33 })
    void nearestIsExact(int d) {
        Random rnd = new Random(42 + d);
        for (int trial = 0; trial < 40; trial++) {
            int n = 1 + rnd.nextInt(150);
            int m = 1 + rnd.nextInt(8);
            float scale = (float) Math.pow(10, rnd.nextInt(7) - 3);
            float[][] pts = new float[n][d];
            float[][] reps = new float[m][d];
            for (float[] r : reps) {
                for (int j = 0; j < d; j++) {
                    r[j] = (float) (scale * rnd.nextGaussian());
                }
            }
            if (m > 2) {
                reps[m - 1] = reps[1].clone(); // exact ties: the first must win
            }
            for (int p = 0; p < n; p++) {
                if (rnd.nextInt(10) == 0) {
                    pts[p] = reps[rnd.nextInt(m)].clone(); // zero distance
                    continue;
                }
                for (int j = 0; j < d; j++) {
                    pts[p][j] = (float) (scale * rnd.nextGaussian());
                }
            }
            float[] flat = new float[d * n];
            for (int p = 0; p < n; p++) {
                for (int j = 0; j < d; j++) {
                    flat[j * n + p] = pts[p][j];
                }
            }
            float[] rflat = new float[m * d];
            for (int r = 0; r < m; r++) {
                System.arraycopy(reps[r], 0, rflat, r * d, d);
            }
            for (boolean l2 : new boolean[] { true, false }) {
                int from = rnd.nextInt(n);
                int to = from + 1 + rnd.nextInt(n - from);
                int off = rnd.nextInt(5);
                double[] first = new double[n + 8];
                double[] best = new double[n + 8];
                int[] arg = new int[n + 8];
                (l2 ? BatchDistance.L2 : BatchDistance.L1).nearest(d, flat, n, from, to, rflat, 0, m, first, best, arg,
                        off, new double[BatchDistance.WORK_PER_POINT * (to - from)]);
                for (int p = from; p < to; p++) {
                    double f = l2 ? VectorSupportLegacy.L2distance(reps[0], pts[p])
                            : VectorSupportLegacy.L1distance(reps[0], pts[p]);
                    double b = f;
                    int a = 0;
                    for (int r = 1; r < m; r++) {
                        double t = l2 ? VectorSupportLegacy.L2distance(reps[r], pts[p])
                                : VectorSupportLegacy.L1distance(reps[r], pts[p]);
                        if (t < b) {
                            b = t;
                            a = r;
                        }
                    }
                    int o = off + p - from;
                    assertEquals(Double.doubleToRawLongBits(f), Double.doubleToRawLongBits(first[o]));
                    assertEquals(Double.doubleToRawLongBits(b), Double.doubleToRawLongBits(best[o]));
                    assertEquals(a, arg[o]);
                }
            }
        }
    }

    @Test
    void blocksChangeNoResult() {
        Random rnd = new Random(7);
        for (int d : new int[] { 1, 2, 3, 5 }) {
            if (!BatchDistance.L2.exactAt(d)) {
                continue; // the summarizer would not use the blocks here
            }
            for (int trial = 0; trial < 6; trial++) {
                int n = 200 + rnd.nextInt(2000);
                int blobs = 2 + rnd.nextInt(6);
                float[][] centers = new float[blobs][d];
                for (float[] c : centers) {
                    for (int j = 0; j < d; j++) {
                        c[j] = (float) (10 * rnd.nextGaussian());
                    }
                }
                List<Weighted<float[]>> points = new ArrayList<>();
                for (int i = 0; i < n; i++) {
                    float[] c = centers[rnd.nextInt(blobs)];
                    float[] x = new float[d];
                    for (int j = 0; j < d; j++) {
                        x[j] = c[j] + (float) rnd.nextGaussian();
                    }
                    points.add(new Weighted<>(x, rnd.nextFloat() + 0.01f));
                }
                for (boolean l2 : new boolean[] { true, false }) {
                    long seed = rnd.nextLong();
                    double shrinkage = 0.2 * rnd.nextDouble();
                    BiFunction<float[], Float, ICluster<float[]>> init = (x, w) -> MultiCenter.initialize(x, w,
                            shrinkage, 5);
                    BatchDistance distance = l2 ? BatchDistance.L2 : BatchDistance.L1;
                    // the blocks must be the path taken
                    List<Weighted<Integer>> refs = new ArrayList<>();
                    for (int i = 0; i < points.size(); i++) {
                        refs.add(new Weighted<>(i, points.get(i).weight));
                    }
                    @SuppressWarnings("unchecked")
                    ICluster<float[]>[] probe = new ICluster[] { init.apply(points.get(0).index, 0f) };
                    assertTrue(new Summarizer.Batch().prepare(refs, i -> points.get(i).index, distance, probe));

                    List<ICluster<float[]>> blocks = Summarizer.summarize(points, 12, 48, 1, true, 0.8, distance, init,
                            seed, false, null);
                    List<ICluster<float[]>> pointwise = Summarizer.summarize(points, 12, 48, 1, true, 0.8,
                            l2 ? (a, b) -> VectorSupport.L2distance(a, b) : (a, b) -> VectorSupport.L1distance(a, b),
                            init, seed, false, null);
                    assertEquals(pointwise.size(), blocks.size());
                    for (int k = 0; k < blocks.size(); k++) {
                        assertEquals(Double.doubleToRawLongBits(pointwise.get(k).getWeight()),
                                Double.doubleToRawLongBits(blocks.get(k).getWeight()));
                        assertEquals(Double.doubleToRawLongBits(pointwise.get(k).averageRadius()),
                                Double.doubleToRawLongBits(blocks.get(k).averageRadius()));
                        List<Weighted<float[]>> a = blocks.get(k).getRepresentatives();
                        List<Weighted<float[]>> b = pointwise.get(k).getRepresentatives();
                        assertEquals(b.size(), a.size());
                        for (int r = 0; r < a.size(); r++) {
                            for (int j = 0; j < d; j++) {
                                assertEquals(Float.floatToRawIntBits(b.get(r).index[j]),
                                        Float.floatToRawIntBits(a.get(r).index[j]));
                            }
                            assertEquals(Float.floatToRawIntBits(b.get(r).weight),
                                    Float.floatToRawIntBits(a.get(r).weight));
                        }
                    }
                }
            }
        }
    }
}
