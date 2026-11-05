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

package org.streamingalgorithms.randomcutforest.imputation;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.Random;

import org.junit.jupiter.api.Test;
import org.streamingalgorithms.randomcutforest.RandomCutForest;
import org.streamingalgorithms.randomcutforest.returntypes.RangeVector;

/**
 * The sequential path keeps imputation's working state across calls; it must
 * return, bit for bit, what the summary path returns, whatever the calls
 * before. The summary path is reached directly for imputation and through
 * multiRead for extrapolation.
 */
public class ImputationReuseTest {

    static RandomCutForest forest(int input, int shingle, boolean rotation, double cache, long seed) {
        RandomCutForest forest = RandomCutForest.builder().numberOfTrees(30).sampleSize(128).dimensions(input * shingle)
                .shingleSize(shingle).internalShinglingEnabled(true).internalRotationEnabled(rotation)
                .boundingBoxCacheFraction(cache).randomSeed(seed).build();
        Random rng = new Random(seed);
        for (int t = 0; t < 2000; t++) {
            float[] p = new float[input];
            for (int j = 0; j < input; j++) {
                p[j] = (float) ((t % 3) * 4 + rng.nextGaussian() + Math.sin(t / 9.0 + j));
            }
            forest.update(p);
        }
        return forest;
    }

    @Test
    void imputationMatchesTheSummary() {
        Random rng = new Random(17);
        for (int shingle : new int[] { 1, 4 }) {
            for (double cache : new double[] { 0.0, 1.0 }) {
                int input = (shingle == 1) ? 12 : 6;
                RandomCutForest forest = forest(input, shingle, false, cache, rng.nextLong());
                for (int q = 0; q < 40; q++) {
                    float[] p = new float[input];
                    for (int j = 0; j < input; j++) {
                        p[j] = (float) (6 * rng.nextGaussian());
                    }
                    // a different set of missing coordinates each call, some marked NaN
                    int[] missing = new int[1 + rng.nextInt(input - 1)];
                    for (int k = 0; k < missing.length; k++) {
                        missing[k] = rng.nextInt(input);
                        if (rng.nextBoolean()) {
                            p[missing[k]] = Float.NaN;
                        }
                    }
                    float[] reused = forest.imputeMissingValues(p, missing);
                    float[] summary = forest.getConditionalFieldSummary(p, missing, 1, 0, false, false, 1.0, 1).median;
                    assertArrayEquals(summary, reused);
                }
            }
        }
    }

    @Test
    void extrapolationMatchesTheSummary() {
        Random rng = new Random(23);
        for (boolean rotation : new boolean[] { false, true }) {
            RandomCutForest forest = forest(5, 6, rotation, 1.0, rng.nextLong());
            for (double centrality : new double[] { 1.0, 0.5, 0.0 }) {
                for (boolean cyclic : new boolean[] { false, true }) {
                    float[] shingle = forest.lastShingledPoint();
                    int index = rng.nextInt(6);
                    RangeVector reused = forest.extrapolateWithRanges(shingle, 5, 5, cyclic, index, centrality);
                    forest.setMultiRead(true);
                    RangeVector summary = forest.extrapolateWithRanges(shingle, 5, 5, cyclic, index, centrality);
                    forest.setMultiRead(false);
                    assertArrayEquals(summary.values, reused.values);
                    assertArrayEquals(summary.lower, reused.lower);
                    assertArrayEquals(summary.upper, reused.upper);
                }
            }
        }
    }

    @Test
    void aKnownCoordinateThatIsNotFiniteIsRefused() {
        RandomCutForest forest = forest(8, 1, false, 1.0, 5);
        float[] p = new float[8];
        p[0] = Float.NaN;
        assertThrows(IllegalArgumentException.class, () -> forest.imputeMissingValues(p, new int[] { 3 }));
        // and the next call is unaffected
        p[0] = 1;
        assertArrayEquals(forest.getConditionalFieldSummary(p, new int[] { 3 }, 1, 0, false, false, 1.0, 1).median,
                forest.imputeMissingValues(p, new int[] { 3 }));
    }
}
