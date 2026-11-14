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

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.util.Arrays;
import java.util.Random;

import org.junit.jupiter.api.Test;
import org.streamingalgorithms.randomcutforest.returntypes.DiVector;

/**
 * The forest reads float[] points; results that are sums over trees are double.
 * Queries that take a buffer write into it and return it.
 */
public class BufferedQueryTest {

    private static final int BASE = 3;

    private static float[] point(Random random, int length, double shift) {
        float[] p = new float[length];
        for (int j = 0; j < length; j++) {
            p[j] = (float) (random.nextGaussian() + shift);
        }
        return p;
    }

    private static RandomCutForest forest(boolean parallel, int shingleSize) {
        RandomCutForest.Builder<?> builder = RandomCutForest.builder().numberOfTrees(20).sampleSize(128)
                .dimensions(BASE * shingleSize).shingleSize(shingleSize).internalShinglingEnabled(shingleSize > 1)
                .randomSeed(23).parallelExecutionEnabled(parallel);
        if (parallel) {
            builder.threadPoolSize(3);
        }
        RandomCutForest forest = builder.build();
        Random random = new Random(29);
        for (int i = 0; i < 1500; i++) {
            forest.update(point(random, BASE, 0));
        }
        return forest;
    }

    @Test
    public void attributionIntoABufferMatchesTheDiVector() {
        for (boolean parallel : new boolean[] { false, true }) {
            for (int shingleSize : new int[] { 1, 4 }) {
                RandomCutForest forest = forest(parallel, shingleSize);
                int d = forest.getDimensions();
                double[] buffer = new double[2 * d];
                Random random = new Random(31);
                for (int i = 0; i < 40; i++) {
                    float[] p = point(random, BASE, (i % 4 == 0) ? 6 : 0);
                    DiVector expected = forest.getAnomalyAttribution(p);
                    Arrays.fill(buffer, Double.NaN); // every entry must be written
                    double[] out = forest.getAnomalyAttribution(p, buffer);
                    assertTrue(out == buffer);
                    double[] fresh = forest.getAnomalyAttribution(p, null);
                    for (int j = 0; j < d; j++) {
                        assertEquals(expected.high[j], out[j], 1e-9);
                        assertEquals(expected.low[j], out[d + j], 1e-9);
                        assertEquals(out[j], fresh[j], 1e-12);
                        assertEquals(out[d + j], fresh[d + j], 1e-12);
                    }
                    assertEquals(forest.getAnomalyScore(p), Arrays.stream(out).sum(), 1e-6);
                    // the DiVector of the array is the attribution
                    assertEquals(expected.getHighLowSum(), new DiVector(out).getHighLowSum(), 1e-9);
                }
            }
        }
    }

    @Test
    public void attributionBufferIsCheckedAndZeroBeforeOutput() {
        RandomCutForest empty = RandomCutForest.builder().dimensions(BASE).build();
        double[] buffer = new double[2 * BASE];
        Arrays.fill(buffer, 7.0);
        assertTrue(empty.getAnomalyAttribution(new float[BASE], buffer) == buffer);
        assertEquals(0.0, Arrays.stream(buffer).sum(), 0.0);
        assertThrows(IllegalArgumentException.class,
                () -> empty.getAnomalyAttribution(new float[BASE], new double[BASE]));
    }

    @Test
    public void imputeIntoABufferMatches() {
        for (int shingleSize : new int[] { 1, 4 }) {
            RandomCutForest forest = forest(false, shingleSize);
            int d = forest.getDimensions();
            float[] buffer = new float[d];
            Random random = new Random(37);
            for (int i = 0; i < 20; i++) {
                float[] p = point(random, BASE, 0);
                int[] missing = new int[] { i % BASE };
                float[] expected = forest.imputeMissingValues(p, missing);
                Arrays.fill(buffer, Float.NaN);
                float[] out = forest.imputeMissingValues(p, missing, buffer);
                assertTrue(out == buffer);
                assertEquals(d, expected.length);
                for (int j = 0; j < d; j++) {
                    assertEquals(expected[j], out[j], 0.0);
                }
            }
            float[] p = point(random, BASE, 0);
            assertThrows(IllegalArgumentException.class,
                    () -> forest.imputeMissingValues(p, new int[] { 0 }, new float[d + 1]));
            if (shingleSize == 1) {
                assertThrows(IllegalArgumentException.class, () -> forest.imputeMissingValues(p, new int[] { 0 }, p));
            }
        }
    }

    /**
     * No public method of the forest takes a point as double[] or returns one
     * without being deprecated. Sums over trees stay double: the batch scores and
     * the attribution buffer.
     */
    @Test
    public void everyDoublePointMethodIsDeprecated() {
        for (Method m : RandomCutForest.class.getDeclaredMethods()) {
            if (!Modifier.isPublic(m.getModifiers()) || m.isSynthetic()) {
                continue;
            }
            boolean statistics = (m.getName().equals("getAnomalyScore") && m.getParameterTypes().length == 1
                    && m.getParameterTypes()[0] == float[][].class)
                    || (m.getName().equals("getAnomalyAttribution") && m.getParameterTypes().length == 2
                            && m.getParameterTypes()[0] == float[].class);
            boolean doubles = m.getReturnType() == double[].class
                    || Arrays.asList(m.getParameterTypes()).contains(double[].class);
            if (doubles && !statistics) {
                assertTrue(m.isAnnotationPresent(Deprecated.class));
            }
        }
    }
}
