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

package org.streamingalgorithms.randomcutforest.interpolation;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import java.util.Random;

import org.junit.jupiter.api.Test;
import org.streamingalgorithms.randomcutforest.RandomCutForest;
import org.streamingalgorithms.randomcutforest.returntypes.AnisotropicLocalGeometry;
import org.streamingalgorithms.randomcutforest.returntypes.Escape;
import org.streamingalgorithms.randomcutforest.returntypes.LocalGeometryConfig;

public class EscapeVisitorTest {

    private static final int DIMENSIONS = 4;

    private static float[] point(Random random, double shift) {
        float[] p = new float[DIMENSIONS];
        for (int j = 0; j < DIMENSIONS; j++) {
            p[j] = (float) (random.nextGaussian() * (j < 2 ? 3 : 0.3) + shift);
        }
        return p;
    }

    private static RandomCutForest forest(boolean parallel, double cacheFraction) {
        RandomCutForest.Builder<?> builder = RandomCutForest.builder().numberOfTrees(20).sampleSize(128)
                .dimensions(DIMENSIONS).randomSeed(17).boundingBoxCacheFraction(cacheFraction)
                .parallelExecutionEnabled(parallel);
        if (parallel) {
            builder.threadPoolSize(3);
        }
        RandomCutForest forest = builder.build();
        Random random = new Random(3);
        for (int i = 0; i < 2000; i++) {
            forest.update(point(random, 0));
        }
        return forest;
    }

    @Test
    public void notReadyGivesNoTrees() {
        RandomCutForest forest = RandomCutForest.builder().dimensions(DIMENSIONS).build();
        Escape escape = forest.getEscape(new float[DIMENSIONS]);
        assertEquals(0, escape.getTreeCount());
        assertEquals(0.0, escape.getCodisplacement());
        assertFalse(escape.hasRow());
    }

    @Test
    public void readingsAreInRange() {
        RandomCutForest forest = forest(false, 1.0);
        Random random = new Random(5);
        for (int i = 0; i < 50; i++) {
            Escape escape = forest.getEscape(point(random, (i % 5 == 0) ? 20 : 0));
            assertEquals(20, escape.getTreeCount());
            assertTrue(escape.getCodisplacement() >= 0);
            // (E sqrt C)^2 <= E C
            assertTrue(escape.getSmoothedCodisplacement() <= escape.getCodisplacement() + 1e-9);
            assertTrue(escape.getRangeEscape() >= 0 && escape.getRangeEscape() <= 1 + 1e-9);
            assertTrue(escape.getExit() >= 0 && escape.getExit() <= 1 + 1e-9);
            assertTrue(escape.getPassageMass() >= 1 - 1e-9 && escape.getPassageMass() <= 128 + 1e-9);
            double exits = 0;
            for (int f = 0; f < 2 * DIMENSIONS; f++) {
                assertTrue(escape.getFaceExit(f) >= 0);
                assertTrue(escape.getCutGap(f) >= 0);
                exits += escape.getFaceExit(f);
            }
            assertEquals(escape.getExit(), exits, 1e-12);
        }
    }

    @Test
    public void aFarPointEscapesMoreThanACentralOne() {
        RandomCutForest forest = forest(false, 1.0);
        Escape far = forest.getEscape(new float[] { 40, 40, 40, 40 });
        Escape central = forest.getEscape(new float[] { 0, 0, 0, 0 });
        assertTrue(far.getCodisplacement() > central.getCodisplacement());
        assertTrue(far.getRangeEscape() > central.getRangeEscape());
        assertTrue(far.getExit() > central.getExit());
        // beyond every max face, so it leaves only by max faces
        for (int f = DIMENSIONS; f < 2 * DIMENSIONS; f++) {
            assertEquals(0.0, far.getFaceExit(f), 1e-12);
        }
    }

    @Test
    public void cutBoxAndPassageMassAgreeWithTheGeometry() {
        RandomCutForest forest = forest(false, 1.0);
        Random random = new Random(7);
        for (int i = 0; i < 30; i++) {
            float[] p = point(random, (i % 3 == 0) ? 8 : 0);
            Escape escape = forest.getEscape(p);
            AnisotropicLocalGeometry geometry = forest.getAnisotropicGeometry(p);
            for (int f = 0; f < 2 * DIMENSIONS; f++) {
                assertEquals(geometry.cut().gap()[f], escape.getCutGap(f), 1e-6);
            }
        }
    }

    @Test
    public void batchMatchesSingleAcrossExecutorsAndCaches() {
        Random random = new Random(11);
        float[][] points = new float[64][];
        for (int i = 0; i < points.length; i++) {
            points[i] = point(random, (i % 4 == 0) ? 10 : 0);
        }
        RandomCutForest reference = forest(false, 1.0);
        for (boolean parallel : new boolean[] { false, true }) {
            for (double cache : new double[] { 0.0, 0.3, 1.0 }) {
                RandomCutForest forest = forest(parallel, cache);
                List<Escape> batch = forest.getEscape(points, null);
                assertEquals(points.length, batch.size());
                for (int i = 0; i < points.length; i++) {
                    Escape expected = reference.getEscape(points[i]);
                    Escape single = forest.getEscape(points[i]);
                    for (Escape actual : new Escape[] { single, batch.get(i) }) {
                        assertEquals(expected.getTreeCount(), actual.getTreeCount());
                        assertEquals(expected.getCodisplacement(), actual.getCodisplacement(), 1e-6);
                        assertEquals(expected.getSmoothedCodisplacement(), actual.getSmoothedCodisplacement(), 1e-6);
                        assertEquals(expected.getRangeEscape(), actual.getRangeEscape(), 1e-6);
                        assertEquals(expected.getExit(), actual.getExit(), 1e-6);
                        assertEquals(expected.getPassageMass(), actual.getPassageMass(), 1e-6);
                    }
                }
            }
        }
    }

    @Test
    public void mergeAddsTrees() {
        RandomCutForest forest = forest(false, 1.0);
        float[] p = new float[] { 5, -5, 0, 0 };
        Escape whole = forest.getEscape(p);
        Escape twice = Escape.merge(whole.copy(), whole);
        assertEquals(2 * whole.getTreeCount(), twice.getTreeCount());
        assertEquals(whole.getCodisplacement(), twice.getCodisplacement(), 1e-12);
        assertEquals(whole.getExit(), twice.getExit(), 1e-12);
        assertThrows(IllegalArgumentException.class, () -> whole.add(new Escape(DIMENSIONS + 1)));
    }

    @Test
    public void centreMomentsAreOptional() {
        RandomCutForest forest = forest(false, 1.0);
        float[] p = new float[] { 6, 0, 0, 0 };
        AnisotropicLocalGeometry plain = forest.getAnisotropicGeometry(p);
        assertThrows(IllegalStateException.class, plain::lead);
        assertThrows(IllegalStateException.class, plain::swing);
        AnisotropicLocalGeometry full = forest.getAnisotropicGeometry(p,
                LocalGeometryConfig.DEFAULT.withCentreMoments(true));
        assertEquals(DIMENSIONS, full.lead().length);
        assertEquals(DIMENSIONS, full.swing().length);
        for (int f = 0; f < 2 * DIMENSIONS; f++) {
            assertEquals(full.cut().gap()[f], plain.cut().gap()[f], 1e-12);
        }
    }
}
