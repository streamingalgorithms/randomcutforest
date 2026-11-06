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

package org.streamingalgorithms.randomcutforest.projection;

import static org.junit.jupiter.api.Assertions.*;

import java.util.Arrays;
import java.util.Random;

import org.junit.jupiter.api.Test;
import org.streamingalgorithms.randomcutforest.RandomCutForest;

/**
 * The forest's projection is the projector's, tree by tree, whatever was
 * projected before; and it refuses what it cannot hold.
 */

public class ForestProjectionTest {

    static RandomCutForest forest(int d, long seed) {
        RandomCutForest forest = RandomCutForest.builder().numberOfTrees(8).sampleSize(64).dimensions(d)
                .randomSeed(seed).build();
        Random rng = new Random(seed);
        for (int t = 0; t < 1000; t++) {
            float[] p = new float[d];
            for (int j = 0; j < d; j++) {
                p[j] = (float) rng.nextGaussian();
            }
            forest.update(p);
        }
        return forest;
    }

    static void same(TreeProjection a, TreeProjection b) {
        int nodes = a.nodes();
        assertEquals(nodes, b.nodes());
        assertArrayEquals(Arrays.copyOf(a.left, nodes), Arrays.copyOf(b.left, nodes));
        assertArrayEquals(Arrays.copyOf(a.right, nodes), Arrays.copyOf(b.right, nodes));
        assertArrayEquals(Arrays.copyOf(a.lo, nodes), Arrays.copyOf(b.lo, nodes));
        assertArrayEquals(Arrays.copyOf(a.hi, nodes), Arrays.copyOf(b.hi, nodes));
        assertArrayEquals(Arrays.copyOf(a.mass, nodes), Arrays.copyOf(b.mass, nodes));
        assertArrayEquals(Arrays.copyOf(a.depth, nodes), Arrays.copyOf(b.depth, nodes));
        assertArrayEquals(Arrays.copyOf(a.storedPoint, nodes), Arrays.copyOf(b.storedPoint, nodes));
        assertArrayEquals(Arrays.copyOf(a.bounds, nodes * 2 * a.dimensions),
                Arrays.copyOf(b.bounds, nodes * 2 * b.dimensions));
    }

    @Test
    void eachTreeAsTheProjectorProjectsIt() {
        int d = 3, n = 60;
        RandomCutForest forest = forest(d, 7);
        Random rng = new Random(3);
        TreeProjection fromForest = new TreeProjection(n, d), direct = new TreeProjection(n, d);
        TreeProjector projector = new TreeProjector(d);
        for (int round = 0; round < 3; round++) {
            float[][] points = new float[n][d];
            for (float[] p : points) {
                for (int j = 0; j < d; j++) {
                    p[j] = (float) (2 * rng.nextGaussian());
                }
            }
            int m = n - 10 * round;
            // fewer points than before: nothing left over
            for (int t = 0; t < forest.getNumberOfTrees(); t++) {
                forest.projectOnTree(t, points, m, fromForest);
                projector.project(forest.getComponents().get(t), points, m, direct);
                same(direct, fromForest);
            }
            TreeProjection[] all = new TreeProjection[forest.getNumberOfTrees()];
            for (int t = 0; t < all.length; t++) {
                all[t] = new TreeProjection(n, d);
            }
            forest.projectOnTrees(points, m, all);
            for (int t = 0; t < all.length; t++) {
                projector.project(forest.getComponents().get(t), points, m, direct);
                same(direct, all[t]);
            }
        }
    }

    @Test
    void refusesWhatItCannotHold() {
        RandomCutForest forest = forest(2, 5);
        float[][] points = new float[4][2];
        assertThrows(IllegalArgumentException.class,
                () -> forest.projectOnTree(8, points, 4, new TreeProjection(4, 2)));
        assertThrows(IllegalArgumentException.class,
                () -> forest.projectOnTree(0, points, 4, new TreeProjection(4, 3)));
        assertThrows(IllegalArgumentException.class,
                () -> forest.projectOnTree(0, points, 5, new TreeProjection(4, 2)));
        assertThrows(IllegalArgumentException.class,
                () -> forest.projectOnTrees(points, 4, new TreeProjection[] { new TreeProjection(4, 2) }));
    }

    @Test
    void nothingBeforeTheOutputIsReady() {
        RandomCutForest forest = RandomCutForest.builder().numberOfTrees(2).sampleSize(64).dimensions(2).randomSeed(1)
                .build();
        forest.update(new float[] { 1, 2 });
        TreeProjection out = new TreeProjection(2, 2);
        forest.projectOnTree(0, new float[][] { { 1, 2 }, { 3, 4 } }, 2, out);
        assertEquals(0, out.nodes());
    }
}
