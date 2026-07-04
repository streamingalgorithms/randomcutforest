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

package org.streamingalgorithms.randomcutforest.benchmark.operations;

import org.streamingalgorithms.randomcutforest.CommonUtils;
import org.streamingalgorithms.randomcutforest.RandomCutForest;
import org.streamingalgorithms.randomcutforest.testutils.NormalMixtureTestData;
import org.streamingalgorithms.randomcutforest.testutils.ShingledMultiDimDataWithKeys;

/**
 * The two shared datasets, re-partitioned on OPERATION rather than
 * distribution. Both are 100 effective dimensions so the JOL retained-size
 * comparison is honest; they differ only in structure:
 * 
 * <pre>
 *   D1 = 100-dim normal mixture, shingle 1   (dense, no shingle sharing)
 *   D2 = 10 base-dim x shingle 10            (shingle sharing -> should be far smaller in the store)
 * </pre>
 *
 * <p>
 * {@link #prepare} builds a forest, drives {@link #INITIAL} points through
 * {@code update()} to push the sampler PAST fill-up into dynamic equilibrium
 * (this is what makes a frozen-forest JMH measurement stationary), and returns
 * the {@link #SCORED}-point hold-out in both double[] and float[] form,
 * converted once. Nothing here is timed.
 */
public final class Datasets {

    public enum Id {
        D1, D2;

        /** Effective (post-shingle) dimensionality; both are 100 by design. */
        public int effectiveDims() {
            return 100;
        }
    }

    public static final int INITIAL = 25_000;
    public static final int SCORED = 10_000;

    private static final int NUMBER_OF_TREES = 50;
    private static final int SAMPLE_SIZE = 256;
    // note that setting the seed 0 makes D1 rotate/generate freshly at random
    // that is not a bug -- the goal is to measure from a class and not specifics
    // even if this is set to non-zero the anomaly injection is randomized
    // per run in testUtils using Math.random() -- we suggest that the randomization
    // is the truth to depend on; but this remark and scaffolding is left in place
    private static int FIXED_SEED = 0;

    private Datasets() {
    }

    public static final class Prepared {
        public final RandomCutForest forest; // saturated, ready to score (frozen unless a driver updates)
        public final double[][] scoreD; // hold-out, double form (DEFAULT paths)
        public final float[][] scoreF; // hold-out, float form (DYNAMIC / DENSITY paths)
        public final int dims;

        Prepared(RandomCutForest forest, double[][] scoreD, float[][] scoreF, int dims) {
            this.forest = forest;
            this.scoreD = scoreD;
            this.scoreF = scoreF;
            this.dims = dims;
        }
    }

    public static Prepared prepare(Id id, double cacheFraction, long seed) {
        return prepare(id, cacheFraction, seed, INITIAL); // benchmark: full equilibrium
    }

    public static Prepared prepare(Id id, double cacheFraction, long seed, int saturation) {
        final int total = INITIAL + SCORED;
        final double[][] data;
        final RandomCutForest forest;

        switch (id) {
        case D1: {
            int dimensions = 100;
            data = new NormalMixtureTestData().generateTestData(total, dimensions, FIXED_SEED);
            forest = base(dimensions, 1, cacheFraction, seed).build();
            break;
        }
        case D2: {
            int baseDim = 10, shingleSize = 10, dimensions = baseDim * shingleSize;
            data = ShingledMultiDimDataWithKeys.getMultiDimData(total, 60, 100, 5, FIXED_SEED, baseDim).data;
            forest = base(dimensions, shingleSize, cacheFraction, seed).build();
            break;
        }
        default:
            throw new IllegalStateException("unknown dataset " + id);
        }

        // Saturate: drive the sampler to not have too many updates
        for (int i = 0; i < saturation; i++) {
            forest.update(data[i]);
        }
        double[][] scoreD = new double[SCORED][];
        float[][] scoreF = new float[SCORED][];
        for (int i = 0; i < SCORED; i++) {
            scoreD[i] = data[saturation + i];
            scoreF[i] = CommonUtils.toFloatArray(data[saturation + i]);
        }
        return new Prepared(forest, scoreD, scoreF, id.effectiveDims());
    }

    private static RandomCutForest.Builder<?> base(int dimensions, int shingleSize, double cacheFraction, long seed) {
        return RandomCutForest.builder().numberOfTrees(NUMBER_OF_TREES).dimensions(dimensions).shingleSize(shingleSize)
                .boundingBoxCacheFraction(cacheFraction).randomSeed(seed).outputAfter(SAMPLE_SIZE)
                .internalShinglingEnabled(true).initialAcceptFraction(0.1).sampleSize(SAMPLE_SIZE)
                .parallelExecutionEnabled(false);
    }

    /**
     * Raw D1/D2 rows (INITIAL+SCORED), the single generation source shared by all
     * suites.
     */
    public static double[][] rawData(Id id) {
        final int total = INITIAL + SCORED;
        switch (id) {
        case D1:
            return new NormalMixtureTestData().generateTestData(total, 100, FIXED_SEED);
        case D2:
            return ShingledMultiDimDataWithKeys.getMultiDimData(total, 60, 100, 5, FIXED_SEED, 10).data;
        default:
            throw new IllegalStateException("unknown dataset " + id);
        }
    }
}
