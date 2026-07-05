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

package org.streamingalgorithms.randomcutforest.examples.datasets;

import static java.lang.Math.PI;

import java.util.Arrays;
import java.util.Random;

public class MultiDimData {
    public float[][] data;
    public int[] changeIndices;
    public float[][] changes;

    public MultiDimData(float[][] data, int[] changeIndices, float[][] changes) {
        this.data = data;
        this.changeIndices = changeIndices;
        this.changes = changes;
    }

    public static MultiDimData multiDimData(int num, int period, double amplitude, double noise, long seed,
            int baseDimension, double anomalyFactor, boolean useSlope) {
        float[][] data = new float[num][];
        float[][] changes = new float[num][];
        int[] changedIndices = new int[num];
        int counter = 0;
        Random prg = new Random(seed);
        Random noiseprg = new Random(prg.nextLong());
        double[] phase = new double[baseDimension];
        double[] amp = new double[baseDimension];
        double[] slope = new double[baseDimension];
        double[] shift = new double[baseDimension];

        for (int i = 0; i < baseDimension; i++) {
            phase[i] = prg.nextInt(period);
            if (useSlope) {
                shift[i] = (4 * prg.nextDouble() - 1) * amplitude;
            }
            amp[i] = (1 + 0.2 * prg.nextDouble()) * amplitude;
            if (useSlope) {
                slope[i] = (0.25 - prg.nextDouble() * 0.5) * amplitude / period;
            }
        }

        for (int i = 0; i < num; i++) {
            data[i] = new float[baseDimension];
            boolean flag = (noiseprg.nextDouble() < 0.01);
            float[] newChange = new float[baseDimension];
            boolean used = false;
            for (int j = 0; j < baseDimension; j++) {
                data[i][j] = (float) (amp[j] * Math.cos(2 * PI * (i + phase[j]) / period) + slope[j] * i + shift[j]);
                if (flag && noiseprg.nextDouble() < 0.3) {
                    double factor = anomalyFactor * (1 + noiseprg.nextDouble());
                    double change = noiseprg.nextDouble() < 0.5 ? factor * noise : -factor * noise;
                    newChange[j] = (float) change;
                    data[i][j] = (float) (data[i][j] + change);
                    used = true;
                } else {
                    data[i][j] = (float) (data[i][j] + noise * (2 * noiseprg.nextDouble() - 1));
                }
            }
            if (used) {
                changedIndices[counter] = i;
                changes[counter++] = newChange;
            }
        }

        return new MultiDimData(data, Arrays.copyOf(changedIndices, counter), Arrays.copyOf(changes, counter));
    }

}
