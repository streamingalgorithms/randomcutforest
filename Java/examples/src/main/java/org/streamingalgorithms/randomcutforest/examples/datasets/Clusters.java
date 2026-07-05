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

import static java.lang.Math.*;

import java.util.Arrays;
import java.util.Random;
import java.util.function.BiFunction;

public class Clusters {

    public static float[][] getData(int dataSize, int newDimensions, int seed,
            BiFunction<float[], float[], Double> distance) {
        double baseMu = 0.0;
        double baseSigma = 1.0;
        double anomalyMu = 0.0;
        double anomalySigma = 1.0;
        double transitionToAnomalyProbability = 0.0;
        // ignoring anomaly cluster for now
        double transitionToBaseProbability = 1.0;
        Random prg = new Random(0);
        NormalMixture generator = new NormalMixture(baseMu, baseSigma, anomalyMu, anomalySigma,
                transitionToAnomalyProbability, transitionToBaseProbability);
        float[][] data = generator.generateData(dataSize, newDimensions, seed).data;

        float[] allZero = new float[newDimensions];
        float[] sigma = new float[newDimensions];
        Arrays.fill(sigma, 1f);
        double scale = distance.apply(allZero, sigma);

        for (int i = 0; i < dataSize; i++) {
            // shrink, shift at random
            int nextD = prg.nextInt(newDimensions);
            for (int j = 0; j < newDimensions; j++) {
                data[i][j] *= 1.0 / (3.0);
                // standard deviation adds up across dimension; taking square root
                // and using s 3 sigma ball
                if (j == nextD) {
                    if (prg.nextDouble() < 0.5)
                        data[i][j] += 2.0 * scale;
                    else
                        data[i][j] -= 2.0 * scale;
                }
            }
        }
        return data;
    }
}