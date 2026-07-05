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

import java.util.Random;

public class Fan {
    public static float[][] getData(int dataSize, int seed, int fans) {
        Random prg = new Random(0);
        NormalMixture generator = new NormalMixture(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
        int newDimensions = 2;
        float[][] data = generator.generateData(dataSize, newDimensions, seed).data;

        for (int i = 0; i < dataSize; i++) {
            int nextFan = prg.nextInt(fans);
            // scale, make an ellipse
            data[i][1] *= 1.0f / fans;
            data[i][0] *= 2.0;
            // shift
            data[i][0] += 5.0 + fans / 2;
            data[i] = rotateClockWise(data[i], 2 * PI * nextFan / fans);
        }

        return data;
    }

    public static float[] rotateClockWise(float[] point, double theta) {
        float[] result = new float[2];
        result[0] = (float) (cos(theta) * point[0] + sin(theta) * point[1]);
        result[1] = (float) (-sin(theta) * point[0] + cos(theta) * point[1]);
        return result;
    }
}