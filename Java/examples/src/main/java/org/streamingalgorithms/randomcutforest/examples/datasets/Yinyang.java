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

public class Yinyang {
    public static float[][] generate(int size) {
        Random prg = new Random();
        float[][] data = new float[size][2];

        for (int i = 0; i < size; i++) {
            boolean test = false;
            while (!test) {
                double x = 2 * prg.nextDouble() - 1;
                double y = 2 * prg.nextDouble() - 1;
                if (x * x + y * y <= 1) {
                    if (y > 0) {
                        if (x > 0 && ((x - 0.5) * (x - 0.5) + y * y) <= 0.25) {
                            test = ((x - 0.5) * (x - 0.5) + y * y > 1.0 / 32) && (prg.nextDouble() < 0.6);
                        }
                    } else {
                        if (x > 0) {
                            if ((x - 0.5) * (x - 0.5) + y * y > 1.0 / 32) {
                                test = ((x - 0.5) * (x - 0.5) + y * y < 0.25) || (prg.nextDouble() < 0.4);
                            }
                        } else {
                            test = ((x + 0.5) * (x + 0.5) + y * y > 0.25) && (prg.nextDouble() < 0.2);
                        }
                    }
                }
                if (test) {
                    data[i][0] = (float) x;
                    data[i][1] = (float) y;
                }
            }
        }
        return data;
    }
}