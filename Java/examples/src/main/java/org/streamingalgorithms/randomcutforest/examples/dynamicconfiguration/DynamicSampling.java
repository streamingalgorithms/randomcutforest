/*
 * Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

package org.streamingalgorithms.randomcutforest.examples.dynamicconfiguration;

import org.streamingalgorithms.randomcutforest.RandomCutForest;
import org.streamingalgorithms.randomcutforest.examples.Example;
import org.streamingalgorithms.randomcutforest.examples.datasets.NormalMixture;
import org.streamingalgorithms.randomcutforest.state.RandomCutForestMapper;

public class DynamicSampling implements Example {

    public static void main(String[] args) throws Exception {
        new DynamicSampling().run();
    }

    @Override
    public String command() {
        return "dynamic_sampling";
    }

    @Override
    public String description() {
        return "check dynamic sampling";
    }

    @Override
    public void run() throws Exception {
        // Create and populate a random cut forest

        int dimensions = 4;
        int numberOfTrees = 50;
        int sampleSize = 256;
        int dataSize = 4 * sampleSize;
        NormalMixture data = new NormalMixture();

        RandomCutForest forest = RandomCutForest.builder().dimensions(dimensions).randomSeed(0)
                .numberOfTrees(numberOfTrees).sampleSize(sampleSize).build();
        RandomCutForest forest2 = RandomCutForest.builder().dimensions(dimensions).randomSeed(0)
                .numberOfTrees(numberOfTrees).sampleSize(sampleSize).build();

        int first_anomalies = 0;
        int second_anomalies = 0;
        forest2.setTimeDecay(10 * forest2.getTimeDecay());

        // will vary with seed == 0; use nonzero for deterministic
        for (var point : data.generateData(dataSize, dimensions, 0).data) {
            if (forest.getAnomalyScore(point) > 1.0) {
                first_anomalies++;
            }
            if (forest2.getAnomalyScore(point) > 1.0) {
                second_anomalies++;
            }
            forest.update(point);
            forest2.update(point);
        }
        System.out.println("Unusual scores: forest one " + first_anomalies + ", second one " + second_anomalies);
        // should be roughly equal

        first_anomalies = second_anomalies = 0;
        data = new NormalMixture(-3, 40);
        for (var point : data.generateData(dataSize, dimensions, 0).data) {
            if (forest.getAnomalyScore(point) > 1.0) {
                first_anomalies++;
            }
            if (forest2.getAnomalyScore(point) > 1.0) {
                second_anomalies++;
            }
            forest.update(point);
            forest2.update(point);
        }
        System.out.println("Unusual scores: forest one " + first_anomalies + ", second one " + second_anomalies);
        // forest2 should adapt faster

        first_anomalies = second_anomalies = 0;
        RandomCutForestMapper mapper = new RandomCutForestMapper();
        mapper.setSaveExecutorContextEnabled(true);
        RandomCutForest copyForest = mapper.toModel(mapper.toState(forest));
        copyForest.setTimeDecay(50 * forest.getTimeDecay());
        // force an adjustment to catch up
        data = new NormalMixture(-10, -40);
        int forced_change_anomalies = 0;
        for (var point : data.generateData(dataSize, dimensions, 0).data) {
            if (forest.getAnomalyScore(point) > 1.0) {
                first_anomalies++;
            }
            if (forest2.getAnomalyScore(point) > 1.0) {
                second_anomalies++;
            }
            if (copyForest.getAnomalyScore(point) > 1.0) {
                forced_change_anomalies++;
            }
            copyForest.update(point);
            forest.update(point);
            forest2.update(point);
        }
        // both should show the similar rate of adjustment
        System.out.println("Unusual scores: forest one " + first_anomalies + ", second one " + second_anomalies
                + ", forced (first) " + forced_change_anomalies);

    }
}
