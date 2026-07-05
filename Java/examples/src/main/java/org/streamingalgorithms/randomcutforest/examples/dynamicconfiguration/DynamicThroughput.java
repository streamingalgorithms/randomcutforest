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

import java.time.Duration;
import java.time.Instant;

import org.streamingalgorithms.randomcutforest.RandomCutForest;
import org.streamingalgorithms.randomcutforest.examples.Example;
import org.streamingalgorithms.randomcutforest.examples.datasets.NormalMixture;

public class DynamicThroughput implements Example {

    public static void main(String[] args) throws Exception {
        new DynamicThroughput().run();
    }

    @Override
    public String command() {
        return "dynamic_caching";
    }

    @Override
    public String description() {
        return "serialize a Random Cut Forest as a JSON string";
    }

    @Override
    public void run() throws Exception {
        // Create and populate a random cut forest

        int dimensions = 4;
        int numberOfTrees = 50;
        int sampleSize = 256;
        int dataSize = 10 * sampleSize;
        NormalMixture data = new NormalMixture();
        // generate data once to eliminate caching issues
        data.generateData(dataSize, dimensions, 0);
        data.generateData(sampleSize, dimensions, 0);

        for (int i = 0; i < 5; i++) {

            RandomCutForest forest = RandomCutForest.builder().dimensions(dimensions).randomSeed(0)
                    .numberOfTrees(numberOfTrees).sampleSize(sampleSize).build();
            RandomCutForest forest2 = RandomCutForest.builder().dimensions(dimensions).randomSeed(0)
                    .numberOfTrees(numberOfTrees).sampleSize(sampleSize).build();
            forest2.setBoundingBoxCacheFraction(i * 0.25);

            int anomalies = 0;

            for (var point : data.generateData(dataSize, dimensions, 0).data) {
                double score = forest.getAnomalyScore(point);
                double score2 = forest2.getAnomalyScore(point);

                if (Math.abs(score - score2) > 1e-10) {
                    anomalies++;
                }
                forest.update(point);
                forest2.update(point);
            }

            Instant start = Instant.now();

            for (float[] point : data.generateData(sampleSize, dimensions, 0).data) {
                double score = forest.getAnomalyScore(point);
                double score2 = forest2.getAnomalyScore(point);

                if (Math.abs(score - score2) > 1e-10) {
                    anomalies++;
                }
                forest.update(point);
                forest2.update(point);
            }

            Instant finish = Instant.now();

            // first validate that this was a nontrivial test
            if (anomalies > 0) {
                throw new IllegalStateException("score mismatch");
            }

            System.out.println("So far so good! Caching fraction = " + (i * 0.25) + ", Time ="
                    + Duration.between(start, finish).toMillis() + " ms (note only one forest is changing)");
        }

    }

}
