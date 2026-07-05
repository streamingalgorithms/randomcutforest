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

package org.streamingalgorithms.randomcutforest.examples.summarization;

import static java.lang.Math.abs;

import java.util.Random;

import org.streamingalgorithms.randomcutforest.examples.Example;
import org.streamingalgorithms.randomcutforest.examples.datasets.Clusters;
import org.streamingalgorithms.randomcutforest.returntypes.SampleSummary;
import org.streamingalgorithms.randomcutforest.summarization.Summarizer;

/**
 * The following example is based off a test of summarization and provides an
 * example use of summarization based on centroidal representation. The
 * clustering takes a distance function from (float[],float []) into double as
 * input, along with a maximum number of allowed clusters and provides a summary
 * which contains the list of cluster centers as "typical points" along with
 * relative likelihood. The specific example below corresponds to 2*d clusters
 * (one each in +ve and -ve axis for each of the d dimensions) where d is chosen
 * at random between 3 and 13. The clusters are designed to almost touch -- but
 * are separable (with high probability) and should be discoverable separately,
 * in particular different projections would discover them separately. Note that
 * the algorithm does not require the knowledge of the true number of clusters
 * (2*d) but is run with a maximum allowed number 5*d.
 */
/*
 * mvn -pl examples -am install -DskipTests java --add-modules
 * jdk.incubator.vector -jar examples/target/*-jar-with-dependencies.jar
 * rcf_summarize
 */

public class CentroidSummarize implements Example {

    public static void main(String[] args) throws Exception {
        new CentroidSummarize().run();
    }

    @Override
    public String command() {
        return "centroid_summarize";
    }

    @Override
    public String description() {
        return "Example of using centroid Summarization";
    }

    @Override
    public void run() throws Exception {
        long seed = new Random().nextLong();
        Random random = new Random(seed);
        int newDimensions = random.nextInt(10) + 3;
        int dataSize = 200000;

        float[][] points = Clusters.getData(dataSize, newDimensions, random.nextInt(), Summarizer::L2distance);

        SampleSummary summary = Summarizer.l2summarize(points, 5 * newDimensions, 42);
        System.out.println(
                summary.summaryPoints.length + " clusters for " + newDimensions + " dimensions, seed : " + seed);
        double epsilon = 0.01;
        System.out.println("Total weight " + summary.weightOfSamples + " rounding to multiples of " + epsilon);
        System.out.println();
        for (int i = 0; i < summary.summaryPoints.length; i++) {
            long t = Math.round(summary.relativeWeight[i] / epsilon);
            System.out.print("Cluster " + i + " relative weight " + ((float) t * epsilon) + " center (approx): ");
            printArray(summary.summaryPoints[i], epsilon);
            System.out.println();
        }

    }

    void printArray(float[] values, double epsilon) {
        System.out.print(" [");
        if (abs(values[0]) < epsilon) {
            System.out.print("0");
        } else {
            if (epsilon <= 0) {
                System.out.print(values[0]);
            } else {
                long t = (int) Math.round(values[0] / epsilon);
                System.out.print((float) t * epsilon);
            }
        }
        for (int i = 1; i < values.length; i++) {
            if (abs(values[i]) < epsilon) {
                System.out.print(", 0");
            } else {
                if (epsilon <= 0) {
                    System.out.print(", " + values[i]);
                } else {
                    long t = Math.round(values[i] / epsilon);
                    System.out.print(", " + ((float) t * epsilon));
                }
            }
        }
        System.out.print("]");
    }

}
