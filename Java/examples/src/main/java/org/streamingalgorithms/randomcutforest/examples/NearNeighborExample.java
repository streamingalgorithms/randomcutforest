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

package org.streamingalgorithms.randomcutforest.examples;

import static java.lang.Math.PI;
import static org.streamingalgorithms.randomcutforest.examples.datasets.Fan.rotateClockWise;

import java.awt.*;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;

import org.streamingalgorithms.randomcutforest.RandomCutForest;
import org.streamingalgorithms.randomcutforest.examples.datasets.Yinyang;
import org.streamingalgorithms.randomcutforest.examples.plot.GifWriter;
import org.streamingalgorithms.randomcutforest.examples.plot.Layer;
import org.streamingalgorithms.randomcutforest.examples.plot.Layers;
import org.streamingalgorithms.randomcutforest.examples.plot.Plot2D;

public class NearNeighborExample implements Example {

    public static void main(String[] args) throws Exception {
        new NearNeighborExample().run();
    }

    @Override
    public String command() {
        return "near_neighbor";
    }

    @Override
    public String description() {
        return "example of dynamic near neighbor computation";
    }

    @Override
    public void run() throws Exception {
        int newDimensions = 2;
        long randomSeed = 123;
        double range = 1.6;

        RandomCutForest newForest = RandomCutForest.builder().numberOfTrees(100).sampleSize(256)
                .dimensions(newDimensions).randomSeed(randomSeed).timeDecay(1.0 / 800).centerOfMassEnabled(true)
                .storeSequenceIndexesEnabled(true).build();

        boolean printFile = false;
        boolean livePlot = true;
        boolean saveGif = true;
        int frameDelayMs = 12;
        int gifSizePx = 700;
        int gifDelayMs = 50;
        int gifEvery = 1;
        int reportEvery = 30;

        BufferedWriter file = printFile ? new BufferedWriter(new FileWriter("dynamic_near_neighbor_example")) : null;

        float[][] data = Yinyang.generate(1000);
        float[] queryPoint = new float[] { 0.5f, 0.6f };

        Plot2D plot = livePlot ? Plot2D.open("Dynamic Near Neighbor", range, 820) : Plot2D.offscreen(range);
        GifWriter gif = saveGif ? new GifWriter(new File("dynamic_near_neighbor.gif"), gifDelayMs, true) : null;

        Instant start = Instant.now();
        long queryNanos = 0;
        int frame = 0;

        for (int degree = 0; degree < 360; degree += 2) {
            float[][] bg = new float[data.length][2];
            int n = 0;
            for (var datum : data) {
                float[] transformed = rotateClockWise(datum, -2 * PI * degree / 360);
                bg[n][0] = transformed[0];
                bg[n][1] = transformed[1];
                n++;
                if (printFile) {
                    file.append(transformed[0] + " " + transformed[1] + "\n");
                }
                newForest.update(transformed);
            }
            if (printFile) {
                file.append("\n");
                file.append("\n");
            }

            float[] movingQuery = rotateClockWise(queryPoint, -3 * PI * degree / 360);
            Instant q0 = Instant.now();
            float[] neighbor = newForest.getNearNeighborsInSample(movingQuery, 1).get(0).point;
            queryNanos += Duration.between(q0, Instant.now()).toNanos();

            if (printFile) {
                file.append(movingQuery[0] + " " + movingQuery[1] + " " + (neighbor[0] - movingQuery[0]) + " "
                        + (neighbor[1] - movingQuery[1]) + "\n");
                file.append("\n");
                file.append("\n");
            }

            List<Layer> scene = new ArrayList<>();
            scene.add(Layers.dots(bg, new Color(120, 120, 120), 2.2));
            scene.add(Layers.arrows(new double[][] { { movingQuery[0], movingQuery[1] } },
                    new double[][] { { neighbor[0] - movingQuery[0], neighbor[1] - movingQuery[1] } }, Layers.color(1),
                    2.0f));
            scene.add(Layers.dots(new float[][] { { neighbor[0], neighbor[1] } }, Layers.color(0), 5));
            scene.add(Layers.dots(new float[][] { { movingQuery[0], movingQuery[1] } }, Color.BLACK, 4));

            if (livePlot) {
                plot.render(scene);
                if (frameDelayMs > 0) {
                    Thread.sleep(frameDelayMs);
                }
            }
            if (saveGif && frame % gifEvery == 0) {
                gif.writeFrame(plot.renderImage(gifSizePx, gifSizePx, scene));
            }

            if (++frame % reportEvery == 0) {
                System.out.printf("[%3d deg] total %d ms | query %.3f ms/frame%n", degree,
                        Duration.between(start, Instant.now()).toMillis(), queryNanos / 1e6 / frame);
            }
        }

        if (gif != null) {
            gif.close();
            System.out.println("wrote dynamic_near_neighbor.gif");
        }
        if (file != null) {
            file.close();
        }
    }
}
