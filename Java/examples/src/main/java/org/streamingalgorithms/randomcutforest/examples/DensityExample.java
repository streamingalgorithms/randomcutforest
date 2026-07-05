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
import org.streamingalgorithms.randomcutforest.returntypes.DensityOutput;

public class DensityExample implements Example {

    public static void main(String[] args) throws Exception {
        new DensityExample().run();
    }

    @Override
    public String command() {
        return "density";
    }

    @Override
    public String description() {
        return "directional dynamic density";
    }

    @Override
    public void run() throws Exception {
        int newDimensions = 2;
        long randomSeed = 123;
        double range = 1.6;

        RandomCutForest newForest = RandomCutForest.builder().numberOfTrees(100).sampleSize(256)
                .dimensions(newDimensions).randomSeed(randomSeed).timeDecay(1.0 / 800).centerOfMassEnabled(true)
                .build();

        boolean printFile = false;
        boolean livePlot = true;
        boolean saveGif = true;
        int frameDelayMs = 12;
        int gifSizePx = 700;
        int gifDelayMs = 50;
        int gifEvery = 1;
        int reportEvery = 30;

        BufferedWriter file = printFile ? new BufferedWriter(new FileWriter("dynamic_density_example")) : null;

        float[][] data = Yinyang.generate(2000);
        Plot2D plot = livePlot ? Plot2D.open("Dynamic Density + Directional Field", range, 820)
                : Plot2D.offscreen(range);
        GifWriter gif = saveGif ? new GifWriter(new File("dynamic_density.gif"), gifDelayMs, true) : null;

        Instant start = Instant.now();
        long densityNanos = 0;
        int frame = 0;

        for (int degree = 0; degree < 360; degree += 2) {
            for (float[] datum : data) {
                newForest.update(rotateClockWise(datum, -2 * PI * degree / 360));
            }

            // density at each rotated data point
            float[][] pts = new float[data.length][2];
            double[] dens = new double[data.length];
            Instant d0 = Instant.now();
            for (int i = 0; i < data.length; i++) {
                float[] q = rotateClockWise(data[i], -2 * PI * degree / 360);
                pts[i] = q;
                dens[i] = newForest.getSimpleDensity(q).getDensity(0.001, 2);
                if (printFile) {
                    file.append(q[0] + " " + q[1] + " " + dens[i] + "\n");
                }
            }
            densityNanos += Duration.between(d0, Instant.now()).toNanos();
            if (printFile) {
                file.append("\n");
                file.append("\n");
            }

            // directional field on a grid (one getDirectionalDensity call per cell)
            List<double[]> go = new ArrayList<>();
            List<double[]> gd = new ArrayList<>();
            for (double x = -0.95; x < 1; x += 0.1) {
                for (double y = -0.95; y < 1; y += 0.1) {
                    DensityOutput density = newForest.getSimpleDensity(new double[] { x, y });
                    var dir = density.getDirectionalDensity(0.001, 2);
                    double aboveInY = dir.low[1], belowInY = dir.high[1];
                    double toTheLeft = dir.high[0], toTheRight = dir.low[0];
                    double len = Math.sqrt(aboveInY * aboveInY + belowInY * belowInY + toTheLeft * toTheLeft
                            + toTheRight * toTheRight);
                    double dx = (toTheRight - toTheLeft) * 0.05 / len;
                    double dy = (aboveInY - belowInY) * 0.05 / len;
                    go.add(new double[] { x, y });
                    gd.add(new double[] { dx, dy });
                    if (printFile) {
                        file.append(x + " " + y + " " + dx + " " + dy + "\n");
                    }
                }
            }
            if (printFile) {
                file.append("\n");
                file.append("\n");
            }

            List<Layer> scene = new ArrayList<>();
            scene.add(Layers.scalarDots(pts, dens, 3.0));
            scene.add(Layers.arrows(go.toArray(new double[0][]), gd.toArray(new double[0][]), new Color(70, 70, 70),
                    1.2f));

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
                System.out.printf("[%3d deg] total %d ms | density %.2f ms/frame%n", degree,
                        Duration.between(start, Instant.now()).toMillis(), densityNanos / 1e6 / frame);
            }
        }

        if (gif != null) {
            gif.close();
            System.out.println("wrote dynamic_density.gif");
        }
        if (file != null) {
            file.close();
        }
    }

}
