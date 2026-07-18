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
 *
 */

package org.streamingalgorithms.randomcutforest.examples.summarization;

import static java.lang.Math.PI;
import static org.streamingalgorithms.randomcutforest.examples.datasets.Fan.rotateClockWise;

import java.awt.*;
import java.awt.geom.Ellipse2D;
import java.awt.geom.Path2D;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import javax.swing.*;

import org.streamingalgorithms.randomcutforest.RandomCutForest;
import org.streamingalgorithms.randomcutforest.examples.Example;
import org.streamingalgorithms.randomcutforest.examples.datasets.Fan;
import org.streamingalgorithms.randomcutforest.examples.datasets.NormalMixture;
import org.streamingalgorithms.randomcutforest.examples.plot.GifWriter;
import org.streamingalgorithms.randomcutforest.examples.plot.Layer;
import org.streamingalgorithms.randomcutforest.examples.plot.Layers;
import org.streamingalgorithms.randomcutforest.examples.plot.Plot2D;
import org.streamingalgorithms.randomcutforest.summarization.ICluster;
import org.streamingalgorithms.randomcutforest.summarization.Summarizer;
import org.streamingalgorithms.randomcutforest.util.Weighted;

public class Summarization implements Example {

    public static void main(String[] args) throws Exception {
        new Summarization().run();
    }

    @Override
    public String command() {
        return "summarization";
    }

    @Override
    public String description() {
        return "shows a potential use of dynamic clustering/summarization";
    }

    private static final Color[] PALETTE = { new Color(214, 39, 40), // red
            new Color(31, 119, 180), // blue
            new Color(44, 160, 44), // green
            new Color(255, 127, 14), // orange
            new Color(148, 103, 189), // purple
            new Color(140, 86, 75), // brown
            new Color(227, 26, 158), // magenta
            new Color(23, 142, 150), // teal
            new Color(8, 48, 107), // navy
            new Color(177, 89, 40), // sienna
            new Color(106, 61, 154), // deep purple
            new Color(99, 99, 99), // gray
    };

    static final class ClusterView extends JPanel {
        private final double range;
        private volatile double[][] bg = new double[0][];
        private volatile java.util.List<double[][]> blades = new ArrayList<>();
        private volatile java.util.List<double[]> weights = new ArrayList<>();
        private volatile java.util.List<Color> colors = new ArrayList<>();

        ClusterView(double range) {
            this.range = range;
            setBackground(Color.WHITE);
        }

        void setFrame(double[][] bg, java.util.List<double[][]> blades, java.util.List<double[]> weights,
                java.util.List<Color> colors) {
            this.bg = bg;
            this.blades = blades;
            this.weights = weights;
            this.colors = colors;
            repaint(); // thread-safe, fine to call from the worker loop
        }

        @Override
        protected void paintComponent(Graphics g) {
            super.paintComponent(g);
            Graphics2D g2 = (Graphics2D) g;
            g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

            int w = getWidth(), h = getHeight(), m = 24;
            double pw = w - 2.0 * m, ph = h - 2.0 * m;

            // faint frame + origin axes for orientation
            g2.setColor(new Color(235, 235, 235));
            for (int u = -15; u <= 15; u += 5) {
                int px = (int) (m + (u + range) / (2 * range) * pw);
                int py = (int) (m + (range - u) / (2 * range) * ph);
                g2.drawLine(px, m, px, h - m);
                g2.drawLine(m, py, w - m, py);
            }
            g2.setColor(new Color(200, 200, 200));
            g2.drawRect(m, m, (int) pw, (int) ph);

            // background: the raw rotating points
            g2.setColor(new Color(218, 218, 218));
            for (double[] p : bg) {
                int px = (int) (m + (p[0] + range) / (2 * range) * pw);
                int py = (int) (m + (range - p[1]) / (2 * range) * ph);
                g2.fillOval(px - 1, py - 1, 2, 2);
            }

            // clusters: filled blade + connecting line + weight-sized centers
            for (int b = 0; b < blades.size(); b++) {
                double[][] pts = blades.get(b);
                double[] ww = weights.get(b);
                Color col = colors.get(b);

                Path2D path = new Path2D.Double();
                double[][] px = new double[pts.length][2];
                for (int i = 0; i < pts.length; i++) {
                    px[i][0] = m + (pts[i][0] + range) / (2 * range) * pw;
                    px[i][1] = m + (range - pts[i][1]) / (2 * range) * ph;
                    if (i == 0)
                        path.moveTo(px[i][0], px[i][1]);
                    else
                        path.lineTo(px[i][0], px[i][1]);
                }
                if (pts.length > 2) {
                    path.closePath();
                    g2.setColor(new Color(col.getRed(), col.getGreen(), col.getBlue(), 55));
                    g2.fill(path);
                }
                g2.setColor(col);
                g2.setStroke(new BasicStroke(1.6f));
                g2.draw(path);

                for (int i = 0; i < pts.length; i++) {
                    double r = 3.0 + 13.0 * Math.sqrt(ww[i]); // area ~ weight; min radius keeps lone centers visible
                    g2.fill(new Ellipse2D.Double(px[i][0] - r, px[i][1] - r, 2 * r, 2 * r));
                }
            }
        }
    }

    @Override
    public void run() throws Exception {
        int newDimensions = 2;
        long randomSeed = 123;
        int dataSize = 1350;
        int numberOfBlades = 5;
        double range = 15.0;

        RandomCutForest newForest = RandomCutForest.builder().numberOfTrees(100).sampleSize(256)
                .dimensions(newDimensions).randomSeed(randomSeed).timeDecay(1.0 / 800).centerOfMassEnabled(true)
                .build();

        boolean printFile = false; // old gnuplot text dump
        boolean livePlot = true; // on-screen window
        boolean saveGif = true; // animated gif output
        int frameDelayMs = 8; // on-screen pacing
        int gifSizePx = 700;
        int gifDelayMs = 40; // ~25 fps in the file
        int gifEvery = 1; // write every Nth frame to the gif; raise to shrink the file
        int reportEvery = 30;

        BufferedWriter file = printFile ? new BufferedWriter(new FileWriter("dynamic_summarization_example")) : null;

        float[][] data = Fan.getData(dataSize, 0, numberOfBlades);

        Plot2D plot = livePlot ? Plot2D.open("Dynamic Summarization (" + numberOfBlades + " blades)", range, 860)
                : Plot2D.offscreen(range);
        GifWriter gif = saveGif ? new GifWriter(new File("dynamic_summarization.gif"), gifDelayMs, true) : null;

        List<ICluster<float[]>> oldSummary = null;
        int[] oldColors = null;

        int count = 0, sum = 0;
        Instant start = Instant.now();
        long updateNanos = 0, summarizeNanos = 0;

        for (int degree = 0; degree < 360; degree += 1) {
            float[][] bg = new float[data.length][2];
            int n = 0;

            Instant u0 = Instant.now();
            for (var datum : data) {
                float[] vec = rotateClockWise(datum, -2 * PI * degree / 360);
                bg[n][0] = vec[0];
                bg[n][1] = vec[1];
                n++;
                if (printFile) {
                    file.append(vec[0] + " " + vec[1] + "\n");
                }
                newForest.update(vec);
            }
            updateNanos += Duration.between(u0, Instant.now()).toNanos();
            if (printFile) {
                file.append("\n");
                file.append("\n");
            }

            Instant s0 = Instant.now();
            List<ICluster<float[]>> summary = newForest.summarize(2 * numberOfBlades + 2, 0.05, 5, 0.8,
                    Summarizer::L2distance, oldSummary);
            summarizeNanos += Duration.between(s0, Instant.now()).toNanos();

            sum += summary.size();
            System.out.println(degree + " " + summary.size());
            if (summary.size() == numberOfBlades) {
                ++count;
            }
            int[] colors = align(summary, oldSummary, oldColors);

            // ---- build the scene ----
            List<Layer> scene = new ArrayList<>();
            scene.add(Layers.dots(bg, new Color(150, 150, 150), 1.8));
            //scene.add(Layers.dots(bg, new Color(218, 218, 218), 1.0));
            for (int i = 0; i < summary.size(); i++) {
                double weight = summary.get(i).getWeight();
                List<double[]> rp = new ArrayList<>();
                List<Double> rw = new ArrayList<>();
                for (Weighted<float[]> rep : summary.get(i).getRepresentatives()) {
                    double t = rep.weight / weight;
                    if (t > 0.05) {
                        if (printFile) {
                            file.append(rep.index[0] + " " + rep.index[1] + " " + t + " " + colors[i] + "\n");
                        }
                        rp.add(new double[] { rep.index[0], rep.index[1] });
                        rw.add(t);
                    }
                }
                if (rp.isEmpty()) {
                    continue;
                }
                int[] ord = angleOrder(rp);
                double[][] ring = new double[ord.length][];
                double[] w = new double[ord.length];
                for (int k = 0; k < ord.length; k++) {
                    ring[k] = rp.get(ord[k]);
                    w[k] = rw.get(ord[k]);
                }
                Color col = Layers.color(colors[i]);
                scene.add(Layers.polyline(ring, col, true, 55, 1.6f));
                scene.add(Layers.weightedDots(ring, w, col, 3.0, 13.0));
            }
            if (printFile) {
                file.append("\n");
                file.append("\n");
            }

            // ---- output the scene ----
            if (livePlot) {
                plot.render(scene);
                if (frameDelayMs > 0) {
                    Thread.sleep(frameDelayMs);
                }
            }
            if (saveGif && degree % gifEvery == 0) {
                gif.writeFrame(plot.renderImage(gifSizePx, gifSizePx, scene));
            }

            if ((degree + 1) % reportEvery == 0) {
                long ms = Duration.between(start, Instant.now()).toMillis();
                System.out.printf("[%3d deg] total %d ms | summarize %.1f ms (%.2f ms/frame) | update %.1f ms%n",
                        degree + 1, ms, summarizeNanos / 1e6, summarizeNanos / 1e6 / (degree + 1), updateNanos / 1e6);
            }

            if (summary.size() == numberOfBlades) {
                oldSummary = summary;
                oldColors = colors;
            }
        }

        System.out.printf("Exact detection: %.2f fraction, avg clusters %.2f%n", Math.round(count / 3.6) * 0.01,
                Math.round(sum / 3.6) * 0.01);

        if (gif != null) {
            gif.close();
            System.out.println("wrote dynamic_summarization.gif");
        }
        if (file != null) {
            file.close();
        }
    }

    private static int[] angleOrder(List<double[]> pts) {
        double cx = 0, cy = 0;
        for (double[] p : pts) {
            cx += p[0];
            cy += p[1];
        }
        final double fcx = cx / pts.size(), fcy = cy / pts.size();
        Integer[] idx = new Integer[pts.size()];
        for (int k = 0; k < idx.length; k++) {
            idx[k] = k;
        }
        Arrays.sort(idx, (a, b) -> Double.compare(Math.atan2(pts.get(a)[1] - fcy, pts.get(a)[0] - fcx),
                Math.atan2(pts.get(b)[1] - fcy, pts.get(b)[0] - fcx)));
        int[] out = new int[idx.length];
        for (int k = 0; k < idx.length; k++) {
            out[k] = idx[k];
        }
        return out;
    }

    private Color colorFor(int c, int max) {
        return Color.getHSBColor((float) c / max, 0.78f, 0.9f);
    }

    public float[][] getData(int dataSize, int seed, int fans) {
        Random prg = new Random(0);
        NormalMixture generator = new NormalMixture(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
        int newDimensions = 2;
        float[][] data = generator.generateData(dataSize, newDimensions, seed).data;

        for (int i = 0; i < dataSize; i++) {
            int nextFan = prg.nextInt(fans);
            // scale, make an ellipse
            data[i][1] *= 1.0 / fans;
            data[i][0] *= 2.0;
            // shift
            data[i][0] += 5.0 + fans / 2;
            data[i] = rotateClockWise(data[i], 2 * PI * nextFan / fans);
        }

        return data;
    }

    int[] align(List<ICluster<float[]>> current, List<ICluster<float[]>> previous, int[] oldColors) {
        int[] nearest = new int[current.size()];

        if (previous == null || previous.size() == 0) {
            for (int i = 0; i < current.size(); i++) {
                nearest[i] = i;
            }
        } else {
            Arrays.fill(nearest, previous.size() + 1);
            for (int i = 0; i < current.size(); i++) {
                double dist = previous.get(0).distance(current.get(i), Summarizer::L1distance);
                nearest[i] = oldColors[0];
                for (int j = 1; j < previous.size(); j++) {
                    double t = previous.get(j).distance(current.get(i), Summarizer::L1distance);
                    if (t < dist) {
                        dist = t;
                        nearest[i] = oldColors[j];
                    }
                }
            }
        }
        return nearest;
    }
}
