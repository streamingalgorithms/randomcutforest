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

import static org.streamingalgorithms.randomcutforest.CommonUtils.toDoubleArray;
import static org.streamingalgorithms.randomcutforest.examples.datasets.MultiDimData.multiDimData;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.geom.Ellipse2D;
import java.io.File;
import java.util.ArrayList;
import java.util.List;

import org.streamingalgorithms.randomcutforest.config.TransformMethod;
import org.streamingalgorithms.randomcutforest.examples.datasets.MultiDimData;
import org.streamingalgorithms.randomcutforest.examples.plot.GifWriter;
import org.streamingalgorithms.randomcutforest.examples.plot.Layer;
import org.streamingalgorithms.randomcutforest.examples.plot.Layers;
import org.streamingalgorithms.randomcutforest.examples.plot.Plot2D;
import org.streamingalgorithms.randomcutforest.parkservices.AnomalyDescriptor;
import org.streamingalgorithms.randomcutforest.parkservices.ThresholdedRandomCutForest;
import org.streamingalgorithms.randomcutforest.parkservices.config.CorrectionMode;

/**
 * Animated companion to
 * {@code parkservices/ThresholdedMultiDimensionalExample}.
 *
 * With baseDimensions == 2 the two attributes are cosines of the same period
 * differing only in phase and amplitude, so the (x,y) phase portrait is a
 * closed ellipse and the observation travels around it once per period. Noise
 * thickens the ellipse into an annulus and an injected anomaly is a radial
 * excursion off it, which is far easier to see here than in two stacked time
 * series.
 *
 * Note that the phase gap is drawn independently per dimension, as a uniform
 * multiple of 360/period. It is 0 or 180 degrees for roughly 8% of seeds, and
 * the portrait then collapses to a straight line rather than an ellipse. Pick a
 * different seed if that happens; a gap of exactly 90 degrees with matched
 * amplitudes gives a circle.
 *
 * Markers are drawn at the timestamp the detector BLAMED, not the timestamp it
 * fired at. When relativeIndex is negative those differ, and by the time the
 * alarm arrives the series has usually returned to the ring -- plotting at the
 * firing timestamp would put the marker on a perfectly ordinary point. The lag
 * is shown instead as a dashed leader from the firing point back to the blamed
 * one.
 *
 * A detection can legitimately arrive without an expected value, when the
 * forest cannot produce a confident imputation. Those are drawn as hollow
 * rings, so "located and explained" reads differently from "flagged but
 * unexplained".
 */
public class ThresholdedRCFMovie implements Example {

    /** Mirrors the private margin field in Plot2D; used to keep pixels square. */
    private static final int PLOT_MARGIN = 38;

    /**
     * CorrectionMode tokens are coloured by ordinal, shifted clear of the two
     * palette slots already spoken for: 0 (red) is a detection and 2 (green) is an
     * expected value. Without the shift MULTI_MODE collides with green.
     */
    private static final int TOKEN_COLOR_OFFSET = 2;

    private static final Color DETECTION = Layers.color(0);
    private static final Color EXPECTED = Layers.color(2);
    private static final Color GROUND_TRUTH = new Color(30, 30, 30);
    private static final Color LAG = new Color(140, 140, 140);

    public static void main(String[] args) throws Exception {
        new ThresholdedRCFMovie().run();
    }

    @Override
    public String command() {
        return "Thresholded_RCF_movie";
    }

    @Override
    public String description() {
        return "Thresholded Multi Dimensional Example, rendered as a rotating phase-portrait GIF";
    }

    @Override
    public void run() throws Exception {

        int shingleSize = 8;
        int numberOfTrees = 50;
        int sampleSize = 256;
        int dataSize = 4 * sampleSize;
        int period = 24;

        // the phase portrait only makes sense for exactly 2 base dimensions
        int baseDimensions = 2;
        int dimensions = baseDimensions * shingleSize;

        double amplitude = 100.0;
        double noise = 5.0;
        double anomalyFactor = 5;
        boolean useSlope = false;
        boolean autoAdjust = true;

        long seed = 2225213261242380303L;

        // ---- rendering knobs -------------------------------------------------
        // the full run is 0 .. dataSize + shingleSize - 1; a window keeps the GIF
        // small. Pass 1 prints every firing, so pick the window from that.
        int fromTime = 740;
        int toTime = 1000;
        int heightPx = 760;
        int legendPx = 210; // gutter reserved for the legend, in pixels
        int gifDelayMs = 60;
        int holdFrames = 14; // extra frames parked on a detection
        int tailLength = period; // one full revolution of comet tail
        int markerWindow = 4 * period; // events older than this stop being drawn
        int arrowLinger = 12; // frames the explanation arrow stays up
        boolean livePlot = false;
        String outputFile = "thresholded_multi_dim.gif";
        // ----------------------------------------------------------------------

        ThresholdedRandomCutForest forest = ThresholdedRandomCutForest.builder().dimensions(dimensions)
                .shingleSize(shingleSize).randomSeed(0).numberOfTrees(numberOfTrees).sampleSize(sampleSize)
                .transformMethod(TransformMethod.NORMALIZE).autoAdjust(autoAdjust).build();

        System.out.println("seed = " + seed);
        MultiDimData dataWithKeys = multiDimData(dataSize + shingleSize - 1, period, amplitude, noise, seed,
                baseDimensions, anomalyFactor, useSlope);

        int n = dataWithKeys.data.length;
        float[][] xy = dataWithKeys.data;

        // ---- pass 1: run the detector, record everything ----------------------
        double[] grade = new double[n];
        double[] score = new double[n];
        CorrectionMode[] mode = new CorrectionMode[n];
        double[][] blamed = new double[n][]; // value held responsible
        double[][] expected = new double[n][]; // value expected instead, may be absent
        int[] relative = new int[n];

        for (int t = 0; t < n; t++) {
            AnomalyDescriptor result = forest.process(toDoubleArray(xy[t]), 0L);
            grade[t] = result.getAnomalyGrade();
            score[t] = result.getRCFScore();
            mode[t] = result.getCorrectionMode();
            relative[t] = result.getRelativeIndex();

            // blame is recorded whether or not an expected value came with it
            if (grade[t] > 0) {
                blamed[t] = result.getPastValues();
                if (result.isExpectedValuesPresent()) {
                    expected[t] = result.getExpectedValuesList()[0];
                }
                System.out.printf("t %4d  score %.4f  grade %.2f  relative %d  expected %s%n", t, score[t], grade[t],
                        relative[t], result.isExpectedValuesPresent() ? "yes" : "NO");
            }
        }

        // ---- viewport: square data area plus an empty strip for the legend ----
        double range = 0;
        for (float[] p : xy) {
            range = Math.max(range, Math.max(Math.abs(p[0]), Math.abs(p[1])));
        }
        range *= 1.12;

        double unitsPerPixel = 2.0 * range / (heightPx - 2.0 * PLOT_MARGIN);
        double gutter = legendPx * unitsPerPixel;
        double xmin = -range, xmax = range + gutter, ymin = -range, ymax = range;
        int widthPx = heightPx + legendPx; // keeps pixels square, so circles stay circular

        Plot2D plot = livePlot
                ? Plot2D.openRect("Thresholded Multi Dimensional", xmin, xmax, ymin, ymax, widthPx, heightPx)
                : Plot2D.offscreenRect(xmin, xmax, ymin, ymax);
        GifWriter gif = new GifWriter(new File(outputFile), gifDelayMs, true);

        CorrectionMode[] tokens = CorrectionMode.values();
        Layer legend = legendFor(tokens);

        // ---- pass 2: render ---------------------------------------------------
        for (int t = fromTime; t < Math.min(toTime, n); t++) {

            int since = t - markerWindow;
            List<Layer> scene = new ArrayList<>();

            // everything observed so far -- the annulus fills in as it runs
            scene.add(Layers.dots(head(xy, t + 1), new Color(212, 212, 212), 1.8));

            // one revolution of history, so the travel around the ring is visible
            scene.add(Layers.polyline(trail(xy, t - tailLength + 1, t), new Color(130, 130, 130), false, 0, 1.3f));

            // suppressed timestamps, coloured by token
            for (CorrectionMode token : tokens) {
                if (token == CorrectionMode.NONE) {
                    continue;
                }
                List<float[]> marks = new ArrayList<>();
                for (int i = Math.max(0, since); i <= t; i++) {
                    if (grade[i] == 0 && mode[i] == token) {
                        marks.add(xy[i]);
                    }
                }
                if (!marks.isEmpty()) {
                    scene.add(Layers.dots(marks.toArray(new float[0][]), tokenColor(token), 3.6));
                }
            }

            // detection lag: firing point back to the point it blamed
            for (int i = Math.max(0, since); i <= t; i++) {
                if (grade[i] > 0 && relative[i] != 0 && blamed[i] != null) {
                    scene.add(Layers.dashedPolyline(
                            new double[][] { { xy[i][0], xy[i][1] }, { blamed[i][0], blamed[i][1] } }, LAG, 1.1f,
                            new float[] { 3f, 4f }));
                }
            }

            // ground truth: where noise was actually replaced by an anomaly
            List<float[]> injected = new ArrayList<>();
            for (int index : dataWithKeys.changeIndices) {
                if (index >= since && index <= t) {
                    injected.add(xy[index]);
                }
            }
            if (!injected.isEmpty()) {
                scene.add(rings(injected.toArray(new float[0][]), GROUND_TRUTH, 7.5, 1.6f));
            }

            // detections, drawn at the blamed timestamp; hollow when unexplained
            List<float[]> explained = new ArrayList<>();
            List<float[]> unexplained = new ArrayList<>();
            for (int i = Math.max(0, since); i <= t; i++) {
                if (grade[i] > 0) {
                    (expected[i] != null ? explained : unexplained).add(blameSite(xy, blamed, i));
                }
            }
            if (!explained.isEmpty()) {
                scene.add(Layers.dots(explained.toArray(new float[0][]), DETECTION, 5.2));
            }
            if (!unexplained.isEmpty()) {
                scene.add(rings(unexplained.toArray(new float[0][]), DETECTION, 5.2, 2.0f));
            }

            // the most recent explanation, held for a few frames
            int fired = lastFiring(grade, t, arrowLinger);
            if (fired >= 0 && expected[fired] != null && blamed[fired] != null) {
                double[] from = blamed[fired];
                double[] to = expected[fired];
                scene.add(Layers.arrows(new double[][] { { from[0], from[1] } },
                        new double[][] { { to[0] - from[0], to[1] - from[1] } }, DETECTION, 2.2f));
                scene.add(Layers.dots(new float[][] { { (float) to[0], (float) to[1] } }, EXPECTED, 4.6));
            }

            // current observation
            scene.add(Layers.dots(new float[][] { xy[t] }, Color.BLACK, 5.0));

            scene.add(Layers.label(xmin + 0.02 * (xmax - xmin), ymax - 0.04 * (ymax - ymin),
                    status(t, score, grade, mode, relative, expected), new Color(40, 40, 40)));
            scene.add(legend);

            if (livePlot) {
                plot.render(scene);
            }
            gif.writeFrame(plot.renderImage(widthPx, heightPx, scene));

            if (grade[t] > 0) {
                for (int k = 0; k < holdFrames; k++) {
                    gif.writeFrame(plot.renderImage(widthPx, heightPx, scene));
                }
            }
        }

        gif.close();
        System.out.println("wrote " + outputFile + " (" + widthPx + "x" + heightPx + ")");
    }

    // ------------------------------------------------------------------ helpers

    private static Color tokenColor(CorrectionMode token) {
        return Layers.color(token.ordinal() + TOKEN_COLOR_OFFSET);
    }

    /** Where the detector placed blame, falling back to the firing point. */
    private static float[] blameSite(float[][] xy, double[][] blamed, int i) {
        return blamed[i] == null ? xy[i] : new float[] { (float) blamed[i][0], (float) blamed[i][1] };
    }

    /** Open circles; the library has filled dots but no rings. */
    private static Layer rings(float[][] xy, Color color, double radius, float stroke) {
        return (g, vp) -> {
            g.setColor(color);
            g.setStroke(new BasicStroke(stroke));
            for (float[] p : xy) {
                g.draw(new Ellipse2D.Double(vp.px(p[0]) - radius, vp.py(p[1]) - radius, 2 * radius, 2 * radius));
            }
        };
    }

    private static Layer legendFor(CorrectionMode[] tokens) {
        List<String> labels = new ArrayList<>();
        List<Color> colors = new ArrayList<>();
        List<Layers.Swatch> swatches = new ArrayList<>();

        add(labels, colors, swatches, "observation", Color.BLACK, Layers.Swatch.DOTS);
        add(labels, colors, swatches, "injected change", GROUND_TRUTH, Layers.Swatch.LINE);
        add(labels, colors, swatches, "anomaly grade > 0", DETECTION, Layers.Swatch.BOX);
        add(labels, colors, swatches, "expected value", EXPECTED, Layers.Swatch.BOX);
        add(labels, colors, swatches, "detection lag", LAG, Layers.Swatch.DASHED);

        for (CorrectionMode token : tokens) {
            if (token != CorrectionMode.NONE) {
                add(labels, colors, swatches, token.name(), tokenColor(token), Layers.Swatch.BOX);
            }
        }
        return Layers.legend(labels.toArray(new String[0]), colors.toArray(new Color[0]),
                swatches.toArray(new Layers.Swatch[0]));
    }

    private static void add(List<String> labels, List<Color> colors, List<Layers.Swatch> swatches, String label,
            Color color, Layers.Swatch swatch) {
        labels.add(label);
        colors.add(color);
        swatches.add(swatch);
    }

    private static String status(int t, double[] score, double[] grade, CorrectionMode[] mode, int[] relative,
            double[][] expected) {
        StringBuilder sb = new StringBuilder();
        sb.append(String.format("t = %4d    score %.3f", t, score[t]));
        if (grade[t] > 0) {
            sb.append(String.format("    ANOMALY grade %.2f", grade[t]));
            if (relative[t] != 0) {
                sb.append(String.format(" (%d steps ago)", -relative[t]));
            }
            if (expected[t] == null) {
                sb.append("  no expected value");
            }
        } else if (mode[t] != CorrectionMode.NONE) {
            sb.append("    suppressed: ").append(mode[t].name());
        }
        return sb.toString();
    }

    private static float[][] head(float[][] xy, int count) {
        float[][] out = new float[count][];
        System.arraycopy(xy, 0, out, 0, count);
        return out;
    }

    private static double[][] trail(float[][] xy, int from, int to) {
        int lo = Math.max(0, from);
        double[][] out = new double[to - lo + 1][2];
        for (int i = lo; i <= to; i++) {
            out[i - lo][0] = xy[i][0];
            out[i - lo][1] = xy[i][1];
        }
        return out;
    }

    private static int lastFiring(double[] grade, int upto, int linger) {
        for (int i = upto; i >= Math.max(0, upto - linger); i--) {
            if (grade[i] > 0) {
                return i;
            }
        }
        return -1;
    }
}
