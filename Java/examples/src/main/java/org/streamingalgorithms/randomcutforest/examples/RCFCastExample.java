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

import java.awt.*;
import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.streamingalgorithms.randomcutforest.config.Precision;
import org.streamingalgorithms.randomcutforest.config.TransformMethod;
import org.streamingalgorithms.randomcutforest.examples.datasets.MultiDimData;
import org.streamingalgorithms.randomcutforest.examples.plot.GifWriter;
import org.streamingalgorithms.randomcutforest.examples.plot.Layer;
import org.streamingalgorithms.randomcutforest.examples.plot.Layers;
import org.streamingalgorithms.randomcutforest.examples.plot.Plot2D;
import org.streamingalgorithms.randomcutforest.parkservices.ForecastDescriptor;
import org.streamingalgorithms.randomcutforest.parkservices.RCFCaster;
import org.streamingalgorithms.randomcutforest.parkservices.config.Calibration;
import org.streamingalgorithms.randomcutforest.returntypes.DiVector;
import org.streamingalgorithms.randomcutforest.returntypes.RangeVector;

public class RCFCastExample implements Example {

    public static void main(String[] args) throws Exception {
        new RCFCastExample().run();
    }

    @Override
    public String command() {
        return "rcf_cast";
    }

    @Override
    public String description() {
        return "Calibrated RCFCast Example";
    }

    @Override
    public void run() throws Exception {
        int numberOfTrees = 50;
        int sampleSize = 256;
        Precision precision = Precision.FLOAT_32;
        int dataSize = 2 * sampleSize;

        int baseDimensions = 2;
        // note that the stream is 2-dimensional and only dimension 0 is plotted
        int vizDim = 0; // the shifted, plotted dimension
        int forecastHorizon = 15;
        int shingleSize = 20;
        int outputAfter = 64;
        long seed = 2023L;

        float[][] fulldata = new float[2 * dataSize][];
        double shiftForViz = 200;
        System.out.println("seed = " + seed);

        MultiDimData dataWithKeys = multiDimData(dataSize, 50, 50, 5, seed, baseDimensions, 0.01, true);
        for (int i = 0; i < dataSize; i++) {
            fulldata[i] = Arrays.copyOf(dataWithKeys.data[i], baseDimensions);
            fulldata[i][0] += shiftForViz;
        }
        MultiDimData second = multiDimData(dataSize, 70, 30, 5, seed + 1, baseDimensions, 0.01, true);
        for (int i = 0; i < dataSize; i++) {
            fulldata[dataSize + i] = Arrays.copyOf(second.data[i], baseDimensions);
            fulldata[dataSize + i][0] += shiftForViz;
        }

        int dimensions = baseDimensions * shingleSize;
        TransformMethod transformMethod = TransformMethod.NORMALIZE;
        RCFCaster caster = RCFCaster.builder().dimensions(dimensions).randomSeed(seed + 1).numberOfTrees(numberOfTrees)
                .shingleSize(shingleSize).sampleSize(sampleSize).internalShinglingEnabled(true).precision(precision)
                .anomalyRate(0.01).outputAfter(outputAfter).calibration(Calibration.SIMPLE).transformDecay(0.02)
                .forecastHorizon(forecastHorizon).initialAcceptFraction(0.125).useRCFCallibration(true).build();

        // ---- viz config ----
        boolean printFile = false;
        boolean livePlot = true;
        boolean saveGif = true;
        int frameEvery = 3;
        int frameDelayMs = 8;
        int gifDelayMs = 40;
        int N = fulldata.length;
        double ymin = -100, ymax = 650;

        java.io.BufferedWriter file = printFile ? new java.io.BufferedWriter(new java.io.FileWriter("example")) : null;

        Plot2D plot = livePlot ? Plot2D.openRect("RCFCaster — calibrated forecast", 0, N + forecastHorizon, ymin, ymax, 1100, 620)
                                                   : Plot2D.offscreenRect(0, N + forecastHorizon, ymin, ymax);

        GifWriter gif = saveGif ? new GifWriter(new File("rcf_cast.gif"), gifDelayMs, true) : null;

        // the full data series (dim vizDim) — static background, computed once
        double[][] dataLine = new double[N][2];
        for (int j = 0; j < N; j++) {
            dataLine[j][0] = j;
            dataLine[j][1] = fulldata[j][vizDim];
        }

        Color cData = Color.BLACK;
        Color cForecast = new Color(31, 119, 180);
        Color cPast = new Color(140, 86, 75);
        Color cAcc = new Color(200, 0, 160);
        Color cNow = new Color(110, 110, 110);
        Color cGuide = new Color(150, 150, 150);

        Instant start = Instant.now();
        long processNanos = 0;

        for (int current = 0; current < N; current++) {
            Instant p0 = Instant.now();
            ForecastDescriptor result = caster.process(toDoubleArray(fulldata[current]), current);
            processNanos += Duration.between(p0, Instant.now()).toNanos();

            if (printFile) {
                printResult(file, result, current, baseDimensions);
            }

            boolean ready = caster.getForest().isOutputReady()
                    && result.getTimedForecast().rangeVector.values.length >= baseDimensions;
            if (!(livePlot || saveGif) || !ready || current % frameEvery != 0) {
                continue;
            }

            RangeVector forecast = result.getTimedForecast().rangeVector;
            int horizon = forecast.values.length / baseDimensions;

            // future: forecast + calibrated interval
            double[] fx = new double[horizon], fv = new double[horizon], fl = new double[horizon],
                    fu = new double[horizon];
            for (int i = 0; i < horizon; i++) {
                int k = i * baseDimensions + vizDim;
                fx[i] = current + i;
                fv[i] = forecast.values[k];
                fu[i] = forecast.upper[k];
                fl[i] = forecast.lower[k];
            }

            List<Layer> scene = new ArrayList<>();
            //scene.add(Layers.hline(0, cGuide));
            //scene.add(Layers.hline(80, cGuide));
            //scene.add(Layers.hline(100, cGuide));
            scene.add(Layers.polyline(dataLine, cData, false, 0, 1.0f));

            // past: observed error distribution + interval accuracy %
            float[] eUp = result.getObservedErrorDistribution().upper;
            float[] eLo = result.getObservedErrorDistribution().lower;
            float[] iPrec = result.getIntervalPrecision();
            if (eUp.length >= horizon * baseDimensions && iPrec.length >= horizon * baseDimensions) {
                double[] px = new double[horizon], pu = new double[horizon], pl = new double[horizon],
                        acc = new double[horizon];
                for (int idx = 0; idx < horizon; idx++) {
                    int i = horizon - 1 - idx;
                    int k = i * baseDimensions + vizDim;
                    px[idx] = current - i;
                    pu[idx] = eUp[k];
                    pl[idx] = eLo[k];
                    acc[idx] = 100.0 * iPrec[k];
                }
                scene.add(Layers.band(px, pl, pu, cPast, 130));
                double[][] accLine = new double[horizon][2];
                for (int idx = 0; idx < horizon; idx++) {
                    accLine[idx][0] = px[idx];
                    accLine[idx][1] = acc[idx];
                }
                scene.add(Layers.polyline(accLine, cAcc, false, 0, 2.0f));
            }

            scene.add(Layers.band(fx, fl, fu, cForecast, 70));
            double[][] fLine = new double[horizon][2];
            for (int i = 0; i < horizon; i++) {
                fLine[i][0] = fx[i];
                fLine[i][1] = fv[i];
            }
            double labelX = 8;
            scene.add(Layers.hline(0, cGuide));
            scene.add(Layers.label(labelX, 0, "Accuracy 0.0", cAcc));
            scene.add(Layers.hline(80, cGuide));
            scene.add(Layers.label(labelX, 80, "Accuracy 0.8", cAcc));
            scene.add(Layers.hline(100, cGuide));
            scene.add(Layers.label(labelX, 100, "Accuracy 1.0", cAcc));
            scene.add(Layers.polyline(fLine, cForecast, false, 0, 2.0f));
            scene.add(Layers.vline(current, cNow, 1.5f));

            scene.add(
                    Layers.legend(
                            new String[] { "Data", "Forecast", "Uncertainty (future)", "Error dist (past)",
                                                               "Interval acc (fraction)" },
                                                new Color[] { cData, cForecast, cForecast.brighter(), cPast, cAcc },
                                      new Layers.Swatch[] { Layers.Swatch.LINE, Layers.Swatch.LINE, Layers.Swatch.BOX,
                                                        Layers.Swatch.BOX, Layers.Swatch.LINE }));


            if (livePlot) {
                javax.swing.SwingUtilities.invokeLater(() -> {
                    plot.render(scene);
                    plot.paintImmediately(0, 0, plot.getWidth(), plot.getHeight());
                });
                if (frameDelayMs > 0) {
                    Thread.sleep(frameDelayMs);
                }
            }
            /*
             * if (livePlot) { plot.render(scene); if (frameDelayMs > 0) {
             * Thread.sleep(frameDelayMs); } }
             *
             */
            if (saveGif) {
                gif.writeFrame(plot.renderImage(1000, 560, scene));
            }

            if (current % 128 == 0) {
                System.out.printf("[t=%4d] total %d ms | process %.3f ms/step%n", current,
                        Duration.between(start, Instant.now()).toMillis(), processNanos / 1e6 / (current + 1));
            }
        }

        if (gif != null) {
            gif.close();
            System.out.println("wrote rcf_cast.gif");
        }
        if (file != null) {
            file.close();
        }
    }

    void printResult(BufferedWriter file, ForecastDescriptor result, int current, int inputLength) throws IOException {
        RangeVector forecast = result.getTimedForecast().rangeVector;
        float[] errorP50 = result.getObservedErrorDistribution().values;
        float[] upperError = result.getObservedErrorDistribution().upper;
        float[] lowerError = result.getObservedErrorDistribution().lower;
        DiVector rmse = result.getErrorRMSE();
        float[] mean = result.getErrorMean();
        float[] intervalPrecision = result.getIntervalPrecision();
        file.append(current + " " + 1000 + "\n");
        file.append("\n");
        file.append("\n");

        // block corresponding to the past; print the errors
        for (int i = forecast.values.length / inputLength - 1; i >= 0; i--) {
            file.append((current - i) + " ");
            for (int j = 0; j < inputLength; j++) {
                int k = i * inputLength + j;
                file.append(mean[k] + " " + rmse.high[k] + " " + rmse.low[k] + " " + errorP50[k] + " " + upperError[k]
                        + " " + lowerError[k] + " " + intervalPrecision[k] + " ");
            }
            file.append("\n");
        }
        file.append("\n");
        file.append("\n");

        // block corresponding to the future; the projections and the projected errors
        for (int i = 0; i < forecast.values.length / inputLength; i++) {
            file.append((current + i) + " ");
            for (int j = 0; j < inputLength; j++) {
                int k = i * inputLength + j;
                file.append(forecast.values[k] + " " + forecast.upper[k] + " " + forecast.lower[k] + " ");
            }
            file.append("\n");
        }
        file.append("\n");
        file.append("\n");
    }

}
