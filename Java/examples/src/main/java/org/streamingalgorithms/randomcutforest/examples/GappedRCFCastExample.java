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
import java.util.Random;

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

public class GappedRCFCastExample implements Example {

    public static void main(String[] args) throws Exception {
        new GappedRCFCastExample().run();
    }

    @Override
    public String command() {
        return "gapped_rcf_cast";
    }

    @Override
    public String description() {
        return "Gapped RCFCast Example";
    }

    @Override
    public void run() throws Exception {
        int numberOfTrees = 50;
        int sampleSize = 256;
        int dataSize = 2 * sampleSize;

        int baseDimensions = 1; // single-variate forecasting
        int vizDim = 0;
        int forecastHorizon = 20;
        int shingleSize = 24;
        int outputAfter = 256;
        long seed = 2023L;

        float[][] fulldata = new float[2 * dataSize][];
        int N = fulldata.length;
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
        RCFCaster caster = RCFCaster.builder().dimensions(dimensions).randomSeed(seed + 1).numberOfTrees(numberOfTrees)
                .shingleSize(shingleSize).sampleSize(sampleSize).internalShinglingEnabled(true).anomalyRate(0.01)
                .outputAfter(outputAfter).calibration(Calibration.SIMPLE).transformDecay(0.02)
                .forecastHorizon(forecastHorizon).initialAcceptFraction(0.125).useRCFCallibration(true).build();

        // lowerLimit(new float[baseDimensions])
        // will force the forecast to be nonnegative in every dimension

        // ---- gaps: coin-toss contiguous runs. Keep maxGap <= forecastHorizon so the
        // frozen forecast can cover the whole gap; longer gaps get a flat hold (see
        // below).
        boolean[] missing = new boolean[N];
        Random gapRng = new Random(2030L);
        double gapProb = 0.02;
        int minGap = 3, maxGap = 3 * forecastHorizon;
        for (int j = 0; j < N;) {
            if (gapRng.nextDouble() < gapProb) {
                int len = minGap + gapRng.nextInt(maxGap - minGap + 1);
                for (int g = 0; g < len && j < N; g++, j++) {
                    missing[j] = true;
                }
            } else {
                j++;
            }
        }
        List<double[]> gapSpans = new ArrayList<>();
        for (int j = 0; j < N; j++) {
            if (missing[j]) {
                int s = j;
                while (j < N && missing[j])
                    j++;
                gapSpans.add(new double[] { s, j - 1 });
            }
        }

        // ---- viz config ----
        boolean printFile = false;
        boolean livePlot = true;
        boolean saveGif = true;
        int frameEvery = 3;
        int frameDelayMs = 8;
        int gifDelayMs = 40;
        double ymin = -100, ymax = 500;

        java.io.BufferedWriter file = printFile ? new java.io.BufferedWriter(new java.io.FileWriter("example")) : null;

        Plot2D plot = livePlot ? Plot2D.openRect("Gapped RCFCaster — freeze + impute", 0, N, ymin, ymax, 1100, 620)
                : Plot2D.offscreenRect(0, N, ymin, ymax);
        GifWriter gif = saveGif ? new GifWriter(new File("gapped_rcf_cast.gif"), gifDelayMs, true) : null;

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

        List<float[]> imputedPts = new ArrayList<>();
        double gapSe = 0;
        int gapN = 0;

        ForecastDescriptor lastResult = null; // forecast from the most recent OBSERVED step
        int lastObserved = -1;

        Instant start = Instant.now();
        long processNanos = 0;

        for (int current = 0; current < N; current++) {
            if (!missing[current]) {
                // observed: process normally (updates forest, calibrates errors, forecasts)
                Instant p0 = Instant.now();
                lastResult = caster.process(toDoubleArray(fulldata[current]), current); // timestamp = current
                processNanos += Duration.between(p0, Instant.now()).toNanos();
                lastObserved = current;
                if (printFile) {
                    printResult(file, lastResult, current, baseDimensions);
                }
            } else if (lastResult != null) {
                RangeVector f = lastResult.getTimedForecast().rangeVector;
                int h = f.values.length / baseDimensions;
                int offset = current - lastObserved;
                if (h > 0 && offset < h) { // only within the forecast horizon
                    float imputedVal = f.values[offset * baseDimensions + vizDim];
                    imputedPts.add(new float[] { current, imputedVal });
                    double e = imputedVal - fulldata[current][vizDim];
                    gapSe += e * e;
                    gapN++;
                }
                // beyond horizon: no dot, no score — the model has nothing to say here
            }

            boolean ready = lastResult != null && lastObserved >= outputAfter
                    && lastResult.getTimedForecast().rangeVector.values.length >= baseDimensions;
            if (!(livePlot || saveGif) || !ready || current % frameEvery != 0) {
                continue;
            }

            // forecast is anchored at lastObserved; during a gap it stays put and the
            // now-line sweeps into it
            RangeVector forecast = lastResult.getTimedForecast().rangeVector;
            int horizon = forecast.values.length / baseDimensions;
            double[] fx = new double[horizon], fv = new double[horizon], fl = new double[horizon],
                    fu = new double[horizon];
            for (int i = 0; i < horizon; i++) {
                int k = i * baseDimensions + vizDim;
                fx[i] = lastObserved + i;
                fv[i] = forecast.values[k];
                fu[i] = forecast.upper[k];
                fl[i] = forecast.lower[k];
            }

            List<Layer> scene = new ArrayList<>();
            scene.add(Layers.xBands(gapSpans.toArray(new double[0][]), new Color(255, 180, 0), 40));

            double labelX = 8;
            scene.add(Layers.hline(0, cGuide));
            scene.add(Layers.label(labelX, 0, "Accuracy 0.0", cAcc));
            scene.add(Layers.hline(80, cGuide));
            scene.add(Layers.label(labelX, 80, "Accuracy 0.8", cAcc));
            scene.add(Layers.hline(100, cGuide));
            scene.add(Layers.label(labelX, 100, "Accuracy 1.0", cAcc));

            scene.add(Layers.polyline(dataLine, cData, false, 0, 1.0f));
            scene.add(Layers.dots(imputedPts.toArray(new float[0][]), new Color(230, 120, 0), 3.5));

            // past: observed error distribution + interval accuracy %, anchored at
            // lastObserved
            float[] eUp = lastResult.getObservedErrorDistribution().upper;
            float[] eLo = lastResult.getObservedErrorDistribution().lower;
            float[] iPrec = lastResult.getIntervalPrecision();
            if (eUp.length >= horizon * baseDimensions && iPrec.length >= horizon * baseDimensions) {
                double[] px = new double[horizon], pu = new double[horizon], pl = new double[horizon],
                        acc = new double[horizon];
                for (int idx = 0; idx < horizon; idx++) {
                    int i = horizon - 1 - idx;
                    int k = i * baseDimensions + vizDim;
                    px[idx] = lastObserved - i;
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
            scene.add(Layers.polyline(fLine, cForecast, false, 0, 2.0f));
            scene.add(Layers.vline(current, cNow, 1.5f)); // now-line at current, may be inside a gap
            scene.add(Layers.legend(
                    new String[] { "Data", "Forecast (frozen in gap)", "Uncertainty", "Error dist (past)",
                            "Imputed / acc %" },
                    new Color[] { cData, cForecast, cForecast, cPast, new Color(230, 120, 0) }));

            if (livePlot) {
                javax.swing.SwingUtilities.invokeLater(() -> {
                    plot.render(scene);
                    plot.paintImmediately(0, 0, plot.getWidth(), plot.getHeight());
                });
                if (frameDelayMs > 0) {
                    Thread.sleep(frameDelayMs);
                }
            }
            if (saveGif) {
                gif.writeFrame(plot.renderImage(1000, 560, scene));
            }

            if (current % 128 == 0 && lastObserved > 0) {
                System.out.printf("[t=%4d] total %d ms | process %.3f ms/observed-step%n", current,
                        Duration.between(start, Instant.now()).toMillis(), processNanos / 1e6 / (lastObserved + 1));
            }
        }

        if (gapN > 0) {
            System.out.printf("imputation RMSE over %d masked points: %.3f%n", gapN, Math.sqrt(gapSe / gapN));
        }
        if (gif != null) {
            gif.close();
            System.out.println("wrote gapped_rcf_cast.gif");
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
