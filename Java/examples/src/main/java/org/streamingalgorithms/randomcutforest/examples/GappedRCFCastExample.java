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

/**
 * Forecasting a stream with occlusion: contiguous spans of the input are never
 * shown to the model. During a span the forecast issued at the last observed
 * point stays frozen and the now-line sweeps into it; the frozen forecast is
 * read off as an imputation for each masked step within the horizon.
 *
 * <p>
 * Two things this example is careful about, because both are easy to get wrong:
 *
 * <ol>
 * <li><b>Readiness is a property of the forest, not of the clock.</b>
 * {@code outputAfter} counts <i>updates delivered to the forest</i>. Roughly
 * 39% of this stream is withheld, so the forest reaches 256 updates at t=412,
 * not at t=256. Gating on a timestamp would record ~30% of the imputations
 * while the forecast is still identically zero -- they plot at y=0 and poison
 * the RMSE. Ask the forest:
 * {@link org.streamingalgorithms.randomcutforest.RandomCutForest#isOutputReady()}.</li>
 * <li><b>The plot must distinguish what the model saw from what it did not.</b>
 * The withheld series is drawn dotted. It is ground truth for scoring only; the
 * model never receives it.</li>
 * </ol>
 */
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
        return "Forecasting and imputation over a stream with occluded segments";
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

        // two regimes back to back: the generating process changes at t = dataSize
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

        // ---- occlusion: coin-toss contiguous runs -------------------------------
        // maxGap deliberately EXCEEDS forecastHorizon. A frozen forecast reaches
        // exactly forecastHorizon-1 steps past the last observation, so long spans
        // run out of forecast partway through. That is not a defect to be hidden:
        // the model declining to guess past its horizon is worth showing, so the
        // unreachable tail of each span is drawn in a different shade and gets no
        // imputed point.
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

        // occluded spans, split into the part the frozen forecast can reach and the
        // part it cannot
        List<double[]> reachSpans = new ArrayList<>();
        List<double[]> beyondSpans = new ArrayList<>();
        for (int j = 0; j < N; j++) {
            if (missing[j]) {
                int s = j;
                while (j < N && missing[j]) {
                    j++;
                }
                int e = j - 1;
                // offset = current - lastObserved, imputed while offset < forecastHorizon
                int reachEnd = Math.min(e, s + forecastHorizon - 2);
                reachSpans.add(new double[] { s, reachEnd });
                if (e > reachEnd) {
                    beyondSpans.add(new double[] { reachEnd + 1, e });
                }
            }
        }

        // ---- the series, split into runs the model saw and runs it did not ------
        List<double[][]> seenRuns = new ArrayList<>();
        List<double[][]> withheldRuns = new ArrayList<>();
        for (int j = 0; j < N;) {
            boolean m = missing[j];
            int s = j;
            while (j < N && missing[j] == m) {
                j++;
            }
            // overlap each run by one index so consecutive runs meet on screen
            int lo = Math.max(0, s - 1), hi = Math.min(N - 1, j);
            double[][] seg = new double[hi - lo + 1][2];
            for (int k = lo; k <= hi; k++) {
                seg[k - lo][0] = k;
                seg[k - lo][1] = fulldata[k][vizDim];
            }
            (m ? withheldRuns : seenRuns).add(seg);
        }

        // ---- viz config ----
        boolean printFile = false;
        boolean livePlot = true;
        boolean saveGif = true;
        int frameEvery = 3;
        int frameDelayMs = 8;
        int gifDelayMs = 40;
        // ymax lifted 500 -> 650: the 9-row legend is 152px tall and at ymax=500 its
        // bottom edge lands at data y=299, straight through a series that peaks at
        // 355.6. At 650 it clears by 43. RCFCastExample uses the same box so the two
        // figures stay comparable.
        double ymin = -100, ymax = 650;

        BufferedWriter file = printFile ? new BufferedWriter(new java.io.FileWriter("example")) : null;

        Plot2D plot = livePlot
                ? Plot2D.openRect("Gapped RCFCaster — freeze + impute", 0, N + forecastHorizon, ymin, ymax, 1100, 620)
                : Plot2D.offscreenRect(0, N + forecastHorizon, ymin, ymax);
        GifWriter gif = saveGif ? new GifWriter(new File("gapped_rcf_cast.gif"), gifDelayMs, true) : null;

        Color cData = Color.BLACK;
        // bright red: the withheld series is the one thing here the model never
        // receives, and grey put it 3.8 dE from the occlusion band. Matches
        // Summarization.PALETTE[0]; dE 40.3 from its nearest neighbour (cImputed).
        Color cWithheld = new Color(214, 39, 40);
        Color cReach = new Color(255, 180, 0);
        Color cBeyond = new Color(150, 150, 150);
        Color cForecast = new Color(31, 119, 180);
        Color cPast = new Color(140, 86, 75);
        Color cImputed = new Color(55, 200, 200);
        Color cAcc = new Color(200, 0, 160);
        Color cNow = new Color(110, 110, 110);
        Color cGuide = new Color(150, 150, 150);
        float[] dash = new float[] { 2f, 3f };

        List<float[]> imputedPts = new ArrayList<>();
        double gapSe = 0;
        int gapN = 0;
        int skippedNotReady = 0, skippedBeyondHorizon = 0;

        ForecastDescriptor lastResult = null; // forecast from the most recent OBSERVED step
        int lastObserved = -1;
        boolean lastResultValid = false; // was the forest output-ready when it was produced?

        Instant start = Instant.now();
        long processNanos = 0;

        for (int current = 0; current < N; current++) {
            if (!missing[current]) {
                Instant p0 = Instant.now();
                lastResult = caster.process(toDoubleArray(fulldata[current]), current);
                processNanos += Duration.between(p0, Instant.now()).toNanos();
                lastObserved = current;
                // readiness is a property of the forest (updates delivered), never of
                // `current` -- 39% of timestamps here never reach the forest at all
                lastResultValid = caster.getForest().isOutputReady();
                if (printFile) {
                    printResult(file, lastResult, current, baseDimensions);
                }
            } else if (lastResult != null) {
                if (!lastResultValid) {
                    skippedNotReady++; // forecast is still identically zero; not an imputation
                } else {
                    RangeVector f = lastResult.getTimedForecast().rangeVector;
                    int h = f.values.length / baseDimensions;
                    int offset = current - lastObserved;
                    if (h > 0 && offset < h) {
                        float imputedVal = f.values[offset * baseDimensions + vizDim];
                        imputedPts.add(new float[] { current, imputedVal });
                        double e = imputedVal - fulldata[current][vizDim];
                        gapSe += e * e;
                        gapN++;
                    } else {
                        skippedBeyondHorizon++; // the model has nothing to say this far out
                    }
                }
            }

            boolean ready = lastResultValid
                    && lastResult.getTimedForecast().rangeVector.values.length >= baseDimensions;
            if (!(livePlot || saveGif) || !ready || current % frameEvery != 0) {
                continue;
            }

            // the forecast is anchored at lastObserved; inside a span it stays put and
            // the now-line sweeps into it
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
            scene.add(Layers.xBands(reachSpans.toArray(new double[0][]), cReach, 40));
            scene.add(Layers.xBands(beyondSpans.toArray(new double[0][]), cBeyond, 45));

            double labelX = 8;
            scene.add(Layers.hline(0, cGuide));
            scene.add(Layers.label(labelX, 0, "Accuracy 0.0", cAcc));
            scene.add(Layers.hline(80, cGuide));
            scene.add(Layers.label(labelX, 80, "Accuracy 0.8", cAcc));
            scene.add(Layers.hline(100, cGuide));
            scene.add(Layers.label(labelX, 100, "Accuracy 1.0", cAcc));

            // what the model consumed, and what it never saw
            for (double[][] seg : seenRuns) {
                scene.add(Layers.polyline(seg, cData, false, 0, 1.0f));
            }
            for (double[][] seg : withheldRuns) {
                scene.add(Layers.dashedPolyline(seg, cWithheld, 1.0f, dash));
            }
            scene.add(Layers.dots(imputedPts.toArray(new float[0][]), cImputed, 3.5));

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
            scene.add(Layers.vline(current, cNow, 1.5f));
            scene.add(Layers.legend(
                    new String[] { "Data (seen by model)", "Withheld (never seen)", "Occluded, within horizon",
                            "Occluded, beyond horizon", "Forecast (frozen in gap)", "Uncertainty", "Error dist (past)",
                            "Imputed", "Interval acc (fraction)" },
                    new Color[] { cData, cWithheld, cReach, cBeyond, cForecast, cForecast.brighter(), cPast, cImputed,
                            cAcc },
                    new Layers.Swatch[] { Layers.Swatch.LINE, Layers.Swatch.DASHED, Layers.Swatch.BOX,
                            Layers.Swatch.BOX, Layers.Swatch.LINE, Layers.Swatch.BOX, Layers.Swatch.BOX,
                            Layers.Swatch.DOTS, Layers.Swatch.LINE }));

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

        int withheld = 0;
        for (boolean m : missing) {
            if (m) {
                withheld++;
            }
        }
        System.out.printf("%nwithheld %d / %d steps (%.1f%% of the stream)%n", withheld, N, 100.0 * withheld / N);
        System.out.printf("  imputed          : %d%n", gapN);
        System.out.printf("  skipped, warmup  : %d (forest not output-ready)%n", skippedNotReady);
        System.out.printf("  skipped, beyond  : %d (further than %d steps past the last observation)%n",
                skippedBeyondHorizon, forecastHorizon);
        if (gapN > 0) {
            System.out.printf("imputation RMSE over %d imputed points: %.3f%n", gapN, Math.sqrt(gapSe / gapN));
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
