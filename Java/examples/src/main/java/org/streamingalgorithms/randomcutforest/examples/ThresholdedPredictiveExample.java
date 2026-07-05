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
import static java.lang.Math.cos;
import static org.streamingalgorithms.randomcutforest.CommonUtils.toDoubleArray;
import static org.streamingalgorithms.randomcutforest.examples.datasets.MultiDimData.multiDimData;

import java.awt.*;
import java.io.File;
import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.streamingalgorithms.randomcutforest.config.TransformMethod;
import org.streamingalgorithms.randomcutforest.examples.datasets.MultiDimData;
import org.streamingalgorithms.randomcutforest.examples.plot.GifWriter;
import org.streamingalgorithms.randomcutforest.examples.plot.Layer;
import org.streamingalgorithms.randomcutforest.examples.plot.Layers;
import org.streamingalgorithms.randomcutforest.examples.plot.Plot2D;
import org.streamingalgorithms.randomcutforest.parkservices.ThresholdedRandomCutForest;
import org.streamingalgorithms.randomcutforest.returntypes.RangeVector;

/* the following example is an alerting example based on the sum of multiple time series where each time series is forecasted and then we add the forecasts; the final alerting corresponds to a predicted "cross". The graphics shows the colored lines for the predicted and actual crossings of the (dynamic) threshold.
*/
public class ThresholdedPredictiveExample implements Example {

    public static void main(String[] args) throws Exception {
        new ThresholdedPredictiveExample().run();
    }

    @Override
    public String command() {
        return "Thresholded_Predictive_example";
    }

    @Override
    public String description() {
        return "Example of predictive forecast across multiple time series using ThresholdedRCF";
    }

    double alertThreshold(int i) {
        return 100 * cos(2 * PI * i / 200) - 100; // sweeps [-200, 0]
    }

    @Override
    public void run() throws Exception {
        int sampleSize = 256;
        int baseDimensions = 1;
        int length = 4 * sampleSize;
        int outputAfter = 128;
        long seed = 2026L;
        Random random = new Random(seed);
        int numberOfModels = 5;
        int shingleSize = 10;
        int horizon = 20;
        // NOTE: no more `double alertThreshold = -200;` — the method is the threshold
        // now

        MultiDimData[] dataWithKeys = new MultiDimData[numberOfModels];
        ThresholdedRandomCutForest[] forests = new ThresholdedRandomCutForest[numberOfModels];
        int anomalies = 0;
        for (int k = 0; k < numberOfModels; k++) {
            int period = (int) Math.round(40 + 30 * random.nextDouble());
            dataWithKeys[k] = multiDimData(length, period, 100, 10, seed, baseDimensions, 0.01, true);
            anomalies += dataWithKeys[k].changes.length;
        }
        System.out.println(anomalies + " anomalies injected");
        for (int k = 0; k < numberOfModels; k++) {
            forests[k] = new ThresholdedRandomCutForest.Builder<>().dimensions(baseDimensions * shingleSize)
                    .randomSeed(seed + k).internalShinglingEnabled(true).shingleSize(shingleSize)
                    .outputAfter(outputAfter).transformMethod(TransformMethod.NORMALIZE).build();
        }

        // actual aggregate (background + y-range)
        double[][] aggLine = new double[length][2];
        double minAgg = Double.MAX_VALUE, maxAgg = -Double.MAX_VALUE;
        for (int i = 0; i < length; i++) {
            double s = 0;
            for (int k = 0; k < numberOfModels; k++) {
                s += dataWithKeys[k].data[i][0];
            }
            aggLine[i][0] = i;
            aggLine[i][1] = s;
            minAgg = Math.min(minAgg, s);
            maxAgg = Math.max(maxAgg, s);
        }

        // threshold curve, sampled once
        double[][] thrLine = new double[length][2];
        double thrMin = Double.MAX_VALUE, thrMax = -Double.MAX_VALUE;
        for (int i = 0; i < length; i++) {
            double th = alertThreshold(i);
            thrLine[i][0] = i;
            thrLine[i][1] = th;
            thrMin = Math.min(thrMin, th);
            thrMax = Math.max(thrMax, th);
        }
        // pad 12% on each side; keep the whole threshold curve in view
        double lo = Math.min(minAgg, thrMin);
        double hi = Math.max(maxAgg, thrMax);
        double pad = 0.12 * (hi - lo);
        double ymin = lo - pad, ymax = hi + pad;

        boolean livePlot = true, saveGif = true;
        int frameEvery = 2, frameDelayMs = 8, gifDelayMs = 40;

        Plot2D plot = livePlot
                ? Plot2D.openRect("ThresholdedPredictive — aggregate crossing", 0, length, ymin, ymax, 1100, 600)
                : Plot2D.offscreenRect(0, length, ymin, ymax);
        GifWriter gif = saveGif ? new GifWriter(new File("thresholded_predictive.gif"), gifDelayMs, true) : null;

        Color cAgg = Color.BLACK, cFore = new Color(31, 119, 180), cThresh = new Color(150, 150, 150);
        Color cPred = new Color(0, 220, 60, 110); // bright green, translucent
        Color cActual = new Color(230, 30, 30, 110); // red, translucent

        List<double[]> predictedCrossings = new ArrayList<>(); // {predictedAtIndex, predictedCrossingIndex}
        List<Integer> actualCrossings = new ArrayList<>();
        boolean predictNextCrossing = true, actualCrossingAlerted = false;
        double lastActualSum = 0;
        double half = 1.5; // half-width of the crossing bands in x-units

        Instant start = Instant.now();
        for (int i = 0; i < length; i++) {
            double[] prediction = new double[horizon];
            if (i > 2 * sampleSize) {
                for (int k = 0; k < numberOfModels; k++) {
                    RangeVector forecast = forests[k].extrapolate(horizon).rangeVector;
                    for (int t = 0; t < horizon; t++) {
                        prediction[t] += forecast.values[t];
                    }
                }
                // compare the forecast endpoint against the threshold AT THE PREDICTED TIME
                double predThr = alertThreshold(i + horizon - 1);
                if (prediction[horizon - 1] > predThr && predictNextCrossing) {
                    System.out.println("At " + i + ", predict crossing " + predThr + " near " + (i + horizon - 1));
                    predictedCrossings.add(new double[] { i, i + horizon - 1 });
                    predictNextCrossing = false;
                } else if (prediction[horizon - 1] < predThr && !predictNextCrossing) {
                    predictNextCrossing = true;
                }
            }

            double sumValue = aggLine[i][1];
            // each point compared against its OWN threshold; prev against prev's threshold
            double thrNow = alertThreshold(i);
            double thrPrev = alertThreshold(i - 1);
            if (lastActualSum > thrPrev && sumValue > thrNow) {
                if (!actualCrossingAlerted) {
                    System.out.println("Actual crossing near " + (i - 1) + " " + i);
                    if (i > 2 * sampleSize) { // not enough data earlier
                        actualCrossings.add(i);
                        actualCrossingAlerted = true;
                    } else {
                        actualCrossingAlerted = false;
                    }
                }
            } else if (sumValue < thrNow) {
                actualCrossingAlerted = false;
            }
            lastActualSum = sumValue;

            for (int k = 0; k < numberOfModels; k++) {
                forests[k].process(toDoubleArray(dataWithKeys[k].data[i]), i);
            }

            if ((livePlot || saveGif) && i % frameEvery == 0) {
                List<Layer> scene = new ArrayList<>();
                scene.add(Layers.polyline(thrLine, cThresh, false, 0, 1.0f));
                scene.add(Layers.polyline(aggLine, cAgg, false, 0, 1.0f));
                if (i > sampleSize) {
                    double[][] fLine = new double[horizon][2];
                    for (int t = 0; t < horizon; t++) {
                        fLine[t][0] = i + t;
                        fLine[t][1] = prediction[t];
                    }
                    scene.add(Layers.polyline(fLine, cFore, false, 0, 2.0f));
                }

                for (double[] pc : predictedCrossings) {
                    scene.add(
                            Layers.xBands(new double[][] { { pc[1] - half, pc[1] + half } }, cPred, cPred.getAlpha()));
                }
                for (int ac : actualCrossings) {
                    scene.add(Layers.xBands(new double[][] { { ac - half, ac + half } }, cActual, cActual.getAlpha()));
                }

                scene.add(Layers.vline(i, new Color(110, 110, 110), 1.5f));
                scene.add(Layers.legend(new String[] { "Aggregate", "Forecast (sum)", "Threshold", "Predicted crossing",
                        "Actual crossing" }, new Color[] { cAgg, cFore, cThresh, cPred, cActual }));

                if (livePlot) {
                    plot.render(scene);
                    if (frameDelayMs > 0) {
                        Thread.sleep(frameDelayMs);
                    }
                }
                if (saveGif) {
                    gif.writeFrame(plot.renderImage(1000, 540, scene));
                }
            }
        }

        if (gif != null) {
            gif.close();
            System.out.println("wrote thresholded_predictive.gif");
        }

        System.out.println("\n=== predictive scorecard ===");
        int matched = 0;
        for (int ac : actualCrossings) {
            double bestLead = Double.NaN;
            for (double[] pc : predictedCrossings) {
                if (pc[0] <= ac && Math.abs(pc[1] - ac) <= horizon) {
                    double lead = ac - pc[0];
                    if (Double.isNaN(bestLead) || lead < bestLead) {
                        bestLead = lead;
                    }
                }
            }
            if (!Double.isNaN(bestLead)) {
                matched++;
                System.out.printf("actual crossing @ %4d  predicted %d steps ahead%n", ac, (int) bestLead);
            } else {
                System.out.printf("actual crossing @ %4d  MISSED%n", ac);
            }
        }
        System.out.printf("%d/%d actual crossings caught in advance; %d prediction events fired (total %d ms)%n",
                matched, actualCrossings.size(), predictedCrossings.size(),
                Duration.between(start, Instant.now()).toMillis());
    }

}