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
 * The file is substantially modified from its prior version which had the
 * following notice.
 *
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

package org.streamingalgorithms.randomcutforest.parkservices.calibration;

import static java.lang.Math.max;
import static java.lang.Math.min;
import static org.streamingalgorithms.randomcutforest.CommonUtils.checkArgument;
import static org.streamingalgorithms.randomcutforest.CommonUtils.toFloatArray;
import static org.streamingalgorithms.randomcutforest.parkservices.RCFCaster.DEFAULT_ERROR_PERCENTILE;

import java.util.Arrays;
import java.util.Optional;

import org.streamingalgorithms.randomcutforest.PredictiveRandomCutForest;
import org.streamingalgorithms.randomcutforest.config.TransformMethod;
import org.streamingalgorithms.randomcutforest.parkservices.ForecastDescriptor;
import org.streamingalgorithms.randomcutforest.parkservices.config.Calibration;
import org.streamingalgorithms.randomcutforest.returntypes.DiVector;
import org.streamingalgorithms.randomcutforest.returntypes.RangeVector;
import org.streamingalgorithms.randomcutforest.returntypes.SampleSummary;
import org.streamingalgorithms.randomcutforest.statistics.Deviation;

import lombok.AccessLevel;
import lombok.Getter;
import lombok.Setter;

// we recommend the article "Regret in the On-Line Decision Problem", by Foster and Vohra,
// Games and Economic Behavior, Vol=29 (1-2), 1999
// the discussion is applicable to non-regret scenarios as well; but essentially boils down to
// fixed point/minimax computation. One could use multiplicative update type methods which would be
// uniform over all quantiles, provided sufficient data and a large enough calibration horizon.
// Multiplicative updates are scale free -- but providing scale free forecasting over a stream raises the
// issue "what is the current scale of the stream". While such questions can be answered, that discussion
// can be involved and out of current scope of this library. We simplify the issue to calibrating two
// fixed quantiles and hence additive updates are reasonable.

/**
 * Keeps the errors of the forecasts that were published and turns them into the
 * interval of the next forecast. This is the algorithm in outline; the methods
 * give the details.
 * <ul>
 * <li>It stores the last forecastHorizon published forecasts (point, upper and
 * lower), in a ring, and the last two inputs.</li>
 * <li>When an actual arrives, for each step k it takes the forecast made k + 1
 * steps earlier. It records a hit if the actual fell inside that forecast's
 * published interval. It splits the error, actual minus the published point,
 * into its positive part and its negative part; each part goes to its own
 * discounted statistic (discount 1/errorHorizon), and the other side receives a
 * zero.</li>
 * <li>For a starting range of forecasts, 10 x forecastHorizon (or the
 * estimator's outputAfter, 50 as built), both widths are the data's smoothed
 * deviation. Over the next as many forecasts the widths move linearly to the
 * error deviations (without the estimator). After that, the upward width is the
 * standard deviation of the positive parts and the downward width that of the
 * negative parts, so that the interval is asymmetric.</li>
 * <li>With the estimator (useRCF), a second forest trained on the last two
 * inputs, the step, and the two parts of the error supplies the widths of the
 * near steps, moving linearly to the observed widths as the step grows.</li>
 * <li>A control loop: when a step's hit rate is below 1 - percentile, both of
 * its widths are multiplied by 1/(hit rate + 0.1).</li>
 * <li>The mean error is the mean of the positive parts plus the mean of the
 * negative parts. The upper offset is mean + 1.3 x upward width and the lower
 * offset is mean - 1.3 x downward width.</li>
 * </ul>
 * {@link #calibrate(double[], Calibration, RangeVector)} describes how the
 * offsets meet the forecast. The reported errorRMSE holds the widths before the
 * estimator and the control loop: after the starting range, the standard
 * deviations of the two parts of the error, not root mean squares.
 */
@Getter
@Setter
public class ErrorHandler {

    int sequenceIndex;
    double percentile;
    int forecastHorizon;
    int errorHorizon;
    // the following arrays store the state of the sequential computation
    // these can be optimized -- for example once could store the errors; which
    // would see much fewer increments.
    // However, for a small enough errorHorizon, the generality of
    // changing the error function
    // outweighs the benefit of recomputation. The search in the ensemble tree is
    // still a larger bottleneck than
    // these computations at the moment; not to mention issues of saving and
    // restoring state.
    protected RangeVector[] pastForecasts;

    RangeVector errorDistribution;
    DiVector errorRMSE;
    float[] errorMean;
    Deviation[] intervalPrecision;
    Deviation[] rmseHighDeviations;
    Deviation[] rmseLowDeviations;
    float[] lowerLimit;
    float[] upperLimit;
    double[] lastInputs;
    PredictiveRandomCutForest estimator;
    float[] lastDataDeviations;

    // scratch arrays, reused so that an update allocates nothing; they carry no
    // state and are not saved
    @Getter(AccessLevel.NONE)
    private final float[] errorTuple;
    @Getter(AccessLevel.NONE)
    private final float[] query;
    @Getter(AccessLevel.NONE)
    private final int[] missing;
    @Getter(AccessLevel.NONE)
    private final float[] errorHigh;
    @Getter(AccessLevel.NONE)
    private final float[] errorLow;
    @Getter(AccessLevel.NONE)
    private final float[] precision;

    // We keep the multipliers defined for potential
    // future use.

    RangeVector multipliers;
    RangeVector adders;

    public ErrorHandler(Builder builder) {
        checkArgument(builder.forecastHorizon > 0, "has to be positive");
        checkArgument(builder.errorHorizon >= builder.forecastHorizon,
                "intervalPrecision horizon should be at least as large as forecast horizon");
        checkArgument(builder.errorHorizon <= 1024, "reduce error horizon");
        forecastHorizon = builder.forecastHorizon;
        errorHorizon = builder.errorHorizon;
        int inputLength = (builder.dimensions / builder.shingleSize);
        int length = inputLength * forecastHorizon;
        percentile = builder.percentile;
        pastForecasts = new RangeVector[forecastHorizon];
        for (int i = 0; i < forecastHorizon; i++) {
            pastForecasts[i] = new RangeVector(length);
        }
        sequenceIndex = 0;
        lastInputs = new double[2 * inputLength];
        errorTuple = new float[lastInputs.length + 2 * inputLength + 2];
        query = new float[lastInputs.length + 2 * inputLength + 2];
        missing = new int[inputLength];
        errorHigh = new float[length];
        errorLow = new float[length];
        precision = new float[length];
        rmseHighDeviations = new Deviation[length];
        rmseLowDeviations = new Deviation[length];
        intervalPrecision = new Deviation[length];
        for (int i = 0; i < length; i++) {
            rmseHighDeviations[i] = new Deviation(1.0 / errorHorizon);
            rmseLowDeviations[i] = new Deviation(1.0 / errorHorizon);
            intervalPrecision[i] = new Deviation(1.0 / errorHorizon);
        }
        errorMean = new float[length];
        errorRMSE = new DiVector(length);
        lastDataDeviations = new float[inputLength];
        errorDistribution = new RangeVector(length);
        if (builder.upperLimit.isPresent()) {
            checkArgument(builder.upperLimit.get().length == inputLength, "incorrect length");
            upperLimit = Arrays.copyOf(builder.upperLimit.get(), inputLength);
        } else {
            upperLimit = new float[inputLength];
            Arrays.fill(upperLimit, Float.MAX_VALUE);
        }
        if (builder.lowerLimit.isPresent()) {
            checkArgument(builder.lowerLimit.get().length == inputLength, "incorrect length");
            for (int y = 0; y < inputLength; y++) {
                checkArgument(builder.lowerLimit.get()[y] <= upperLimit[y], "incorrect limits");
            }
            lowerLimit = Arrays.copyOf(builder.lowerLimit.get(), inputLength);
        } else {
            lowerLimit = new float[inputLength];
            Arrays.fill(lowerLimit, -Float.MAX_VALUE);
        }
        // uses lastInputs as a markov input, the +2 corresponds to lookahead and
        // forecasthorizon - lookahed; the 2*inputlength correspond to the
        // (plausibly correlated) positive and negative errors
        // There are potentially many different variations -- use a difference encoding
        // for the
        // last two values (keeps lastInputs the same length) -- or change lastInputs to
        // be a full
        // on wavelet transform, etc.
        //
        if (builder.useRCF) {
            int inputDimensions = lastInputs.length + 2 * inputLength + 2;
            double[] weights = new double[inputDimensions];
            Arrays.fill(weights, 1.0);
            // the lookahead has 1/3 the weight of the total
            weights[lastInputs.length] = lastInputs.length;
            weights[lastInputs.length + 1] = lastInputs.length;
            estimator = new PredictiveRandomCutForest.Builder<>().inputDimensions(inputDimensions).weights(weights)
                    .randomSeed(13).outputAfter(50).transformMethod(TransformMethod.NORMALIZE).startNormalization(49)
                    .build();
        }
    }

    // for mappers
    public ErrorHandler(int errorHorizon, int forecastHorizon, int sequenceIndex, double percentile, int inputLength,
            float[] pastForecastsFlattened, float[] lastDataDeviations, double[] lastInput, Deviation[] deviations,
            PredictiveRandomCutForest estimator, float[] auxiliary) {
        checkArgument(forecastHorizon > 0, " incorrect forecast horizon");
        checkArgument(errorHorizon >= forecastHorizon, "incorrect error horizon");
        checkArgument(inputLength > 0, "incorrect parameters");
        checkArgument(sequenceIndex >= 0, "cannot be negative");
        checkArgument(Math.abs(percentile - 0.25) < 0.24, "has to be between (0,0.5) ");
        checkArgument(deviations.length == 3 * inputLength * forecastHorizon, "incorrect length");
        checkArgument(lastInput.length == 2 * inputLength, "incorrect length");

        this.sequenceIndex = sequenceIndex;
        this.errorHorizon = errorHorizon;
        this.percentile = percentile;
        this.forecastHorizon = forecastHorizon;
        this.pastForecasts = new RangeVector[forecastHorizon];
        this.lastInputs = Arrays.copyOf(lastInput, lastInput.length);

        int length = forecastHorizon * inputLength;
        errorTuple = new float[lastInputs.length + 2 * inputLength + 2];
        query = new float[lastInputs.length + 2 * inputLength + 2];
        missing = new int[inputLength];
        errorHigh = new float[length];
        errorLow = new float[length];
        precision = new float[length];
        checkArgument(lastDataDeviations.length >= inputLength, "incorrect length");
        this.lastDataDeviations = Arrays.copyOf(lastDataDeviations, lastDataDeviations.length);
        this.errorMean = new float[length];
        this.errorRMSE = new DiVector(length);
        this.errorDistribution = new RangeVector(length);
        this.intervalPrecision = new Deviation[inputLength * forecastHorizon];
        this.rmseHighDeviations = new Deviation[inputLength * forecastHorizon];
        this.rmseLowDeviations = new Deviation[inputLength * forecastHorizon];
        for (int y = 0; y < inputLength * forecastHorizon; y++) {
            this.intervalPrecision[y] = deviations[y].copy();
            this.rmseHighDeviations[y] = deviations[y + inputLength * forecastHorizon].copy();
            this.rmseLowDeviations[y] = deviations[y + 2 * inputLength * forecastHorizon].copy();
        }
        lowerLimit = new float[inputLength];
        Arrays.fill(lowerLimit, -Float.MAX_VALUE);
        upperLimit = new float[inputLength];
        Arrays.fill(upperLimit, Float.MAX_VALUE);
        this.estimator = estimator;
        int arrayLength = pastForecastsFlattened.length / (3 * length);
        checkArgument(arrayLength * 3 * length == pastForecastsFlattened.length, " has to be multiple of 3");
        for (int i = 0; i < arrayLength; i++) {
            float[] values = Arrays.copyOfRange(pastForecastsFlattened, i * 3 * length, (i * 3 + 1) * length);
            float[] upper = Arrays.copyOfRange(pastForecastsFlattened, (i * 3 + 1) * length, (i * 3 + 2) * length);
            float[] lower = Arrays.copyOfRange(pastForecastsFlattened, (i * 3 + 2) * length, (i * 3 + 3) * length);
            // a state saved before calibrate kept each end on its side of the point can
            // hold an interval that misses its point; it is repaired as calibrate now
            // would have published it
            for (int y = 0; y < length; y++) {
                if (upper[y] < values[y]) {
                    upper[y] = values[y];
                }
                if (lower[y] > values[y]) {
                    lower[y] = values[y];
                }
            }
            pastForecasts[i] = new RangeVector(values, upper, lower);

        }
        for (int i = arrayLength; i < forecastHorizon; i++) {
            pastForecasts[i] = new RangeVector(length);
        }
        recomputeErrors(lastInputs, inputLength);
    }

    public void setUpperLimit(float[] upperLimit) {
        if (upperLimit != null) {
            checkArgument(upperLimit.length == this.upperLimit.length, "incorrect Length");
            System.arraycopy(upperLimit, 0, this.upperLimit, 0, upperLimit.length);
        }
    }

    public void setLowerLimit(float[] lowerLimit) {
        if (lowerLimit != null) {
            checkArgument(lowerLimit.length == this.lowerLimit.length, "incorrect Length");
            for (int i = 0; i < lowerLimit.length; i++) {
                checkArgument(lowerLimit[i] <= this.upperLimit[i], "lower limit is higher than upper limit");
                this.lowerLimit[i] = lowerLimit[i];
            }
        }
    }

    /**
     * Updates the statistics with an actual and recomputes the offsets. The input
     * is first shifted into the last inputs, so that the estimator's training row
     * pairs an error with the two inputs ending at its actual; the estimator's
     * query in recomputeErrors uses the two inputs ending when the next forecast is
     * made. For each step k with a stored forecast, the forecast made k + 1 steps
     * earlier is compared with the actual: a hit if the actual lies within its
     * published interval, and the error, actual minus its published point, split
     * into a positive and a negative part.
     *
     * @param input      the actual input
     * @param deviations the deviations (post the current input)
     */
    public void updateActuals(double[] input, double[] deviations) {
        int arrayLength = pastForecasts.length;
        int inputLength = input.length;

        for (int i = 0; i < lastInputs.length - inputLength; i++) {
            lastInputs[i] = lastInputs[i + inputLength];
        }
        System.arraycopy(input, 0, lastInputs, lastInputs.length - inputLength, inputLength);

        if (sequenceIndex > 0) {
            // sequenceIndex indicates the first empty place for input
            // note the predictions have already been stored
            int inputIndex = (sequenceIndex + arrayLength - 1) % arrayLength;
            float[] tuple = null;
            if (estimator != null) {
                int tupleLength = lastInputs.length + 2 * inputLength + 2;
                tuple = (errorTuple.length == tupleLength) ? errorTuple : new float[tupleLength];
                for (int y = 0; y < lastInputs.length; y++) {
                    tuple[y] = (float) lastInputs[y];
                }
            }
            for (int i = 0; i < forecastHorizon; i++) {
                if (sequenceIndex > i) {
                    RangeVector a = pastForecasts[inputIndex];
                    int offset = i * inputLength;
                    for (int j = 0; j < inputLength; j++) {
                        boolean hit = input[j] <= a.upper[offset + j] && input[j] >= a.lower[offset + j];
                        intervalPrecision[offset + j].update(hit ? 1.0 : 0);
                        double error = input[j] - a.values[offset + j];
                        if (error >= 0) {
                            rmseHighDeviations[offset + j].update(error);
                            rmseLowDeviations[offset + j].update(0);
                        } else {
                            rmseLowDeviations[offset + j].update(error);
                            rmseHighDeviations[offset + j].update(0);
                        }
                    }
                    if (tuple != null) {
                        tuple[lastInputs.length] = i;
                        tuple[lastInputs.length + 1] = forecastHorizon - i;
                        for (int j = 0; j < inputLength; j++) {
                            double error = input[j] - a.values[offset + j];
                            if (error >= 0) {
                                tuple[lastInputs.length + 2 + j] = (float) error;
                                tuple[lastInputs.length + inputLength + 2 + j] = 0;
                            } else {
                                tuple[lastInputs.length + inputLength + 2 + j] = (float) (error);
                                tuple[lastInputs.length + 2 + j] = 0;
                            }
                        }
                        estimator.update(tuple, 0L);
                    }
                }
                inputIndex = (inputIndex + arrayLength - 1) % arrayLength;
            }
        }
        if (deviations != null && lastDataDeviations.length == deviations.length) {
            // as toFloatArray, in place
            for (int i = 0; i < deviations.length; i++) {
                lastDataDeviations[i] = (deviations[i] == 0) ? 0 : (float) deviations[i];
            }
        } else {
            lastDataDeviations = toFloatArray(deviations);
        }
        recomputeErrors(lastInputs, inputLength);
    }

    /**
     * Recomputes the mean error and the two offsets from the statistics: the
     * starting range, the linear move to the error deviations, the estimator if
     * there is one, and the control loop; see the class description.
     */
    void recomputeErrors(double[] lastInputs, int inputLength) {
        double a;
        if (estimator != null) {
            a = (double) (sequenceIndex) / (estimator.getForest().getOutputAfter());
        } else {
            a = (double) (sequenceIndex) / (10 * forecastHorizon);
        }
        if (a < 1) {
            for (int y = 0; y < intervalPrecision.length; y++) {
                errorRMSE.high[y] = errorRMSE.low[y] = lastDataDeviations[y % inputLength];
                errorHigh[y] = errorLow[y] = lastDataDeviations[y % inputLength];
            }
        } else {
            if (a < 2) {
                for (int y = 0; y < errorRMSE.high.length; y++) {
                    double offset = (2 - a) * lastDataDeviations[y % inputLength];
                    errorRMSE.high[y] = (offset + (a - 1) * rmseHighDeviations[y].getDeviation());
                    errorRMSE.low[y] = (offset + (a - 1) * rmseLowDeviations[y].getDeviation());
                }
            } else {
                for (int y = 0; y < errorRMSE.high.length; y++) {
                    errorRMSE.high[y] = rmseHighDeviations[y].getDeviation();
                    errorRMSE.low[y] = rmseLowDeviations[y].getDeviation();
                }
            }

            if (estimator != null) {
                int queryLength = lastInputs.length + inputLength * 2 + 2;
                // the error coordinates of the query stay zero; they are the missing ones
                float[] query = (this.query.length == queryLength) ? this.query : new float[queryLength];
                for (int y = 0; y < lastInputs.length; y++) {
                    // as toFloatArray
                    query[y] = (lastInputs[y] == 0) ? 0 : (float) lastInputs[y];
                }
                int[] missing = (this.missing.length == inputLength) ? this.missing : new int[inputLength];
                for (int i = 0; i < forecastHorizon; i++) {
                    query[lastInputs.length] = i;
                    query[lastInputs.length + 1] = forecastHorizon - i;
                    for (int j = 0; j < inputLength; j++) {
                        missing[j] = lastInputs.length + 2 + j;
                    }
                    // at this moment we use the PredictiveRCF more for the shorter term estimation,
                    // and use an interpolation
                    // with the observed error for the longer term
                    SampleSummary answer = estimator.predict(query, 0, missing, 1, 0.5, 0.7);
                    for (int j = 0; j < inputLength; j++) {

                        errorHigh[i * inputLength + j] = (forecastHorizon - i)
                                * max(0, answer.deviation[lastInputs.length + 2 + j]) / forecastHorizon
                                + (float) (i * rmseHighDeviations[i * inputLength + j].getDeviation()
                                        / forecastHorizon);

                    }
                    for (int j = 0; j < inputLength; j++) {
                        missing[j] = lastInputs.length + inputLength + 2 + j;
                    }
                    answer = estimator.predict(query, 0, missing, 1, 0.5, 0.7);
                    for (int j = 0; j < inputLength; j++) {

                        errorLow[i * inputLength + j] = (forecastHorizon - i)
                                * max(0, answer.deviation[lastInputs.length + inputLength + 2 + j]) / forecastHorizon
                                + (float) (i * rmseLowDeviations[i * inputLength + j].getDeviation() / forecastHorizon);

                    }
                }
            } else {
                for (int y = 0; y < errorRMSE.high.length; y++) {
                    errorHigh[y] = (float) errorRMSE.high[y];
                    errorLow[y] = (float) errorRMSE.low[y];
                }
            }
        }
        // a control loop
        for (int y = 0; y < intervalPrecision.length; y++) {
            double hitRate = intervalPrecision[y].getMean();
            if (hitRate < 1.0 - percentile) {
                float factor = (float) max(1.0, 1.0 / (hitRate + 0.1));
                errorHigh[y] = factor * errorHigh[y];
                errorLow[y] = factor * errorLow[y];
            }
        }

        for (int i = 0; i < errorMean.length; i++) {
            errorMean[i] = (float) (rmseHighDeviations[i].getMean() + rmseLowDeviations[i].getMean());
            errorDistribution.values[i] = errorMean[i];
            errorDistribution.upper[i] = errorMean[i] + (float) (1.3 * errorHigh[i]);
            errorDistribution.lower[i] = errorMean[i] - (float) (1.3 * errorLow[i]);
        }
    }

    public void augmentDescriptor(ForecastDescriptor descriptor) {
        int inputLength = descriptor.getInputLength();
        // the descriptor copies what it is given
        float[] iPrecision = (precision.length == inputLength * forecastHorizon) ? precision
                : new float[inputLength * forecastHorizon];
        for (int i = 0; i < errorMean.length; i++) {
            iPrecision[i] = (float) intervalPrecision[i].getMean();
        }
        descriptor.setErrorMean(errorMean);
        descriptor.setErrorRMSE(errorRMSE);
        descriptor.setObservedErrorDistribution(errorDistribution);
        descriptor.setIntervalPrecision(iPrecision);
    }

    /**
     * saves the forecast -- note that this section assumes that updateActuals() has
     * been invoked prior (to recompute the deviations)
     *
     * @param vector the forecast
     */
    public void updateForecasts(RangeVector vector) {
        ++sequenceIndex;
        int arrayLength = pastForecasts.length;
        int storedForecastIndex = (sequenceIndex + arrayLength - 1) % (arrayLength);
        int length = pastForecasts[0].values.length;
        System.arraycopy(vector.values, 0, pastForecasts[storedForecastIndex].values, 0, length);
        System.arraycopy(vector.upper, 0, pastForecasts[storedForecastIndex].upper, 0, length);
        System.arraycopy(vector.lower, 0, pastForecasts[storedForecastIndex].lower, 0, length);
    }

    public RangeVector getErrorDistribution() {
        return new RangeVector(errorDistribution);
    }

    public float[] getErrorMean() {
        return Arrays.copyOf(errorMean, errorMean.length);
    }

    public DiVector getErrorRMSE() {
        return new DiVector(errorRMSE);
    }

    public Deviation[] getDeviationList() {
        Deviation[] list = new Deviation[3 * intervalPrecision.length];
        for (int i = 0; i < intervalPrecision.length; i++) {
            list[i] = intervalPrecision[i].copy();
            list[i + intervalPrecision.length] = rmseHighDeviations[i].copy();
            list[i + 2 * intervalPrecision.length] = rmseLowDeviations[i].copy();
        }
        return list;
    }

    public float[] getIntervalPrecision() {
        float[] iPrecision = new float[intervalPrecision.length];
        for (int i = 0; i < iPrecision.length; i++) {
            iPrecision[i] = (float) (intervalPrecision[i].getMean());
        }
        return iPrecision;
    }

    /**
     * Calibrates a forecast in place. SIMPLE moves the point by the mean error and
     * MINIMAL keeps it; both hold it within the limits. The published upper end is
     * the larger of RCF's own upper end and the point plus the upper offset, and
     * the lower end is the smaller of RCF's own lower end and the point plus the
     * lower offset: the envelope of the forecast's own range and the range of the
     * past errors, not their sum. NONE leaves the forecast as RCF made it.
     * <p>
     * The errors are measured against the published point, never against the
     * bounds. Under SIMPLE the point has been moved by the mean error, so the mean
     * error settles near half of RCF's own mean error; the offsets contain the mean
     * error again, so the band sits around RCF's forecast plus all of its mean
     * error, while the point carries half of it. Under MINIMAL the mean error is
     * RCF's own, and the band sits at the same place.
     * <p>
     * Each end is then kept on its side of the point. An end is bounded only by its
     * own limit, so without this a lower end could lie above an upper limit that
     * holds the point (or an upper end below a lower limit), and a saved state
     * holding that interval could not be restored.
     *
     * @param input       the last input (its length is checked)
     * @param calibration the calibration
     * @param ranges      the forecast; calibrated on return
     */
    public void calibrate(double[] input, Calibration calibration, RangeVector ranges) {
        if (calibration != Calibration.NONE) {
            int inputLength = intervalPrecision.length / forecastHorizon;
            checkArgument(input.length == inputLength, "incorrect input");
            checkArgument(intervalPrecision.length == ranges.values.length, "mismatched lengths");
            for (int y = 0; y < intervalPrecision.length; y++) {
                if (calibration == Calibration.SIMPLE) {
                    ranges.values[y] = min(
                            max(ranges.values[y] + errorDistribution.values[y], lowerLimit[y % inputLength]),
                            upperLimit[y % inputLength]);
                } else {
                    ranges.values[y] = min(max(ranges.values[y], lowerLimit[y % inputLength]),
                            upperLimit[y % inputLength]);
                }
                ranges.upper[y] = min(max(ranges.upper[y], ranges.values[y] + errorDistribution.upper[y]),
                        upperLimit[y % inputLength]);
                ranges.lower[y] = max(min(ranges.lower[y], ranges.values[y] + errorDistribution.lower[y]),
                        lowerLimit[y % inputLength]);
                // each end stays on its side of the point
                if (ranges.upper[y] < ranges.values[y]) {
                    ranges.upper[y] = ranges.values[y];
                }
                if (ranges.lower[y] > ranges.values[y]) {
                    ranges.lower[y] = ranges.values[y];
                }
            }
        }
    }

    public int getInputLength() {
        return lastInputs.length / 2;
    }

    /**
     * produces the stored forecasts as a non-null array
     */
    public float[] getPastForecastsFlattened() {
        int arrayLength = min(sequenceIndex, pastForecasts.length);
        int length = intervalPrecision.length;
        float[] answer = new float[3 * length * arrayLength];
        for (int i = 0; i < arrayLength; i++) {
            System.arraycopy(pastForecasts[i].values, 0, answer, 3 * i * length, length);
            System.arraycopy(pastForecasts[i].upper, 0, answer, 3 * i * length + length, length);
            System.arraycopy(pastForecasts[i].lower, 0, answer, 3 * i * length + 2 * length, length);
        }
        return answer;
    }

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        protected int dimensions;
        protected int shingleSize = 1;
        protected int forecastHorizon;
        protected boolean useRCF = true;
        protected int errorHorizon = 100; // easy for percentile
        protected double percentile = DEFAULT_ERROR_PERCENTILE;
        protected Optional<float[]> upperLimit = Optional.empty();
        protected Optional<float[]> lowerLimit = Optional.empty();

        public Builder dimensions(int dimensions) {
            this.dimensions = dimensions;
            return this;
        }

        public Builder shingleSize(int shingleSize) {
            this.shingleSize = shingleSize;
            return this;
        }

        public Builder forecastHorizon(int horizon) {
            this.forecastHorizon = horizon;
            return this;
        }

        public Builder errorHorizon(int errorHorizon) {
            this.errorHorizon = errorHorizon;
            return this;
        }

        public Builder percentile(double percentile) {
            this.percentile = percentile;
            return this;
        }

        public Builder lowerLimit(float[] lowerLimit) {
            this.lowerLimit = Optional.of(lowerLimit);
            return this;
        }

        public Builder upperLimit(float[] upperLimit) {
            this.upperLimit = Optional.of(upperLimit);
            return this;
        }

        public Builder useRCF(boolean use) {
            useRCF = use;
            return this;
        }

        public ErrorHandler build() {
            return new ErrorHandler(this);
        }
    }
}
