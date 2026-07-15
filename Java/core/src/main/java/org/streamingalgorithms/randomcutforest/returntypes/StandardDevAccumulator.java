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

package org.streamingalgorithms.randomcutforest.returntypes;

import static java.lang.Math.max;

import lombok.Getter;

/**
 * This accumulator checks to see if a result is converging by testing the
 * sample mean and standard deviation of a scalar value computed from the
 * result.
 *
 * @param <R> The type of the value being accumulated.
 */
public abstract class StandardDevAccumulator<R> implements ConvergingAccumulator<R> {

    /**
     * When testing for convergence, we use ALPHA times the sample standard
     * deviation to define our interval.
     */
    private static final double ALPHA = 0.5;
    /**
     * The minimum number of values that have to be accepted by this accumulator
     * before we start testing for convergence.
     */
    private final int minValuesAccepted;
    /**
     * The number of witnesses needed to declare convergence.
     */
    private final int convergenceThreshold;

    private final CriticalDirection direction;

    /**
     * The value accumulated until now.
     */
    protected R accumulatedValue;
    /**
     * The number of values accepted by this accumulator until now.
     */
    private int valuesAccepted;
    /**
     * The number of values that are 'witnesses' to convergence until now. See
     * {@link #accept}.
     */
    @Getter
    private int witnesses;
    /**
     * The current sum of the converging scalar value. Used to compute the sample
     * mean.
     */
    private double sumConvergeVal;
    /**
     * The current sum of squares of the converging scalar value. Used to compute
     * the sample standard deviation.
     */
    private double sumSqConvergeVal;

    private final int sign;
    private final boolean useAbsolute;

    /**
     * Create a new converging accumulator that uses a one-sided standard deviation
     * test.
     *
     * @param direction         direction of convergence
     * @param precision         The number of witnesses required before declaring
     *                          convergence will be at least 1.0 / precision.
     * @param minValuesAccepted The user-specified minimum number of values visited
     *                          before returning a result. Note that
     *                          {@link #isConverged()} may return true before
     *                          accepting this number of results if the
     * @param maxValuesAccepted The maximum number of values that will be accepted
     *                          by this accumulator.
     */
    public StandardDevAccumulator(CriticalDirection direction, double precision, int minValuesAccepted,
            int maxValuesAccepted) {

        this.direction = direction;
        this.convergenceThreshold = precision < 1.0 / maxValuesAccepted ? maxValuesAccepted : (int) (1.0 / precision);
        this.minValuesAccepted = Math.min(minValuesAccepted, maxValuesAccepted);
        valuesAccepted = 0;
        witnesses = 0;
        sumConvergeVal = 0.0;
        sumSqConvergeVal = 0.0;
        accumulatedValue = null;
        useAbsolute = (direction == CriticalDirection.BOTH);
        if (direction == CriticalDirection.HIGH) {
            sign = 1;
        } else {
            sign = -1;
        }
    }

    /**
     * Given a new result value, add it to the accumulated value and update
     * convergence statistics.
     *
     * @param result The new value being accumulated.
     */
    @Override
    public void accept(R result) {
        accumulateValue(result);
        acceptValue(getConvergingValue(result));
    }

    @Override
    public void acceptValue(double value) {
        sumConvergeVal += value;
        sumSqConvergeVal += value * value;
        final int n = ++valuesAccepted;

        final double mean = sumConvergeVal / n;
        double var = Math.max(0.0, sumSqConvergeVal / n - mean * mean);
        final double dev = Math.sqrt((double) n * var / (n - 1));

        // (n >= minValuesAccepted) ? 1 : 0, branch-free via sign bit
        final int active = ((n - minValuesAccepted) >> 31) + 1;

        // Apply sign or absolute value directly to the gap
        double gap = value - mean;
        if (useAbsolute) {
            gap = Math.abs(gap);
        } else {
            gap *= sign; // multiplies by 1 for HIGH, -1 for LOW
        }

        // This perfectly preserves the (gap + 1e-6 > ALPHA * dev) logic from your
        // original code
        final int witness = (gap + 1e-6 > ALPHA * dev) ? 1 : 0;

        witnesses += active & witness;
    }

    @Override
    public boolean isPrimitive() {
        return true;
    }

    /**
     * @return the number of values accepted until now.
     */
    @Override
    public int getValuesAccepted() {
        return valuesAccepted;
    }

    /**
     * @return 'true' if the accumulated value has converged, 'false' otherwise.
     */
    @Override
    public boolean isConverged() {
        return witnesses >= convergenceThreshold;
    }

    /**
     * @return the accumulated value.
     */
    @Override
    public R getAccumulatedValue() {
        return accumulatedValue;
    }

    /**
     * Given a new result value, compute its converging scalar value.
     *
     * @param result A new result value computed by a Random Cut Tree.
     * @return the scalar value used to measure convergence for this result type.
     */
    protected abstract double getConvergingValue(R result);

    /**
     * Add the new result to the accumulated value.
     *
     * @param result The new result to add to the accumulated value.
     */
    protected abstract void accumulateValue(R result);

    /**
     * @return the mean of the values
     */
    public double getMean() {
        return (valuesAccepted == 0) ? 0 : sumConvergeVal / valuesAccepted;
    }

    /**
     * it is possible that valuesAccepted is not large hence applying Bessel
     * correction
     * 
     * @return the standard deviation of the accepted values
     */
    public double getDeviation() {
        if (valuesAccepted <= 1) {
            return 0;
        }

        double mean = sumConvergeVal / valuesAccepted;
        double stdev = max(0, sumSqConvergeVal / valuesAccepted - mean * mean);

        stdev = Math.sqrt(valuesAccepted * stdev / (valuesAccepted - 1));
        return stdev;
    }
}
