/*
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

package org.streamingalgorithms.randomcutforest.returntypes;

import static java.lang.Math.round;
import static org.streamingalgorithms.randomcutforest.CommonUtils.checkArgument;
import static org.streamingalgorithms.randomcutforest.CommonUtils.checkNotNull;

import java.util.stream.Collector;

import lombok.Getter;

/**
 * An InterpolationMeasure is used to store certain geometric quantities during
 * a tree traversal and useful for density estimation
 */
public class InterpolationMeasure {

    public final DiVector measure;
    public final DiVector distances;
    public final DiVector probMass;
    @Getter
    protected final int dimensions;
    @Getter
    protected int sampleSize;

    /**
     * Create a new InterpolationMeasure object with the given number of spatial
     * dimensions. Note that the number of half-dimensions will be 2 * dimensions.
     *
     * @param dimensions The number of spatial dimensions.
     * @param sampleSize The samplesize of each tree in forest, which may be used
     *                   for normalization.
     */
    public InterpolationMeasure(int dimensions, int sampleSize) {
        checkArgument(dimensions > 0, "dimensions must be greater than 0");
        this.sampleSize = sampleSize;
        this.dimensions = dimensions;
        measure = new DiVector(dimensions);
        distances = new DiVector(dimensions);
        probMass = new DiVector(dimensions);
    }

    /**
     * A copy constructor that creates a deep copy of the base InterpolationMeasure.
     *
     * @param base An InterpolationMeasure instance that we want to copy.
     */
    public InterpolationMeasure(InterpolationMeasure base, boolean consume) {
        this.sampleSize = base.sampleSize;
        this.dimensions = base.dimensions;
        measure = (consume) ? base.measure : new DiVector(base.measure);
        distances = (consume) ? base.distances : new DiVector(base.distances);
        probMass = (consume) ? base.probMass : new DiVector(base.probMass);
    }

    public InterpolationMeasure(int sampleSize, DiVector measure, DiVector distances, DiVector probMass) {

        checkArgument(measure.getDimensions() == distances.getDimensions(),
                "measure.getDimensions() should be equal to distances.getDimensions()");
        checkArgument(measure.getDimensions() == probMass.getDimensions(),
                "measure.getDimensions() should be equal to probMass.getDimensions()");

        this.sampleSize = sampleSize;
        this.dimensions = measure.getDimensions();
        this.measure = measure;
        this.distances = distances;
        this.probMass = probMass;
    }

    /**
     * Add the values of {@link #measure}, {@link #distances}, and {@link #probMass}
     * from the right InterpolationMeasure to the left InterpolationMeasure and
     * return the left InterpolationMeasure. This method is used to accumulate
     * InterpolationMeasure results.
     *
     * @param left  The InterpolationMeasure we are modifying. After calling this
     *              method, fields in this InterpolationMeasure will contain a sum
     *              of the previous values and the corresponding values from the
     *              right InterpolationMeasure.
     * @param right An InterpolationMeasure that we want to add to the left vector.
     *              This InterpolationMeasure is not modified by the method.
     * @return the modified left vector.
     */
    public static InterpolationMeasure addToLeft(InterpolationMeasure left, InterpolationMeasure right) {
        checkNotNull(left, "left must not be null");
        checkNotNull(right, "right must not be null");
        checkArgument(left.dimensions == right.dimensions, "dimensions must be the same");
        left.sampleSize += right.sampleSize;
        DiVector.addToLeft(left.distances, right.distances);
        DiVector.addToLeft(left.measure, right.measure);
        DiVector.addToLeft(left.probMass, right.probMass);

        return left;
    }

    /**
     * Return a {@link Collector} which can be used to the collect many
     * InterpolationMeasure results into a single, final result.
     *
     * @param dimensions    The number of spatial dimensions in the
     *                      InterpolationMeasures being collected.
     * @param sampleSize    The sample size of the Random Cut Trees that were
     *                      measured.
     * @param numberOfTrees The number of trees whose measures we are collecting
     *                      into a final result. This value is used for scaling.
     * @return an interpolation measure containing the aggregated, scaled result.
     */
    public static Collector<InterpolationMeasure, InterpolationMeasure, InterpolationMeasure> collector(int dimensions,
            int sampleSize, int numberOfTrees) {
        return Collector.of(() -> new InterpolationMeasure(dimensions, sampleSize), InterpolationMeasure::addToLeft,
                InterpolationMeasure::addToLeft, result -> result.scale(1.0 / numberOfTrees));
    }

    /**
     * Return a new InterpolationMeasure will all values scaled by the given factor.
     *
     * @param z The scale factor.
     * @return a new InterpolationMeasure will all values scaled by the given
     *         factor.
     */
    public InterpolationMeasure scale(double z) {
        return new InterpolationMeasure((int) round(sampleSize * z), measure.scale(z), distances.scale(z),
                probMass.scale(z));
    }

    // InterpolationMeasure — in-place variants for the fold path
    public InterpolationMeasure scaleInPlace(double z) {
        sampleSize = (int) round(sampleSize * z);
        measure.scaleInPlace(z); // ASSUME DiVector has this; else measure = measure.scale(z) (allocates)
        distances.scaleInPlace(z);
        probMass.scaleInPlace(z);
        return this;
    }

    public void setSampleSize(int sampleSize) {
        this.sampleSize = sampleSize;
    }
}
