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

package org.streamingalgorithms.randomcutforest.parkservices.returntypes;

import static org.streamingalgorithms.randomcutforest.CommonUtils.checkArgument;

import java.util.Arrays;

import org.streamingalgorithms.randomcutforest.RandomCutForest;
import org.streamingalgorithms.randomcutforest.config.ForestMode;
import org.streamingalgorithms.randomcutforest.config.ImputationMethod;
import org.streamingalgorithms.randomcutforest.config.TransformMethod;
import org.streamingalgorithms.randomcutforest.inputtypes.Point;
import org.streamingalgorithms.randomcutforest.parkservices.config.CorrectionMode;
import org.streamingalgorithms.randomcutforest.parkservices.config.ScoringStrategy;
import org.streamingalgorithms.randomcutforest.preprocessor.IPreprocessor;
import org.streamingalgorithms.randomcutforest.returntypes.DiVector;

import lombok.Getter;
import lombok.Setter;

/**
 * a basic class that is used to store the internal state of the streaming
 * processing in ThresholdedRandomCutForest and others.
 */
@Getter
@Setter
public class RCFComputeDescriptor extends Point {

    ForestMode forestMode = ForestMode.STANDARD;

    TransformMethod transformMethod = TransformMethod.NONE;

    ImputationMethod imputationMethod = ImputationMethod.PREVIOUS;

    ScoringStrategy scoringStrategy = ScoringStrategy.EXPECTED_INVERSE_DEPTH;

    CorrectionMode correctionMode = CorrectionMode.NONE;

    // the most important parameter of the forest
    int shingleSize;

    // the actual dimensions
    int dimension;

    // the input length; useful for standalone analysis
    int inputLength;

    // sequence index (the number of updates to RCF) -- it is possible in imputation
    // that
    // the number of updates more than the input tuples seen by the overall program
    long totalUpdates;

    // determines if values can be input and or expected point calculated
    boolean reasonableForecast;

    // internal timestamp (basically a sequence index, but can be scaled and
    // jittered as in
    // the example);
    // kept as long for potential future use
    long internalTimeStamp;

    // number of trees in the forest
    int numberOfTrees;

    // current missing values, if any
    int[] missingValues;

    // potential number of imputes before processing current point
    int numberOfNewImputes;

    // actual, potentially transformed point on which compute occurs
    float[] RCFPoint;

    // score for various postprocessing
    double RCFScore;

    // the following describes the grade of the anomaly in the range [0:1] where
    // 0 is not an anomaly
    double anomalyGrade;

    // the threshold used in inference
    double threshold;

    // same for attribution; this is basic RCF attribution which has high/low
    // information
    DiVector attribution;

    /**
     * position of the anomaly vis a vis the current time (can be -ve) if anomaly is
     * detected late, which can and should happen sometime; for shingle size 1; this
     * is always 0
     */
    int relativeIndex;

    // useful for detecting noise
    double[] deviations;

    // useful for calibration in RCFCaster
    double[] postDeviations;

    // the multiplication factors to convert RCF representation to actuals/input
    double[] scale;

    // the addition performed (after multiplications) to convert RCF representation
    // to actuals/input
    double[] shift;

    // effects of a specific anomaly
    double[] postShift;

    // how long the effects last
    double transformDecay;

    // expected RCFPoint for the current point
    float[] expectedRCFPoint;

    // internal timestamp of last anomaly
    long lastAnomalyInternalTimestamp;

    // expected point of last anomaly
    float[] lastExpectedRCFPoint;

    // if the anomaly is due to timestamp when it is augmented only for current time
    long expectedTimeStamp;

    // used for streaming imputation
    double[][] imputedPoints;

    public RCFComputeDescriptor(double[] input, long inputTimeStamp) {
        super(input, inputTimeStamp);
    }

    public RCFComputeDescriptor(double[] input, long inputTimeStamp, ForestMode forestMode,
            TransformMethod transformMethod, ImputationMethod imputationMethod) {
        super(input, inputTimeStamp);
        this.forestMode = forestMode;
        this.transformMethod = transformMethod;
        this.imputationMethod = imputationMethod;
    }

    /**
     * Readies the descriptor for reuse on a new input: every field returns to the
     * value of a new descriptor, except the arrays that every processing of an
     * input sets (the transformed point, shift, scale, deviations, post deviations
     * and the expected point of the last anomaly), which are kept so that the
     * setters overwrite them in place. A setter writes into the array the
     * descriptor holds when the lengths agree and allocates one otherwise; null
     * clears the field.
     *
     * @param input          the new input
     * @param inputTimeStamp its timestamp
     */
    public void reset(double[] input, long inputTimeStamp) {
        resetInput(input, inputTimeStamp);
        forestMode = ForestMode.STANDARD;
        transformMethod = TransformMethod.NONE;
        imputationMethod = ImputationMethod.PREVIOUS;
        scoringStrategy = ScoringStrategy.EXPECTED_INVERSE_DEPTH;
        correctionMode = CorrectionMode.NONE;
        shingleSize = 0;
        dimension = 0;
        inputLength = 0;
        totalUpdates = 0;
        reasonableForecast = false;
        internalTimeStamp = 0;
        numberOfTrees = 0;
        missingValues = null;
        numberOfNewImputes = 0;
        RCFScore = 0;
        anomalyGrade = 0;
        threshold = 0;
        attribution = null;
        relativeIndex = 0;
        postShift = null;
        transformDecay = 0;
        expectedRCFPoint = null;
        lastAnomalyInternalTimestamp = 0;
        expectedTimeStamp = 0;
        imputedPoints = null;
    }

    public void setShift(double[] shift) {
        this.shift = copyInto(this.shift, shift);
    }

    public void setPostShift(double[] shift) {
        this.postShift = copyInto(this.postShift, shift);
    }

    public double[] getShift() {
        return copyIfNotnull(shift);
    }

    public void setScale(double[] scale) {
        this.scale = copyInto(this.scale, scale);
    }

    public double[] getScale() {
        return copyIfNotnull(scale);
    }

    public void setDeviations(double[] deviations) {
        this.deviations = copyInto(this.deviations, deviations);
    }

    public void setPostDeviations(double[] deviations) {
        this.postDeviations = copyInto(this.postDeviations, deviations);
    }

    /**
     * The scale, shift and deviations of the preprocessor, written into the arrays
     * the descriptor holds (allocated when they are null or of another length).
     *
     * @param preprocessor the preprocessor
     */
    public void readTransform(IPreprocessor preprocessor) {
        scale = preprocessor.getScale(scale);
        shift = preprocessor.getShift(shift);
        deviations = preprocessor.getSmoothedDeviations(deviations);
    }

    /**
     * The deviations of the preprocessor after its update, written into the array
     * the descriptor holds; null until the preprocessor has output.
     *
     * @param preprocessor the preprocessor
     */
    public void readPostDeviations(IPreprocessor preprocessor) {
        postDeviations = preprocessor.isOutputReady() ? preprocessor.getSmoothedDeviations(postDeviations) : null;
    }

    /**
     * The point the forest sees for this input, its timestamp and missing values,
     * written into the array the descriptor holds; null when the preprocessor has
     * none yet.
     *
     * @param preprocessor the preprocessor
     * @param forest       the forest
     */
    public void readRCFPoint(IPreprocessor preprocessor, RandomCutForest forest) {
        RCFPoint = preprocessor.getScaledShingledInput(currentInput, inputTimestamp, missingValues, forest, RCFPoint);
    }

    public double[] getDeltaShift() {
        if (shift == null || postShift == null) {
            return null;
        }
        double[] answer = new double[shift.length];
        for (int i = 0; i < shift.length; i++) {
            answer[i] = postShift[i] - shift[i];
        }
        return answer;
    }

    public void setExpectedRCFPoint(float[] point) {
        expectedRCFPoint = copyInto(expectedRCFPoint, point);
    }

    public float[] getExpectedRCFPoint() {
        return copyIfNotnull(expectedRCFPoint);
    }

    public void setRCFPoint(float[] point) {
        RCFPoint = copyInto(RCFPoint, point);
    }

    public float[] getRCFPoint() {
        return copyIfNotnull(RCFPoint);
    }

    /**
     * @return whether there is a transformed point, as getRCFPoint() != null
     *         without the copy
     */
    public boolean isRCFPointPresent() {
        return RCFPoint != null;
    }

    /**
     * @return whether there is an expected point, as getExpectedRCFPoint() != null
     *         without the copy
     */
    public boolean isExpectedRCFPointPresent() {
        return expectedRCFPoint != null;
    }

    public void setLastExpectedRCFPoint(float[] point) {
        lastExpectedRCFPoint = copyInto(lastExpectedRCFPoint, point);
    }

    public void setLastExpectedRCFdPoint(float[] point) {
        setLastExpectedRCFPoint(point);
    }

    public float[] getLastExpectedRCFPoint() {
        return copyIfNotnull(lastExpectedRCFPoint);
    }

    public void setAttribution(DiVector attribution) {
        if (attribution == null) {
            this.attribution = null;
        } else if (this.attribution != null && this.attribution.getDimensions() == attribution.getDimensions()) {
            System.arraycopy(attribution.high, 0, this.attribution.high, 0, attribution.getDimensions());
            System.arraycopy(attribution.low, 0, this.attribution.low, 0, attribution.getDimensions());
        } else {
            this.attribution = new DiVector(attribution);
        }
    }

    public DiVector getAttribution() {
        return (attribution == null) ? null : new DiVector(attribution);
    }

    public int[] getMissingValues() {
        return (missingValues == null) ? null : Arrays.copyOf(missingValues, missingValues.length);
    }

    public void setMissingValues(int[] values) {
        missingValues = (values == null) ? null : Arrays.copyOf(values, values.length);
    }

    protected float[] copyIfNotnull(float[] array) {
        return array == null ? null : Arrays.copyOf(array, array.length);
    }

    /**
     * A copy of source: written into target when the lengths agree, a new array
     * otherwise, and null when source is null.
     *
     * @param target the array to reuse, or null
     * @param source the values
     * @return the copy
     */
    protected static float[] copyInto(float[] target, float[] source) {
        if (source == null) {
            return null;
        }
        if (target == null || target.length != source.length) {
            return Arrays.copyOf(source, source.length);
        }
        System.arraycopy(source, 0, target, 0, source.length);
        return target;
    }

    public void setImputedPoint(int index, double[] impute) {
        checkArgument(numberOfNewImputes > 0, " no imputation is indicated");
        checkArgument(impute != null && impute.length == inputLength, "incorrect length");
        if (imputedPoints == null) {
            imputedPoints = new double[Math.min(numberOfNewImputes, shingleSize - 1)][];
        }
        checkArgument(imputedPoints.length > index && index >= 0 && imputedPoints[index] == null, "already set!");
        imputedPoints[index] = Arrays.copyOf(impute, inputLength);
    }

    // an explicit copy operation to control the stored state
    public RCFComputeDescriptor copyOf() {
        return copyOf(null);
    }

    /**
     * As {@link #copyOf()}, written into target when it is not null: target is
     * reset and its arrays are overwritten in place when the lengths agree. The
     * copy is a separate object from this descriptor; target must not be this.
     *
     * @param target a descriptor to overwrite, or null for a new one
     * @return the copy: target, if one was given
     */
    public RCFComputeDescriptor copyOf(RCFComputeDescriptor target) {
        checkArgument(target != this, "a descriptor cannot be copied onto itself");
        RCFComputeDescriptor answer;
        if (target == null) {
            answer = new RCFComputeDescriptor(currentInput, inputTimestamp, forestMode, transformMethod,
                    imputationMethod);
        } else {
            answer = target;
            answer.reset(currentInput, inputTimestamp);
            answer.setForestMode(forestMode);
            answer.setTransformMethod(transformMethod);
            answer.setImputationMethod(imputationMethod);
            // a copy carries neither (they are kept by reset for processing)
            answer.setDeviations(null);
            answer.setPostDeviations(null);
        }
        answer.setShingleSize(shingleSize);
        answer.setDimension(dimension);
        answer.setInputLength(inputLength);
        answer.setReasonableForecast(reasonableForecast);
        answer.setAttribution(attribution);
        answer.setRCFPoint(RCFPoint);
        answer.setRCFScore(RCFScore);
        answer.setInternalTimeStamp(internalTimeStamp);
        answer.setExpectedRCFPoint(expectedRCFPoint);
        answer.setNumberOfTrees(numberOfTrees);
        answer.setTotalUpdates(totalUpdates);
        answer.setNumberOfNewImputes(numberOfNewImputes);
        answer.setLastAnomalyInternalTimestamp(lastAnomalyInternalTimestamp);
        answer.setLastExpectedRCFPoint(lastExpectedRCFPoint);
        answer.setScoringStrategy(scoringStrategy);
        answer.setShift(shift);
        answer.setScale(scale);
        answer.setPostShift(postShift);
        answer.setTransformDecay(transformDecay);
        answer.setAnomalyGrade(anomalyGrade);
        answer.setThreshold(threshold);
        answer.setCorrectionMode(correctionMode);
        return answer;
    }
}