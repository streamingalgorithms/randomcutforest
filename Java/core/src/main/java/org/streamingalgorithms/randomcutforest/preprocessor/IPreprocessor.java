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

package org.streamingalgorithms.randomcutforest.preprocessor;

import org.streamingalgorithms.randomcutforest.RandomCutForest;
import org.streamingalgorithms.randomcutforest.config.ImputationMethod;
import org.streamingalgorithms.randomcutforest.returntypes.RangeVector;
import org.streamingalgorithms.randomcutforest.returntypes.SampleSummary;
import org.streamingalgorithms.randomcutforest.returntypes.TimedRangeVector;

public interface IPreprocessor {

    boolean isOutputReady();

    int getShingleSize();

    int getInputLength();

    float[] getLastShingledPoint();

    double[] getShift();

    double[] getScale();

    double[] getSmoothedDeviations();

    int getInternalTimeStamp();

    int getValuesSeen();

    ImputationMethod getImputationMethod();

    double dataQuality();

    float[] getScaledShingledInput(double[] point, long timestamp, int[] missing, RandomCutForest forest);

    // the following are as the methods above, written into buffer when it has
    // the length of the answer and into a new array otherwise (so null asks for
    // a new array); they are for callers that keep the arrays

    default double[] getShift(double[] buffer) {
        return copyInto(buffer, getShift());
    }

    default double[] getScale(double[] buffer) {
        return copyInto(buffer, getScale());
    }

    default double[] getSmoothedDeviations(double[] buffer) {
        return copyInto(buffer, getSmoothedDeviations());
    }

    default float[] getScaledShingledInput(double[] point, long timestamp, int[] missing, RandomCutForest forest,
            float[] buffer) {
        float[] answer = getScaledShingledInput(point, timestamp, missing, forest);
        if (answer == null || buffer == null || buffer.length != answer.length) {
            return answer;
        }
        System.arraycopy(answer, 0, buffer, 0, answer.length);
        return buffer;
    }

    private static double[] copyInto(double[] buffer, double[] answer) {
        if (answer == null || buffer == null || buffer.length != answer.length) {
            return answer;
        }
        System.arraycopy(answer, 0, buffer, 0, answer.length);
        return buffer;
    }

    SampleSummary invertInPlaceRecentSummaryBlock(SampleSummary summary);

    void update(double[] point, float[] rcfPoint, long timestamp, int[] missing, RandomCutForest forest);

    double[] getExpectedValue(int relativeBlockIndex, double[] reference, float[] point, float[] newPoint);

    double[] getShingledInput(int index);

    double[] getShingledInput();

    double[] getDefaultFill();

    void setDefaultFill(double[] fill);

    long getTimeStamp(int index);

    double getTransformDecay();

    int numberOfImputes(long timestamp);

    TimedRangeVector invertForecastRange(RangeVector ranges, long lastTimeStamp, double[] delta, boolean useExpected,
            long expectedTimeStamp);

}