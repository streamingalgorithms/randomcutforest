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

package org.streamingalgorithms.randomcutforest.parkservices;

import static org.streamingalgorithms.randomcutforest.CommonUtils.checkArgument;

import java.util.Arrays;

import org.streamingalgorithms.randomcutforest.parkservices.returntypes.RCFComputeDescriptor;

import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
public class AnomalyDescriptor extends RCFComputeDescriptor {

    public static int NUMBER_OF_EXPECTED_VALUES = 1;

    // confidence, for both anomalies/non-anomalies
    double dataConfidence;

    // flag indicating if the anomaly is the start of an anomaly or part of a run of
    // anomalies
    boolean startOfAnomaly;

    // flag indicating if the time stamp is in elevated score region to be
    // considered as anomaly
    boolean inHighScoreRegion;

    // a flattened version denoting the basic contribution of each input variable
    // (not shingled) for the
    // time slice indicated by relativeIndex
    double[] relevantAttribution;

    // when time is appended for the anomalous time slice
    double timeAttribution;

    // the values being replaced; may correspond to past
    double[] pastValues;

    // older timestamp if that is replaced
    long pastTimeStamp;

    // expected values, currently set to maximum 1
    double[][] expectedValuesList;

    // likelihood values for the list
    double[] likelihoodOfValues;

    public AnomalyDescriptor(double[] input, long inputTimeStamp) {
        super(input, inputTimeStamp);
    }

    /**
     * Readies the descriptor for reuse on a new input, as
     * {@link RCFComputeDescriptor#reset(double[], long)}; the fields of an anomaly
     * return to those of a new descriptor.
     *
     * @param input          the new input
     * @param inputTimeStamp its timestamp
     */
    @Override
    public void reset(double[] input, long inputTimeStamp) {
        super.reset(input, inputTimeStamp);
        dataConfidence = 0;
        startOfAnomaly = false;
        inHighScoreRegion = false;
        relevantAttribution = null;
        timeAttribution = 0;
        pastValues = null;
        pastTimeStamp = 0;
        expectedValuesList = null;
        likelihoodOfValues = null;
    }

    public void setPastValues(double[] values) {
        pastValues = copyInto(pastValues, values);
    }

    public boolean isExpectedValuesPresent() {
        return expectedValuesList != null;
    }

    public void setRelevantAttribution(double[] values) {
        this.relevantAttribution = copyInto(relevantAttribution, values);
    }

    public void setExpectedValues(int position, double[] values, double likelihood) {
        checkArgument(position < NUMBER_OF_EXPECTED_VALUES, "Increase size of expected array");
        if (expectedValuesList == null) {
            expectedValuesList = new double[NUMBER_OF_EXPECTED_VALUES][];
        }
        if (likelihoodOfValues == null) {
            likelihoodOfValues = new double[NUMBER_OF_EXPECTED_VALUES];
        }
        expectedValuesList[position] = Arrays.copyOf(values, values.length);
        likelihoodOfValues[position] = likelihood;
    }

    public void setDataConfidence(double timeDecay, long valuesSeen, long outputAfter, double dataQuality) {
        long total = valuesSeen;
        double lambda = timeDecay;
        double totalExponent = total * lambda;
        if (totalExponent == 0) {
            dataConfidence = 0.0;
        } else if (totalExponent >= 20) {
            dataConfidence = Math.min(1.0, dataQuality);
        } else {
            double eTotal = Math.exp(totalExponent);
            double confidence = dataQuality * (eTotal - Math.exp(lambda * Math.min(total, outputAfter))) / (eTotal - 1);
            dataConfidence = Math.max(0, confidence);
        }
    }

}