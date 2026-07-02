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

package org.streamingalgorithms.randomcutforest.parkservices.state.returntypes;

import static org.streamingalgorithms.randomcutforest.CommonUtils.toDoubleArrayNullable;
import static org.streamingalgorithms.randomcutforest.CommonUtils.toFloatArrayNullable;

import org.streamingalgorithms.randomcutforest.parkservices.config.CorrectionMode;
import org.streamingalgorithms.randomcutforest.parkservices.config.ScoringStrategy;
import org.streamingalgorithms.randomcutforest.parkservices.returntypes.RCFComputeDescriptor;
import org.streamingalgorithms.randomcutforest.state.IStateMapper;
import org.streamingalgorithms.randomcutforest.state.returntypes.DiVectorMapper;

import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
public class ComputeDescriptorMapper implements IStateMapper<RCFComputeDescriptor, ComputeDescriptorState> {

    @Override
    public RCFComputeDescriptor toModel(ComputeDescriptorState state, long seed) {

        RCFComputeDescriptor descriptor = new RCFComputeDescriptor(state.getCurrentInput(), state.getInputTimeStamp());
        descriptor.setRCFScore(state.getScore());
        descriptor.setInternalTimeStamp(state.getInternalTimeStamp());
        descriptor.setAttribution(new DiVectorMapper().toModel(state.getAttribution()));
        descriptor.setRCFPoint(toFloatArrayNullable(state.getPoint()));
        descriptor.setExpectedRCFPoint(toFloatArrayNullable(state.getExpectedPoint()));
        descriptor.setRelativeIndex(state.getRelativeIndex());
        descriptor.setScoringStrategy(ScoringStrategy.valueOf(state.getStrategy()));
        descriptor.setShift(state.getShift());
        descriptor.setPostShift(state.getPostShift());
        descriptor.setTransformDecay(state.getTransformDecay());
        descriptor.setPostDeviations(state.getPostDeviations());
        descriptor.setScale(state.getScale());
        descriptor.setAnomalyGrade(state.getAnomalyGrade());
        descriptor.setThreshold(state.getThreshold());
        descriptor.setCorrectionMode(CorrectionMode.valueOf(state.getCorrectionMode()));
        return descriptor;
    }

    @Override
    public ComputeDescriptorState toState(RCFComputeDescriptor descriptor) {

        ComputeDescriptorState state = new ComputeDescriptorState();
        state.setInternalTimeStamp(descriptor.getInternalTimeStamp());
        state.setScore(descriptor.getRCFScore());
        state.setAttribution(new DiVectorMapper().toState(descriptor.getAttribution()));
        state.setPoint(toDoubleArrayNullable(descriptor.getRCFPoint()));
        state.setExpectedPoint(toDoubleArrayNullable(descriptor.getExpectedRCFPoint()));
        state.setRelativeIndex(descriptor.getRelativeIndex());
        state.setStrategy(descriptor.getScoringStrategy().name());
        state.setShift(descriptor.getShift());
        state.setPostShift(descriptor.getPostShift());
        state.setTransformDecay(descriptor.getTransformDecay());
        state.setPostDeviations(descriptor.getPostDeviations());
        state.setScale(descriptor.getScale());
        state.setAnomalyGrade(descriptor.getAnomalyGrade());
        state.setThreshold(descriptor.getThreshold());
        state.setCorrectionMode(descriptor.getCorrectionMode().name());
        state.setInputTimeStamp(descriptor.getInputTimestamp());
        state.setCurrentInput(descriptor.getCurrentInput());
        return state;
    }

}
