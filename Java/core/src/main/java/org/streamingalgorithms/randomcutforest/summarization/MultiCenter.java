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

package org.streamingalgorithms.randomcutforest.summarization;

import static org.streamingalgorithms.randomcutforest.CommonUtils.checkArgument;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.Function;
import java.util.function.ToDoubleBiFunction;

import org.streamingalgorithms.randomcutforest.util.Weighted;

public class MultiCenter extends GenericMultiCenter<float[]> {

    // the points assigned in the current pass, as parallel primitive arrays that
    // are reused across passes instead of a list of boxed entries
    int[] assignedIndex = new int[16];
    float[] assignedWeight = new float[16];
    // the distance of each assigned point to this cluster, as given to addPoint
    double[] assignedDist = new double[16];
    int assignedCount;

    MultiCenter(float[] coordinate, float weight, double shrinkage, int numberOfRepresentatives) {
        super(coordinate, weight, shrinkage, numberOfRepresentatives);
    }

    public static MultiCenter initialize(float[] coordinate, float weight, double shrinkage,
            int numberOfRepresentatives) {
        checkArgument(shrinkage >= 0 && shrinkage <= 1.0, " parameter has to be in [0,1]");
        checkArgument(numberOfRepresentatives > 0 && numberOfRepresentatives <= 100,
                " the number of representatives has to be in (0,100]");
        return new MultiCenter(coordinate, weight, shrinkage, numberOfRepresentatives);
    }

    public void addPoint(int index, float weight, double dist, float[] point,
            ToDoubleBiFunction<float[], float[]> distance) {
        super.addPoint(index, weight, dist, point, distance);
        keep(index, weight, dist);
    }

    /**
     * addPoint when the caller already knows the closest representative, by
     * position: the same updates as GenericMultiCenter.addPoint, without measuring
     * the point again. Used by Summarizer's block assignment.
     */
    void addPointAt(int index, float weight, double dist, int closest) {
        representatives.get(closest).weight += weight;
        this.weight += weight;
        this.sumOfRadius += weight * dist;
        keep(index, weight, dist);
    }

    private void keep(int index, float weight, double dist) {
        if (assignedCount == assignedIndex.length) {
            assignedIndex = Arrays.copyOf(assignedIndex, 2 * assignedCount);
            assignedWeight = Arrays.copyOf(assignedWeight, 2 * assignedCount);
            assignedDist = Arrays.copyOf(assignedDist, 2 * assignedCount);
        }
        assignedIndex[assignedCount] = index;
        assignedWeight[assignedCount] = weight;
        assignedDist[assignedCount++] = dist;
    }

    // the following sets up reassignment of the coordinate based on the points
    // assigned to the center
    public void reset() {
        super.reset();
        assignedCount = 0;
    }

    // a standard reassignment using the median values and NOT the mean; the mean is
    // unlikely to
    // provide robust convergence
    public double recompute(Function<Integer, float[]> getPoint, boolean force,
            ToDoubleBiFunction<float[], float[]> distanceFunction) {
        if (assignedCount == 0 || weight == 0.0 || !force) {
            return 0;
        }

        previousSumOFRadius = sumOfRadius;
        sumOfRadius = 0;
        for (int j = 0; j < assignedCount; j++) {
            // distance will check for -negative internally
            double addTerm = distance(getPoint.apply(assignedIndex[j]), distanceFunction) * assignedWeight[j];
            sumOfRadius += addTerm;
        }
        return (previousSumOFRadius - sumOfRadius);

    }

    /**
     * recompute(getPoint, true, distance) right after a block assignment: there the
     * distance of every assigned point to this cluster is the one addPointAt was
     * given, so the same sum is formed from those, in the same order.
     */
    double recomputeAssigned() {
        if (assignedCount == 0 || weight == 0.0) {
            return 0;
        }
        previousSumOFRadius = sumOfRadius;
        sumOfRadius = 0;
        for (int j = 0; j < assignedCount; j++) {
            double addTerm = assignedDist[j] * assignedWeight[j];
            sumOfRadius += addTerm;
        }
        return (previousSumOFRadius - sumOfRadius);
    }

    /**
     * A copy of the current assignment; the cluster itself keeps primitive arrays.
     */
    @Override
    public List<Weighted<Integer>> getAssignedPoints() {
        ArrayList<Weighted<Integer>> answer = new ArrayList<>(assignedCount);
        for (int j = 0; j < assignedCount; j++) {
            answer.add(new Weighted<>(assignedIndex[j], assignedWeight[j]));
        }
        return answer;
    }
}
