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

package org.streamingalgorithms.randomcutforest.tree;

import java.util.HashMap;

public interface INodeView {

    float[] expanded();

    boolean isLeaf();

    int getMass();

    IBoundingBoxView getBoundingBox();

    IBoundingBoxView getSiblingBoundingBox(float[] point);

    // current sibling
    IBoundingBoxView getSiblingBoundingBox();

    int getCutDimension();

    double getCutValue();

    float[] getLeafPoint();

    default float[] getLiftedLeafPoint() {
        return getLeafPoint();
    };

    /**
     * for a leaf node, return the sequence indices corresponding leaf point. If
     * this method is invoked on a non-leaf node then it throws an
     * IllegalStateException.
     */
    HashMap<Long, Integer> getSequenceIndexes();

    // computes the probability from an expanded point and if the vector is non-null
    // fills it with the components over the 2*dimension half-dimensions
    // the first half correspond to the max values and the second half for min
    // values
    double probabilityAndSeparation(float[] point, float[] components);

    double probabilityAndSeparation(float[] components);

    double probabilityAndSeparation(ArrayBox box, float[] components);

    float[] separation(double[] ranges);

    /**
     * for a leaf node, return the index in the point store for the leaf point. If
     * this method is invoked on a non-leaf node then it throws an
     * IllegalStateException.
     */
    int getLeafPointIndex();

}
