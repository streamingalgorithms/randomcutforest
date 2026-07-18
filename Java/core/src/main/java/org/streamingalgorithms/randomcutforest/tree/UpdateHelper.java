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

package org.streamingalgorithms.randomcutforest.tree;

import lombok.Getter;

@Getter
public class UpdateHelper<P> {
    protected final int dimension;
    protected final float[] pointScratch;
    protected int[] nodeScratch; // trees of different sizes can reassign if needed
    protected final Cut cutScratch;
    protected final ArrayBox boxScratch;
    protected final ArrayBox updateBox;

    public UpdateHelper(int dimension) {
        this(dimension, 256);
    }

    public UpdateHelper(int dimension, int maxNumberOfLeaves) {
        this.dimension = dimension;
        pointScratch = new float[dimension];
        nodeScratch = new int[maxNumberOfLeaves];
        cutScratch = new Cut(0, 0);
        boxScratch = new ArrayBox(dimension);
        updateBox = new ArrayBox(dimension);
    }

    public void resize(int treeSize) {
        if (treeSize > nodeScratch.length) {
            nodeScratch = new int[treeSize];
        }
    }

}
