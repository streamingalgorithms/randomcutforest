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

package org.streamingalgorithms.randomcutforest;

import org.streamingalgorithms.randomcutforest.tree.ArrayBox;
import org.streamingalgorithms.randomcutforest.tree.ITree;

import lombok.Getter;

public abstract class RFVisitor<R> implements IRFVisitor<R> {
    protected float[] pointToScore;
    protected int treeMass; // the tree-bound scalar; sampleSize at full
    @Getter
    protected boolean pointInsideBox;
    protected ArrayBox shadowBox;
    protected boolean shadowBoxActive;

    @Override
    public final void prepare(ITree<?, ?> tree, float[] rawPoint) {
        float[] p = tree.projectToTree(rawPoint);
        if (p != pointToScore)
            System.arraycopy(p, 0, pointToScore, 0, pointToScore.length);
        this.treeMass = tree.getMass();
        reset();
    }

    protected abstract void reset();

    @Override
    public final boolean isConverged() {
        return pointInsideBox;
    }
}
