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

import java.util.Arrays;

import org.streamingalgorithms.randomcutforest.tree.ITree;

/**
 * Abstract base for reusable MultiVisitors. Analog of RFVisitor, but
 * deliberately thinner: it owns only what is universal to ANY reusable
 * multi-visitor -- the restore of the working query -- because that is the one
 * obligation the reuse lifecycle imposes on every multi-visitor, and (unlike
 * RFVisitor's pointToScore) it is a real hazard: the multi query is MUTATED
 * during the walk.
 *
 * With ImputeVisitor as the sole implementor today, everything else stays in
 * the concrete visitor; this base grows only when a second reusable
 * multi-visitor shares more state.
 */
public abstract class RFMultiVisitor<R> implements IRFMultiVisitor<R> {

    protected float[] pointToScore;
    protected final float[] pristinePoint;

    protected RFMultiVisitor(float[] projectedQuery) {
        this.pointToScore = Arrays.copyOf(projectedQuery, projectedQuery.length);
        this.pristinePoint = Arrays.copyOf(projectedQuery, projectedQuery.length);
    }

    protected RFMultiVisitor(RFMultiVisitor<R> original) {
        this.pointToScore = Arrays.copyOf(original.pointToScore, original.pointToScore.length);
        this.pristinePoint = original.pristinePoint;
    }

    @Override
    public final void prepare(ITree<?, ?> tree, float[] rawPoint) {
        System.arraycopy(pristinePoint, 0, pointToScore, 0, pointToScore.length);
        reset(tree);
    }

    protected abstract void reset(ITree<?, ?> tree);
}
