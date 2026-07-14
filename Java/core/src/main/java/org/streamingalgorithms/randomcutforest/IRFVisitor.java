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

import org.streamingalgorithms.randomcutforest.tree.ITree;

public interface IRFVisitor<R> extends Visitor<R> {
    default void prepare(ITree<?, ?> tree, float[] rawPoint) {
        throw new UnsupportedOperationException("not a reusable visitor");
    }

    default R foldInto(R acc, ITree<?, ?> tree) {
        throw new UnsupportedOperationException("not foldable");
    }

    default double convergingValue() { // for approximate/early-stop forests
        throw new UnsupportedOperationException("not a converging visitor");
    }

    // IRFVisitor
    default void foldOut() {
        throw new UnsupportedOperationException("not foldable");
    }

    // IRFVisitor
    default R getFoldResult() {
        throw new UnsupportedOperationException("not foldable");
    }

    default void resetAcrossQueries(float[] point) {
        throw new UnsupportedOperationException("not reusable yet");
    }
}
