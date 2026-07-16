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

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.Test;
import org.streamingalgorithms.randomcutforest.tree.INodeView;
import org.streamingalgorithms.randomcutforest.tree.ITree;

class VisitorTest {

    private final IRFVisitor<String> defaultVisitor = new IRFMultiVisitor<String>() {
        @Override
        public boolean trigger(INodeView node) {
            return false;
        }

        @Override
        public MultiVisitor<String> newPartialCopy() {
            return null;
        }

        @Override
        public void combine(MultiVisitor<String> other) {

        }

        @Override
        public void accept(INodeView node, int depthOfNode) {
        }

        @Override
        public String getResult() {
            return "";
        }
    };

    private final IMultiVisitorFactory<String> defaultFactory = new IMultiVisitorFactory<String>() {
        @Override
        public MultiVisitor<String> newVisitor(ITree<?, ?> tree, float[] point) {
            return null;
        }
    };

    @Test
    void testPrepareThrowsByDefault() {
        assertThrows(UnsupportedOperationException.class, () -> {
            defaultVisitor.prepare(null, new float[] { 1.0f });
        });
    }

    @Test
    void testFoldIntoThrowsByDefault() {
        assertThrows(UnsupportedOperationException.class, () -> {
            defaultVisitor.foldInto("acc", null);
        });
    }

    @Test
    void testConvergingValueThrowsByDefault() {
        assertThrows(UnsupportedOperationException.class, () -> {
            defaultVisitor.convergingValue();
        });
    }

    @Test
    void testFoldOutThrowsByDefault() {
        assertThrows(UnsupportedOperationException.class, () -> {
            defaultVisitor.foldOut();
        });
    }

    @Test
    void testGetFoldResultThrowsByDefault() {
        assertThrows(UnsupportedOperationException.class, () -> {
            defaultVisitor.getFoldResult();
        });
    }

    @Test
    void testResetAcrossQueriesThrowsByDefault() {
        assertThrows(UnsupportedOperationException.class, () -> {
            defaultVisitor.resetAcrossQueries(new float[] { 1.0f });
        });
    }

    @Test
    void testPrepare() {
        assertThrows(UnsupportedOperationException.class, () -> {
            defaultVisitor.prepare(null, new float[] { 1.0f });
        });
    }

    @Test
    void testFactory() {
        assertFalse(defaultFactory.isReusable());
        assertFalse(defaultFactory.isFoldable());
        assertFalse(defaultFactory.isReusableAcrossQueries());
        assertThrows(UnsupportedOperationException.class, () -> defaultFactory.newReusableVisitor(new float[2]));
        assertThrows(UnsupportedOperationException.class, () -> defaultFactory.newReusableMultiVisitor(null));
        assertEquals(defaultFactory.liftResult(null, "test"), "test");
    }
}
