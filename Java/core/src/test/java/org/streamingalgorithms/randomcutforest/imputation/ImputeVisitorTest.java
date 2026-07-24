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

package org.streamingalgorithms.randomcutforest.imputation;

import static org.junit.jupiter.api.Assertions.*;
import static org.streamingalgorithms.randomcutforest.CommonUtils.defaultScoreSeenFunction;
import static org.streamingalgorithms.randomcutforest.CommonUtils.defaultScoreUnseenFunction;

import java.util.HashMap;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.streamingalgorithms.randomcutforest.returntypes.ConditionalTreeSample;
import org.streamingalgorithms.randomcutforest.tree.ArrayBox;
import org.streamingalgorithms.randomcutforest.tree.IBoundingBoxView;
import org.streamingalgorithms.randomcutforest.tree.INodeView;

/**
 * Mechanics unit tests for the rewritten ImputeVisitor. No Mockito: a tiny
 * FakeNode implements the INodeView surface the visitor actually calls.
 * Reuse/prepare and the store-backed lift are exercised by the reuse-vs-legacy
 * parity/integration tests on a real forest; this suite pins the per-visitor
 * mechanics so a regression fails loudly.
 *
 * In-package so it can read the protected fields (queryPoint, anomalyRank, ...)
 * that getResult() no longer exposes (leafPoint is null now; coordinates live
 * in the store).
 */
public class ImputeVisitorTest {

    private static final double EPS = 1e-6;

    // second coordinate is the missing one; indices 99/-888 are out of the query
    // and
    // must be ignored by construction.
    private float[] queryPoint;
    private int[] missingIndexes;
    private ImputeVisitor visitor; // centrality = 1.0 (convenience ctor)

    @BeforeEach
    public void setUp() {
        queryPoint = new float[] { -1.0f, 1000.0f, 3.0f };
        missingIndexes = new int[] { 1 };
        visitor = new ImputeVisitor(queryPoint, 1, missingIndexes);
    }

    // ---- a minimal INodeView fake (settable fields; unused methods throw) ----
    // If INodeView declares members beyond these, add one-line stubs; the visitor
    // only
    // calls getBoundingBox / getMass / getLeafPoint / getLeafPointIndex /
    // getCutDimension.
    static final class FakeNode implements INodeView {
        ArrayBox boundingBox;
        int mass;
        float[] leafPoint;
        int leafPointIndex;
        int cutDimension;

        public int getMass() {
            return mass;
        }

        public IBoundingBoxView getBoundingBox() {
            return boundingBox;
        }

        public IBoundingBoxView getSiblingBoundingBox(float[] point) {
            throw new UnsupportedOperationException();
        }

        @Override
        public IBoundingBoxView getSiblingBoundingBox() {
            return null;
        }

        public int getCutDimension() {
            return cutDimension;
        }

        public double getCutValue() {
            throw new UnsupportedOperationException();
        }

        public float[] getLeafPoint() {
            return leafPoint;
        }

        public float[] getLiftedLeafPoint() {
            return leafPoint;
        }

        public int getLeafPointIndex() {
            return leafPointIndex;
        }

        @Override
        public float[] expanded() {
            return new float[0];
        }

        public boolean isLeaf() {
            throw new UnsupportedOperationException();
        }

        public HashMap<Long, Integer> getSequenceIndexes() {
            throw new UnsupportedOperationException();
        }

        public double probabilityAndSeparation(float[] point) {
            throw new UnsupportedOperationException();
        }

        @Override
        public double probabilityAndSeparation(ArrayBox box, float[] components) {
            return 0;
        }

        @Override
        public float[] separation(double[] ranges) {
            return new float[0];
        }

        public double probabilityAndSeparation(float[] point, float[] components) {
            throw new UnsupportedOperationException();
        }

    }

    private FakeNode leaf(float[] point, int index, int mass) {
        FakeNode n = new FakeNode();
        n.leafPoint = point;
        n.leafPointIndex = index;
        n.mass = mass;
        return n;
    }

    // ---- construction ----

    @Test
    public void testConstruction() {
        // query is copied, not aliased; pristine copy retained for reuse-restore
        assertNotSame(queryPoint, visitor.queryPoint);
        assertArrayEquals(queryPoint, visitor.queryPoint);
        assertNotSame(visitor.queryPoint, visitor.queryPointOriginal);
        assertArrayEquals(queryPoint, visitor.queryPointOriginal);

        assertTrue(visitor.missing[1]);
        assertFalse(visitor.missing[0]);
        assertFalse(visitor.missing[2]);

        assertEquals(ImputeVisitor.DEFAULT_INIT_VALUE, visitor.getAnomalyRank());
        assertEquals(Double.MAX_VALUE, visitor.getDistance());
        assertFalse(visitor.isConverged());
    }

    @Test
    public void testConstructionValidation() {
        assertThrows(IllegalArgumentException.class,
                () -> new ImputeVisitor(queryPoint, queryPoint, null, null, -1.0, 42));
        assertThrows(IllegalArgumentException.class,
                () -> new ImputeVisitor(queryPoint, queryPoint, null, null, 2.0, 42));
        assertThrows(IllegalArgumentException.class,
                () -> new ImputeVisitor(queryPoint, queryPoint, null, new int[] { -1 }, 1.0, 42));
        assertThrows(IllegalArgumentException.class,
                () -> new ImputeVisitor(queryPoint, queryPoint, null, new int[] { 3 }, 1.0, 42));
        assertDoesNotThrow(() -> new ImputeVisitor(queryPoint, queryPoint, null, null, 1.0, 42));
    }

    // ---- getResult contract: index + distance only; leafPoint/box null; fresh
    // object ----

    @Test
    public void testGetResultContract() {
        FakeNode leaf = leaf(new float[] { -1.0f, 2.0f, 3.0f }, 77, 10);
        visitor.acceptLeaf(leaf, 5);

        ConditionalTreeSample s = visitor.getResult();
        assertEquals(77, s.pointStoreIndex);
        assertEquals(visitor.getDistance(), s.distance, EPS);
        assertNull(s.leafPoint); // coordinates recovered from the store by index
        assertNull(s.parentOfLeafBox); // dead field
        assertNotSame(s, visitor.getResult()); // fresh sample each call
    }

    // ---- acceptLeaf: missing overwritten from leaf, distance = L1 over
    // non-missing ----

    @Test
    public void testAcceptLeafImputesMissingOnly() {
        // leaf differs from query at the missing coord (2) and matches elsewhere
        FakeNode leaf = leaf(new float[] { -1.0f, 2.0f, 3.0f }, 5, 10);
        visitor.acceptLeaf(leaf, 100);

        // missing coord took the leaf value; non-missing untouched
        assertArrayEquals(new float[] { -1.0f, 2.0f, 3.0f }, visitor.queryPoint);
        assertEquals(5, visitor.getResult().pointStoreIndex);
        assertEquals(0.0, visitor.getDistance(), EPS); // matches on all non-missing -> distance 0
        assertTrue(visitor.isConverged());
        assertEquals(defaultScoreSeenFunction(100, 10), visitor.getAnomalyRank(), EPS);
    }

    @Test
    public void testAcceptLeafExactMatchZeroDepth() {
        FakeNode leaf = leaf(new float[] { -1.0f, 2.0f, 3.0f }, 5, 10);
        visitor.acceptLeaf(leaf, 0);
        assertEquals(0.0, visitor.getDistance(), EPS);
        assertEquals(0.0, visitor.getAnomalyRank());
    }

    @Test
    public void testAcceptLeafMismatchScoresUnseen() {
        // differ at a NON-missing coord -> distance > 0 -> unseen score
        FakeNode leaf = leaf(new float[] { -1.0f, 2.0f, -111.11f }, 5, 10);
        visitor.acceptLeaf(leaf, 100);

        assertArrayEquals(new float[] { -1.0f, 2.0f, 3.0f }, visitor.queryPoint); // non-missing kept
        assertTrue(visitor.getDistance() > 0);
        assertEquals(defaultScoreUnseenFunction(100, 10), visitor.getAnomalyRank(), EPS);
    }

    // ---- accept: p==0 short-circuits (converged, rank unchanged); p>0 applies
    // recurrence ----

    @Test
    public void testAcceptConvergesWhenInsideBox() {
        visitor.acceptLeaf(leaf(new float[] { -1.0f, 2.0f, -50.0f }, 5, 10), 100);
        double oldRank = visitor.getAnomalyRank();

        // box that CONTAINS the (imputed) query -> probabilityOfCut == 0
        float[] lo = new float[] { -10, -10, -100 };
        float[] hi = new float[] { 10, 10, 100 };
        ArrayBox box = new ArrayBox(lo, hi);
        double p = box.probabilityOfCut(visitor.expandedPoint, null, null);
        assertEquals(0.0, p, EPS); // precondition: this box gives p==0

        FakeNode node = new FakeNode();
        node.boundingBox = box;
        node.mass = 12;
        visitor.accept(node, 99);

        assertTrue(visitor.isConverged());
        assertEquals(oldRank, visitor.getAnomalyRank(), EPS); // early return, no update
    }

    @Test
    public void testAcceptAppliesRecurrenceWhenSeparable() {
        visitor.acceptLeaf(leaf(new float[] { -1.0f, 2.0f, -50.0f }, 5, 10), 100);
        double oldRank = visitor.getAnomalyRank();

        // box far from the query -> probabilityOfCut > 0
        float[] lo = new float[] { 500, 500, 500 };
        float[] hi = new float[] { 600, 600, 600 };
        ArrayBox box = new ArrayBox(lo, hi);
        double p = box.probabilityOfCut(visitor.expandedPoint, null, null);
        assertTrue(p > 0); // precondition

        FakeNode node = new FakeNode();
        node.boundingBox = box;
        node.mass = 12;
        visitor.accept(node, 99);

        double expected = p * defaultScoreUnseenFunction(99, 12) + (1 - p) * oldRank;
        assertEquals(expected, visitor.getAnomalyRank(), EPS);
        assertFalse(visitor.isConverged());
    }

    // ---- trigger: splits on a missing cut dimension; counts the dimension ----

    @Test
    public void testTrigger() {
        FakeNode onMissing = new FakeNode();
        onMissing.cutDimension = 1; // the missing coord
        assertTrue(visitor.trigger(onMissing));
        assertEquals(1, visitor.dimensionsUsed[1]);

        FakeNode onPresent = new FakeNode();
        onPresent.cutDimension = 0;
        assertFalse(visitor.trigger(onPresent));
        assertEquals(1, visitor.dimensionsUsed[0]);
    }

    // ---- FORK: shares the working buffer (no copy). This is the load-bearing
    // invariant:
    // if someone reintroduces Arrays.copyOf in the fork ctor, this fails. ----

    @Test
    public void testForkSharesBuffer() {
        ImputeVisitor fork = (ImputeVisitor) visitor.newPartialCopy();

        assertSame(visitor.queryPoint, fork.queryPoint); // SHARED, not copied
        assertSame(visitor.queryPointOriginal, fork.queryPointOriginal);
        assertSame(visitor.missing, fork.missing);
        assertSame(visitor.dimensionsUsed, fork.dimensionsUsed);
        assertSame(visitor.rng, fork.rng);

        // fork starts with fresh scalar state
        assertEquals(ImputeVisitor.DEFAULT_INIT_VALUE, fork.getAnomalyRank());
        assertEquals(ImputeVisitor.DEFAULT_INIT_VALUE, fork.getDistance());
    }

    // ---- combine: pick the lower adjustedRank; copy scalars; no-op otherwise ----

    @Test
    public void testCombinePicksLowerAdjustedRank() {
        // centrality == 1.0 so adjustedRank == anomalyRank
        ImputeVisitor a = new ImputeVisitor(queryPoint, 1, missingIndexes);
        ImputeVisitor b = new ImputeVisitor(queryPoint, 1, missingIndexes);
        a.anomalyRank = 5.0;
        a.distance = 9.0;
        a.pointIndex = 1;
        b.anomalyRank = 2.0;
        b.distance = 4.0;
        b.pointIndex = 2;

        a.combine(b); // b is better (2 < 5) -> a adopts b's scalars
        assertEquals(2.0, a.getAnomalyRank(), EPS);
        assertEquals(4.0, a.getDistance(), EPS);
        assertEquals(2, a.getResult().pointStoreIndex);
    }

    @Test
    public void testCombineKeepsWhenNotBetter() {
        ImputeVisitor a = new ImputeVisitor(queryPoint, 1, missingIndexes);
        ImputeVisitor b = new ImputeVisitor(queryPoint, 1, missingIndexes);
        a.anomalyRank = 2.0;
        a.distance = 4.0;
        a.pointIndex = 1;
        b.anomalyRank = 5.0;
        b.distance = 9.0;
        b.pointIndex = 2;

        a.combine(b); // b is worse -> a unchanged
        assertEquals(2.0, a.getAnomalyRank(), EPS);
        assertEquals(4.0, a.getDistance(), EPS);
        assertEquals(1, a.getResult().pointStoreIndex);
    }
}
