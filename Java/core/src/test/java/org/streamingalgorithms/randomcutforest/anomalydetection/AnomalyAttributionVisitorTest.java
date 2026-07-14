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

package org.streamingalgorithms.randomcutforest.anomalydetection;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;
import static org.streamingalgorithms.randomcutforest.CommonUtils.defaultScalarNormalizerFunction;
import static org.streamingalgorithms.randomcutforest.CommonUtils.defaultScoreUnseenFunction;
import static org.streamingalgorithms.randomcutforest.TestUtils.EPSILON;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.streamingalgorithms.randomcutforest.CommonUtils;
import org.streamingalgorithms.randomcutforest.returntypes.DiVector;
import org.streamingalgorithms.randomcutforest.tree.ArrayBox;
import org.streamingalgorithms.randomcutforest.tree.BoundingBox;
import org.streamingalgorithms.randomcutforest.tree.INodeView;
import org.streamingalgorithms.randomcutforest.tree.NodeView;

public class AnomalyAttributionVisitorTest {

    @Test
    public void testNew() {
        float[] point = new float[] { 1.1f, -2.2f, 3.3f };
        int treeMass = 99;
        AttributionVisitor visitor = new AttributionVisitor(point, treeMass);

        assertFalse(visitor.isConverged());

        assertFalse(visitor.ignoreLeaf);
        assertEquals(0, visitor.ignoreLeafMassThreshold);
        DiVector result = visitor.getResult();
        double[] zero = new double[point.length];
        assertArrayEquals(zero, result.high);
        assertArrayEquals(zero, result.low);
    }

    @Test
    public void testNewWithIgnoreOptions() {
        float[] point = new float[] { 1.1f, -2.2f, 3.3f };
        int treeMass = 99;
        AttributionVisitor visitor = new AttributionVisitor(point, treeMass, 7);

        assertFalse(visitor.isConverged());

        assertTrue(visitor.ignoreLeaf);
        assertEquals(7, visitor.ignoreLeafMassThreshold);
        DiVector result = visitor.getResult();
        double[] zero = new double[point.length];
        assertArrayEquals(zero, result.high);
        assertArrayEquals(zero, result.low);
    }

    @Test
    public void testAcceptLeafEquals() {
        float[] point = { 1.1f, -2.2f, 3.3f };
        INodeView leafNode = mock(NodeView.class);
        when(leafNode.getLeafPoint()).thenReturn(point);
        when(leafNode.getBoundingBox()).thenReturn(new BoundingBox(point, point));

        int leafDepth = 100;
        int leafMass = 10;
        when(leafNode.getMass()).thenReturn(leafMass);

        int treeMass = 21;
        AttributionVisitor visitor = new AttributionVisitor(point, treeMass, 0);
        visitor.acceptLeaf(leafNode, leafDepth);

        assertTrue(visitor.hitDuplicates);
        double expectedScoreSum = CommonUtils.defaultDampFunction(leafMass, treeMass)
                / (leafDepth + Math.log(leafMass + 1) / Math.log(2));
        double expectedScore = expectedScoreSum / (2 * point.length);
        DiVector result = visitor.observeResult();
        for (int i = 0; i < point.length; i++) {
            assertEquals(defaultScalarNormalizerFunction(expectedScore, treeMass), result.low[i], EPSILON);
            assertEquals(defaultScalarNormalizerFunction(expectedScore, treeMass), result.high[i], EPSILON);
        }
    }

    @Test
    public void testAcceptLeafNotEquals() {
        float[] point = new float[] { 1.1f, -2.2f, 3.3f };
        float[] anotherPoint = new float[] { -4.0f, 5.0f, 6.0f };

        INodeView leafNode = mock(NodeView.class);
        when(leafNode.getLeafPoint()).thenReturn(anotherPoint);
        when(leafNode.getBoundingBox()).thenReturn(new BoundingBox(anotherPoint, anotherPoint));
        int leafDepth = 100;
        int leafMass = 4;
        when(leafNode.getMass()).thenReturn(leafMass);

        int treeMass = 21;
        AttributionVisitor visitor = new AttributionVisitor(point, treeMass, 0);
        visitor.acceptLeaf(leafNode, leafDepth);

        double expectedScoreSum = defaultScoreUnseenFunction(leafDepth, leafMass);
        double sumOfNewRange = (1.1 - (-4.0)) + (5.0 - (-2.2)) + (6.0 - 3.3);

        DiVector result = visitor.observeResult();
        assertEquals(defaultScalarNormalizerFunction(expectedScoreSum * (1.1 - (-4.0)) / sumOfNewRange, treeMass),
                result.high[0], EPSILON);
        assertEquals(0.0, result.low[0]);
        assertEquals(0.0, result.high[1]);
        assertEquals(defaultScalarNormalizerFunction(expectedScoreSum * (5.0 - (-2.2)) / sumOfNewRange, treeMass),
                result.low[1], EPSILON);
        assertEquals(0.0, result.high[2]);
        assertEquals(defaultScalarNormalizerFunction(expectedScoreSum * (6.0 - 3.3) / sumOfNewRange, treeMass),
                result.low[2], EPSILON);

        visitor = new AttributionVisitor(point, treeMass, 3);
        visitor.acceptLeaf(leafNode, leafDepth);
        result = visitor.observeResult();
        assertEquals(defaultScalarNormalizerFunction(expectedScoreSum * (1.1 - (-4.0)) / sumOfNewRange, treeMass),
                result.high[0], EPSILON);
        assertEquals(0.0, result.low[0]);
        assertEquals(0.0, result.high[1]);
        assertEquals(defaultScalarNormalizerFunction(expectedScoreSum * (5.0 - (-2.2)) / sumOfNewRange, treeMass),
                result.low[1], EPSILON);
        assertEquals(0.0, result.high[2]);
        assertEquals(defaultScalarNormalizerFunction(expectedScoreSum * (6.0 - 3.3) / sumOfNewRange, treeMass),
                result.low[2], EPSILON);

        visitor = new AttributionVisitor(point, treeMass, 4);
        visitor.acceptLeaf(leafNode, leafDepth);
        double expectedScore = expectedScoreSum / (2 * point.length);
        result = visitor.observeResult();
        for (int i = 0; i < point.length; i++) {
            assertEquals(defaultScalarNormalizerFunction(expectedScore, treeMass), result.low[i], EPSILON);
            assertEquals(defaultScalarNormalizerFunction(expectedScore, treeMass), result.high[i], EPSILON);
        }
    }

    @Test
    public void testAccept() {
        float[] pointToScore = { 0.0f, 0.0f };
        int treeMass = 50;
        AttributionVisitor visitor = new AttributionVisitor(pointToScore, treeMass, 0);

        // ---- leaf ----
        INodeView leafNode = mock(NodeView.class);
        float[] point = new float[] { 1.0f, -2.0f };
        when(leafNode.getLeafPoint()).thenReturn(point); // acceptLeaf now uses getLeafPoint, not getBoundingBox
        int leafMass = 3;
        when(leafNode.getMass()).thenReturn(leafMass);
        int depth = 4;
        visitor.acceptLeaf(leafNode, depth);
        DiVector result = visitor.observeResult(); // non-destructive peek

        double expectedScoreSum = defaultScoreUnseenFunction(depth, leafNode.getMass());
        double sumOfNewRange = 1.0 + 2.0;
        double[] expectedUnnormalizedLow = new double[] { expectedScoreSum * 1.0 / sumOfNewRange, 0.0 };
        double[] expectedUnnormalizedHigh = new double[] { 0.0, expectedScoreSum * 2.0 / sumOfNewRange };

        for (int i = 0; i < pointToScore.length; i++) {
            assertEquals(defaultScalarNormalizerFunction(expectedUnnormalizedLow[i], treeMass), result.low[i], EPSILON);
            assertEquals(defaultScalarNormalizerFunction(expectedUnnormalizedHigh[i], treeMass), result.high[i],
                    EPSILON);
        }

        // ---- parent does not contain pointToScore ----
        // The visitor now asks the node for probabilityOfSeparation; stub that
        // (delegating
        // to the box) instead of getBoundingBox(), which the new accept no longer
        // calls.
        depth--;
        INodeView parent = mock(NodeView.class);
        int parentMass = leafMass + 2; // sibling mass = 2
        when(parent.getMass()).thenReturn(parentMass);
        ArrayBox parentBox = new ArrayBox(point, new float[] { 2.0f, -0.5f });
        when(parent.probabilityOfSeparationSimd(any(), any()))
                .thenAnswer(inv -> parentBox.probabilityOfCutSimd(inv.getArgument(0), inv.getArgument(1)));

        visitor.accept(parent, depth);
        result = visitor.observeResult();

        double expectedSumOfNewRange2 = 2.0 + 2.0;
        double expectedProbOfCut2 = (1.0 + 0.5) / expectedSumOfNewRange2;
        double[] expectedDifferenceInRangeVector2 = { 0.0, 1.0, 0.5, 0.0 };

        double expectedScore2 = defaultScoreUnseenFunction(depth, parent.getMass());
        double[] expectedUnnormalizedLow2 = new double[pointToScore.length];
        double[] expectedUnnormalizedHigh2 = new double[pointToScore.length];

        for (int i = 0; i < pointToScore.length; i++) {
            double prob = expectedDifferenceInRangeVector2[2 * i] / expectedSumOfNewRange2;
            expectedUnnormalizedHigh2[i] = prob * expectedScore2
                    + (1 - expectedProbOfCut2) * expectedUnnormalizedHigh[i];

            prob = expectedDifferenceInRangeVector2[2 * i + 1] / expectedSumOfNewRange2;
            expectedUnnormalizedLow2[i] = prob * expectedScore2 + (1 - expectedProbOfCut2) * expectedUnnormalizedLow[i];
        }

        for (int i = 0; i < pointToScore.length; i++) {
            assertEquals(defaultScalarNormalizerFunction(expectedUnnormalizedLow2[i], treeMass), result.low[i],
                    EPSILON);
            assertEquals(defaultScalarNormalizerFunction(expectedUnnormalizedHigh2[i], treeMass), result.high[i],
                    EPSILON);
        }

        // ---- grandparent contains pointToScore -> converge, values frozen ----
        assertFalse(visitor.isConverged()); // not yet converged after the parent
        depth--;
        INodeView grandParent = mock(NodeView.class);
        when(grandParent.getMass()).thenReturn(parentMass + 2);
        ArrayBox grandParentBox = parentBox
                .getMergedBox(new ArrayBox(new float[] { -1.0f, 1.0f }).getMergedBox(new float[] { -0.5f, -1.5f }));
        when(grandParent.probabilityOfSeparationSimd(any(), any()))
                .thenAnswer(inv -> grandParentBox.probabilityOfCutSimd(inv.getArgument(0), inv.getArgument(1)));

        visitor.accept(grandParent, depth);
        assertTrue(visitor.isConverged()); // NOW it converged — the containment short-circuit fired
        result = visitor.observeResult();

        for (int i = 0; i < pointToScore.length; i++) {
            assertEquals(defaultScalarNormalizerFunction(expectedUnnormalizedLow2[i], treeMass), result.low[i],
                    EPSILON);
            assertEquals(defaultScalarNormalizerFunction(expectedUnnormalizedHigh2[i], treeMass), result.high[i],
                    EPSILON);
        }
    }

    @ParameterizedTest
    @ValueSource(ints = { 3, 5 })
    public void reNormalizeNotEqual(int mass) {
        float[] pointToScore = { 0.0f, 0.0f };
        int treeMass = 50;
        AttributionVisitor visitor = new AttributionVisitor(pointToScore, treeMass, 4);

        INodeView leafNode = mock(NodeView.class);
        float[] point = new float[] { 1.0f, -2.0f };
        when(leafNode.getLeafPoint()).thenReturn(point);
        when(leafNode.getBoundingBox()).thenReturn(new ArrayBox(point));

        int leafMass = mass;
        when(leafNode.getMass()).thenReturn(leafMass);
        visitor.acceptLeaf(leafNode, 1);
        INodeView parent = mock(NodeView.class);
        int parentMass = leafMass + 2;
        when(parent.getMass()).thenReturn(parentMass);
        ArrayBox boundingBox = new ArrayBox(point, new float[] { 2.0f, 2.0f });
        when(parent.getBoundingBox()).thenReturn(boundingBox);
        when(parent.getSiblingBoundingBox(any())).thenReturn(new ArrayBox(new float[] { 2.0f, 2.0f }));
        visitor.accept(parent, 0);
        DiVector result = new DiVector(visitor.directionalAttribution);
        assertEquals(result.getHighLowSum(), visitor.savedScore, 1e-6);
    }

    @ParameterizedTest
    @ValueSource(ints = { 3, 5 })
    public void reNormalize(int mass) {
        float[] pointToScore = { 0.0f, 0.0f };
        int treeMass = 50;
        AttributionVisitor visitor = new AttributionVisitor(pointToScore, treeMass, 4);

        INodeView leafNode = mock(NodeView.class);
        float[] point = pointToScore;
        when(leafNode.getLeafPoint()).thenReturn(point);
        when(leafNode.getBoundingBox()).thenReturn(new BoundingBox(point, point));

        int leafMass = mass;
        when(leafNode.getMass()).thenReturn(leafMass);
        visitor.acceptLeaf(leafNode, 1);
        INodeView parent = mock(NodeView.class);
        int parentMass = leafMass + 2;
        when(parent.getMass()).thenReturn(parentMass);
        ArrayBox boundingBox = new ArrayBox(point, new float[] { 2.0f, 2.0f });
        when(parent.getBoundingBox()).thenReturn(boundingBox);
        when(parent.getSiblingBoundingBox(any())).thenReturn(new ArrayBox(new float[] { 2.0f, 2.0f }));
        visitor.accept(parent, 0);
        DiVector result = new DiVector(visitor.directionalAttribution);
        assertEquals(result.getHighLowSum(), visitor.savedScore, 1e-6);
    }

}
