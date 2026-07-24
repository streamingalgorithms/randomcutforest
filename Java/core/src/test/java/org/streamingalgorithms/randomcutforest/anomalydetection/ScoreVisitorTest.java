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

package org.streamingalgorithms.randomcutforest.anomalydetection;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.closeTo;
import static org.hamcrest.Matchers.is;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.*;
import static org.streamingalgorithms.randomcutforest.TestUtils.EPSILON;

import org.junit.jupiter.api.Test;
import org.streamingalgorithms.randomcutforest.CommonUtils;
import org.streamingalgorithms.randomcutforest.tree.ArrayBox;
import org.streamingalgorithms.randomcutforest.tree.INodeView;
import org.streamingalgorithms.randomcutforest.tree.NodeView;
import org.streamingalgorithms.randomcutforest.tree.VectorSupport;

/**
 * Port of {@code AnomalyScoreVisitorTest} to the unified {@code ScoreVisitor}.
 *
 * <p>
 * The visitor no longer exposes {@code coordInsideBox},
 * {@code ignoreLeafEquals}, or {@code getProbabilityOfSeparation}: the
 * probability computation moved into {@code ArrayBox}/{@code NodeView}, and the
 * internal-node path now consumes
 * {@code node.probabilityOfSeparation(point, components)} directly. Tests that
 * pinned those members are either dropped (they tested a seam that no longer
 * exists) or rewritten to stub {@code probabilityOfSeparation} and assert the
 * observable result. The score recurrence itself is unchanged, so every
 * expected value carries over byte-for-byte.
 */
public class ScoreVisitorTest {

    @Test
    public void testNew() {
        float[] point = new float[] { 1.0f, 2.0f };
        int sampleSize = 9;
        ScoreVisitor visitor = new ScoreVisitor(point, sampleSize);

        assertFalse(visitor.isConverged());
        assertFalse(visitor.ignoreLeaf);
        assertEquals(0, visitor.ignoreLeafMassThreshold);
        assertThat(visitor.getResult(), is(0.0));
    }

    @Test
    public void testNewWithIgnoreOptions() {
        float[] point = new float[] { 1.0f, 2.0f };
        int sampleSize = 9;
        ScoreVisitor visitor = new ScoreVisitor(point, sampleSize, 7);

        assertFalse(visitor.isConverged());
        assertTrue(visitor.ignoreLeaf);
        assertEquals(7, visitor.ignoreLeafMassThreshold);
        assertThat(visitor.getResult(), is(0.0));
    }

    @Test
    public void testAcceptLeafEquals() {
        float[] point = { 1.0f, 2.0f, 3.0f };
        INodeView leafNode = mock(NodeView.class);
        when(leafNode.getLeafPoint()).thenReturn(point);
        when(leafNode.expanded()).thenReturn(new float[] { 1.0f, 2.0f, 3.0f, -1f, -2f, -3.0f });
        int leafDepth = 100;
        int leafMass = 10;
        when(leafNode.getMass()).thenReturn(leafMass);

        int subSampleSize = 21;
        ScoreVisitor visitor = new ScoreVisitor(point, subSampleSize);
        visitor.acceptLeaf(leafNode, leafDepth);
        double expectedScore = CommonUtils.defaultDampFunction(leafMass, subSampleSize)
                / (leafDepth + Math.log(leafMass + 1) / Math.log(2));
        assertThat(visitor.getResult(),
                closeTo(CommonUtils.defaultScalarNormalizerFunction(expectedScore, subSampleSize), EPSILON));
        assertTrue(visitor.isConverged());

        visitor = new ScoreVisitor(point, subSampleSize);
        visitor.acceptLeaf(leafNode, 0);
        expectedScore = CommonUtils.defaultDampFunction(leafMass, subSampleSize)
                / (Math.log(leafMass + 1) / Math.log(2.0));
        assertThat(visitor.getResult(),
                closeTo(CommonUtils.defaultScalarNormalizerFunction(expectedScore, subSampleSize), EPSILON));
        assertTrue(visitor.isConverged());

        // ignoreLeafMassThreshold = 7 < leafMass = 10 -> leaf is NOT ignored, so the
        // damped-seen score is used, matching the default visitor.
        ScoreVisitor anotherVisitor = new ScoreVisitor(point, subSampleSize, 7);
        anotherVisitor.acceptLeaf(leafNode, 0);
        assertEquals(anotherVisitor.savedScore, visitor.savedScore);

        // ignoreLeafMassThreshold = 12 >= leafMass = 10 -> leaf IS ignored, scoreUnseen
        // is used instead, so the base score differs.
        ScoreVisitor yetAnotherVisitor = new ScoreVisitor(point, subSampleSize, 12);
        yetAnotherVisitor.acceptLeaf(leafNode, 0);
        assertNotEquals(yetAnotherVisitor.savedScore, visitor.savedScore);
    }

    @Test
    public void testAcceptLeafNotEquals() {
        float[] point = new float[] { 1.0f, 2.0f, 3.0f };
        float[] anotherPoint = new float[] { 4.0f, 5.0f, 6.0f };
        INodeView leafNode = mock(NodeView.class);
        when(leafNode.getLeafPoint()).thenReturn(anotherPoint);
        when(leafNode.expanded()).thenReturn(new float[] { 1.0f, 2.0f, 3.0f, -1, -2, -3 });
        int leafDepth = 100;

        ScoreVisitor visitor = new ScoreVisitor(point, 2);
        visitor.acceptLeaf(leafNode, leafDepth);
        double expectedScore = 1.0 / (leafDepth + 1);
        assertThat(visitor.getResult(),
                closeTo(CommonUtils.defaultScalarNormalizerFunction(expectedScore, 2), EPSILON));
        assertFalse(visitor.isConverged());

        int leafMass = 10;
        when(leafNode.getMass()).thenReturn(leafMass);
        // when the point is not equal to the leaf, scoreUnseen is used regardless of
        // the
        // ignore threshold, so all three visitors agree.
        ScoreVisitor anotherVisitor = new ScoreVisitor(point, 2, 7);
        anotherVisitor.acceptLeaf(leafNode, 100);
        assertEquals(anotherVisitor.savedScore, visitor.savedScore);

        ScoreVisitor yetAnotherVisitor = new ScoreVisitor(point, 2, 12);
        yetAnotherVisitor.acceptLeaf(leafNode, 100);
        assertEquals(yetAnotherVisitor.savedScore, visitor.savedScore);
    }

    @Test
    public void testAccept() {
        float[] pointToScore = new float[] { 0.0f, 0.0f };
        int sampleSize = 50;
        ScoreVisitor visitor = new ScoreVisitor(pointToScore, sampleSize);

        float[] expandedPoint = new float[pointToScore.length * 2];
        VectorSupport.expandInto(pointToScore, 0, expandedPoint, 0, pointToScore.length);

        // ---- leaf, not equal to the query ----
        INodeView leafNode = mock(NodeView.class);
        float[] otherPoint = new float[] { 1.0f, 1.0f };
        when(leafNode.getLeafPoint()).thenReturn(otherPoint);
        when(leafNode.expanded()).thenReturn(expandedPoint);
        int depth = 4;
        visitor.acceptLeaf(leafNode, depth);
        double expectedScore = 1.0 / (depth + 1);
        assertThat(visitor.getResult(),
                closeTo(CommonUtils.defaultScalarNormalizerFunction(expectedScore, sampleSize), EPSILON));
    }

    /**
     * Regression guard for the mis-ported double-update: with a leaf ignored
     * (threshold &gt; mass) and an internal node whose separation probability is
     * positive, the score must be updated exactly once, not twice.
     */
    @Test
    public void testIgnoreLeafSingleUpdate() {
        float[] pointToScore = new float[] { 0.0f, 0.0f };
        int sampleSize = 50;
        int threshold = 12;
        ScoreVisitor visitor = new ScoreVisitor(pointToScore, sampleSize, threshold);

        float[] expandedPoint = new float[pointToScore.length * 2];
        VectorSupport.expandInto(pointToScore, 0, expandedPoint, 0, pointToScore.length);

        INodeView leafNode = mock(NodeView.class);
        float[] leafPoint = new float[] { 0.0f, 0.0f }; // equal to query
        when(leafNode.getLeafPoint()).thenReturn(leafPoint);
        when(leafNode.getMass()).thenReturn(3); // <= threshold -> ignored -> scoreUnseen
        when(leafNode.expanded()).thenReturn(expandedPoint);
        int depth = 3;
        visitor.acceptLeaf(leafNode, depth);
        double baseScore = visitor.savedScore; // scoreUnseen(depth, mass)

        // shadow path (ignoreLeaf true): one growing sibling box
        depth--;
        INodeView parent = mock(NodeView.class);
        int parentMass = 5;
        when(parent.getMass()).thenReturn(parentMass);
        when(parent.expanded()).thenReturn(expandedPoint);
        when(parent.getSiblingBoundingBox())
                .thenReturn(new ArrayBox(new float[] { 1.0f, 1.0f }, new float[] { 2.0f, 2.0f }));
        when(parent.probabilityAndSeparation(any(ArrayBox.class), any())).thenAnswer(inv -> {
            ArrayBox box = inv.getArgument(0);
            float[] comp = inv.getArgument(1);
            return box.probabilityOfCut(expandedPoint, comp, null);
        });

        visitor.accept(parent, depth);

        // expected: exactly one update score = p*scoreUnseen + (1-p)*baseScore
        ArrayBox shadow = new ArrayBox(new float[] { 1.0f, 1.0f }, new float[] { 2.0f, 2.0f });
        double p = shadow.probabilityOfCut(expandedPoint, null, null);
        double nodeScore = CommonUtils.defaultScoreUnseenFunction(depth, parentMass);
        double expected = p * nodeScore + (1 - p) * baseScore;
        assertEquals(expected, visitor.savedScore, EPSILON);
    }

    /**
     * Cross-check: on the same mock path, the sum of the attribution DiVector
     * equals the scalar score, confirming the dual accumulator stays consistent.
     */
    @Test
    public void testScoreMatchesAttributionSum() {
        float[] pointToScore = new float[] { 0.0f, 0.0f };
        int sampleSize = 50;

        float[] leafPoint = new float[] { 1.0f, -2.0f };
        ArrayBox parentBox = new ArrayBox(leafPoint, new float[] { 2.0f, -0.5f });

        ScoreVisitor scoreVisitor = new ScoreVisitor(pointToScore, sampleSize);
        AttributionVisitor attrVisitor = new AttributionVisitor(pointToScore, sampleSize);

        for (var visitor : new Object[] { scoreVisitor, attrVisitor }) {
            INodeView leaf = mock(NodeView.class);
            when(leaf.getLeafPoint()).thenReturn(leafPoint);
            when(leaf.getMass()).thenReturn(3);
            when(leaf.expanded()).thenReturn(new float[4]);
            INodeView parent = mock(NodeView.class);
            when(parent.getMass()).thenReturn(5);
            when(parent.expanded()).thenReturn(new float[4]);
            when(parent.probabilityAndSeparation(any(float[].class), any()))
                    .thenAnswer(inv -> parentBox.probabilityOfCut(inv.getArgument(0), inv.getArgument(1), null));
            if (visitor == scoreVisitor) {
                scoreVisitor.acceptLeaf(leaf, 4);
                scoreVisitor.accept(parent, 3);
            } else {
                attrVisitor.acceptLeaf(leaf, 4);
                attrVisitor.accept(parent, 3);
            }
        }

        double scalar = scoreVisitor.getResult();
        var attribution = attrVisitor.observeResult();
        double sum = 0;
        for (int i = 0; i < pointToScore.length; i++) {
            sum += attribution.high[i] + attribution.low[i];
        }
        assertEquals(scalar, sum, EPSILON);
    }
}
