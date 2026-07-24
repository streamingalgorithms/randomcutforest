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

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.closeTo;
import static org.hamcrest.Matchers.is;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;
import static org.streamingalgorithms.randomcutforest.TestUtils.EPSILON;
import static org.streamingalgorithms.randomcutforest.tree.NodeStore.Null;

import java.util.Arrays;
import java.util.Random;

import org.junit.jupiter.api.Test;
import org.streamingalgorithms.randomcutforest.CommonUtils;
import org.streamingalgorithms.randomcutforest.RandomCutForest;
import org.streamingalgorithms.randomcutforest.store.IPointStoreView;
import org.streamingalgorithms.randomcutforest.store.PointStore;
import org.streamingalgorithms.randomcutforest.tree.*;

public class AnomalyScoreVisitorTest {

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
        when(leafNode.getBoundingBox()).thenReturn(new BoundingBox(point, point));
        when(leafNode.expanded()).thenReturn(new float[] { 1.0f, 2.0f, 3.0f, -1, -2, -3 });
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

        ScoreVisitor anotherVisitor = new ScoreVisitor(point, subSampleSize, 7);
        anotherVisitor.acceptLeaf(leafNode, 0);
        assertEquals(anotherVisitor.savedScore, visitor.savedScore);

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
        when(leafNode.getBoundingBox()).thenReturn(new BoundingBox(anotherPoint, anotherPoint));
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
        ScoreVisitor anotherVisitor = new ScoreVisitor(point, 2, 7);
        anotherVisitor.acceptLeaf(leafNode, 100);
        assertEquals(anotherVisitor.savedScore, visitor.savedScore);

        ScoreVisitor yetAnotherVisitor = new ScoreVisitor(point, 2, 12);
        yetAnotherVisitor.acceptLeaf(leafNode, 100);
        assertEquals(yetAnotherVisitor.savedScore, visitor.savedScore);

    }

    @Test
    public void testAcceptEqualsLeafPoint() {
        float[] pointToScore = { 0.0f, 0.0f };
        int sampleSize = 50;
        ScoreVisitor visitor = new ScoreVisitor(pointToScore, sampleSize);

        float[] point = Arrays.copyOf(pointToScore, pointToScore.length);
        INodeView node = mock(NodeView.class);
        when(node.getLeafPoint()).thenReturn(point);
        when(node.getBoundingBox()).thenReturn(new BoundingBox(point, point));
        when(node.expanded()).thenReturn(new float[4]);
        int depth = 2;
        visitor.acceptLeaf(node, depth);
        double expectedScore = CommonUtils.defaultDampFunction(node.getMass(), sampleSize)
                / (depth + Math.log(node.getMass() + 1) / Math.log(2));
        assertThat(visitor.getResult(),
                closeTo(CommonUtils.defaultScalarNormalizerFunction(expectedScore, sampleSize), EPSILON));

        depth--;
        IBoundingBoxView boundingBox = node.getBoundingBox().getMergedBox(new float[] { 1.0f, 1.0f });
        IPointStoreView<float[]> pointStoreView = new PointStore.Builder<>().dimensions(2).capacity(2).build();
        node = new NodeView(null, pointStoreView, Null, pointToScore);
        visitor.accept(node, depth);
        assertThat(visitor.getResult(),
                closeTo(CommonUtils.defaultScalarNormalizerFunction(expectedScore, sampleSize), EPSILON));

        depth--;
        boundingBox = boundingBox.getMergedBox(new float[] { -1.0f, -1.0f });
        node = new NodeView(null, pointStoreView, Null, pointToScore);
        visitor.accept(node, depth);
        assertThat(visitor.getResult(),
                closeTo(CommonUtils.defaultScalarNormalizerFunction(expectedScore, sampleSize), EPSILON));
    }

    @Test
    public void testScoreOnRealTree() {
        int dims = 2, sampleSize = 64;
        RandomCutForest forest = RandomCutForest.builder().dimensions(dims).numberOfTrees(1).sampleSize(sampleSize)
                .randomSeed(42).build();
        Random rng = new Random(7);
        for (int i = 0; i < 500; i++) {
            forest.update(new float[] { (float) rng.nextGaussian(), (float) rng.nextGaussian() });
        }
        double score = forest.getAnomalyScore(new float[] { 8.0f, 8.0f }); // clear outlier
        assertTrue(score > 1.0, "a far outlier should score > 1");

        double inlier = forest.getAnomalyScore(new float[] { 0.0f, 0.0f });
        assertTrue(inlier < score, "inlier scores below the outlier");
    }

    @Test
    public void test_getProbabilityOfSeparation_leafNode() {
        float[] point = new float[] { 1.0f, 2.0f, 3.0f };
        float[] leafPoint = Arrays.copyOf(point, point.length);
        BoundingBox boundingBox = new BoundingBox(leafPoint);

        ScoreVisitor visitor = new ScoreVisitor(point, 2);
        TransductiveScalarScoreVisitor esotericVisitor = new TransductiveScalarScoreVisitor(leafPoint, 2,
                CommonUtils::defaultScoreSeenFunction, CommonUtils::defaultScoreUnseenFunction,
                CommonUtils::defaultDampFunction, b -> new double[3]);
        assertThrows(IllegalStateException.class, () -> esotericVisitor.getProbabilityOfSeparation(boundingBox));

    }
}
