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

package org.streamingalgorithms.randomcutforest.interpolation;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import java.util.Arrays;

import org.junit.jupiter.api.Test;
import org.streamingalgorithms.randomcutforest.returntypes.InterpolationMeasure;
import org.streamingalgorithms.randomcutforest.tree.ArrayBox;
import org.streamingalgorithms.randomcutforest.tree.INodeView;
import org.streamingalgorithms.randomcutforest.tree.NodeView;

/**
 * Tests for the ArrayBox / two-pass {@code InterpolationVisitor}.
 *
 * <p>
 * The centerpiece is {@link #testParityWithSimple()}: it runs the new visitor
 * and the legacy {@code SimpleInterpolationVisitor} over the same mock path and
 * asserts coordinate-wise equality across all six accumulators. That is the
 * real guarantee that the box/allocation rewrite preserved behavior; the
 * remaining tests are the ported unit cases, retained so a failure localizes.
 *
 * <p>
 * Mocks stub {@code getBoundingBox}/{@code getSiblingBoundingBox} to return
 * {@code ArrayBox} (the new visitor casts the sibling box to {@code ArrayBox}
 * and reads the node box through {@code IBoundingBoxView}); the legacy visitor
 * reads the same boxes through the interface, so a single stub feeds both.
 */
public class InterpolationVisitorTest {

    private static double TOL = 1e-9;

    // ---------- helpers ----------

    private static InterpolationVisitor newVisitor(float[] point, int sampleSize) {
        return new InterpolationVisitor(point, sampleSize, 1, false);
    }

    private static void assertMeasuresClose(InterpolationMeasure a, InterpolationMeasure b, double tol) {
        assertArrayClose(a.measure.high, b.measure.high, tol);
        assertArrayClose(a.measure.low, b.measure.low, tol);
        assertArrayClose(a.probMass.high, b.probMass.high, tol);
        assertArrayClose(a.probMass.low, b.probMass.low, tol);
        assertArrayClose(a.distances.high, b.distances.high, tol);
        assertArrayClose(a.distances.low, b.distances.low, tol);
    }

    private static void assertArrayClose(double[] expected, double[] actual, double tol) {
        assertEquals(expected.length, actual.length);
        for (int i = 0; i < expected.length; i++) {
            assertEquals(expected[i], actual[i], tol, "index " + i);
        }
    }

    // ---------- ported unit tests ----------

    @Test
    public void testNew() {
        float[] point = { 1.0f, 2.0f };
        int sampleSize = 9;
        InterpolationVisitor visitor = newVisitor(point, sampleSize);

        assertFalse(visitor.pointInsideBox);

        InterpolationMeasure output = visitor.getResult();
        double[] zero = new double[point.length];
        assertArrayEquals(zero, output.measure.high);
        assertArrayEquals(zero, output.distances.high);
        assertArrayEquals(zero, output.probMass.high);
        assertArrayEquals(zero, output.measure.low);
        assertArrayEquals(zero, output.distances.low);
        assertArrayEquals(zero, output.probMass.low);
    }

    @Test
    public void testAcceptLeafEquals() {
        float[] point = { 1.0f, 2.0f, 3.0f };
        INodeView leafNode = mock(NodeView.class);
        when(leafNode.getLeafPoint()).thenReturn(point);
        int leafMass = 10;
        when(leafNode.getMass()).thenReturn(leafMass);

        int sampleSize = 21;
        InterpolationVisitor visitor = newVisitor(point, sampleSize);
        visitor.acceptLeaf(leafNode, 100);

        InterpolationMeasure result = visitor.getResult();

        double[] expected = new double[point.length];
        Arrays.fill(expected, 0.5 * (1 + leafMass) / point.length);
        assertArrayEquals(expected, result.measure.high);
        assertArrayEquals(expected, result.measure.low);

        Arrays.fill(expected, 0.5 / point.length);
        assertArrayEquals(expected, result.probMass.high);
        assertArrayEquals(expected, result.probMass.low);

        Arrays.fill(expected, 0.0);
        assertArrayEquals(expected, result.distances.high);
        assertArrayEquals(expected, result.distances.low);
    }

    @Test
    public void testAcceptLeafNotEquals() {
        float[] point = { 1.0f, 9.0f, 4.0f };
        float[] anotherPoint = { 4.0f, 5.0f, 6.0f };

        INodeView leafNode = mock(NodeView.class);
        when(leafNode.getLeafPoint()).thenReturn(anotherPoint);
        when(leafNode.getMass()).thenReturn(4);
        int sampleSize = 99;

        InterpolationVisitor visitor = newVisitor(point, sampleSize);
        visitor.acceptLeaf(leafNode, 100);
        InterpolationMeasure result = visitor.getResult();

        double expectedSumOfNewRange = 3.0 + 4.0 + 2.0;
        double[] expectedDifferenceInRangeVector = { 0.0, 3.0, 4.0, 0.0, 0.0, 2.0 };
        double[] expectedProbVector = Arrays.stream(expectedDifferenceInRangeVector).map(x -> x / expectedSumOfNewRange)
                .toArray();
        double[] expectedMeasure = Arrays.stream(expectedProbVector).map(x -> x * 5).toArray();
        double[] expectedDistances = new double[2 * point.length];
        for (int i = 0; i < 2 * point.length; i++) {
            expectedDistances[i] = expectedProbVector[i] * expectedDifferenceInRangeVector[i];
        }

        for (int i = 0; i < point.length; i++) {
            assertEquals(expectedProbVector[2 * i], result.probMass.high[i], TOL);
            assertEquals(expectedProbVector[2 * i + 1], result.probMass.low[i], TOL);
            assertEquals(expectedMeasure[2 * i], result.measure.high[i], TOL);
            assertEquals(expectedMeasure[2 * i + 1], result.measure.low[i], TOL);
            assertEquals(expectedDistances[2 * i], result.distances.high[i], TOL);
            assertEquals(expectedDistances[2 * i + 1], result.distances.low[i], TOL);
        }
    }

    // ---------- the parity harness: new vs legacy over the same path ----------

    @Test
    public void testParityWithSimple() {
        float[] pointToScore = { 0.0f, 0.0f };
        int sampleSize = 50;

        InterpolationVisitor newVisitor = newVisitor(pointToScore, sampleSize);
        SimpleInterpolationVisitor oldVisitor = new SimpleInterpolationVisitor(pointToScore, sampleSize, 1, false);

        // ---- leaf, not equal ----
        float[] leafPoint = { 1.0f, -2.0f };
        int leafMass = 3;
        int depth = 4;
        runLeaf(newVisitor, leafPoint, leafMass, depth);
        runLeaf(oldVisitor, leafPoint, leafMass, depth);
        assertMeasuresClose(oldVisitor.getResult(), newVisitor.getResult(), TOL);

        // ---- parent, point outside ----
        depth--;
        int parentMass = leafMass + 2;
        float[] parentMax = { 2.0f, -0.5f };
        runInternal(newVisitor, leafPoint, parentMax, null, parentMass, depth);
        runInternal(oldVisitor, leafPoint, parentMax, null, parentMass, depth);
        assertMeasuresClose(oldVisitor.getResult(), newVisitor.getResult(), TOL);
        assertFalse(newVisitor.pointInsideBox);

        // ---- grandparent, point inside -> both converge, values frozen ----
        depth--;
        // grandparent box contains (0,0): coord0 [-1,2], coord1 [-2,1]
        float[] gpMin = { -1.0f, -1.5f };
        float[] gpMax = { 2.0f, 1.0f };
        int gpMass = parentMass + 2;
        runInternal(newVisitor, gpMin, gpMax, null, gpMass, depth);
        runInternal(oldVisitor, gpMin, gpMax, null, gpMass, depth);
        assertMeasuresClose(oldVisitor.getResult(), newVisitor.getResult(), TOL);
    }

    /**
     * Parity on the shadow / equalsLeaf path, where the query equals the leaf and
     * subsequent nodes use the sibling-merged counterfactual. Exercises the
     * "max-side-wins" directional-distance quirk explicitly.
     */
    @Test
    public void testParityShadowPath() {
        float[] pointToScore = { 0.0f, 0.0f };
        int sampleSize = 50;

        InterpolationVisitor newVisitor = newVisitor(pointToScore, sampleSize);
        SimpleInterpolationVisitor oldVisitor = new SimpleInterpolationVisitor(pointToScore, sampleSize, 1, false);

        // leaf equals the query -> pointEqualsLeaf, self measures
        float[] leafPoint = { 0.0f, 0.0f };
        runLeaf(newVisitor, leafPoint, 1, 2);
        runLeaf(oldVisitor, leafPoint, 1, 2);
        assertMeasuresClose(oldVisitor.getResult(), newVisitor.getResult(), TOL);

        // parent: shadow path, sibling box at {1,-2}, node box spans both points
        float[] siblingPoint = { 1.0f, -2.0f };
        INodeView pNew = mock(NodeView.class);
        INodeView pOld = mock(NodeView.class);
        for (INodeView p : new INodeView[] { pNew, pOld }) {
            when(p.getMass()).thenReturn(3);
            when(p.getBoundingBox()).thenReturn(new ArrayBox(leafPoint, siblingPoint));
            when(p.getSiblingBoundingBox(any())).thenReturn(new ArrayBox(siblingPoint, siblingPoint));
        }
        newVisitor.accept(pNew, 1);
        oldVisitor.accept(pOld, 1);
        assertMeasuresClose(oldVisitor.getResult(), newVisitor.getResult(), TOL);
    }

    // ---------- path drivers (feed both visitor types the same mocks) ----------

    private static void runLeaf(INodeView visitorTarget, float[] leafPoint, int mass, int depth) {
        // visitorTarget is actually the visitor; kept generic for symmetry below
        throw new UnsupportedOperationException("use the typed overloads");
    }

    private static void runLeaf(InterpolationVisitor visitor, float[] leafPoint, int mass, int depth) {
        INodeView leaf = mock(NodeView.class);
        when(leaf.getLeafPoint()).thenReturn(leafPoint);
        when(leaf.getMass()).thenReturn(mass);
        visitor.acceptLeaf(leaf, depth);
    }

    private static void runLeaf(SimpleInterpolationVisitor visitor, float[] leafPoint, int mass, int depth) {
        INodeView leaf = mock(NodeView.class);
        when(leaf.getLeafPoint()).thenReturn(leafPoint);
        when(leaf.getBoundingBox()).thenReturn(new ArrayBox(leafPoint, leafPoint));
        when(leaf.getMass()).thenReturn(mass);
        visitor.acceptLeaf(leaf, depth);
    }

    private static void runInternal(InterpolationVisitor visitor, float[] boxA, float[] boxB, float[] sibling, int mass,
            int depth) {
        INodeView node = mock(NodeView.class);
        when(node.getMass()).thenReturn(mass);
        when(node.getBoundingBox()).thenReturn(new ArrayBox(boxA, boxB));
        if (sibling != null) {
            when(node.getSiblingBoundingBox(any())).thenReturn(new ArrayBox(sibling, sibling));
        }
        visitor.accept(node, depth);
    }

    private static void runInternal(SimpleInterpolationVisitor visitor, float[] boxA, float[] boxB, float[] sibling,
            int mass, int depth) {
        INodeView node = mock(NodeView.class);
        when(node.getMass()).thenReturn(mass);
        when(node.getBoundingBox()).thenReturn(new ArrayBox(boxA, boxB));
        if (sibling != null) {
            when(node.getSiblingBoundingBox(any())).thenReturn(new ArrayBox(sibling, sibling));
        }
        visitor.accept(node, depth);
    }
}
