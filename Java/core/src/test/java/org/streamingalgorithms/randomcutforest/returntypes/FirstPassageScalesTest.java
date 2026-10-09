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

package org.streamingalgorithms.randomcutforest.returntypes;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

public class FirstPassageScalesTest {

    // Cut boxes in the double[2 * d] layout: high toward minus, then low toward
    // plus.
    private static final double[] LOWER_LEFT = { 3.0, 2.0, 1.0, 0.0 }; // centre (-1, -1)
    private static final double[] UPPER_RIGHT = { 1.0, 0.0, 3.0, 2.0 }; // centre (1, 1)
    private static final double[] NONE = new double[4];

    @Test
    public void testCutSpreadIsTheCovarianceOfTheCentres() {
        FirstPassageScales scales = new FirstPassageScales(2);
        scales.observeTree(LOWER_LEFT, NONE, NONE, NONE, 1.0);
        scales.observeTree(UPPER_RIGHT, NONE, NONE, NONE, 1.0);

        // the averaged box is centred on the query, the mean of the two centres
        assertArrayEquals(new double[] { 2.0, 1.0, 2.0, 1.0 }, scales.cutBox(), 1e-12);
        double[][] spread = scales.cutSpread();
        assertArrayEquals(new double[] { 1.0, 1.0 }, spread[0], 1e-12);
        assertArrayEquals(new double[] { 1.0, 1.0 }, spread[1], 1e-12);
    }

    @Test
    public void testCutSpreadSurvivesTheMerge() {
        FirstPassageScales left = new FirstPassageScales(2);
        left.observeTree(LOWER_LEFT, NONE, NONE, NONE, 1.0);
        FirstPassageScales right = new FirstPassageScales(2);
        right.observeTree(UPPER_RIGHT, NONE, NONE, NONE, 1.0);
        assertEquals(0.0, left.cutSpread()[0][0], 1e-12); // one tree, no spread

        FirstPassageScales merged = FirstPassageScales.addToLeft(new FirstPassageScales(left), right);
        assertArrayEquals(new double[] { 1.0, 1.0 }, merged.cutSpread()[0], 1e-12);
        assertArrayEquals(new double[] { 1.0, 1.0 }, merged.cutSpread()[1], 1e-12);
    }
}
