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

package org.streamingalgorithms.randomcutforest.benchmark.operations;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;

import java.util.concurrent.TimeUnit;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

/**
 * Datasets is the single generation source every suite depends on; drift here
 * silently invalidates ALL numbers. These assert STRUCTURAL facts within one
 * generation -- counts, widths, float/double consistency -- not reproducibility
 * ACROSS generations.
 *
 * <p>
 * Deliberately NOT tested here: "same seed -> identical data." That is a
 * testutils generator contract, and the benchmark is better off not relying on
 * it -- we measure over a distribution CLASS, not a specific realized dataset,
 * which is more robust (throughput shouldn't care which anomaly path was
 * drawn). (Aside: NormalMixtureTestData's transitions use Math.random(), which
 * is process-global and unseeded, so identity across calls does not currently
 * hold for D1 anyway -- that's a testutils finding, fixed in testutils, not
 * here.)
 *
 * <p>
 * Machinery test, not a benchmark: saturates with {@link #SAT} points, not the
 * 25k equilibrium. The class timeout is a backstop so a genuine wedge fails
 * loud in seconds instead of hanging the reactor.
 */

@Timeout(value = 60, unit = TimeUnit.SECONDS, threadMode = Timeout.ThreadMode.SEPARATE_THREAD)
public class DatasetsTest {

    private static final int SAT = 1000; // machinery-test saturation, not benchmark equilibrium

    @ParameterizedTest
    @EnumSource(Datasets.Id.class)
    public void rawDataHasFullLength(Datasets.Id id) {
        double[][] data = Datasets.rawData(id);
        assertEquals(Datasets.INITIAL + Datasets.SCORED, data.length, id + ": row count");
        for (double[] row : data) {
            assertNotNull(row, id + ": null row");
        }
    }

    @Test
    public void d1IsHundredWideD2IsBaseTen() {
        // D1 = 100-dim direct (shingle 1); D2 = 10 base dims (forest shingles
        // internally).
        // If the D2 generator ever returns pre-shingled 100-wide rows, this fails
        // loudly.
        assertEquals(100, Datasets.rawData(Datasets.Id.D1)[0].length, "D1 row width");
        assertEquals(10, Datasets.rawData(Datasets.Id.D2)[0].length, "D2 row width (base dim)");
    }

    @ParameterizedTest
    @EnumSource(Datasets.Id.class)
    public void effectiveDimsAreHundred(Datasets.Id id) {
        assertEquals(100, id.effectiveDims(), id + ": effective dims");
    }

    @ParameterizedTest
    @EnumSource(Datasets.Id.class)
    public void prepareHoldoutShapeAndFloatConsistency(Datasets.Id id) {
        // Build ONCE, assert many times over the realized rows -- structural properties
        // that
        // hold regardless of which anomaly path was drawn (the robust way to test the
        // data).
        Datasets.Prepared p = Datasets.prepare(id, 0.2, 99L, SAT);
        assertNotNull(p.forest, id + ": forest");
        assertEquals(Datasets.SCORED, p.scoreD.length, id + ": scoreD length");
        assertEquals(Datasets.SCORED, p.scoreF.length, id + ": scoreF length");

        // The float hold-out must be the float cast of the double hold-out, element for
        // element -- the two feed DEFAULT (double) vs DYNAMIC/DENSITY (float) paths and
        // must
        // describe the SAME points, or checksums compared across those paths are
        // meaningless.
        for (int i = 0; i < Datasets.SCORED; i++) {
            assertEquals(p.scoreD[i].length, p.scoreF[i].length, id + ": width mismatch at " + i);
            for (int j = 0; j < p.scoreD[i].length; j++) {
                assertEquals((float) p.scoreD[i][j], p.scoreF[i][j], id + ": float/double mismatch at " + i + "," + j);
            }
        }
    }
}
