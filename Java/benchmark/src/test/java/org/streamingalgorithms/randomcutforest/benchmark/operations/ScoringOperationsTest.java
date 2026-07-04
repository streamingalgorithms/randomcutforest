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

import static java.lang.Math.log;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.concurrent.TimeUnit;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.streamingalgorithms.randomcutforest.benchmark.operations.ScoringOperations.Func;
import org.streamingalgorithms.randomcutforest.benchmark.operations.ScoringOperations.Kind;
import org.streamingalgorithms.randomcutforest.benchmark.operations.ScoringOperations.Mode;

/**
 * scoreOne is the anti-drift core both scoring drivers call. The full sweep
 * showed invariants; these turn "observed" into "enforced" so a refactor that
 * breaks one is loud instead of silent.
 *
 * <p>
 * One shared forest built in {@link BeforeAll} at {@link #SAT} points (not
 * benchmark equilibrium); the timeout is a backstop.
 */

@Timeout(value = 60, unit = TimeUnit.SECONDS, threadMode = Timeout.ThreadMode.SEPARATE_THREAD)
public class ScoringOperationsTest {

    private static final int SAT = 1000;
    private static Datasets.Prepared p;

    @BeforeAll
    static void setUp() {
        p = Datasets.prepare(Datasets.Id.D2, 0.2, 99L, SAT);
    }

    @Test
    public void everyCellIsFiniteAndNonDegenerate() {
        for (Mode mode : Mode.values()) {
            for (Kind kind : Kind.values()) {
                for (Func func : Func.values()) {
                    double sum = 0;
                    for (int i = 0; i < 200; i++) {
                        double s = ScoringOperations.scoreOne(p.forest, p.scoreD[i], p.scoreF[i], mode, kind, func);
                        assertTrue(Double.isFinite(s),
                                "non-finite at " + mode + "/" + kind + "/" + func + " point " + i);
                        sum += s;
                    }
                    assertTrue(sum != 0.0, "degenerate (all-zero) sum at " + mode + "/" + kind + "/" + func);
                }
            }
        }
    }

    @Test
    public void approxDensityMirrorsExactDensity() {
        // No approximate-density entry point exists yet, so APPROX x DENSITY is a
        // reserved
        // cell that mirrors EXACT x DENSITY (bit-identical B/op in the full sweep). If
        // someone
        // adds a real approx-density path, THIS TEST SHOULD FAIL -- that's the signal
        // to update
        // ScoringOperations' reserved-cell comment and the drivers, not to delete this.
        for (int i = 0; i < 200; i++) {
            double exact = ScoringOperations.scoreOne(p.forest, p.scoreD[i], p.scoreF[i], Mode.EXACT, Kind.DENSITY,
                    Func.DEFAULT);
            double approx = ScoringOperations.scoreOne(p.forest, p.scoreD[i], p.scoreF[i], Mode.APPROX, Kind.DENSITY,
                    Func.DEFAULT);
            assertEquals(exact, approx, 0.0, "APPROX density diverged from EXACT at point " + i);
        }
    }

    @Test
    public void approxIsCloseToExactForScalar() {
        // APPROX averages over fewer trees; close to EXACT but not bit-exact. Guards
        // against an
        // approx path that silently returns something unrelated.
        double exactSum = 0, approxSum = 0;
        for (int i = 0; i < 200; i++) {
            exactSum += ScoringOperations.scoreOne(p.forest, p.scoreD[i], p.scoreF[i], Mode.EXACT, Kind.SCALAR,
                    Func.DEFAULT);
            approxSum += ScoringOperations.scoreOne(p.forest, p.scoreD[i], p.scoreF[i], Mode.APPROX, Kind.SCALAR,
                    Func.DEFAULT);
        }
        assertEquals(exactSum, approxSum, Math.abs(exactSum) * 0.10, "APPROX scalar too far from EXACT");
    }

    /**
     * DEFAULT and DYNAMIC scalar scores diverge because the DYNAMIC (dynamic-score)
     * path is UNNORMALIZED relative to DEFAULT. The normalization is per tree, so
     * the transformation below is an approximation.
     */
    @Test
    public void dynamicWithDefaultFunctionsTracksDefaultPath() {
        double defSum = 0, dynSum = 0;
        for (int i = 0; i < 200; i++) {
            defSum += ScoringOperations.scoreOne(p.forest, p.scoreD[i], p.scoreF[i], Mode.EXACT, Kind.SCALAR,
                    Func.DEFAULT);
            dynSum += ScoringOperations.scoreOne(p.forest, p.scoreD[i], p.scoreF[i], Mode.EXACT, Kind.SCALAR,
                    Func.DYNAMIC);
        }
        assertTrue(defSum != 0.0, "degenerate default sum");
        assertEquals(defSum, dynSum * log(p.forest.getSampleSize()) / log(2.0), Math.abs(defSum) * 0.01,
                "DYNAMIC default-functions path drifted from DEFAULT");
    }
}
