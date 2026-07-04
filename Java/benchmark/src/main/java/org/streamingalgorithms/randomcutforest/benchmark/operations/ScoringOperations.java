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

import org.streamingalgorithms.randomcutforest.CommonUtils;
import org.streamingalgorithms.randomcutforest.RandomCutForest;
import org.streamingalgorithms.randomcutforest.returntypes.DiVector;
import org.streamingalgorithms.randomcutforest.returntypes.InterpolationMeasure;

/**
 * THE scoring operation, defined exactly once. Both drivers -- the JMH driver
 * ({@code ScoringBenchmark}) and the cold driver ({@code ScoringColdMain}) --
 * call {@link #scoreOne} and nothing else. Because the label of every measured
 * cell is DERIVED from (mode, kind, func) and the work is this one method, a
 * printed label can never disagree with what ran. This is the anti-drift
 * property the old copy/paste benchmark methods kept losing.
 *
 * <p>
 * The (mode, kind, func) axes:
 * 
 * <pre>
 *   mode in { EXACT, APPROX }              -- full traversal vs early-stopping
 *   kind in { SCALAR, ATTRIBUTION, DENSITY}-- double / DiVector / InterpolationMeasure
 *   func in { DEFAULT, DYNAMIC }           -- fixed CommonUtils fns vs supplied fns
 * </pre>
 * 
 * Every cell is collapsed to ONE finite double so the drivers can checksum it
 * (finiteness / non-degeneracy oracle) and so a Blackhole/return consumes it
 * without dead-code elimination. The scalar is a checksum, NOT a meaningful
 * metric -- do not interpret it.
 *
 * <p>
 * DEFAULT paths take the {@code double[]} point; DYNAMIC paths take the
 * {@code float[]} point. This asymmetry is inherited from the library's
 * overload set and is deliberate -- both drivers therefore hold each hold-out
 * point in both representations, converted once, outside any timed region.
 */

public final class ScoringOperations {

    public enum Mode {
        EXACT, APPROX
    }

    public enum Kind {
        SCALAR, ATTRIBUTION, DENSITY
    }

    public enum Func {
        DEFAULT, DYNAMIC
    }

    // ---- approximate / dynamic parameters (only consulted on the relevant paths)
    // ----
    public static final double APPROX_PRECISION = 0.1;
    public static final boolean APPROX_HIGH_IS_CRITICAL = true;
    public static final int IGNORE_LEAF_MASS_THRESHOLD = 0;

    private ScoringOperations() {
    }

    /**
     * One point -> one comparable finite scalar. Monomorphic per call site when a
     * driver fixes (mode, kind, func), so the JIT inlines it; the enum switch cost
     * is negligible against an O(trees x depth) traversal.
     */
    public static double scoreOne(RandomCutForest f, double[] pD, float[] pF, Mode mode, Kind kind, Func func) {
        switch (kind) {
        case SCALAR:
            if (func == Func.DEFAULT) {
                return (mode == Mode.EXACT) ? f.getAnomalyScore(pD) : f.getApproximateAnomalyScore(pD);
            }
            return (mode == Mode.EXACT)
                    ? f.getDynamicScore(pF, IGNORE_LEAF_MASS_THRESHOLD, CommonUtils::defaultScoreSeenFunction,
                            CommonUtils::defaultScoreUnseenFunction, CommonUtils::defaultDampFunction)
                    : f.getApproximateDynamicScore(pF, APPROX_PRECISION, APPROX_HIGH_IS_CRITICAL,
                            IGNORE_LEAF_MASS_THRESHOLD, CommonUtils::defaultScoreSeenFunction,
                            CommonUtils::defaultScoreUnseenFunction, CommonUtils::defaultDampFunction);

        case ATTRIBUTION: {
            DiVector v;
            if (func == Func.DEFAULT) {
                v = (mode == Mode.EXACT) ? f.getAnomalyAttribution(pD) : f.getApproximateAnomalyAttribution(pD);
            } else {
                v = (mode == Mode.EXACT)
                        ? f.getDynamicAttribution(pF, IGNORE_LEAF_MASS_THRESHOLD, CommonUtils::defaultScoreSeenFunction,
                                CommonUtils::defaultScoreUnseenFunction, CommonUtils::defaultDampFunction)
                        : f.getApproximateDynamicAttribution(pF, APPROX_PRECISION, APPROX_HIGH_IS_CRITICAL,
                                IGNORE_LEAF_MASS_THRESHOLD, CommonUtils::defaultScoreSeenFunction,
                                CommonUtils::defaultScoreUnseenFunction, CommonUtils::defaultDampFunction);
            }
            return v.getHighLowSum();
        }

        case DENSITY:
            // No approximate-density entry point exists yet, so the APPROX x DENSITY
            // cell currently MIRRORS EXACT x DENSITY (a reserved slot, not wasted: when
            // approx density lands, branch on `mode` here and the cell becomes live with
            // zero driver changes). `func` does not apply to density.
            return reduce(f.getSimpleDensity(pF));

        default:
            throw new IllegalStateException("unhandled kind " + kind);
        }
    }

    /**
     * Collapse an InterpolationMeasure to one stable scalar, purely for checksum /
     * DCE-avoidance.
     */
    private static double reduce(InterpolationMeasure im) {
        return im.measure.getHighLowSum();
    }
}
