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

import java.util.function.BiFunction;

import org.streamingalgorithms.randomcutforest.tree.IBoundingBoxView;

public class DefaultScoreFunctions {
    @FunctionalInterface
    public interface ScoreFn {
        double of(int depth, int mass);
    }

    @FunctionalInterface
    public interface DampFn {
        double of(int leafMass, int treeMass);
    }

    public interface Normalizer {
        double scale(int treeMass);

        default boolean isAffine() {
            return true;
        }

        Normalizer IDENTITY = m -> 1.0;
    }

    // the transductive box->vector shape, used by getDynamicSimulatedScore:
    @FunctionalInterface
    public interface GVecFn {
        double[] of(IBoundingBoxView box);
    }

    public static final int DEFAULT_IGNORE_LEAF_MASS_THRESHOLD = 0;
    // defaults — delegate to CommonUtils, do NOT re-derive (avoids transcription
    // drift):
    public static final ScoreFn DEFAULT_SCORE_SEEN = CommonUtils::defaultScoreSeenFunction;
    public static final ScoreFn DEFAULT_SCORE_UNSEEN = CommonUtils::defaultScoreUnseenFunction;
    public static final DampFn DEFAULT_DAMP = CommonUtils::defaultDampFunction;
    public static final Normalizer DEFAULT_NORMALIZER = CommonUtils::defaultAffineNormalizer; // affine k
    public static final Normalizer NO_NORMALIZER = Normalizer.IDENTITY;

    // adapters — the bridge the public BiFunction APIs cross, ONCE, at
    // construction:
    public static ScoreFn score(BiFunction<Double, Double, Double> f) {
        return (d, m) -> f.apply((double) d, (double) m);
    }

    public static DampFn damp(BiFunction<Double, Double, Double> f) {
        return (a, b) -> f.apply((double) a, (double) b);
    }
}
