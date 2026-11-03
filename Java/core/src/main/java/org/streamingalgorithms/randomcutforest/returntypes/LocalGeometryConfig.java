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

import static org.streamingalgorithms.randomcutforest.CommonUtils.checkArgument;

/**
 * Where each tree reads the three boxes of an {@link AnisotropicLocalGeometry}.
 * Every result carries the configuration it was read under.
 *
 * <p>
 * Walking up from the leaf, the walk stops at the first vertex above the leaf
 * whose own cut probability (the chance that one cut of the box holding the
 * query separates the query) is at most {@code stopProbability}. From there
 * down, the separation law is read at two quantiles: the cut box at
 * {@code cutQuantile} and the passage box at {@code passageQuantile}. Both are
 * capped at the first ancestor holding the cap mass, where the mass dial
 * starts. The reading box is the first ancestor holding the reading mass, a
 * hitting mass that is the same for every query and every tree.
 *
 * <p>
 * Masses are {@code factor * psi^exponent}, with psi the tree's mass. The
 * defaults are the paper's: stop at 0.05, quantiles 1/2 and 0.05, cap
 * {@code psi^(1/2)} and reading mass {@code 2 psi^(1/2)}. The constructor
 * refuses settings under which the three boxes would not nest: the passage
 * quantile must be below the cut quantile, and the reading mass at least the
 * cap mass for every psi of at least 1.
 */
public record LocalGeometryConfig(double stopProbability, double cutQuantile, double passageQuantile, double capFactor,
        double capExponent, double readingFactor, double readingExponent) {

    public static final LocalGeometryConfig DEFAULT = new LocalGeometryConfig(0.05, 0.5, 0.05, 1.0, 0.5, 2.0, 0.5);

    public LocalGeometryConfig {
        checkArgument(stopProbability > 0 && stopProbability < 1, "stopProbability must be in (0, 1)");
        checkArgument(passageQuantile > 0 && passageQuantile < cutQuantile && cutQuantile < 1,
                "quantiles must satisfy 0 < passageQuantile < cutQuantile < 1");
        checkArgument(capFactor > 0 && capExponent >= 0 && capExponent <= 1,
                "cap must be factor > 0, exponent in [0, 1]");
        checkArgument(readingFactor >= capFactor && readingExponent >= capExponent && readingExponent <= 1,
                "the reading mass must be at least the cap mass for every tree mass");
    }

    /** The mass that caps the cut and passage boxes of a tree of this mass. */
    public double capMass(double treeMass) {
        return capFactor * Math.pow(Math.max(1.0, treeMass), capExponent);
    }

    /** The mass at which a tree of this mass reads its third box. */
    public double readingMass(double treeMass) {
        return readingFactor * Math.pow(Math.max(1.0, treeMass), readingExponent);
    }
}
