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

/**
 * DensityOutput plus tree-averaged boxes recorded at local cut-probability or
 * mass crossings.
 */
public class AnisotropicDensityOutput extends DensityOutput {

    public AnisotropicDensityOutput(int dimensions, double sampleSize) {
        super(dimensions, (int) sampleSize);
        this.scales = new FirstPassageScales(dimensions);
    }

    /**
     * Wraps a folded measure. The scales ride on the base class, so they survive an
     * accumulator seeded with a plain InterpolationMeasure; this constructor only
     * fixes the type at the end, exactly as DensityOutput does.
     */
    public AnisotropicDensityOutput(InterpolationMeasure base) {
        super(base);
        if (this.scales == null) {
            // no anisotropic visitor ran, or every tree converged immediately
            this.scales = new FirstPassageScales(getDimensions());
        }
    }

    @Override
    public FirstPassageScales getScales() {
        return scales;
    }

    // ------------------------------------------------------------------
    // the box
    // ------------------------------------------------------------------

    /** Query-relative extents, high followed by low, averaged across trees. */
    public double[] cutBox() {
        return scales.cutBox();
    }

    public double[] passageBox() {
        return scales.passageBox();
    }

    public double[] stopBox() {
        return scales.stopBox();
    }

    /** Full width along each axis: L_high + L_low. */
    public double[] getAxisExtents() {
        int d = getDimensions();
        double[] box = cutBox();
        double[] widths = new double[d];
        for (int i = 0; i < d; i++) {
            widths[i] = box[i] + box[i + d];
        }
        return widths;
    }

    public double[] passageAxis() {
        double[] box = passageBox();
        int d = getDimensions();
        double[] widths = new double[d];
        for (int i = 0; i < d; i++) {
            widths[i] = box[i] + box[i + d];
        }
        return widths;
    }

    /**
     * Per-axis asymmetry (L_high - L_low) / (L_high + L_low), in [-1, 1]. This is
     * the odd-parity part of the neighbourhood, which no symmetric rank-2 density
     * tensor can represent: it says the mass sits preferentially on one side rather
     * than merely that the axis is stretched.
     */
    public double[] getDrift() {
        double[] box = cutBox();
        int d = getDimensions();
        double[] drift = new double[d];
        for (int i = 0; i < d; i++) {
            double sum = box[i] + box[i + d];
            drift[i] = (sum > 0.0) ? (box[i] - box[i + d]) / sum : 0.0;
        }
        return drift;
    }

    /**
     * Condition number of the local frame: the ratio of the largest to the smallest
     * axis extent. 1 is isotropic. Note this is anisotropy <i>relative to the
     * coordinate axes</i> — the cuts are axis-aligned, so a ridge at 45 degrees
     * inflates every axis equally and reads as isotropic.
     */
    public double getAnisotropy() {
        double[] widths = getAxisExtents();
        double max = 0.0, min = Double.MAX_VALUE;
        for (double w : widths) {
            if (w > 0.0) {
                max = Math.max(max, w);
                min = Math.min(min, w);
            }
        }
        return (min < Double.MAX_VALUE && min > 0.0) ? max / min : 1.0;
    }

    // ------------------------------------------------------------------
    // density from the box
    // ------------------------------------------------------------------

    /**
     * Displaced mass over a product volume built from the crossing half-lengths.
     * Where the submanifold dimension is below the ambient one, the product is
     * taken over the k most contracted axes, since those are the ones that actually
     * shrink along the path.
     */
    public double getCutDensity(double q, int manifoldDimension) {
        double weight = getSampleWeight();
        if (!(weight > 0.0)) {
            return 0.0;
        }
        double sumOfPts = measure.getHighLowSum() / weight;
        if (!(sumOfPts > 0.0)) {
            return 0.0;
        }

        // Product of the selected axis widths of the averaged cut box.
        double[] sorted = getAxisExtents();
        java.util.Arrays.sort(sorted);
        double volume = 1.0;
        int used = 0;
        for (double width : sorted) {
            if (used == manifoldDimension) {
                break;
            }
            if (width > 0.0) {
                volume *= width;
                used++;
            }
        }
        if (manifoldDimension <= 0 || used < manifoldDimension) {
            return 0.0;
        }

        if (!(volume > 0)) {
            return 0.0;
        }
        return sumOfPts / (q * sumOfPts + volume);
    }

    public double getCutDensity() {
        return getCutDensity(DEFAULT_SUM_OF_POINTS_SCALING_FACTOR, getDimensions());
    }

    private double[] getDirectionalDensity(double q) {
        int d = getDimensions();
        double[] out = new double[2 * d];
        double total = measure.getHighLowSum();
        double scalar = passageDensity(q);
        if (!(total > 0.0) || !(scalar > 0.0)) {
            return out;
        }
        for (int i = 0; i < d; i++) {
            out[i] = scalar * measure.high[i] / total;
            out[i + d] = scalar * measure.low[i] / total;
        }
        return out;
    }

    private double[] directionalDensity() {
        return getDirectionalDensity(DEFAULT_SUM_OF_POINTS_SCALING_FACTOR);
    }

    public double[] getDensityGradient() {
        int d = getDimensions();
        double[] c = directionalDensity();
        double norm = 0;
        for (double v : c) {
            norm += v * v;
        }
        norm = Math.sqrt(norm);
        double[] out = new double[d];
        if (norm > 0) {
            for (int i = 0; i < d; i++) {
                out[i] = (c[i + d] - c[i]) / norm;
            }
        }
        return out;
    }

    public double passageDensity(double q) {
        double weight = getSampleWeight();
        double volume = scales.meanVolume(false);
        if (!(weight > 0.0) || !(volume > 0.0)) {
            return 0.0;
        }
        double sumOfPts = measure.getHighLowSum() / weight;
        return (sumOfPts > 0.0) ? sumOfPts / (q * sumOfPts + volume) : 0.0;
    }

    /** Density using the volume of the averaged recorded stopping box. */
    public double getStoppingDensity(double q) {
        double weight = getSampleWeight();
        if (!(weight > 0.0)) {
            return 0.0;
        }
        double sumOfPts = measure.getHighLowSum() / weight;
        double volume = volumeOf(stopBox());
        return (sumOfPts > 0.0 && volume > 0.0) ? sumOfPts / (q * sumOfPts + volume) : 0.0;
    }

    public double getStoppingDensity() {
        return getStoppingDensity(DEFAULT_SUM_OF_POINTS_SCALING_FACTOR);
    }

    /** Volume of a DiVector-layout box: the product of its axis widths. */
    private double volumeOf(double[] box) {
        int d = getDimensions();
        double v = 1.0;
        for (int i = 0; i < d; i++) {
            double w = box[i] + box[i + d];
            if (!(w > 0)) {
                return 0.0;
            }
            v *= w;
        }
        return v;
    }

    /**
     * Ratio of cut-box density to the inherited mixture density; no fixed bound is
     * assumed.
     */
    public double getVolumeAnisotropy() {
        double mixture = getDensity();
        return (mixture > 0.0) ? getCutDensity() / mixture : 1.0;
    }

    /**
     * Whether any traversal supplied box geometry; this is not a statistical
     * validity test.
     */
    public boolean isReliable() {
        return scales.getTreeCount() > 0;
    }
}
