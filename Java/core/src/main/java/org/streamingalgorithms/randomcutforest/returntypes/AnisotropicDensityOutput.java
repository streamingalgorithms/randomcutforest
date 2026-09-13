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
 *
 * <p>
 * <b>An axis of zero gap carries no extent.</b> A coordinate the cuts never
 * fall in, because the bounding box has no width there, contributes no factor
 * to a volume and reduces that volume's dimension by one. Every density here
 * follows that rule and {@link #getActiveDimensions} reports how many axes
 * survived it. Volumes of different active dimension are not comparable, since
 * a product over k axes has units of length^k, so read the count beside any
 * density and prefer {@link #meanLogVolume} when comparing across queries.
 *
 * <p>
 * <b>Where the submanifold dimension lives.</b> Selecting a subset of axes is
 * only meaningful when fewer are asked for than are active, and it belongs to
 * the mixture in {@link #getDensity(double, int)}, which already carries the
 * parameter. {@link #getCutDensity(double)} takes its product over every active
 * axis and needs neither parameter nor sort: asking for all of them is asking
 * for each of them, and order does not affect a product.
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
     * Axes of the cut box with positive width. This is the dimension of the volume
     * {@link #getCutDensity(double)} divides by, below the ambient dimension
     * whenever a coordinate carries no extent.
     */
    public int getActiveDimensions() {
        int active = 0;
        for (double w : getAxisExtents()) {
            if (w > 0.0) {
                active++;
            }
        }
        return active;
    }

    /**
     * Displaced mass over the cut-box volume, the product over every axis of
     * positive width. No sort and no submanifold parameter; see the class note. The
     * dimension of the volume is {@link #getActiveDimensions}.
     */
    public double getCutDensity(double q) {
        double weight = getSampleWeight();
        if (!(weight > 0.0)) {
            return 0.0;
        }
        double sumOfPts = measure.getHighLowSum() / weight;
        if (!(sumOfPts > 0.0)) {
            return 0.0;
        }
        double volume = 1.0;
        int active = 0;
        for (double width : getAxisExtents()) {
            if (width > 0.0) {
                volume *= width;
                active++;
            }
        }
        // No active axis is a point, not a unit box: an empty product reads as
        // volume 1 and would pass the guard below unnoticed.
        if (active == 0 || !(volume > 0.0)) {
            return 0.0;
        }
        return sumOfPts / (q * sumOfPts + volume);
    }

    public double getCutDensity() {
        return getCutDensity(DEFAULT_SUM_OF_POINTS_SCALING_FACTOR);
    }

    /**
     * Mixture density, restricted to a submanifold dimension when one is asked for.
     *
     * <p>
     * Each active axis proposes a length scale t = distance / probability mass and
     * the volume is estimated as t raised to the number of axes forming it,
     * weighted by that axis's probability mass. An axis of zero gap is never cut,
     * so its probability mass is zero and it leaves both the sum and the count with
     * no special case: the weights do what a product cannot.
     *
     * <p>
     * When fewer axes are asked for than are active, the axes are sorted and the
     * ones of <b>largest</b> length scale kept. That inverts the wording of the old
     * cut-density comment, which sorted ascending and took the most contracted. For
     * data near a lower-dimensional set the neighbourhood is wide along the
     * submanifold and thin across it, so the axes spanning it are the wide ones and
     * taking the contracted ones selects the normal directions instead. The old
     * behaviour only looked right because exact zeros were skipped, which hides the
     * difference for a perfectly flat axis and not for a merely thin one.
     *
     * @param q                 smoothing parameter
     * @param manifoldDimension axes to include; at or above the active count all
     *                          active axes are used, 0 or less yields 0
     */
    @Override
    public double getDensity(double q, int manifoldDimension) {
        if (manifoldDimension <= 0 || sampleSize == 0) {
            return 0.0;
        }
        double sumOfPts = measure.getHighLowSum() / sampleSize;
        if (!(sumOfPts > 0.0)) {
            return 0.0;
        }
        double[] scale = new double[dimensions];
        double[] mass = new double[dimensions];
        int active = collectActive(scale, mass);
        if (active == 0) {
            return 0.0;
        }
        int used = Math.min(manifoldDimension, active);
        double cutoff = (used < active) ? kthLargest(scale, active, used) : Double.NEGATIVE_INFINITY;

        double sumOfFactors = 0.0;
        int taken = 0;
        for (int i = 0; i < active && taken < used; i++) {
            if (scale[i] >= cutoff) {
                sumOfFactors += mass[i] * Math.exp(Math.log(scale[i]) * used);
                taken++;
            }
        }
        return sumOfPts / (q * sumOfPts + sumOfFactors);
    }

    /**
     * Packs the length scale and probability mass of every axis that is cut at all
     * into the fronts of the two arrays.
     *
     * @return how many were packed
     */
    private int collectActive(double[] scale, double[] mass) {
        int active = 0;
        for (int i = 0; i < dimensions; i++) {
            double p = probMass.getHighLowSum(i);
            if (p > 0.0) {
                double t = distances.getHighLowSum(i) / p;
                if (t > 0.0) {
                    scale[active] = t;
                    mass[active] = p;
                    active++;
                }
            }
        }
        return active;
    }

    /** The k-th largest of the first n entries, k >= 1; sorts a copy. */
    private static double kthLargest(double[] values, int n, int k) {
        double[] sorted = java.util.Arrays.copyOf(values, n);
        java.util.Arrays.sort(sorted);
        return sorted[n - k];
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

    /**
     * Displaced mass over the passage-box volume. Whether an axis of zero gap
     * annihilates this is decided by FirstPassageScales.meanVolume, not here.
     */
    public double passageDensity(double q) {
        double weight = getSampleWeight();
        double volume = scales.meanVolume(false);
        if (!(weight > 0.0) || !(volume > 0.0)) {
            return 0.0;
        }
        double sumOfPts = measure.getHighLowSum() / weight;
        return (sumOfPts > 0.0) ? 1.0 / (sampleSize * (q + volume / sumOfPts)) : 0.0;
    }

    /**
     * Probability-weighted mean log volume over the selected axes, in the same
     * convention as {@link #getDensity(double, int)}: same active set, same
     * selection, same exponent. NaN when no axis is active. A sum of logs rather
     * than the log of a product, so an axis of zero gap is an absent term rather
     * than an annihilating factor, and narrow axes do not underflow.
     */
    public double meanLogVolume() {
        return meanLogVolume(dimensions);
    }

    public double meanLogVolume(int manifoldDimension) {
        if (manifoldDimension <= 0) {
            return Double.NaN;
        }
        double[] scale = new double[dimensions];
        double[] mass = new double[dimensions];
        int active = collectActive(scale, mass);
        if (active == 0) {
            return Double.NaN;
        }
        int used = Math.min(manifoldDimension, active);
        double cutoff = (used < active) ? kthLargest(scale, active, used) : Double.NEGATIVE_INFINITY;

        double acc = 0.0, wsum = 0.0;
        int taken = 0;
        for (int i = 0; i < active && taken < used; i++) {
            if (scale[i] >= cutoff) {
                acc += mass[i] * used * Math.log(scale[i]);
                wsum += mass[i];
                taken++;
            }
        }
        return (wsum > 0.0) ? acc / wsum : Double.NaN;
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

    /**
     * Volume of a DiVector-layout box: the product of its axis widths of positive
     * width. 0 only when no axis is active, since an empty product would read as a
     * unit box.
     */
    private double volumeOf(double[] box) {
        int d = getDimensions();
        double v = 1.0;
        int active = 0;
        for (int i = 0; i < d; i++) {
            double w = box[i] + box[i + d];
            if (w > 0.0) {
                v *= w;
                active++;
            }
        }
        return (active > 0) ? v : 0.0;
    }

    /**
     * Ratio of cut-box density to the inherited mixture density; no fixed bound is
     * assumed.
     */
    public double getVolumeAnisotropy() {
        int active = getActiveDimensions();
        if (active <= 0) {
            return 1.0;
        }
        double mixture = getDensity(DEFAULT_SUM_OF_POINTS_SCALING_FACTOR, active);
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