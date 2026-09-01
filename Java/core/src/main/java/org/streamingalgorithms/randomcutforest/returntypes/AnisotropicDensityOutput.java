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
 * DensityOutput plus the local extent box.
 *
 * <p>
 * The scales travel inside the measure rather than beside it, so
 * InterpolationMeasure.addToLeft merges them along with everything else and the
 * parallel and fold paths need no special handling. Each tree contributes its
 * own crossing estimate in data units and the merge is a vector addition, so
 * the result is the ensemble mean of per-tree estimates -- the same thing
 * measure, distances and score are.
 *
 * <p>
 * <b>Two densities.</b> {@link #getDensity} is inherited: a
 * probability-weighted mixture of k-cubes whose edges are the <i>mean</i>
 * conditional lengths. Box ranges grow geometrically up the tree, so that mean
 * is dominated by rare early-separation events with enormous boxes.
 * {@link #getBoxDensity} uses the median crossing lengths and a genuine product
 * volume instead. With uniform weights and k = d, AM-GM gives V_mixture &gt;=
 * V_box with equality exactly when the neighbourhood is isotropic, so the ratio
 * of the two densities is itself an anisotropy measurement.
 *
 * <p>
 * <b>Read the validity fields.</b> A wide crossing means the neighbourhood has
 * no single scale and the box is an arbitrary representative of a flat region;
 * a large degenerate fraction means the point is masked by duplicates and has
 * no direction at all. Neither shows up as noise in the box itself.
 */
public class AnisotropicDensityOutput extends DensityOutput {

    /** Above this crossing width, in octaves, the box has no single scale. */
    public static final double DEFAULT_CROSSING_TOLERANCE = 3.0;

    public AnisotropicDensityOutput(int dimensions, double sampleSize) {
        super(dimensions, (int) sampleSize);
        this.scales = new DirectionalScales(dimensions);
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
            this.scales = new DirectionalScales(getDimensions());
        }
    }

    @Override
    public DirectionalScales getScales() {
        return scales;
    }

    // ------------------------------------------------------------------
    // the box
    // ------------------------------------------------------------------

    /**
     * The 2 * dimensions half-separation lengths in data units, laid out like a
     * DiVector: high in [0, d), low in [d, 2d). This is the local gauge — the
     * length that makes each coordinate dimensionless at this point.
     *
     * <p>
     * All 2d values come from one level of each tree's path, the level at which the
     * cumulative separation probability crosses one half, so the box is the
     * enlarged bounding box of a real node rather than a per-face mixture that
     * corresponds to no node at all.
     */
    public double[] getExtentBox() {
        return scales.halfSeparationBox();
    }

    /**
     * The boxes at the bracketing quantiles. There is no arbitrary-lambda variant:
     * lambda is fixed when the walk runs, since the crossing is scanned along the
     * path rather than read back out of a stored distribution.
     */
    public double[] getLowExtentBox() {
        return scales.lowBox();
    }

    public double[] getHighExtentBox() {
        return scales.highBox();
    }

    /** Full width along each axis: L_high + L_low. */
    public double[] getAxisExtents() {
        double[] box = getExtentBox();
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
        double[] box = getExtentBox();
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
    public double getBoxDensity(double q, int manifoldDimension) {
        double weight = getSampleWeight();
        if (!(weight > 0.0)) {
            return 0.0;
        }
        double sumOfPts = measure.getHighLowSum() / weight;
        if (!(sumOfPts > 0.0)) {
            return 0.0;
        }

        double[] widths = getAxisExtents();
        int k = Math.min(manifoldDimension, widths.length);
        double[] sorted = widths.clone();
        java.util.Arrays.sort(sorted);

        double volume = 1.0;
        int used = 0;
        for (int i = 0; i < sorted.length && used < k; i++) {
            if (sorted[i] > 0.0) {
                volume *= sorted[i];
                used++;
            }
        }
        if (used < k) {
            return 0.0;
        }
        return sumOfPts / (q * sumOfPts + volume);
    }

    public double getBoxDensity() {
        return getBoxDensity(DEFAULT_SUM_OF_POINTS_SCALING_FACTOR, getDimensions());
    }

    /**
     * Ratio of the box density to the inherited mixture density. Bounded below by 1
     * under uniform weights at k = d, and equal to 1 exactly when isotropic, so it
     * reads as a scalar anisotropy index derived from volumes rather than from the
     * frame directly.
     */
    public double getVolumeAnisotropy() {
        double mixture = getDensity();
        return (mixture > 0.0) ? getBoxDensity() / mixture : 1.0;
    }

    // ------------------------------------------------------------------
    // calibration: measured exponent vs the exponent the density assumes
    // ------------------------------------------------------------------

    /**
     * Local Holder exponent: the slope of log node mass against log length, fitted
     * per tree along that tree's own path and then averaged.
     *
     * <p>
     * There is no moment argument. The earlier binned form took one so that q could
     * reweight which bins dominated; with the fit done directly on the path there
     * are no bins to reweight. It is also no longer quantized -- a synthetic
     * geometric walk of known dimension 1, 2, 3 now returns 1.0000, 2.0000, 3.0000,
     * where the binned version gave 3.169 at d = 3.
     *
     * <p>
     * A slope is invariant to any rescaling of either mass or length, so nothing
     * needs normalizing, and it carries no intercept term of the kind that made a
     * ratio estimator report d(1 + 1/k).
     */
    public double getHolderExponent() {
        return scales.holderExponent();
    }

    /**
     * Fraction of trees whose path had enough spread on the length axis to fit an
     * exponent at all. Low coverage means the exponent above rests on few trees;
     * trees that cannot fit one contribute nothing rather than contributing a zero,
     * since averaging in those zeros is what biased an earlier version toward 0.2
     * when the answer was near the ambient dimension.
     */
    public double getExponentCoverage() {
        return scales.exponentCoverage();
    }

    /**
     * Agreement in [0, 1] between the measured exponent and the exponent the
     * density formula assumes.
     *
     * <p>
     * This is the calibration that matters. getDensity divides displaced mass by a
     * volume built as length^k with k supplied by the caller. That is only a volume
     * if the measure actually scales with exponent k here. The exponent is
     * estimated independently -- from mass against length, never touching the
     * density formula -- so |alpha - k| is a direct check on the power the density
     * is being raised to, against a k that is known rather than fitted.
     *
     * <p>
     * On a d-dimensional support with k = d it should sit near 1, and fall where
     * the local measure is filamentary, since a stretched neighbourhood is closer
     * to one-dimensional and alpha drops below k there. That is a spatial
     * prediction, not just a number, so a plot of it is falsifiable at a glance.
     */
    public double getExponentAgreement(int manifoldDimension) {
        double alpha = getHolderExponent();
        if (!(alpha > 0.0) || manifoldDimension <= 0) {
            return 0.0;
        }
        return Math.max(0.0, 1.0 - Math.abs(alpha - manifoldDimension) / manifoldDimension);
    }

    // ------------------------------------------------------------------
    // validity
    // ------------------------------------------------------------------

    public double[] getCrossingWidths() {
        return scales.crossingWidths();
    }

    public double getMaxCrossingWidth() {
        double max = 0.0;
        for (double w : getCrossingWidths()) {
            max = Math.max(max, w);
        }
        return max;
    }

    /**
     * Fraction of first-passage weight with no defined length, because the query
     * point coincided with a leaf. Now a single scalar: the degenerate case has no
     * direction by definition, so a per-face breakdown of it was reporting the same
     * number 2d times.
     */
    public double getDegenerateFraction() {
        return scales.degenerateFraction();
    }

    /**
     * Whether the box means anything here. False when the crossing is flat
     * (genuinely multi-scale neighbourhood, no single gauge) or when most of the
     * weight sits on duplicates (no direction exists).
     */
    public boolean isReliable() {
        return getMaxCrossingWidth() <= DEFAULT_CROSSING_TOLERANCE && getDegenerateFraction() < 0.5;
    }

    /**
     * Mean first-passage weight per tree, which must be 1.0 to within rounding. It
     * checks the survival product, the crossing scan, the degenerate routing and
     * the fold in one number; 0 means no scales reached the result, and a tree
     * count of 1 where many trees ran means the merge is not happening.
     */
    public double getTotalMass() {
        return scales.totalMass();
    }
}