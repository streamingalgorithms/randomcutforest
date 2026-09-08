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
 *
 * This hile has been modified substantially from the original which had the
 * following notice.
 *
 * Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import static java.lang.Math.round;
import static org.streamingalgorithms.randomcutforest.CommonUtils.checkArgument;
import static org.streamingalgorithms.randomcutforest.CommonUtils.checkNotNull;

import java.util.stream.Collector;

/**
 * The accumulated result of an interpolation walk.
 *
 * <p>
 * <b>Displacement.</b> The high-low sum of {@link #measure} is, per tree,
 * exactly DISP(x, S) + pointMass, where DISP is the bit-displacement of Guha et
 * al. (2016, Definition 2): the expected mass of the sibling subtree of the
 * leaf that would hold x. The walk computes E[DISP | T] in closed form rather
 * than sampling an insertion, so it is a Rao-Blackwellized estimate.
 * {@link #measure} is its decomposition over the 2 * dimensions oriented faces,
 * and {@link #getDisplacement()} recovers the scalar.
 *
 * <p>
 * <b>Two normalizers, not one.</b> {@link #sampleSize} is a mass and
 * {@link #treeCount} is a count. A single int field carried the sum of tree
 * masses in the fold path and their mean after the collector applied scale(1 /
 * numberOfTrees); consistent as a ratio against measure, but unable to report
 * how many trees contributed, and liable to round to zero under a small scale
 * factor, at which point getDensity silently returned 0. Both are doubles here
 * because both are only ever used as divisors.
 *
 * <p>
 * <b>pointMass is a parameter, not an accumulator.</b> It is adopted on merge
 * and left alone by scale(). Summing or scaling it would corrupt the
 * displacement correction.
 */
public class InterpolationMeasure {

    public final DiVector measure;
    public final DiVector distances;
    public final DiVector probMass;

    protected final int dimensions;

    /** Accumulated tree mass; a weight, used only as a divisor. */
    protected double sampleSize;

    /** Number of trees folded into this result. */
    protected double treeCount;

    /** The probe mass used for the walk. Adopted, never summed or scaled. */
    protected double pointMass;

    /**
     * The scalar accumulated with the primary field function. With the base field
     * function (node mass) this is identically measure.getHighLowSum(), since the
     * scalar and directional recurrences share a scalar fieldVal. It diverges only
     * if a subclass supplies a per-face field.
     */
    protected double score;

    /**
     * The scalar accumulated with the height field function, by default the score
     * function at the separating node: the expected inverse height, i.e. the
     * ordinary RCF anomaly score. Independent of {@link #score} because it runs its
     * own recurrence with its own weight.
     */
    protected double heightScore;

    /**
     * Optional per-face local scales, null unless the walk produced them.
     *
     */
    protected FirstPassageScales scales;

    public InterpolationMeasure(int dimensions, double sampleSize) {
        checkArgument(dimensions > 0, "dimensions must be greater than 0");
        this.dimensions = dimensions;
        this.sampleSize = sampleSize;
        measure = new DiVector(dimensions);
        distances = new DiVector(dimensions);
        probMass = new DiVector(dimensions);
    }

    /** Retained for source compatibility with the int-valued constructor. */
    public InterpolationMeasure(int dimensions, int sampleSize) {
        this(dimensions, (double) sampleSize);
    }

    public InterpolationMeasure(InterpolationMeasure base, boolean consume) {
        this.dimensions = base.dimensions;
        this.sampleSize = base.sampleSize;
        this.treeCount = base.treeCount;
        this.pointMass = base.pointMass;
        this.score = base.score;
        this.heightScore = base.heightScore;
        this.scales = base.scales;
        measure = consume ? base.measure : new DiVector(base.measure);
        distances = consume ? base.distances : new DiVector(base.distances);
        probMass = consume ? base.probMass : new DiVector(base.probMass);
    }

    public InterpolationMeasure(double sampleSize, DiVector measure, DiVector distances, DiVector probMass) {
        this(sampleSize, measure, distances, probMass, 0.0, 0.0, 0.0, 0.0);
    }

    public InterpolationMeasure(double sampleSize, DiVector measure, DiVector distances, DiVector probMass,
            double treeCount, double pointMass, double score, double heightScore) {
        checkArgument(measure.getDimensions() == distances.getDimensions(),
                "measure.getDimensions() should be equal to distances.getDimensions()");
        checkArgument(measure.getDimensions() == probMass.getDimensions(),
                "measure.getDimensions() should be equal to probMass.getDimensions()");

        this.dimensions = measure.getDimensions();
        this.sampleSize = sampleSize;
        this.measure = measure;
        this.distances = distances;
        this.probMass = probMass;
        this.treeCount = treeCount;
        this.pointMass = pointMass;
        this.score = score;
        this.heightScore = heightScore;
    }

    // ------------------------------------------------------------------
    // derived quantities
    // ------------------------------------------------------------------

    /**
     * The bit-displacement DISP(x, S) of Definition 2, per tree, with the probe
     * mass removed. Valid in both the collector path (where measure and treeCount
     * have both been scaled by 1 / numberOfTrees) and the fold path (where both are
     * sums), since only their ratio is used.
     */
    public double getDisplacement() {
        return (treeCount > 0.0) ? measure.getHighLowSum() / treeCount - pointMass : 0.0;
    }

    /**
     * Per-tree directional displacement over the 2 * dimensions faces. The probe
     * mass is spread across faces in proportion to the separation probabilities and
     * so cannot be subtracted componentwise; run the walk with pointMass = 0 for a
     * pure directional DISP.
     */
    public DiVector getDirectionalDisplacement() {
        return (treeCount > 0.0) ? measure.scale(1.0 / treeCount) : new DiVector(dimensions);
    }

    /** Expected inverse height: the ordinary RCF anomaly score, per tree. */
    public double getExpectedInverseHeight() {
        return (treeCount > 0.0) ? heightScore / treeCount : 0.0;
    }

    /** The primary field scalar, per tree. */
    public double getScore() {
        return (treeCount > 0.0) ? score / treeCount : 0.0;
    }

    public int getDimensions() {
        return dimensions;
    }

    /** Accumulated tree mass as a weight. */
    public double getSampleWeight() {
        return sampleSize;
    }

    /** Rounded view, for callers written against the previous int field. */
    public int getSampleSize() {
        return (int) round(sampleSize);
    }

    public double getTreeCount() {
        return treeCount;
    }

    public double getPointMass() {
        return pointMass;
    }

    public void setSampleSize(double sampleSize) {
        this.sampleSize = sampleSize;
    }

    public void setPointMass(double pointMass) {
        this.pointMass = pointMass;
    }

    /**
     * May be null; callers wanting an extent box should go through
     * AnisotropicDensityOutput.
     */
    public FirstPassageScales getScales() {
        return scales;
    }

    public void setScales(FirstPassageScales scales) {
        this.scales = scales;
    }

    // ------------------------------------------------------------------
    // accumulation
    // ------------------------------------------------------------------

    public static InterpolationMeasure addToLeft(InterpolationMeasure left, InterpolationMeasure right) {
        checkNotNull(left, "left must not be null");
        checkNotNull(right, "right must not be null");
        checkArgument(left.dimensions == right.dimensions, "dimensions must be the same");

        // pointMass is a parameter of the walk, identical across trees, so it is
        // adopted from the first non-empty operand rather than summed. Tested BEFORE
        // treeCount is advanced: an earlier version compared the two counts after the
        // increment, which happened to be equivalent but said nothing about the intent
        // and would have broken silently if the lines were reordered.
        boolean leftIsEmpty = left.treeCount == 0.0;

        left.sampleSize += right.sampleSize;
        left.treeCount += right.treeCount;
        left.score += right.score;
        left.heightScore += right.heightScore;
        if (leftIsEmpty) {
            left.pointMass = right.pointMass;
        }
        DiVector.addToLeft(left.distances, right.distances);
        DiVector.addToLeft(left.measure, right.measure);
        DiVector.addToLeft(left.probMass, right.probMass);
        if (left.scales == null) {
            // adopt: right came from toMeasure, which already copied, so no other
            // holder can observe the subsequent in-place merges
            left.scales = right.scales;
        } else if (right.scales != null) {
            FirstPassageScales.addToLeft(left.scales, right.scales);
        }
        return left;
    }

    public static Collector<InterpolationMeasure, InterpolationMeasure, InterpolationMeasure> collector(int dimensions,
            int sampleSize, int numberOfTrees) {
        return Collector.of(() -> new InterpolationMeasure(dimensions, sampleSize), InterpolationMeasure::addToLeft,
                InterpolationMeasure::addToLeft, result -> result.scale(1.0 / numberOfTrees));
    }

    /**
     * Scales every accumulator, leaving pointMass and scales alone. Any field added
     * later must be added here as well as to addToLeft; the two must stay in step
     * or the derived quantities silently drift between the fold and collector
     * paths.
     *
     * <p>
     * scales is deliberately untouched: it carries its own tree count and divides
     * by it internally, so every quantity it reports is already a per-tree mean.
     * Scaling it here would divide a second time.
     */
    public InterpolationMeasure scale(double z) {
        InterpolationMeasure out = new InterpolationMeasure(sampleSize * z, measure.scale(z), distances.scale(z),
                probMass.scale(z), treeCount * z, pointMass, score * z, heightScore * z);
        out.scales = scales;
        return out;
    }

    public InterpolationMeasure scaleInPlace(double z) {
        sampleSize *= z;
        treeCount *= z;
        score *= z;
        heightScore *= z;
        measure.scaleInPlace(z);
        distances.scaleInPlace(z);
        probMass.scaleInPlace(z);
        return this;
    }
}
