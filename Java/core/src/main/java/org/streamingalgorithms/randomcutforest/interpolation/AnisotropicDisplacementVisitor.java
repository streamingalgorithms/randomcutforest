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

package org.streamingalgorithms.randomcutforest.interpolation;

import java.util.Arrays;

import org.streamingalgorithms.randomcutforest.IRFVisitor;
import org.streamingalgorithms.randomcutforest.IVisitorFactory;
import org.streamingalgorithms.randomcutforest.Visitor;
import org.streamingalgorithms.randomcutforest.returntypes.DirectionalScales;
import org.streamingalgorithms.randomcutforest.returntypes.InterpolationMeasure;
import org.streamingalgorithms.randomcutforest.tree.ITree;

/**
 * Reads a local length scale off each face of the neighbourhood, alongside the
 * ordinary interpolation measure.
 *
 * <p>
 * The walk is unchanged. This adds one buffered copy of (pi, len, decay, mass)
 * per level and reuses the parent's already-computed quantities through the
 * deposit hooks. The measure fields still carry displacement exactly as before;
 * this only annotates it with scale.
 *
 * <p>
 * <b>Everything is computed here, on the path.</b> An earlier version deposited
 * into a per-face log-length histogram and read quantiles back from it. That
 * was re-deriving from this buffer in a worse coordinate: the bin axis was
 * log2(len/L_full) with L_full a single observation that differed per face and
 * per tree, so merged bins meant slightly different things in different trees;
 * quantiles were taken on the pooled mixture rather than averaged per tree,
 * unlike every other accumulator in the forest; and binning quantized, which is
 * exactly the residual that made a synthetic walk of known dimension 3 read
 * 3.169. Scanning the buffer directly removes all three -- the same synthetic
 * now returns 1.0000, 2.0000, 3.0000 -- and shrinks the merged state from 2 *
 * 2d * bins doubles to 3 * 2d plus a few scalars.
 *
 * <p>
 * <b>The crossing is taken on the scalar, not per face.</b> The walk induces a
 * first-passage weight S(v)q_v per level summing to exactly 1. Scanning upward
 * from the leaf to where the cumulative reaches lambda selects one level, so
 * the 2d half-lengths reported all come from the enlarged bounding box of a
 * single real node: the node where the point is, at the median, about to be
 * separated. Letting each face cross at its own level would return a box that
 * corresponds to no node at all.
 *
 * <p>
 * <b>Why the path is buffered.</b> The accumulation runs bottom-up, so when
 * accept() fires on node v the survival product over v's <i>ancestors</i> is
 * not yet known. Rescaling afterwards is valid -- the damping is uniform, so
 * ratios between earlier contributions survive -- but it means dividing by a
 * product that reaches 1e-9 on a deep path and multiplying back at the end,
 * discarding digits exactly where paths are longest. RCF depth is O(log
 * sampleSize), so buffering and making one forward pass is both exact and
 * cheap. The buffer grows by doubling and then allocates nothing.
 *
 * <p>
 * <b>Length reconstruction.</b> The parent computes distComp[j] = pi_j * (gap_j
 * + oldRange_i) and then overwrites gap[] with pi in place, so the length is
 * distComp[j]/pi_j with no bounding box access. A point outside a box can
 * overhang on at most one side per axis, so on every face that receives weight
 * this equals the enlarged box range along that axis, which is monotone up the
 * path.
 */
public class AnisotropicDisplacementVisitor extends InterpolationVisitor {

    private static final int INITIAL_PATH_CAPACITY = 32;

    /** The crossing reported as the extent box, and the pair bracketing it. */
    public static final double LAMBDA = 0.5;
    public static final double LAMBDA_LO = 0.25;
    public static final double LAMBDA_HI = 0.75;

    /** Per-tree; cleared in reset(). */
    private final DirectionalScales scales;
    /** Per-query; NOT cleared in reset(). */
    private final DirectionalScales foldedScales;

    /** Which of the two the next toMeasure call should hand out. */
    private DirectionalScales pending;

    // per-tree path buffer, filled leaf-first by the bottom-up walk
    private double[] pathPi; // capacity * len
    private double[] pathLen; // capacity * len
    private double[] pathDecay; // capacity; 1 - probOfCut at that level
    private double[] pathMass; // capacity; node mass separated from at that level
    private int capacity;
    private int levels;

    // duplicate seed: per-component influence emitted when the query point
    // coincides with a leaf, so no length exists
    private double degenerateSelf;

    private boolean flushed;

    // scratch, reused across trees
    private double[] survivalAt; // capacity
    private double[] levelWeight; // capacity
    private final double[] boxAt; // len
    private final double[] loAt; // len
    private final double[] hiAt; // len

    AnisotropicDisplacementVisitor(int dimension, double pointMass, boolean centerOfMass) {
        super(dimension, pointMass, centerOfMass);
        this.scales = new DirectionalScales(dimension);
        this.foldedScales = new DirectionalScales(dimension);
        this.capacity = INITIAL_PATH_CAPACITY;
        this.pathPi = new double[capacity * len];
        this.pathLen = new double[capacity * len];
        this.pathDecay = new double[capacity];
        this.pathMass = new double[capacity];
        this.survivalAt = new double[capacity];
        this.levelWeight = new double[capacity];
        this.boxAt = new double[len];
        this.loAt = new double[len];
        this.hiAt = new double[len];
        clearPath();
    }

    public AnisotropicDisplacementVisitor(float[] pointToScore, int treeMass, double pointMass, boolean centerOfMass) {
        this(pointToScore.length, pointMass, centerOfMass);
        this.treeMass = treeMass;
    }

    // ------------------------------------------------------------------
    // hooks called by InterpolationVisitor
    // ------------------------------------------------------------------

    @Override
    protected void deposit(float[] prob, float[] lenComp, double decay, double mass) {
        ensureCapacity(levels + 1);
        int off = levels * len;
        for (int j = 0; j < len; j++) {
            double p = prob[j];
            pathPi[off + j] = p;
            pathLen[off + j] = (p > 0.0) ? lenComp[j] / p : 0.0;
        }
        pathDecay[levels] = decay;
        pathMass[levels] = mass;
        levels++;
    }

    @Override
    protected void depositDegenerate(double perComponentInfluence) {
        degenerateSelf = perComponentInfluence;
    }

    // ------------------------------------------------------------------
    // lifecycle
    // ------------------------------------------------------------------

    @Override
    protected void reset() {
        super.reset();
        scales.clear();
        clearPath();
    }

    @Override
    public void foldOut() {
        flush();
        DirectionalScales.addToLeft(foldedScales, scales);
        super.foldOut();
    }

    @Override
    public InterpolationMeasure getResult() {
        flush();
        pending = scales;
        return super.getResult();
    }

    @Override
    public InterpolationMeasure getFoldResult() {
        pending = foldedScales;
        return super.getFoldResult();
    }

    /**
     * Attaches a copy of the scales to an otherwise ordinary measure. No subclass
     * is constructed here: traverseForest seeds its accumulator with a plain
     * InterpolationMeasure, so a subclass-typed result would be merged into a
     * base-typed left operand and the payload would be dropped. The type is fixed
     * once at the end, in RandomCutForest.getAnisotropicDensity.
     */
    @Override
    protected InterpolationMeasure toMeasure(double[] m, double[] d, double[] p, double ss, double trees, double sc,
            double h) {
        InterpolationMeasure base = super.toMeasure(m, d, p, ss, trees, sc, h);
        base.setScales(new DirectionalScales((pending != null) ? pending : foldedScales));
        pending = null;
        return base;
    }

    @Override
    public void resetAcrossQueries(float[] point) {
        super.resetAcrossQueries(point);
        foldedScales.clear();
        scales.clear();
        clearPath();
    }

    private void clearPath() {
        levels = 0;
        degenerateSelf = 0.0;
        flushed = false;
    }

    private void ensureCapacity(int required) {
        if (required <= capacity) {
            return;
        }
        int target = capacity;
        while (target < required) {
            target <<= 1;
        }
        pathPi = Arrays.copyOf(pathPi, target * len);
        pathLen = Arrays.copyOf(pathLen, target * len);
        pathDecay = Arrays.copyOf(pathDecay, target);
        pathMass = Arrays.copyOf(pathMass, target);
        survivalAt = Arrays.copyOf(survivalAt, target);
        levelWeight = Arrays.copyOf(levelWeight, target);
        capacity = target;
    }

    // ------------------------------------------------------------------
    // the forward pass
    // ------------------------------------------------------------------

    /**
     * Survival top-down, then the lambda crossings scanned upward from the leaf,
     * then the exponent fit. Idempotent within a tree; cleared by reset().
     */
    private void flush() {
        if (flushed) {
            return;
        }
        flushed = true;
        if (levels == 0 && degenerateSelf == 0.0) {
            return;
        }

        // S(v) = product of (1 - q) over ancestors
        double s = 1.0;
        for (int lvl = levels - 1; lvl >= 0; lvl--) {
            survivalAt[lvl] = s;
            s *= pathDecay[lvl];
        }
        double leafSurvival = s;

        // scalar first-passage weight per level, and the exponent fit in one pass
        double total = 0.0;
        double sw = 0, sx = 0, sy = 0;
        int fitPoints = 0;
        for (int lvl = 0; lvl < levels; lvl++) {
            int off = lvl * len;
            double q = 0.0;
            for (int j = 0; j < len; j++) {
                q += pathPi[off + j];
            }
            levelWeight[lvl] = survivalAt[lvl] * q;
            total += levelWeight[lvl];

            if (pathMass[lvl] > 0.0) {
                double lnMass = Math.log(pathMass[lvl]);
                for (int j = 0; j < len; j++) {
                    double pi = pathPi[off + j], length = pathLen[off + j];
                    if (pi <= 0.0 || length <= 0.0) {
                        continue;
                    }
                    double w = survivalAt[lvl] * pi;
                    sw += w;
                    sx += w * Math.log(length);
                    sy += w * lnMass;
                    fitPoints++;
                }
            }
        }

        // alpha = d log(mass) / d log(length): a slope, so invariant to any rescaling
        // of either axis and free of the intercept term that biased a ratio estimator
        double alpha = Double.NaN;
        if (fitPoints >= 2 && sw > 0) {
            double xbar = sx / sw, ybar = sy / sw, sxx = 0, sxy = 0;
            for (int lvl = 0; lvl < levels; lvl++) {
                if (pathMass[lvl] <= 0.0) {
                    continue;
                }
                int off = lvl * len;
                double y = Math.log(pathMass[lvl]) - ybar;
                for (int j = 0; j < len; j++) {
                    double pi = pathPi[off + j], length = pathLen[off + j];
                    if (pi <= 0.0 || length <= 0.0) {
                        continue;
                    }
                    double w = survivalAt[lvl] * pi;
                    double x = Math.log(length) - xbar;
                    sxx += w * x * x;
                    sxy += w * x * y;
                }
            }
            if (sxx > 0) {
                alpha = sxy / sxx;
            }
        }

        crossing(LAMBDA, total, boxAt);
        crossing(LAMBDA_LO, total, loAt);
        crossing(LAMBDA_HI, total, hiAt);

        // The duplicate seed sits below every buffered level, so it inherits the whole
        // survival product. It has no length and never enters a crossing; it is
        // reported separately so a masked point is distinguishable from a measured one.
        double degenerate = degenerateSelf * len * leafSurvival;
        scales.observeTree(boxAt, loAt, hiAt, alpha, degenerate, total);
    }

    /**
     * Half-lengths at the level where the cumulative first-passage weight, scanned
     * upward from the leaf, first reaches lambda * total.
     *
     * <p>
     * There are only eight to twenty levels, so a hard crossing quantizes badly.
     * The fraction into the crossing level interpolates between it and the level
     * below, geometrically rather than linearly, because box ranges grow
     * multiplicatively up the path.
     *
     * <p>
     * A face carries no length at a level where the point does not overhang on that
     * side, since the length is recovered as distComp/pi and pi is zero there. Such
     * a face falls back to the nearest level that does have one, so the box stays
     * complete rather than collapsing a side to zero. That is a judgement call; the
     * alternative is reporting zero, which is arguably more honest and produces
     * degenerate boxes downstream.
     */
    private void crossing(double lambda, double total, double[] out) {
        Arrays.fill(out, 0.0);
        if (!(total > 0.0) || levels == 0) {
            return;
        }
        double target = lambda * total;
        double cumulative = 0.0;
        int hit = levels - 1;
        double frac = 1.0;
        for (int lvl = 0; lvl < levels; lvl++) {
            double w = levelWeight[lvl];
            if (cumulative + w >= target) {
                hit = lvl;
                frac = (w > 0.0) ? (target - cumulative) / w : 0.0;
                break;
            }
            cumulative += w;
        }
        int prev = Math.max(0, hit - 1);
        for (int j = 0; j < len; j++) {
            double a = lengthNear(prev, j);
            double b = lengthNear(hit, j);
            out[j] = (a > 0.0 && b > 0.0) ? Math.exp((1 - frac) * Math.log(a) + frac * Math.log(b)) : Math.max(a, b);
        }
    }

    /** Length on face j at that level, or from the nearest level that has one. */
    private double lengthNear(int level, int j) {
        for (int d = 0; d < levels; d++) {
            int up = level + d, down = level - d;
            if (up < levels && pathLen[up * len + j] > 0.0) {
                return pathLen[up * len + j];
            }
            if (down >= 0 && pathLen[down * len + j] > 0.0) {
                return pathLen[down * len + j];
            }
        }
        return 0.0;
    }

    // ------------------------------------------------------------------
    // factory
    // ------------------------------------------------------------------

    /**
     * Each visitor owns its own accumulators; sharing one through the closure would
     * race across trees on the parallel path and double count on the non-fold path.
     *
     * <p>
     * The type argument stays InterpolationMeasure because a visitor's result type
     * is pinned by RFVisitor&lt;R&gt; and cannot be narrowed. The scales travel
     * inside the measure and merge through InterpolationMeasure.addToLeft.
     */
    public static IVisitorFactory<InterpolationMeasure> reusableFactory(double pointMass, boolean centerOfMass) {
        return new IVisitorFactory<InterpolationMeasure>() {
            @Override
            public boolean isReusable() {
                return true;
            }

            @Override
            public boolean isReusableAcrossQueries() {
                return true;
            }

            @Override
            public boolean isFoldable() {
                return true;
            }

            @Override
            public IRFVisitor<InterpolationMeasure> newReusableVisitor(float[] point) {
                return new AnisotropicDisplacementVisitor(point.length, pointMass, centerOfMass);
            }

            @Override
            public Visitor<InterpolationMeasure> newVisitor(ITree<?, ?> tree, float[] point) {
                return new AnisotropicDisplacementVisitor(tree.projectToTree(point), tree.getMass(), pointMass,
                        centerOfMass);
            }
        };
    }
}
