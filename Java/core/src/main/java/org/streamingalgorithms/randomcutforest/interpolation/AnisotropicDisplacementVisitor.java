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
import org.streamingalgorithms.randomcutforest.returntypes.FirstPassageScales;
import org.streamingalgorithms.randomcutforest.returntypes.InterpolationMeasure;
import org.streamingalgorithms.randomcutforest.tree.INodeView;
import org.streamingalgorithms.randomcutforest.tree.ITree;

/**
 * Records boxes at the 0.5 and 0.95 quantiles of the completed first-separation
 * distribution, scanned leaf-to-root. Geometry is copied from growingBox at
 * each level; it is never reconstructed from probability-weighted lengths.
 *
 * First passage ends the upward traversal. Passage is achieved either by the
 * local cut probability falling to Q_STOP or by the node mass reaching
 * treeMass^massTargetExponent. Because the parent's accept() returns
 * immediately once pointInsideBox is set, measure, probMass and distances
 * receive no terms from levels above first passage: the recurrences hold their
 * value as of that level, so the length scale distances.getHighLowSum(i) /
 * probMass.getHighLowSum(i) is the scale at passage rather than a whole-path
 * average dominated by the root. The level that achieves passage is included,
 * since the parent calls deposit() before recur() within a level.
 *
 * Set stopAtFirstPassage false to restore the previous behaviour, in which only
 * Q_STOP terminated the walk and the mass crossing merely capped the three
 * selected levels. The score and height carried by the same
 * InterpolationMeasure are truncated along with the measure arrays, so a caller
 * that needs the whole-path score should use a separate visitor for it.
 *
 * The last visited box is the stopping box, which under truncation is the first
 * passage box. Outputs are averaged equally across trees by scales.
 */
public class AnisotropicDisplacementVisitor extends InterpolationVisitor {

    public static final double LAMBDA_CUT = 0.5;
    public static final double LAMBDA_PASSAGE = 0.95;
    /** Local cut-probability threshold for ending the traversal, not a quantile. */
    public static final double Q_STOP = 0.05;
    public static final double DEFAULT_MASS_EXPONENT = 0.5;
    public static final boolean DEFAULT_STOP_AT_FIRST_PASSAGE = true;
    private static final int INITIAL_CAPACITY = 32;

    private final double massTargetExponent;
    /** When true the mass crossing ends the walk instead of only capping levels. */
    private final boolean stopAtFirstPassage;
    private final double[] cutBox;
    private final double[] passageBox;
    private final double[] stopBox;

    // Leaf-first buffer. pathWeight initially holds q, then S*q after flush().
    private double[] pathBox;
    private double[] pathDecay;
    private double[] pathWeight;
    private int levels;
    private int massLevel = -1;
    private boolean flushed;
    private boolean folded;
    private boolean deposited;

    private final FirstPassageScales scales;
    private final FirstPassageScales foldedScales;
    private FirstPassageScales pending;

    AnisotropicDisplacementVisitor(int dimension, double pointMass, boolean centerOfMass) {
        this(dimension, pointMass, centerOfMass, DEFAULT_MASS_EXPONENT, DEFAULT_STOP_AT_FIRST_PASSAGE);
    }

    AnisotropicDisplacementVisitor(int dimension, double pointMass, boolean centerOfMass, double massTargetExponent) {
        this(dimension, pointMass, centerOfMass, massTargetExponent, DEFAULT_STOP_AT_FIRST_PASSAGE);
    }

    AnisotropicDisplacementVisitor(int dimension, double pointMass, boolean centerOfMass, double massTargetExponent,
            boolean stopAtFirstPassage) {
        super(dimension, pointMass, centerOfMass);
        this.massTargetExponent = massTargetExponent;
        this.stopAtFirstPassage = stopAtFirstPassage;
        cutBox = new double[len];
        passageBox = new double[len];
        stopBox = new double[len];
        pathBox = new double[INITIAL_CAPACITY * len];
        pathDecay = new double[INITIAL_CAPACITY];
        pathWeight = new double[INITIAL_CAPACITY];
        scales = new FirstPassageScales(dimension);
        foldedScales = new FirstPassageScales(dimension);
    }

    public AnisotropicDisplacementVisitor(float[] pointToScore, int treeMass, double pointMass, boolean centerOfMass) {
        this(pointToScore.length, pointMass, centerOfMass);
        this.treeMass = treeMass;
    }

    public AnisotropicDisplacementVisitor(float[] pointToScore, int treeMass, double pointMass, boolean centerOfMass,
            double massTargetExponent, boolean stopAtFirstPassage) {
        this(pointToScore.length, pointMass, centerOfMass, massTargetExponent, stopAtFirstPassage);
        this.treeMass = treeMass;
    }

    @Override
    public void accept(INodeView node, int depthOfNode) {
        if (pointInsideBox) {
            return;
        }
        deposited = false;
        super.accept(node, depthOfNode);
        // The parent updates growingBox but returns before deposit() at q = 0.
        // Keep this endpoint for the stopping box and the mass crossing.
        if (!deposited && pointInsideBox) {
            appendLevel(0.0, 1.0, node.getMass());
        }
    }

    @Override
    public void acceptLeaf(INodeView node, int depthOfNode) {
        deposited = false;
        super.acceptLeaf(node, depthOfNode);
        if (!deposited) {
            // Duplicate leaf has no geometric first-separation event. Keep its
            // zero box for a possible mass crossing, but assign no passage weight.
            // Quantiles are conditional on the nondegenerate passage weight.
            // The walk is deliberately not terminated here: a heavy duplicate can
            // set massLevel at level 0, and stopping would leave every selected
            // box degenerate and every volume zero.
            appendLevel(0.0, 0.0, node.getMass());
        }
    }

    @Override
    protected void deposit(float[] prob, float[] lenComp, double decay, double mass) {
        deposited = true;
        double q = 1.0 - decay;
        boolean massReachedBefore = massLevel >= 0;
        appendLevel(q, decay, mass);
        // First passage. Setting pointInsideBox makes the parent's accept() return
        // on the next level, so no further terms enter measure, probMass or
        // distances. Only a crossing carrying nondegenerate passage weight ends
        // the walk; see acceptLeaf.
        boolean massCrossedHere = stopAtFirstPassage && !massReachedBefore && massLevel == levels - 1;
        if (q <= Q_STOP || massCrossedHere) {
            pointInsideBox = true;
        }
    }

    private void appendLevel(double q, double decay, double mass) {
        if (levels == pathDecay.length) {
            int capacity = pathDecay.length * 2;
            pathBox = Arrays.copyOf(pathBox, capacity * len);
            pathDecay = Arrays.copyOf(pathDecay, capacity);
            pathWeight = Arrays.copyOf(pathWeight, capacity);
        }
        System.arraycopy(growingBox, 0, pathBox, levels * len, len);
        pathDecay[levels] = decay;
        pathWeight[levels] = q;
        if (massLevel < 0 && mass >= Math.pow(Math.max(1.0, treeMass), massTargetExponent)) {
            massLevel = levels;
        }
        levels++;
    }

    private void flush() {
        if (flushed) {
            return;
        }
        flushed = true;
        if (levels == 0) {
            return;
        }

        // Conditional q becomes an unconditional first-separation weight after
        // multiplying by survival through all buffered ancestors, top-down.
        // Under truncation the buffer ends at first passage, so total is the
        // passage weight of the truncated path and the quantiles below are taken
        // within it. This is what bounds passageBox away from the root box.
        double survival = 1.0;
        double total = 0.0;
        for (int level = levels - 1; level >= 0; level--) {
            pathWeight[level] *= survival;
            total += pathWeight[level];
            survival *= pathDecay[level];
        }

        int cutLevel = crossingLevel(LAMBDA_CUT, total);
        int passageLevel = crossingLevel(LAMBDA_PASSAGE, total);
        int stopLevel = levels - 1;
        // Retained for the acceptLeaf path, where massLevel can precede the last
        // buffered level. When deposit() ended the walk, massLevel == levels - 1
        // and these are no-ops.
        if (massLevel >= 0) {
            cutLevel = Math.min(cutLevel, massLevel);
            passageLevel = Math.min(passageLevel, massLevel);
            stopLevel = Math.min(stopLevel, massLevel);
        }
        copyBox(cutLevel, cutBox);
        copyBox(passageLevel, passageBox);
        copyBox(stopLevel, stopBox);
        scales.observeTree(cutBox, passageBox, stopBox);
    }

    /** First leaf-to-root crossing of lambda times the total passage weight. */
    private int crossingLevel(double lambda, double total) {
        if (!(total > 0.0)) {
            // No nondegenerate separation: report the zero leaf geometry.
            return 0;
        }
        double target = lambda * total;
        double cumulative = 0.0;
        int lastPositive = 0;
        for (int level = 0; level < levels; level++) {
            if (pathWeight[level] > 0.0) {
                lastPositive = level;
                cumulative += pathWeight[level];
                if (cumulative >= target) {
                    return level;
                }
            }
        }
        return lastPositive; // floating-point summation fallback
    }

    private void copyBox(int level, double[] out) {
        System.arraycopy(pathBox, level * len, out, 0, len);
    }

    @Override
    protected void reset() {
        super.reset();
        levels = 0;
        massLevel = -1;
        flushed = folded = deposited = false;
        pending = null;
        scales.clear();
        Arrays.fill(cutBox, 0.0);
        Arrays.fill(passageBox, 0.0);
        Arrays.fill(stopBox, 0.0);
        // Buffer slots are overwritten on append; no per-tree allocation/clearing.
    }

    @Override
    public void resetAcrossQueries(float[] point) {
        super.resetAcrossQueries(point);
        foldedScales.clear();
    }

    @Override
    public void foldOut() {
        if (folded) {
            return;
        }
        flush();
        FirstPassageScales.addToLeft(foldedScales, scales);
        super.foldOut();
        folded = true;
    }

    @Override
    public InterpolationMeasure getResult() {
        flush();
        pending = scales;
        InterpolationMeasure result = super.getResult();
        // The base sets its sampleSize in foldOut(), but not on the direct path.
        result.setSampleSize(treeMass);
        return result;
    }

    @Override
    public InterpolationMeasure getFoldResult() {
        pending = foldedScales;
        return super.getFoldResult();
    }

    @Override
    protected InterpolationMeasure toMeasure(double[] m, double[] d, double[] p, double ss, double trees, double sc,
            double h) {
        InterpolationMeasure base = super.toMeasure(m, d, p, ss, trees, sc, h);
        base.setScales(new FirstPassageScales(pending != null ? pending : scales));
        pending = null;
        return base;
    }

    public static IVisitorFactory<InterpolationMeasure> reusableFactory(double pointMass, boolean centerOfMass) {
        return reusableFactory(pointMass, centerOfMass, DEFAULT_MASS_EXPONENT, DEFAULT_STOP_AT_FIRST_PASSAGE);
    }

    public static IVisitorFactory<InterpolationMeasure> reusableFactory(double pointMass, boolean centerOfMass,
            double massTargetExponent, boolean stopAtFirstPassage) {
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
                return new AnisotropicDisplacementVisitor(point.length, pointMass, centerOfMass, massTargetExponent,
                        stopAtFirstPassage);
            }

            @Override
            public Visitor<InterpolationMeasure> newVisitor(ITree<?, ?> tree, float[] point) {
                return new AnisotropicDisplacementVisitor(tree.projectToTree(point), tree.getMass(), pointMass,
                        centerOfMass, massTargetExponent, stopAtFirstPassage);
            }
        };
    }
}