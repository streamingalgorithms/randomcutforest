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
 * Three boxes around the query, indexed by the cumulative probability that a
 * root-to-leaf descent has separated the query by the time it reaches that
 * level.
 *
 * <pre>
 *   cutBox     P[separated] = P_CUT,     bounded by the mass rule
 *   passageBox P[separated] = P_PASSAGE, bounded by the mass rule
 *   stopBox    P[separated] = P_PASSAGE, not bounded
 * </pre>
 *
 * So cutBox is contained in passageBox is contained in stopBox, and stopBox is
 * what passageBox would have been with no mass rule.
 *
 * Each node carries a local cut probability q: the chance that one uniform cut
 * of the box merging that node with the query separates the query alone. It is
 * local, so the cumulative quantity is the survival S = product of (1 - q) over
 * the levels above, and P[separated] = 1 - S. S is only known once the walk
 * reaches the root, which is why the path is buffered rather than thresholded
 * on the way up.
 *
 * The walk ends when q falls to Q_STOP. The mass crossing does not end it: it
 * is recorded and bounds cutBox and passageBox in flush, so the walk length is
 * the same for every query and the InterpolationMeasure accumulators -- and
 * therefore passageDensity -- do not depend on where the mass crossing fell.
 *
 * Geometry is copied from growingBox at each level and never reconstructed from
 * probability-weighted lengths. Outputs are averaged equally across trees by
 * scales.
 */
public class AnisotropicDisplacementVisitor extends InterpolationVisitor {

    /** Separation probability at which cutBox is read. */
    public static final double P_CUT = 0.5;
    /** Separation probability at which passageBox and stopBox are read. */
    public static final double P_PASSAGE = 0.05;
    /**
     * Local cut probability that ends the traversal. Not a separation probability.
     */
    public static final double Q_STOP = 0.05;
    public static final double DEFAULT_MASS_EXPONENT = 0.5;

    private static final int INITIAL_CAPACITY = 32;

    private final double massTargetExponent;
    private final double[] cutBox;
    private final double[] passageBox;
    private final double[] stopBox;

    /** Leaf-first buffer: node geometry and 1 - q at each level. */
    private double[] pathBox;
    private double[] pathDecay;
    private int levels;
    private int massLevel = -1;
    private boolean flushed;
    private boolean folded;
    private boolean deposited;

    private final FirstPassageScales scales;
    private final FirstPassageScales foldedScales;
    private FirstPassageScales pending;

    AnisotropicDisplacementVisitor(int dimension, double pointMass, boolean centerOfMass, double massTargetExponent) {
        super(dimension, pointMass, centerOfMass);
        this.massTargetExponent = massTargetExponent;
        cutBox = new double[len];
        passageBox = new double[len];
        stopBox = new double[len];
        pathBox = new double[INITIAL_CAPACITY * len];
        pathDecay = new double[INITIAL_CAPACITY];
        scales = new FirstPassageScales(dimension);
        foldedScales = new FirstPassageScales(dimension);
    }

    public AnisotropicDisplacementVisitor(float[] pointToScore, int treeMass, double pointMass, boolean centerOfMass,
            double massTargetExponent) {
        this(pointToScore.length, pointMass, centerOfMass, massTargetExponent);
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
        // Keep the endpoint: it is the outermost geometry the walk saw.
        if (!deposited && pointInsideBox) {
            appendLevel(1.0, node.getMass());
        }
    }

    @Override
    public void acceptLeaf(INodeView node, int depthOfNode) {
        deposited = false;
        super.acceptLeaf(node, depthOfNode);
        if (!deposited) {
            // Duplicate leaf: the gap is zero, so probOfCut is zero and the
            // survival through this level is 1. Its box is the degenerate leaf
            // geometry, which is why the walk is not terminated here.
            appendLevel(1.0, node.getMass());
        }
    }

    @Override
    protected void deposit(float[] prob, float[] lenComp, double decay, double mass) {
        deposited = true;
        appendLevel(decay, mass);
        // Setting pointInsideBox makes the parent's accept() return on the next
        // level, so nothing further enters measure, probMass or distances.
        if (1.0 - decay <= Q_STOP) {
            pointInsideBox = true;
        }
    }

    /** @param decay 1 - q at this level */
    private void appendLevel(double decay, double mass) {
        if (levels == pathDecay.length) {
            int capacity = pathDecay.length * 2;
            pathBox = Arrays.copyOf(pathBox, capacity * len);
            pathDecay = Arrays.copyOf(pathDecay, capacity);
        }
        System.arraycopy(growingBox, 0, pathBox, levels * len, len);
        pathDecay[levels] = decay;
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

        // Descend from the root accumulating survival. P[separated] rises as the
        // levels get deeper, so the deepest level still under a threshold is the
        // crossing, and a lower threshold crosses higher up and gives a larger box.
        double survival = 1.0;
        int cutLevel = levels - 1;
        int passageLevel = levels - 1;
        for (int level = levels - 1; level >= 0; level--) {
            double separated = 1.0 - survival;
            if (separated <= P_PASSAGE) {
                passageLevel = level;
            }
            if (separated <= P_CUT) {
                cutLevel = level;
            }
            survival *= pathDecay[level];
        }

        int stopLevel = passageLevel;
        if (massLevel >= 0) {
            cutLevel = Math.min(cutLevel, massLevel);
            passageLevel = Math.min(passageLevel, massLevel);
        }

        copyBox(cutLevel, cutBox);
        copyBox(passageLevel, passageBox);
        copyBox(stopLevel, stopBox);
        scales.observeTree(cutBox, passageBox, stopBox);
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
        // Buffer slots are overwritten on append; no per-tree clearing.
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
        return reusableFactory(pointMass, centerOfMass, DEFAULT_MASS_EXPONENT);
    }

    public static IVisitorFactory<InterpolationMeasure> reusableFactory(double pointMass, boolean centerOfMass,
            double massTargetExponent) {
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
                return new AnisotropicDisplacementVisitor(point.length, pointMass, centerOfMass, massTargetExponent);
            }

            @Override
            public Visitor<InterpolationMeasure> newVisitor(ITree<?, ?> tree, float[] point) {
                return new AnisotropicDisplacementVisitor(tree.projectToTree(point), tree.getMass(), pointMass,
                        centerOfMass, massTargetExponent);
            }
        };
    }
}