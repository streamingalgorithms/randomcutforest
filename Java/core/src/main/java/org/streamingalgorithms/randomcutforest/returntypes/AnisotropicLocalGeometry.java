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
import static org.streamingalgorithms.randomcutforest.CommonUtils.checkNotNull;

import java.util.Arrays;

import org.streamingalgorithms.randomcutforest.tree.VectorSupport;

/**
 * The local geometry of a query, as the trees of a forest read it at three
 * nested boxes on their paths (see {@link LocalGeometryConfig}): the cut box,
 * the passage box, and the reading box. Each is a {@link LocalBox}, averaged
 * over the trees before any ratio is taken.
 *
 * <p>
 * The lead and the swing describe the gap box of each tree's cut box: the void
 * between the query and the face of the data box it lies beyond, into which
 * that tree's next cut would fall. The lead is the mean over trees of the
 * offset from the query to the centre of that box, and the swing the covariance
 * of those centres across trees.
 *
 * <p>
 * Immutable. It holds sums over trees and divides on the way out, so
 * {@link #merge} is exact addition and two partial results merge in any order.
 * Every accessor returns a fresh array. All lengths are in the coordinates of
 * the tree, in double.
 */
public final class AnisotropicLocalGeometry {

    /**
     * Index of the cut box in the per-box arrays of {@link Accumulator#addTree}.
     */
    public static final int CUT = 0;
    /** Index of the passage box. */
    public static final int PASSAGE = 1;
    /** Index of the reading box. */
    public static final int READING = 2;
    public static final int BOXES = 3;

    private final LocalGeometryConfig config;
    private final int dimensions;
    private final int treeCount;
    private final double treeMassSum;
    private final double[] reachSum; // BOXES * 2d, box-major
    private final double[] gapSum;
    private final double[] massSum; // BOXES
    private final double[] cutSum;
    private final int[] capped;
    private final double[] centreSum; // d
    private final double[] centreProducts; // i <= j, packed by rows

    private AnisotropicLocalGeometry(Accumulator a) {
        this.config = a.config;
        this.dimensions = a.dimensions;
        this.treeCount = a.treeCount;
        this.treeMassSum = a.treeMassSum;
        this.reachSum = a.reachSum.clone();
        this.gapSum = a.gapSum.clone();
        this.massSum = a.massSum.clone();
        this.cutSum = a.cutSum.clone();
        this.capped = a.capped.clone();
        this.centreSum = a.centreSum.clone();
        this.centreProducts = a.centreProducts.clone();
    }

    /** No trees: every box is empty and {@link #isEmpty()} is true. */
    public static AnisotropicLocalGeometry empty(int dimensions, LocalGeometryConfig config) {
        return new Accumulator(dimensions, config).toGeometry();
    }

    /**
     * The geometry of the trees of both, which must share dimensions and
     * configuration.
     */
    public static AnisotropicLocalGeometry merge(AnisotropicLocalGeometry left, AnisotropicLocalGeometry right) {
        Accumulator a = new Accumulator(left.dimensions, left.config);
        a.add(left);
        a.add(right);
        return a.toGeometry();
    }

    public LocalGeometryConfig getConfig() {
        return config;
    }

    public int getDimensions() {
        return dimensions;
    }

    public int getTreeCount() {
        return treeCount;
    }

    /** True when no tree contributed. */
    public boolean isEmpty() {
        return treeCount == 0;
    }

    /** Mean tree mass, the psi of the configuration's masses. */
    public double getTreeMass() {
        return (treeCount > 0) ? treeMassSum / treeCount : 0.0;
    }

    public LocalBox cut() {
        return box(CUT);
    }

    public LocalBox passage() {
        return box(PASSAGE);
    }

    public LocalBox reading() {
        return box(READING);
    }

    private LocalBox box(int which) {
        int len = 2 * dimensions;
        double[] reach = new double[len];
        double[] gap = new double[len];
        double n = Math.max(1, treeCount);
        for (int k = 0; k < len; k++) {
            reach[k] = reachSum[which * len + k] / n;
            gap[k] = gapSum[which * len + k] / n;
        }
        return new LocalBox(reach, gap, massSum[which] / n, cutSum[which] / n, capped[which]);
    }

    /**
     * Mean offset, axis by axis, from the query to the centre of each tree's gap
     * box.
     */
    public double[] lead() {
        double[] out = new double[dimensions];
        for (int i = 0; i < dimensions && treeCount > 0; i++) {
            out[i] = centreSum[i] / treeCount;
        }
        return out;
    }

    /** Covariance across trees of the gap-box centre, d by d; zero for one tree. */
    public double[][] swing() {
        double[][] out = new double[dimensions][dimensions];
        if (treeCount == 0) {
            return out;
        }
        double[] mean = lead();
        int k = 0;
        for (int i = 0; i < dimensions; i++) {
            for (int j = i; j < dimensions; j++) {
                double value = centreProducts[k++] / treeCount - mean[i] * mean[j];
                out[i][j] = value;
                out[j][i] = value;
            }
        }
        return out;
    }

    @Override
    public String toString() {
        return "AnisotropicLocalGeometry[trees=" + treeCount + ", cut=" + cut() + ", passage=" + passage()
                + ", reading=" + reading() + ", lead=" + Arrays.toString(lead()) + "]";
    }

    /**
     * The mutable sums a visitor folds trees into; not a result. Allocation free
     * per tree: {@link #addTree} reads its arrays and keeps nothing.
     */
    public static final class Accumulator {
        private final LocalGeometryConfig config;
        private final int dimensions;
        private final int len;
        private int treeCount;
        private double treeMassSum;
        private final double[] reachSum;
        private final double[] gapSum;
        private final double[] massSum;
        private final double[] cutSum;
        private final int[] capped;
        private final double[] centreSum;
        private final double[] centreProducts;

        public Accumulator(int dimensions, LocalGeometryConfig config) {
            checkArgument(dimensions > 0, "dimensions must be greater than 0");
            checkNotNull(config, "config must not be null");
            this.config = config;
            this.dimensions = dimensions;
            this.len = 2 * dimensions;
            reachSum = new double[BOXES * len];
            gapSum = new double[BOXES * len];
            massSum = new double[BOXES];
            cutSum = new double[BOXES];
            capped = new int[BOXES];
            centreSum = new double[dimensions];
            centreProducts = new double[dimensions * (dimensions + 1) / 2];
        }

        public void clear() {
            treeCount = 0;
            treeMassSum = 0.0;
            Arrays.fill(reachSum, 0.0);
            Arrays.fill(gapSum, 0.0);
            Arrays.fill(massSum, 0.0);
            Arrays.fill(cutSum, 0.0);
            Arrays.fill(capped, 0);
            Arrays.fill(centreSum, 0.0);
            Arrays.fill(centreProducts, 0.0);
        }

        /**
         * One tree. {@code reach} and {@code gap} hold the three boxes in the order
         * CUT, PASSAGE, READING, each 2d in half dimensions (see {@link LocalBox}); the
         * other arrays hold one entry per box.
         */
        public void addTree(double treeMass, double[] reach, double[] gap, double[] mass, double[] cutProbability,
                boolean[] isCapped) {
            checkArgument(reach.length == BOXES * len && gap.length == BOXES * len, "expected three boxes of 2d");
            VectorSupport.axpyInto(reachSum, reach, 1.0);
            VectorSupport.axpyInto(gapSum, gap, 1.0);
            for (int b = 0; b < BOXES; b++) {
                massSum[b] += mass[b];
                cutSum[b] += cutProbability[b];
                capped[b] += isCapped[b] ? 1 : 0;
            }
            // The gap box of the cut box: gap[i] > 0 puts the void below the query on
            // axis i, gap[d + i] > 0 above it, so its centre sits at half their
            // difference from the query.
            int base = CUT * len;
            int k = 0;
            for (int i = 0; i < dimensions; i++) {
                double ci = 0.5 * (gap[base + dimensions + i] - gap[base + i]);
                centreSum[i] += ci;
                for (int j = i; j < dimensions; j++) {
                    centreProducts[k++] += ci * 0.5 * (gap[base + dimensions + j] - gap[base + j]);
                }
            }
            treeMassSum += treeMass;
            treeCount++;
        }

        /** Adds the trees of a result. */
        public void add(AnisotropicLocalGeometry g) {
            checkNotNull(g, "geometry must not be null");
            checkArgument(g.dimensions == dimensions, "dimensions must be the same");
            checkArgument(g.config.equals(config), "geometries read under different configurations do not merge");
            VectorSupport.axpyInto(reachSum, g.reachSum, 1.0);
            VectorSupport.axpyInto(gapSum, g.gapSum, 1.0);
            for (int b = 0; b < BOXES; b++) {
                massSum[b] += g.massSum[b];
                cutSum[b] += g.cutSum[b];
                capped[b] += g.capped[b];
            }
            VectorSupport.axpyInto(centreSum, g.centreSum, 1.0);
            VectorSupport.axpyInto(centreProducts, g.centreProducts, 1.0);
            treeMassSum += g.treeMassSum;
            treeCount += g.treeCount;
        }

        public AnisotropicLocalGeometry toGeometry() {
            return new AnisotropicLocalGeometry(this);
        }
    }
}