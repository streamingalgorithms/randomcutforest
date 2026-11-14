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

import java.util.Arrays;

/**
 * What a forest reads of one point's escape: how a random-cut insertion of the
 * point would separate it, tree by tree, summed over the trees and divided on
 * the way out. Faces are half dimensions: face k is the max face of axis k for
 * k &lt; d and the min face of axis k - d otherwise.
 *
 * <ul>
 * <li>codisplacement: with the point attached where the insertion separates it,
 * the largest ratio over its path of a sibling's mass to the mass of the
 * subtree holding the point, averaged over where it attaches and over the
 * trees;</li>
 * <li>smoothed codisplacement: the same, each ratio entering by its square root
 * and the mean squared, so that trees which disagree count for less;</li>
 * <li>range escape: the expected range of the box with the point where it
 * separates, over the range of the root box with the point;</li>
 * <li>face exits: where the insertion separates the point, a walk from the
 * point along the axis of a face it lies beyond, by a gap g, reaches the plane
 * of the root box's face beyond the point, at distance r, before the data box
 * with probability g / (g + r); the face is drawn in proportion to its gap. The
 * exits over the faces sum to at most one, the rest being return;</li>
 * <li>passage mass: the geometric mean over trees of the mass of the passage
 * box, and the mean gaps of the cut box, both under the rules of a
 * {@link LocalGeometryConfig};</li>
 * <li>a reference row, when the walk also read a reference set: see
 * {@link org.streamingalgorithms.randomcutforest.interpolation.ReferenceSums}.</li>
 * </ul>
 *
 * Two partial readings of the same point merge by addition, in any order.
 */
public final class Escape {

    private final int dimensions;
    private int trees;
    private double codisplacement, rootCodisplacement, rangeEscape, logPassageMass;
    private final double[] faceExit, cutGap;
    private double[] row; // null until a reference row is added
    private int rowTrees;

    public Escape(int dimensions) {
        checkArgument(dimensions > 0, "dimensions must be greater than 0");
        this.dimensions = dimensions;
        this.faceExit = new double[2 * dimensions];
        this.cutGap = new double[2 * dimensions];
    }

    public void clear() {
        trees = 0;
        rowTrees = 0;
        codisplacement = rootCodisplacement = rangeEscape = logPassageMass = 0.0;
        Arrays.fill(faceExit, 0.0);
        Arrays.fill(cutGap, 0.0);
        if (row != null) {
            Arrays.fill(row, 0.0);
        }
    }

    /** One tree's reading; the face arrays are read from their start. */
    public void addTree(double codisplacement, double rootCodisplacement, double rangeEscape, double logPassageMass,
            double[] faceExit, double[] cutGap) {
        this.codisplacement += codisplacement;
        this.rootCodisplacement += rootCodisplacement;
        this.rangeEscape += rangeEscape;
        this.logPassageMass += logPassageMass;
        for (int f = 0; f < this.faceExit.length; f++) {
            this.faceExit[f] += faceExit[f];
            this.cutGap[f] += cutGap[f];
        }
        trees++;
    }

    /** One tree's reference row, of the columns of the reference sums. */
    public void addRow(double[] treeRow, int columns) {
        if (row == null || row.length != columns) {
            row = new double[columns];
        }
        for (int c = 0; c < columns; c++) {
            row[c] += treeRow[c];
        }
        rowTrees++;
    }

    /** Adds the trees of another reading of the same point. */
    public Escape add(Escape other) {
        checkArgument(other.dimensions == dimensions, "dimensions must be the same");
        codisplacement += other.codisplacement;
        rootCodisplacement += other.rootCodisplacement;
        rangeEscape += other.rangeEscape;
        logPassageMass += other.logPassageMass;
        for (int f = 0; f < faceExit.length; f++) {
            faceExit[f] += other.faceExit[f];
            cutGap[f] += other.cutGap[f];
        }
        trees += other.trees;
        if (other.row != null && other.rowTrees > 0) {
            if (row == null || row.length != other.row.length) {
                row = new double[other.row.length];
                rowTrees = 0;
            }
            for (int c = 0; c < row.length; c++) {
                row[c] += other.row[c];
            }
            rowTrees += other.rowTrees;
        }
        return this;
    }

    /** Adds right into left and returns left: the accumulator of a forest. */
    public static Escape merge(Escape left, Escape right) {
        return left.add(right);
    }

    public Escape copy() {
        Escape out = new Escape(dimensions);
        out.add(this);
        return out;
    }

    public int getDimensions() {
        return dimensions;
    }

    public int getTreeCount() {
        return trees;
    }

    private double mean(double sum) {
        return (trees > 0) ? sum / trees : 0.0;
    }

    public double getCodisplacement() {
        return mean(codisplacement);
    }

    public double getSmoothedCodisplacement() {
        double root = mean(rootCodisplacement);
        return root * root;
    }

    public double getRangeEscape() {
        return mean(rangeEscape);
    }

    /** The probability of leaving by face f. */
    public double getFaceExit(int face) {
        return mean(faceExit[face]);
    }

    /** The probability of leaving by any face. */
    public double getExit() {
        double sum = 0;
        for (double v : faceExit) {
            sum += v;
        }
        return mean(sum);
    }

    /** The geometric mean over trees of the passage box's mass. */
    public double getPassageMass() {
        return (trees > 0) ? Math.exp(logPassageMass / trees) : 0.0;
    }

    /** The mean over trees of the cut box's gap at face f. */
    public double getCutGap(int face) {
        return mean(cutGap[face]);
    }

    /** True when the walk read a reference set in at least one tree. */
    public boolean hasRow() {
        return row != null && rowTrees > 0;
    }

    /**
     * The reference row: per column, the mean over the trees that hold the
     * reference set of the point's affinity-weighted sums. Null without one.
     */
    public double[] getRow() {
        if (!hasRow()) {
            return null;
        }
        double[] out = new double[row.length];
        for (int c = 0; c < out.length; c++) {
            out[c] = row[c] / rowTrees;
        }
        return out;
    }

    /** Column c of the reference row, without the copy. */
    public double getRow(int column) {
        return row[column] / rowTrees;
    }
}
