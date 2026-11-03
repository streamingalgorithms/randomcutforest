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

package org.streamingalgorithms.randomcutforest.examples;

import static java.lang.Math.log;

import java.util.List;
import java.util.stream.Collectors;

import org.streamingalgorithms.randomcutforest.RandomCutForest;
import org.streamingalgorithms.randomcutforest.interpolation.AnisotropicLocalVisitor;
import org.streamingalgorithms.randomcutforest.returntypes.AnisotropicLocalGeometry;
import org.streamingalgorithms.randomcutforest.returntypes.LocalBox;
import org.streamingalgorithms.randomcutforest.returntypes.LocalGeometryConfig;

/**
 * Readings of an {@link AnisotropicLocalGeometry} that the examples use and the
 * library does not freeze: ratios and summaries still being worked on.
 */
public final class LocalGeometryReaders {

    private LocalGeometryReaders() {
    }

    /**
     * Covariance across trees, d by d, of the centre of each tree's cut box (the
     * box with the query) relative to the query: the shape the isolines give each
     * point. Unlike the swing, which follows the gap box and vanishes deep inside
     * the data, it is positive wherever the trees disagree about the cut box. One
     * pass over the trees, collecting each tree's geometry; zero until the forest
     * is ready to output.
     */
    public static double[][] cutSpread(RandomCutForest forest, float[] point) {
        if (!forest.isOutputReady()) {
            return new double[forest.getDimensions()][forest.getDimensions()];
        }
        List<AnisotropicLocalGeometry> trees = forest.traverseForest(forest.transformToShingledPoint(point),
                AnisotropicLocalVisitor.factory(LocalGeometryConfig.DEFAULT), Collectors.toList());
        int d = trees.isEmpty() ? point.length : trees.get(0).getDimensions();
        double[] mean = new double[d];
        double[][] out = new double[d][d];
        int n = 0;
        for (AnisotropicLocalGeometry tree : trees) {
            if (tree.isEmpty()) {
                continue;
            }
            double[] reach = tree.cut().reach();
            double[] c = new double[d];
            for (int i = 0; i < d; i++) {
                c[i] = 0.5 * (reach[i] - reach[d + i]); // above less below
                mean[i] += c[i];
            }
            for (int i = 0; i < d; i++) {
                for (int j = i; j < d; j++) {
                    out[i][j] += c[i] * c[j];
                }
            }
            n++;
        }
        if (n == 0) {
            return out;
        }
        for (int i = 0; i < d; i++) {
            mean[i] /= n;
        }
        for (int i = 0; i < d; i++) {
            for (int j = i; j < d; j++) {
                out[i][j] = out[i][j] / n - mean[i] * mean[j];
                out[j][i] = out[i][j];
            }
        }
        return out;
    }

    /**
     * Largest over smallest positive axis width of the box with the query; 1 when
     * no axis has width. Relative to the coordinate axes: a ridge at 45 degrees
     * widens every axis equally and reads as isotropic.
     */
    public static double anisotropy(LocalBox box) {
        double max = 0.0, min = Double.MAX_VALUE;
        for (double w : box.widthWithQuery()) {
            if (w > 0.0) {
                max = Math.max(max, w);
                min = Math.min(min, w);
            }
        }
        return (min < Double.MAX_VALUE) ? max / min : 1.0;
    }

    /**
     * Adds to out[0..2], per query, the mean over axes of the log ratios of axis
     * widths: cut box over its gap box, passage box over cut box, reading box over
     * passage box, each over the axes where both widths are positive. The widths
     * are means over trees, so the ratios are of averaged boxes.
     */
    public static void addLogRatios(AnisotropicLocalGeometry geometry, double[] out) {
        if (out == null) {
            return; // the caller does not want the ratios
        }
        LocalBox cut = geometry.cut();
        double[] gap = cut.gap();
        double[] c = cut.widthWithQuery();
        double[] p = geometry.passage().widthWithQuery();
        double[] r = geometry.reading().widthWithQuery();
        int d = c.length;
        double[] acc = new double[3];
        int[] terms = new int[3];
        for (int i = 0; i < d; i++) {
            double g = gap[i] + gap[d + i];
            if (c[i] > 0 && g > 0) {
                acc[0] += log(c[i] / g);
                terms[0]++;
            }
            if (p[i] > 0 && c[i] > 0) {
                acc[1] += log(p[i] / c[i]);
                terms[1]++;
            }
            if (r[i] > 0 && p[i] > 0) {
                acc[2] += log(r[i] / p[i]);
                terms[2]++;
            }
        }
        for (int k = 0; k < 3; k++) {
            if (terms[k] > 0) {
                out[k] += acc[k] / terms[k];
            }
        }
    }
}
