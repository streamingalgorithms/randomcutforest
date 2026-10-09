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

package org.streamingalgorithms.randomcutforest.examples.plot;

import java.awt.Color;
import java.awt.Graphics2D;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Isolines drawn through the cut.
 *
 * <p>
 * Each reference point brings the forest's own ellipse at that point: the
 * covariance, across trees, of where each tree's cut box sits around it --
 * {@code AnisotropicLocalGeometry.cutSpread()}, or its shadow on the plot when
 * the data has more than two dimensions. The density is a sum of unit-mass
 * Gaussians, one per point, centred on the point and shaped by its ellipse.
 *
 * <p>
 * The forest picks the shape and the sample picks the scale. Every ellipse
 * keeps its orientation and aspect, and its area is set to the median over the
 * reference: one point's tree-to-tree spread is a fair shape and a noisy size.
 * Each also gets the grid cell's own variance, a twelfth of the cell squared on
 * either axis, so a node reads the average over its cell and an ellipse thinner
 * than a cell cannot slip between nodes. There is no floor and no probe grid.
 *
 * <p>
 * Take the reference before the points are inserted: it is then a sample of the
 * world at that moment, asked of a forest that has not yet been told about it.
 *
 * <p>
 * The levels are shares of the mass rather than quantiles of the canvas: by
 * default the lines hold 95%, 75%, 50% and 25% of it. A canvas is mostly empty,
 * so quantiles of its nodes would spend the lower levels on the far field and
 * move whenever the data takes up more or less of the picture.
 *
 * <p>
 * Nothing is computed until the layer is first drawn, so one gated off with
 * {@link Layers#when} costs only its reference, and turning it on over a paused
 * frame still draws that frame's lines.
 */
public final class CutIsolines implements Layer {

    /**
     * Largest axis ratio kept. A spread whose tree centres all fall on one line has
     * no second axis, and scaling it to the median area would stretch it without
     * bound.
     */
    private static final double MAX_ASPECT = 12.0;

    private final double lo;
    private final double span;
    private final int grid;
    private final Color color;
    private double[] mass = { 0.95, 0.75, 0.5, 0.25 };
    private Color bandColor;
    private int bandFloor;

    /** {x, y, sxx, sxy, syy} per reference point. */
    private final List<double[]> ellipses = new ArrayList<>();
    private List<Layer> drawn;

    /**
     * @param lo    lower bound of the square on both axes
     * @param span  side of the square
     * @param grid  nodes per axis
     * @param color line colour; alpha ramps with the level
     */
    public CutIsolines(double lo, double span, int grid, Color color) {
        this.lo = lo;
        this.span = span;
        this.grid = grid;
        this.color = color;
    }

    /**
     * The share of the mass inside each line; the default is 0.95, 0.75, 0.5 and
     * 0.25.
     */
    public CutIsolines levels(double... massInside) {
        this.mass = massInside.clone();
        return this;
    }

    /**
     * Shades the bands between the levels from the given one up, under the lines.
     */
    public CutIsolines bands(Color fill, int floor) {
        this.bandColor = fill;
        this.bandFloor = floor;
        return this;
    }

    /**
     * A reference point in the plane, with its d by d spread; the first two axes
     * are drawn.
     */
    public void add(float[] point, double[][] spread) {
        add(point[0], point[1], spread[0][0], spread[0][1], spread[1][1]);
    }

    /**
     * A reference point drawn at (x, y), with its spread seen through the plot's
     * axes u and v: the drawn ellipse is the 2 by 2 matrix [u v]' S [u v]. For a
     * linear view u and v are its two directions; for a camera they are the rows of
     * the projection's Jacobian at the point.
     */
    public void add(double x, double y, double[][] spread, double[] u, double[] v) {
        double uu = 0;
        double uv = 0;
        double vv = 0;
        for (int i = 0; i < u.length; i++) {
            double su = 0;
            double sv = 0;
            for (int j = 0; j < u.length; j++) {
                su += spread[i][j] * u[j];
                sv += spread[i][j] * v[j];
            }
            uu += u[i] * su;
            uv += u[i] * sv;
            vv += v[i] * sv;
        }
        add(x, y, uu, uv, vv);
    }

    public void add(double x, double y, double sxx, double sxy, double syy) {
        ellipses.add(new double[] { x, y, sxx, sxy, syy });
    }

    @Override
    public void draw(Graphics2D g, Plot2D.Viewport vp) {
        if (ellipses.isEmpty()) {
            return;
        }
        for (Layer layer : layers()) {
            layer.draw(g, vp);
        }
    }

    private synchronized List<Layer> layers() {
        if (drawn == null) {
            Contour.Field density = field();
            double[][] v = Contour.sample(density, lo, span, grid);
            double[] levels = Contour.massLevels(v, mass);
            drawn = new ArrayList<>();
            if (bandColor != null) {
                drawn.add(bandLayer(v, levels));
            }
            drawn.addAll(Contour.isolines(density, v, lo, span, levels, color));
        }
        return drawn;
    }

    /**
     * The density the lines are drawn from: every ellipse at the median scale plus
     * the cell's own variance, as unit-mass Gaussians centred on the points. It can
     * be asked anywhere, on the grid or off it.
     */
    public Contour.Field field() {
        double cell = Math.pow(span / (grid - 1.0), 2) / 12.0;
        int n = ellipses.size();
        double[][] capped = new double[n][];
        double[] roots = new double[n];
        int shaped = 0;
        for (double[] e : ellipses) {
            double a = e[2];
            double b = e[3];
            double c = e[4];
            double half = 0.5 * (a + c);
            double disc = Math.sqrt(0.25 * (a - c) * (a - c) + b * b);
            double major = half + disc;
            if (!(major > 0)) {
                continue;
            }
            double lift = Math.max(0.0, major / (MAX_ASPECT * MAX_ASPECT) - (half - disc));
            a += lift;
            c += lift;
            capped[shaped] = new double[] { e[0], e[1], a, b, c };
            roots[shaped++] = Math.sqrt(a * c - b * b);
        }
        double[] sorted = Arrays.copyOf(roots, shaped);
        Arrays.sort(sorted);
        double scale = (shaped > 0) ? sorted[shaped / 2] : 0.0;

        // {x, y, inverse xx, inverse xy, inverse yy, 1 / sqrt(det), reach in x, reach
        // in y}
        double[][] kernels = new double[n][];
        int k = 0;
        for (int i = 0; i < shaped; i++) {
            double[] e = capped[i];
            double f = scale / roots[i];
            kernels[k++] = kernel(e[0], e[1], f * e[2] + cell, f * e[3], f * e[4] + cell);
        }
        for (double[] e : ellipses) {
            double half = 0.5 * (e[2] + e[4]);
            double disc = Math.sqrt(0.25 * (e[2] - e[4]) * (e[2] - e[4]) + e[3] * e[3]);
            if (!(half + disc > 0)) {
                // every tree put the box in the same place: no shape, so round
                kernels[k++] = kernel(e[0], e[1], scale + cell, 0.0, scale + cell);
            }
        }
        double[][] all = Arrays.copyOf(kernels, k);
        return (x, y) -> {
            double sum = 0;
            for (double[] q : all) {
                double dx = x - q[0];
                double dy = y - q[1];
                if (dx * dx > q[6] || dy * dy > q[7]) {
                    continue;
                }
                double m = q[2] * dx * dx + 2 * q[3] * dx * dy + q[4] * dy * dy;
                sum += Math.exp(-0.5 * m) * q[5];
            }
            return sum;
        };
    }

    /**
     * Inverse covariance and normalisation; beyond seven deviations on an axis is
     * zero.
     */
    private static double[] kernel(double x, double y, double a, double b, double c) {
        double det = a * c - b * b;
        return new double[] { x, y, c / det, -b / det, a / det, 1.0 / Math.sqrt(det), 49 * a, 49 * c };
    }

    /**
     * Fills each cell with the band its corner mean falls in, from bandFloor up.
     */
    private Layer bandLayer(double[][] v, double[] levels) {
        double[] sorted = Arrays.copyOf(levels, levels.length);
        Arrays.sort(sorted);
        double h = span / (grid - 1.0);
        int painted = Math.max(1, sorted.length - bandFloor);
        return (g, vp) -> {
            for (int i = 0; i + 1 < grid; i++) {
                for (int j = 0; j + 1 < grid; j++) {
                    double m = 0.25 * (v[i][j] + v[i + 1][j] + v[i][j + 1] + v[i + 1][j + 1]);
                    int band = 0;
                    while (band < sorted.length && m > sorted[band]) {
                        band++;
                    }
                    if (band < bandFloor) {
                        continue;
                    }
                    int x0 = (int) Math.round(vp.px(lo + h * i));
                    int x1 = (int) Math.round(vp.px(lo + h * (i + 1)));
                    int y0 = (int) Math.round(vp.py(lo + h * (j + 1)));
                    int y1 = (int) Math.round(vp.py(lo + h * j));
                    g.setColor(new Color(bandColor.getRed(), bandColor.getGreen(), bandColor.getBlue(),
                            9 + 25 * (band - bandFloor) / painted));
                    // +1 closes the hairline seam that rounding leaves between cells
                    g.fillRect(x0, y0, x1 - x0 + 1, y1 - y0 + 1);
                }
            }
        };
    }
}
