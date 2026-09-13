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

import java.awt.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * Isolines of a scalar field, by marching squares.
 *
 * <p>
 * <b>Ambiguous cells are measured, not interpolated.</b> When all four edges of
 * a cell cross the level there are two valid pairings, and they give different
 * topology: one separates the high corners, the other joins them. The textbook
 * asymptotic decider settles it with the bilinear value at the centre, the mean
 * of the four corners. That is the right tool when grid samples are all you
 * have and the field cannot be queried between them. Here the field is a
 * callable, so the centre is sampled directly and no assumption about sub-cell
 * behaviour is needed. It matters when the field has structure the grid does
 * not resolve -- an estimator built from axis-aligned boxes has no reason to be
 * bilinear at small scale.
 *
 * <p>
 * Note what this does NOT do: segments are emitted independently and never
 * joined into ordered polylines. They abut because neighbouring cells
 * interpolate the shared edge identically, so continuity is automatic. Anything
 * needing connectivity -- smoothing, level labels, enclosed area -- would need
 * that pass.
 *
 * <p>
 * Levels are placed at quantiles of the sampled values rather than evenly,
 * since a density typically spans orders of magnitude and even spacing stacks
 * every contour on the densest region. The low end of the quantile band is the
 * risky one: where the field is flat, a small amount of estimator noise
 * displaces a contour a long way and can close it into a spurious loop.
 *
 * <p>
 * <b>Quantile levels are not comparable across grids.</b> A quantile is taken
 * over whatever node set was sampled, so changing the grid reweights the band
 * and moves the levels even when the field is unchanged. Two runs at different
 * resolutions therefore cannot be compared while both are deriving their own
 * levels. Use {@link #sample} and {@link #quantileLevels} to capture the levels
 * one run picks, then pass those fixed values to the explicit-level
 * {@link #isolines(Field, double, double, int, double[], Color)} for every run
 * being compared. Fixed levels also stop the band drifting frame to frame as a
 * streaming estimator resamples underneath it.
 */
public final class Contour {

    /** A scalar field that can be evaluated anywhere, not merely on a grid. */
    public interface Field {
        double at(double x, double y);
    }

    private Contour() {
    }

    /**
     * Evaluate the field on a grid by grid lattice spanning [lo, lo + span] in both
     * axes. Node (i, j) is at (lo + h*i, lo + h*j) with h = span / (grid - 1), so
     * the corners of the square are sampled and doubling the resolution as grid' =
     * 2*grid - 1 makes the coarse nodes an exact subset of the fine ones.
     *
     * @param f    the field
     * @param lo   lower bound on both axes
     * @param span extent on both axes
     * @param grid nodes per axis, at least 2
     * @return the sampled values, indexed [i][j]
     */
    public static double[][] sample(Field f, double lo, double span, int grid) {
        if (grid < 2) {
            throw new IllegalArgumentException("grid must be at least 2");
        }
        double[][] v = new double[grid][grid];
        double h = span / (grid - 1.0);
        for (int i = 0; i < grid; i++) {
            for (int j = 0; j < grid; j++) {
                v[i][j] = f.at(lo + h * i, lo + h * j);
            }
        }
        return v;
    }

    /**
     * The levels the quantile band selects for an already sampled grid. Returned
     * ascending, one per contour. Non-finite nodes are left out of the order
     * statistics: a field that returns NaN or an infinity off its support would
     * otherwise place those nodes at the top of the sorted array and hand back
     * levels no finite value can cross.
     *
     * @param v          sampled grid, as returned by {@link #sample}
     * @param levelCount number of contours
     * @param qLo        low end of the quantile band
     * @param qHi        high end of the quantile band
     * @return the level values, ascending; empty if no node was finite
     */
    public static double[] quantileLevels(double[][] v, int levelCount, double qLo, double qHi) {
        List<Double> vals = new ArrayList<>(v.length * v.length);
        for (double[] row : v) {
            for (double x : row) {
                if (Double.isFinite(x)) {
                    vals.add(x);
                }
            }
        }
        if (vals.isEmpty()) {
            return new double[0];
        }
        Collections.sort(vals);
        double[] levels = new double[levelCount];
        for (int c = 1; c <= levelCount; c++) {
            levels[c - 1] = vals.get((int) ((qLo + (qHi - qLo) * c / (levelCount + 1.0)) * (vals.size() - 1)));
        }
        return levels;
    }

    /**
     * Isolines at explicit levels on an already sampled grid. The field is still
     * needed: ambiguous cells are resolved by measuring the centre rather than
     * averaging the corners. Levels are drawn darkest-highest regardless of the
     * order given, so the caller's array is copied and sorted rather than mutated.
     *
     * @param f      the field, for centre samples in ambiguous cells
     * @param v      sampled grid, as returned by {@link #sample}
     * @param lo     lower bound on both axes, matching the sample call
     * @param span   extent on both axes, matching the sample call
     * @param levels the level values
     * @param color  base colour; alpha ramps with the level
     * @return one Layer per level
     */
    public static List<Layer> isolines(Field f, double[][] v, double lo, double span, double[] levels, Color color) {
        int grid = v.length;
        if (grid < 2) {
            throw new IllegalArgumentException("grid must be at least 2");
        }
        double h = span / (grid - 1.0);
        double[] sorted = Arrays.copyOf(levels, levels.length);
        Arrays.sort(sorted);

        List<Layer> out = new ArrayList<>();
        for (int c = 1; c <= sorted.length; c++) {
            double level = sorted[c - 1];
            List<double[]> segs = new ArrayList<>();
            for (int i = 0; i + 1 < grid; i++) {
                for (int j = 0; j + 1 < grid; j++) {
                    double x0 = lo + h * i, y0 = lo + h * j;
                    double a = v[i][j], b = v[i + 1][j], cc = v[i + 1][j + 1], d = v[i][j + 1];
                    List<double[]> hits = new ArrayList<>(4);
                    if ((a > level) != (b > level)) {
                        hits.add(new double[] { x0 + h * frac(a, b, level), y0 });
                    }
                    if ((b > level) != (cc > level)) {
                        hits.add(new double[] { x0 + h, y0 + h * frac(b, cc, level) });
                    }
                    if ((d > level) != (cc > level)) {
                        hits.add(new double[] { x0 + h * frac(d, cc, level), y0 + h });
                    }
                    if ((a > level) != (d > level)) {
                        hits.add(new double[] { x0, y0 + h * frac(a, d, level) });
                    }
                    if (hits.size() == 2) {
                        segs.add(seg(hits.get(0), hits.get(1)));
                    } else if (hits.size() == 4) {
                        double centre = f.at(x0 + h / 2, y0 + h / 2); // measured, not averaged
                        boolean joinsA = (centre > level) == (a > level);
                        if (joinsA) {
                            segs.add(seg(hits.get(0), hits.get(1)));
                            segs.add(seg(hits.get(2), hits.get(3)));
                        } else {
                            segs.add(seg(hits.get(0), hits.get(3)));
                            segs.add(seg(hits.get(1), hits.get(2)));
                        }
                    }
                }
            }
            int alpha = 55 + 130 * c / sorted.length;
            out.add(segments(segs, new Color(color.getRed(), color.getGreen(), color.getBlue(), alpha), 1.1f));
        }
        return out;
    }

    /**
     * Counts the ambiguous cells and how often the measured centre disagrees with
     * the bilinear guess, over the same cells and levels
     * {@link #isolines(Field, double[][], double, double, double[], Color)} would
     * walk. A high disagreement rate means the grid does not resolve the field, so
     * contour topology read off it -- closed loops especially -- is unreliable.
     *
     * <p>
     * This repeats the centre samples that isolines already takes, so it roughly
     * doubles the cost of the ambiguous cells. Ambiguous cells are normally a small
     * fraction of the grid, and a static field is built once.
     *
     * @return two counts: ambiguous cells, and of those the ones where measuring
     *         the centre changed the pairing the bilinear guess would have chosen
     */
    public static int[] saddleAudit(Field f, double[][] v, double lo, double span, double[] levels) {
        int grid = v.length;
        if (grid < 2) {
            throw new IllegalArgumentException("grid must be at least 2");
        }
        double h = span / (grid - 1.0);
        int saddles = 0, disagree = 0;
        for (double level : levels) {
            for (int i = 0; i + 1 < grid; i++) {
                for (int j = 0; j + 1 < grid; j++) {
                    double a = v[i][j], b = v[i + 1][j], cc = v[i + 1][j + 1], d = v[i][j + 1];
                    int crossings = 0;
                    if ((a > level) != (b > level)) {
                        crossings++;
                    }
                    if ((b > level) != (cc > level)) {
                        crossings++;
                    }
                    if ((d > level) != (cc > level)) {
                        crossings++;
                    }
                    if ((a > level) != (d > level)) {
                        crossings++;
                    }
                    if (crossings == 4) {
                        saddles++;
                        double x0 = lo + h * i, y0 = lo + h * j;
                        double centre = f.at(x0 + h / 2, y0 + h / 2);
                        double bilinear = 0.25 * (a + b + cc + d);
                        if ((centre > level) != (bilinear > level)) {
                            disagree++;
                        }
                    }
                }
            }
        }
        return new int[] { saddles, disagree };
    }

    /**
     * Isolines at explicit levels, sampling the grid internally. This is the
     * overload to use when comparing two resolutions: pass the same levels to both.
     */
    public static List<Layer> isolines(Field f, double lo, double span, int grid, double[] levels, Color color) {
        return isolines(f, sample(f, lo, span, grid), lo, span, levels, color);
    }

    /**
     * Isolines with levels derived from the quantile band of this grid's own
     * values. Convenient for a single run; see the class note on why the levels
     * cannot be compared with those of a different grid.
     */
    public static List<Layer> isolines(Field f, double lo, double span, int grid, int levelCount, double qLo,
            double qHi, Color color) {
        double[][] v = sample(f, lo, span, grid);
        return isolines(f, v, lo, span, quantileLevels(v, levelCount, qLo, qHi), color);
    }

    private static double[] seg(double[] p, double[] q) {
        return new double[] { p[0], p[1], q[0], q[1] };
    }

    private static double frac(double a, double b, double level) {
        return (b == a) ? 0.5 : (level - a) / (b - a);
    }

    /** Many disconnected segments as one Layer; Layer is a functional interface. */
    public static Layer segments(List<double[]> segs, Color color, float stroke) {
        return (g, vp) -> {
            g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
            g.setColor(color);
            g.setStroke(new BasicStroke(stroke));
            for (double[] s : segs) {
                g.drawLine((int) Math.round(vp.px(s[0])), (int) Math.round(vp.py(s[1])), (int) Math.round(vp.px(s[2])),
                        (int) Math.round(vp.py(s[3])));
            }
        };
    }
}