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
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Isolines on rectangles the estimator sizes, rather than on a fixed square
 * lattice.
 *
 * <p>
 * A probe here returns the value and the local half-extents the estimator
 * itself reports -- the cut box. A cell is refined along an axis when it is
 * wider than the box is on that axis, and only along that axis, so a
 * neighbourhood that is long in x and collapsed in y is cut into tall thin
 * rectangles instead of squares. On a fixed lattice the same region is
 * oversampled along the long axis and undersampled along the short one, which
 * is the wrong way round and is what turns a contour into filaments where the
 * box degenerates.
 *
 * <p>
 * Refinement is by bisection along one axis at a time, so a split costs two
 * evaluations rather than five, and the two children keep their parent's
 * corners. The centre probe a cell needs for the ambiguous case is the same
 * probe that decided whether to refine it, so nothing is evaluated twice.
 *
 * <p>
 * <b>Cells are not comparable, so levels must be fixed.</b> Cell size is a
 * function of the field, so any statistic taken over cells or over corner
 * values is weighted by a density of nodes that the field itself chose.
 * {@link Contour#quantileLevels} on such a sample would be circular. Take the
 * levels from a uniform {@link Contour#sample}, or fix them by hand, and pass
 * them in.
 *
 * <p>
 * Like {@link Contour}, segments are emitted independently and never joined
 * into polylines. Unlike {@link Contour}, they are not guaranteed to abut: two
 * adjacent cells refined to different depths interpolate a shared edge from
 * different endpoints, so a T-junction leaves a gap of at most the finer cell's
 * interpolation error. Closing those would need the usual restricted-quadtree
 * balancing pass, and is not done here.
 */
public final class AnisoContour {

    /** A field that also reports the scale it believes in at each point. */
    public interface Probe {
        /**
         * @param extent length 2, written with the half-extent in x and in y; an axis
         *               the estimator reports as collapsed should be written as 0 and
         *               will not drive refinement
         * @return the field value at (x, y)
         */
        double at(double x, double y, double[] extent);
    }

    private AnisoContour() {
    }

    /**
     * @param probe    field and local extents
     * @param lo       lower bound on both axes
     * @param span     extent on both axes
     * @param seed     cells per axis before adaptive refinement begins
     * @param maxDepth bisections allowed per axis under a seed cell
     * @param cover    a cell is refined on an axis while its side exceeds this many
     *                 times the local half-extent; 2.0 asks for roughly one cell
     *                 per box, smaller values oversample
     * @param levels   explicit level values; see the class note
     * @param color    base colour, alpha ramps with the level
     */
    public static List<Layer> isolines(Probe probe, double lo, double span, int seed, int maxDepth, double cover,
            double[] levels, Color color) {
        double[] sorted = Arrays.copyOf(levels, levels.length);
        Arrays.sort(sorted);
        List<List<double[]>> perLevel = new ArrayList<>();
        for (int i = 0; i < sorted.length; i++) {
            perLevel.add(new ArrayList<>());
        }

        int n = Math.max(1, seed);
        double h = span / n;
        double[] scratch = new double[2];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double x0 = lo + h * i;
                double y0 = lo + h * j;
                double x1 = x0 + h;
                double y1 = y0 + h;
                march(probe, x0, y0, x1, y1, value(probe, x0, y0, scratch), value(probe, x1, y0, scratch),
                        value(probe, x1, y1, scratch), value(probe, x0, y1, scratch), maxDepth, cover, sorted,
                        perLevel);
            }
        }

        List<Layer> out = new ArrayList<>();
        for (int c = 0; c < sorted.length; c++) {
            int alpha = 55 + 130 * (c + 1) / sorted.length;
            out.add(Contour.segments(perLevel.get(c),
                    new Color(color.getRed(), color.getGreen(), color.getBlue(), alpha), 1.1f));
        }
        return out;
    }

    private static double value(Probe probe, double x, double y, double[] scratch) {
        return probe.at(x, y, scratch);
    }

    /**
     * Corners are named a = (x0,y0), b = (x1,y0), c = (x1,y1), d = (x0,y1), the
     * same cycle {@link Contour} uses.
     */
    private static void march(Probe probe, double x0, double y0, double x1, double y1, double a, double b, double c,
            double d, int depth, double cover, double[] levels, List<List<double[]>> perLevel) {
        double[] extent = new double[2];
        double xm = 0.5 * (x0 + x1);
        double ym = 0.5 * (y0 + y1);
        double centre = probe.at(xm, ym, extent);

        if (depth > 0) {
            // Refine the axis that is worse resolved relative to its own extent.
            // A reported extent of zero is a collapsed axis, not a demand for
            // infinite refinement: nothing there can be resolved by subdividing.
            double needX = (extent[0] > 0.0) ? (x1 - x0) / (cover * extent[0]) : 0.0;
            double needY = (extent[1] > 0.0) ? (y1 - y0) / (cover * extent[1]) : 0.0;
            if (needX > 1.0 && needX >= needY) {
                double top = probe.at(xm, y0, extent);
                double bot = probe.at(xm, y1, extent);
                march(probe, x0, y0, xm, y1, a, top, bot, d, depth - 1, cover, levels, perLevel);
                march(probe, xm, y0, x1, y1, top, b, c, bot, depth - 1, cover, levels, perLevel);
                return;
            }
            if (needY > 1.0) {
                double left = probe.at(x0, ym, extent);
                double right = probe.at(x1, ym, extent);
                march(probe, x0, y0, x1, ym, a, b, right, left, depth - 1, cover, levels, perLevel);
                march(probe, x0, ym, x1, y1, left, right, c, d, depth - 1, cover, levels, perLevel);
                return;
            }
        }


        for (int k = 0; k < levels.length; k++) {
            emit(x0, y0, x1, y1, a, b, c, d, centre, levels[k], perLevel.get(k));
        }
    }

    /** Marching squares on one rectangle; the centre value is already measured. */
    private static void emit(double x0, double y0, double x1, double y1, double a, double b, double c, double d,
            double centre, double level, List<double[]> segs) {
        double w = x1 - x0;
        double t = y1 - y0;
        List<double[]> hits = new ArrayList<>(4);
        if ((a > level) != (b > level)) {
            hits.add(new double[] { x0 + w * frac(a, b, level), y0 });
        }
        if ((b > level) != (c > level)) {
            hits.add(new double[] { x1, y0 + t * frac(b, c, level) });
        }
        if ((d > level) != (c > level)) {
            hits.add(new double[] { x0 + w * frac(d, c, level), y1 });
        }
        if ((a > level) != (d > level)) {
            hits.add(new double[] { x0, y0 + t * frac(a, d, level) });
        }
        if (hits.size() == 2) {
            segs.add(seg(hits.get(0), hits.get(1)));
        } else if (hits.size() == 4) {
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

    private static double[] seg(double[] p, double[] q) {
        return new double[] { p[0], p[1], q[0], q[1] };
    }

    private static double frac(double a, double b, double level) {
        return (b == a) ? 0.5 : (level - a) / (b - a);
    }
}
