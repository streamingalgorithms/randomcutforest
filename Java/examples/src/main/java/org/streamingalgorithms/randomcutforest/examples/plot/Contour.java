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
 */
public final class Contour {

    /** A scalar field that can be evaluated anywhere, not merely on a grid. */
    public interface Field {
        double at(double x, double y);
    }

    private Contour() {
    }

    public static List<Layer> isolines(Field f, double lo, double span, int grid, int levelCount, double qLo,
            double qHi, Color color) {
        double[][] v = new double[grid][grid];
        List<Double> vals = new ArrayList<>(grid * grid);
        double h = span / (grid - 1.0);
        for (int i = 0; i < grid; i++) {
            for (int j = 0; j < grid; j++) {
                v[i][j] = f.at(lo + h * i, lo + h * j);
                vals.add(v[i][j]);
            }
        }
        Collections.sort(vals);

        List<Layer> out = new ArrayList<>();
        for (int c = 1; c <= levelCount; c++) {
            double level = vals.get((int) ((qLo + (qHi - qLo) * c / (levelCount + 1.0)) * (vals.size() - 1)));
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
            int alpha = 55 + 130 * c / levelCount;
            out.add(segments(segs, new Color(color.getRed(), color.getGreen(), color.getBlue(), alpha), 1.1f));
        }
        return out;
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
