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

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.geom.Ellipse2D;
import java.awt.geom.Path2D;

/** Factory methods for the common layer types, plus a categorical palette. */
public final class Layers {

    private Layers() {
    }

    /**
     * Strong, well-separated colors: red, blue, green, orange, purple, brown, ...
     */
    public static final Color[] PALETTE = { new Color(214, 39, 40), new Color(31, 119, 180), new Color(44, 160, 44),
            new Color(255, 127, 14), new Color(148, 103, 189), new Color(140, 86, 75), new Color(227, 26, 158),
            new Color(23, 142, 150), new Color(8, 48, 107), new Color(177, 89, 40), new Color(106, 61, 154),
            new Color(99, 99, 99), };

    public static Color color(int i) {
        return PALETTE[((i % PALETTE.length) + PALETTE.length) % PALETTE.length];
    }

    /** Uniform dots — e.g. the raw background cloud. */
    public static Layer dots(float[][] xy, Color color, double radius) {
        return (g, vp) -> {
            g.setColor(color);
            for (float[] p : xy) {
                g.fill(new Ellipse2D.Double(vp.px(p[0]) - radius, vp.py(p[1]) - radius, 2 * radius, 2 * radius));
            }
        };
    }

    /**
     * Dots whose AREA encodes a per-point weight, with a floor so singletons stay
     * visible.
     */
    public static Layer weightedDots(double[][] xy, double[] weight, Color color, double minR, double scaleR) {
        return (g, vp) -> {
            g.setColor(color);
            for (int i = 0; i < xy.length; i++) {
                double r = minR + scaleR * Math.sqrt(Math.max(0, weight[i]));
                g.fill(new Ellipse2D.Double(vp.px(xy[i][0]) - r, vp.py(xy[i][1]) - r, 2 * r, 2 * r));
            }
        };
    }

    /**
     * Dots colored by a scalar through a heat ramp (gnuplot-palette style),
     * auto-scaled to the data.
     */
    public static Layer scalarDots(float[][] xy, double[] value, double radius) {
        double lo = Double.MAX_VALUE, hi = -Double.MAX_VALUE;
        for (double v : value) {
            lo = Math.min(lo, v);
            hi = Math.max(hi, v);
        }
        final double flo = lo, fhi = hi;
        return (g, vp) -> {
            for (int i = 0; i < xy.length; i++) {
                double t = fhi > flo ? (value[i] - flo) / (fhi - flo) : 0.5;
                g.setColor(heat(t));
                g.fill(new Ellipse2D.Double(vp.px(xy[i][0]) - radius, vp.py(xy[i][1]) - radius, 2 * radius,
                        2 * radius));
            }
        };
    }

    /** A polyline; if closed with fillAlpha &gt; 0 the interior is tinted. */
    public static Layer polyline(double[][] xy, Color color, boolean closed, int fillAlpha, float stroke) {
        return (g, vp) -> {
            Path2D path = new Path2D.Double();
            for (int i = 0; i < xy.length; i++) {
                if (i == 0) {
                    path.moveTo(vp.px(xy[i][0]), vp.py(xy[i][1]));
                } else {
                    path.lineTo(vp.px(xy[i][0]), vp.py(xy[i][1]));
                }
            }
            if (closed) {
                path.closePath();
            }
            if (closed && fillAlpha > 0 && xy.length > 2) {
                g.setColor(new Color(color.getRed(), color.getGreen(), color.getBlue(), fillAlpha));
                g.fill(path);
            }
            g.setColor(color);
            g.setStroke(new BasicStroke(stroke));
            g.draw(path);
        };
    }

    /** Arrows from each origin to origin+delta. */
    public static Layer arrows(double[][] origin, double[][] delta, Color color, float stroke) {
        return (g, vp) -> {
            g.setColor(color);
            g.setStroke(new BasicStroke(stroke));
            for (int i = 0; i < origin.length; i++) {
                double x0 = vp.px(origin[i][0]), y0 = vp.py(origin[i][1]);
                double x1 = vp.px(origin[i][0] + delta[i][0]), y1 = vp.py(origin[i][1] + delta[i][1]);
                g.drawLine((int) x0, (int) y0, (int) x1, (int) y1);
                double a = Math.atan2(y1 - y0, x1 - x0), head = 7;
                g.drawLine((int) x1, (int) y1, (int) (x1 - head * Math.cos(a - Math.PI / 7)),
                        (int) (y1 - head * Math.sin(a - Math.PI / 7)));
                g.drawLine((int) x1, (int) y1, (int) (x1 - head * Math.cos(a + Math.PI / 7)),
                        (int) (y1 - head * Math.sin(a + Math.PI / 7)));
            }
        };
    }

    /** blue → cyan → yellow → red. */
    private static Color heat(double t) {
        t = Math.max(0, Math.min(1, t));
        return Color.getHSBColor((float) (0.66 * (1 - t)), 0.85f, 0.95f);
    }

    /** Filled band between lo[i] and hi[i] across x[i] — gnuplot filledcurves. */
    public static Layer band(double[] x, double[] lo, double[] hi, Color color, int alpha) {
        return (g, vp) -> {
            java.awt.geom.Path2D path = new java.awt.geom.Path2D.Double();
            for (int i = 0; i < x.length; i++) {
                if (i == 0)
                    path.moveTo(vp.px(x[i]), vp.py(hi[i]));
                else
                    path.lineTo(vp.px(x[i]), vp.py(hi[i]));
            }
            for (int i = x.length - 1; i >= 0; i--) {
                path.lineTo(vp.px(x[i]), vp.py(lo[i]));
            }
            path.closePath();
            g.setColor(new Color(color.getRed(), color.getGreen(), color.getBlue(), alpha));
            g.fill(path);
        };
    }

    /** Horizontal reference line spanning the full x-range. */
    public static Layer hline(double y, Color color) {
        return (g, vp) -> {
            g.setColor(color);
            g.setStroke(new BasicStroke(1f));
            g.drawLine((int) vp.px(vp.xmin()), (int) vp.py(y), (int) vp.px(vp.xmax()), (int) vp.py(y));
        };
    }

    /** Vertical line spanning the full y-range (the "now" marker). */
    public static Layer vline(double x, Color color, float stroke) {
        return (g, vp) -> {
            g.setColor(color);
            g.setStroke(new BasicStroke(stroke));
            g.drawLine((int) vp.px(x), (int) vp.py(vp.ymin()), (int) vp.px(x), (int) vp.py(vp.ymax()));
        };
    }

    public static Layer legend(String[] labels, Color[] colors) {
        return (g, vp) -> {
            int boxRight = (int) vp.px(vp.xmax());
            int boxTop = (int) vp.py(vp.ymax());
            int w = 210;
            int x = boxRight - w - 10; // 10px inside the right edge of the plot box
            int y = boxTop + 10; // 10px below the top edge
            g.setColor(new Color(255, 255, 255, 210)); // translucent backing so lines don't bleed through
            g.fillRect(x - 6, y - 4, w, labels.length * 16 + 8);
            g.setColor(new Color(205, 205, 205));
            g.drawRect(x - 6, y - 4, w, labels.length * 16 + 8);
            for (int i = 0; i < labels.length; i++) {
                g.setColor(colors[i]);
                g.fillRect(x, y + i * 16, 12, 10);
                g.setColor(new Color(40, 40, 40));
                g.drawString(labels[i], x + 18, y + i * 16 + 9);
            }
        };
    }

    /** Vertical shaded bands over x-intervals — here, the missing-data gaps. */
    public static Layer xBands(double[][] spans, Color color, int alpha) {
        return (g, vp) -> {
            g.setColor(new Color(color.getRed(), color.getGreen(), color.getBlue(), alpha));
            int top = (int) vp.py(vp.ymax()), h = (int) (vp.py(vp.ymin()) - vp.py(vp.ymax()));
            for (double[] s : spans) {
                int x0 = (int) vp.px(s[0]), x1 = (int) vp.px(s[1]);
                g.fillRect(x0, top, Math.max(1, x1 - x0), h);
            }
        };
    }

    /** A short text label anchored at a data coordinate. */
    public static Layer label(double x, double y, String text, Color color) {
        return (g, vp) -> {
            g.setColor(color);
            g.drawString(text, (int) vp.px(x) + 4, (int) vp.py(y) - 3);
        };
    }
}
