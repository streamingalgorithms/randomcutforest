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
import java.awt.geom.Path2D;

/**
 * Axis-aligned extent boxes in three dimensions, drawn under a perspective
 * camera.
 *
 * <p>
 * Only the three faces the camera can see are filled. For an axis-aligned box
 * that needs no depth sort: on each axis, exactly one of the two faces points
 * toward the eye, and which one is decided by the sign of eye[i] - centre[i].
 * Three quads and twelve edges then read unambiguously as a solid box.
 *
 * 
 * <p>
 * <b>Direction convention.</b> The array is DiVector-ordered, high components
 * first, and the two halves point in OPPOSITE directions from the query:
 *
 * <pre>
 *   high_i  -&gt; extent in the MINUS i direction
 *   low_i   -&gt; extent in the PLUS  i direction
 * </pre>
 *
 * because high_i accumulates when the point overhangs the box ABOVE on axis i,
 * which puts the box, and the mass, below it. DensityExample names the same
 * thing directly: dir.high[0] is "toTheLeft" and dir.low[0] is "toTheRight".
 * Getting this backwards draws the box on the far side of the query from the
 * data, with the query at a corner -- invisible when the query sits inside the
 * cloud and both faces are active, obvious the moment it sits outside.
 *
 * <p>
 * The six half-lengths are independent, so the box is not centred on the query
 * point. In three dimensions that offset is a genuine direction in space rather
 * than a per-axis number, and it is the part of the measurement no symmetric
 * density tensor can represent, so it is drawn rather than symmetrised away.
 */
public class Box3DLayer implements Layer {

    /**
     * Maps a world point to plot data coordinates, or null if behind the camera.
     */
    public interface Projector {
        double[] project(double x, double y, double z);
    }

    private final double[][] origins; // world, length 3
    private final double[][] boxes; // {hx, hy, hz, lx, ly, lz}
    private final double[] shade; // 0..1 -> outline ramp
    private final double[] fillLevel; // normalized 0..1, negative for no fill
    private final Projector projector;
    private final double[] eye;
    private final float fillAlpha;

    public Box3DLayer(double[][] origins, double[][] boxes, double[] shade, double[] fillLevel, Projector projector,
            double[] eye, float fillAlpha) {
        this.origins = origins;
        this.boxes = boxes;
        this.shade = shade;
        this.fillLevel = fillLevel;
        this.projector = projector;
        this.eye = eye;
        this.fillAlpha = fillAlpha;
    }

    private static final int[][] EDGES = { { 0, 1 }, { 1, 3 }, { 3, 2 }, { 2, 0 }, { 4, 5 }, { 5, 7 }, { 7, 6 },
            { 6, 4 }, { 0, 4 }, { 1, 5 }, { 2, 6 }, { 3, 7 } };

    @Override
    public void draw(Graphics2D g, Plot2D.Viewport vp) {
        g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

        for (int i = 0; i < origins.length; i++) {
            double[] o = origins[i];
            double[] b = boxes[i];

            // b[0..2] are the high faces and extend toward minus; b[3..5] are the low
            // faces and extend toward plus
            double[] lo = { o[0] - b[0], o[1] - b[1], o[2] - b[2] };
            double[] hi = { o[0] + b[3], o[1] + b[4], o[2] + b[5] };

            // corner c has bit k set when it takes hi on axis k
            double[][] px = new double[8][];
            boolean visible = true;
            for (int c = 0; c < 8; c++) {
                double x = ((c & 1) != 0) ? hi[0] : lo[0];
                double y = ((c & 2) != 0) ? hi[1] : lo[1];
                double z = ((c & 4) != 0) ? hi[2] : lo[2];
                double[] p = projector.project(x, y, z);
                if (p == null) {
                    visible = false;
                    break;
                }
                px[c] = new double[] { vp.px(p[0]), vp.py(p[1]) };
            }
            if (!visible) {
                continue;
            }

            Color fill = BoxLayer.fillColor(fillLevel[i], fillAlpha);
            if (fill.getAlpha() > 0) {
                g.setColor(fill);
                for (int axis = 0; axis < 3; axis++) {
                    double centre = 0.5 * (lo[axis] + hi[axis]);
                    int bit = 1 << axis;
                    int side = (eye[axis] > centre) ? bit : 0;
                    // the four corners agreeing with `side` on `axis`
                    int[] quad = new int[4];
                    int n = 0;
                    for (int c = 0; c < 8; c++) {
                        if ((c & bit) == side) {
                            quad[n++] = c;
                        }
                    }
                    // reorder to a ring: swap the last two so the path does not cross
                    int t = quad[2];
                    quad[2] = quad[3];
                    quad[3] = t;
                    Path2D.Double face = new Path2D.Double();
                    face.moveTo(px[quad[0]][0], px[quad[0]][1]);
                    for (int k = 1; k < 4; k++) {
                        face.lineTo(px[quad[k]][0], px[quad[k]][1]);
                    }
                    face.closePath();
                    g.fill(face);
                }
            }

            g.setColor(BoxLayer.outlineColor(shade[i]));
            g.setStroke(new BasicStroke(1.1f));
            for (int[] e : EDGES) {
                g.drawLine((int) Math.round(px[e[0]][0]), (int) Math.round(px[e[0]][1]), (int) Math.round(px[e[1]][0]),
                        (int) Math.round(px[e[1]][1]));
            }

            double[] c0 = projector.project(o[0], o[1], o[2]);
            if (c0 != null) {
                g.setColor(new Color(50, 50, 50, 190));
                g.fillOval((int) Math.round(vp.px(c0[0])) - 2, (int) Math.round(vp.py(c0[1])) - 2, 4, 4);
            }
        }
    }
}
