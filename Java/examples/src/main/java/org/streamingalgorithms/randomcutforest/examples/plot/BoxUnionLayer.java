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
import java.util.List;
import java.util.function.BooleanSupplier;

/**
 * The union of a cluster's density boxes, projected.
 *
 * <p>
 * One polygon per box, all boxes of a cluster in that cluster's colour, drawn
 * translucent so the union reads as the accumulation of its parts and the
 * region two clusters both claim reads darker still. That double-counting is
 * the point: an overlap is the picture of two clusters failing to be separated
 * as measures, and {@code ClusterBoxes.overlapFraction} puts a number on the
 * same thing.
 *
 * <p>
 * The outline is drawn at low alpha as well. A union of twenty boxes outlined
 * at full strength is a thicket, and what one wants to see is the silhouette.
 *
 * <p>
 * <b>Why the gate lives inside the layer.</b> The visibility flag is consulted
 * at draw time rather than when the scene is built, so a bare repaint honours
 * the toggle. A render loop that is paused, or one that only rebuilds its scene
 * every few hundred milliseconds, would otherwise ignore the key until its next
 * frame -- which reads as the key being broken.
 */
public class BoxUnionLayer implements Layer {

    private final List<double[][]> polygons;
    private final int[] cluster;
    private final Color[] palette;
    private final BooleanSupplier visible;
    private final int fillAlpha;
    private final int outlineAlpha;

    /**
     * @param polygons     projected outlines, any vertex count
     * @param cluster      index into {@code palette} for each polygon
     * @param palette      one colour per cluster
     * @param visible      consulted on every draw
     * @param fillAlpha    0-255; low, because these are meant to stack
     * @param outlineAlpha 0-255
     */
    public BoxUnionLayer(List<double[][]> polygons, int[] cluster, Color[] palette, BooleanSupplier visible,
            int fillAlpha, int outlineAlpha) {
        this.polygons = polygons;
        this.cluster = cluster;
        this.palette = palette;
        this.visible = visible;
        this.fillAlpha = fillAlpha;
        this.outlineAlpha = outlineAlpha;
    }

    @Override
    public void draw(Graphics2D g, Plot2D.Viewport vp) {
        if (!visible.getAsBoolean()) {
            return;
        }
        g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        Stroke saved = g.getStroke();
        g.setStroke(new BasicStroke(1.0f));
        for (int i = 0; i < polygons.size(); i++) {
            double[][] poly = polygons.get(i);
            if (poly.length == 0) {
                continue;
            }
            Color base = palette[cluster[i] % palette.length];

            // A box with one axis of zero extent projects to a segment and one with
            // none to a point. Both are drawn rather than skipped: a box that
            // collapsed is a real statement about the neighbourhood, and dropping it
            // silently is indistinguishable from the boxes never having been
            // computed -- which is a much harder thing to debug from a screenshot.
            if (poly.length < 3) {
                g.setColor(new Color(base.getRed(), base.getGreen(), base.getBlue(), Math.max(outlineAlpha, 140)));
                if (poly.length == 1) {
                    int x = (int) Math.round(vp.px(poly[0][0]));
                    int y = (int) Math.round(vp.py(poly[0][1]));
                    g.fillOval(x - 2, y - 2, 4, 4);
                } else {
                    g.drawLine((int) Math.round(vp.px(poly[0][0])), (int) Math.round(vp.py(poly[0][1])),
                            (int) Math.round(vp.px(poly[1][0])), (int) Math.round(vp.py(poly[1][1])));
                }
                continue;
            }
            Path2D path = new Path2D.Double();
            path.moveTo(vp.px(poly[0][0]), vp.py(poly[0][1]));
            for (int k = 1; k < poly.length; k++) {
                path.lineTo(vp.px(poly[k][0]), vp.py(poly[k][1]));
            }
            path.closePath();
            g.setColor(new Color(base.getRed(), base.getGreen(), base.getBlue(), fillAlpha));
            g.fill(path);
            g.setColor(new Color(base.getRed(), base.getGreen(), base.getBlue(), outlineAlpha));
            g.draw(path);
        }
        g.setStroke(saved);
    }
}
