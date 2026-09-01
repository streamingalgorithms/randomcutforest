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

/**
 * The 3D counterpart of {@link Layers#arrows}: a directional field drawn under
 * a perspective camera.
 *
 * <p>
 * The shaft is projected from its two world endpoints, so it foreshortens
 * correctly and an arrow pointing at the camera collapses to a dot — which is
 * the honest rendering, and the reason this is a strict improvement on the 2D
 * version rather than a decoration. In two dimensions a directional density
 * field can only ever show the component of the gradient that lies in the plane
 * being plotted; here the third component is visible as foreshortening and as
 * motion out of the plane when the field turns.
 *
 * <p>
 * The head is built in screen space after projection. Constructing it in world
 * space would need the arrow's own local frame and would make near-camera
 * arrows grow heads out of proportion to their shafts.
 */
public class Arrow3DLayer implements Layer {

    private final double[][] origins; // world, length 3
    private final double[][] deltas; // world displacement, length 3
    private final Color color;
    private final float stroke;
    private final double headPixels;

    public Arrow3DLayer(double[][] origins, double[][] deltas, Color color, float stroke) {
        this(origins, deltas, color, stroke, 5.0);
    }

    public Arrow3DLayer(double[][] origins, double[][] deltas, Color color, float stroke, double headPixels) {
        this.origins = origins;
        this.deltas = deltas;
        this.color = color;
        this.stroke = stroke;
        this.headPixels = headPixels;
    }

    private final Box3DLayer.Projector[] holder = new Box3DLayer.Projector[1];

    public Arrow3DLayer withProjector(Box3DLayer.Projector projector) {
        holder[0] = projector;
        return this;
    }

    @Override
    public void draw(Graphics2D g, Plot2D.Viewport vp) {
        Box3DLayer.Projector projector = holder[0];
        if (projector == null) {
            return;
        }
        g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        g.setColor(color);
        g.setStroke(new BasicStroke(stroke, BasicStroke.CAP_ROUND, BasicStroke.JOIN_ROUND));

        for (int i = 0; i < origins.length; i++) {
            double[] o = origins[i];
            double[] d = deltas[i];
            double[] pa = projector.project(o[0], o[1], o[2]);
            double[] pb = projector.project(o[0] + d[0], o[1] + d[1], o[2] + d[2]);
            if (pa == null || pb == null) {
                continue;
            }
            double ax = vp.px(pa[0]), ay = vp.py(pa[1]);
            double bx = vp.px(pb[0]), by = vp.py(pb[1]);
            g.drawLine((int) Math.round(ax), (int) Math.round(ay), (int) Math.round(bx), (int) Math.round(by));

            double len = Math.hypot(bx - ax, by - ay);
            if (len < 1.5) {
                continue; // pointing at or away from the camera; a head would be a lie
            }
            double ux = (bx - ax) / len, uy = (by - ay) / len;
            double h = Math.min(headPixels, 0.4 * len);
            for (double sign : new double[] { 1, -1 }) {
                double ca = Math.cos(Math.toRadians(150 * sign)), sa = Math.sin(Math.toRadians(150 * sign));
                double hx = bx + h * (ux * ca - uy * sa);
                double hy = by + h * (ux * sa + uy * ca);
                g.drawLine((int) Math.round(bx), (int) Math.round(by), (int) Math.round(hx), (int) Math.round(hy));
            }
        }
    }
}
