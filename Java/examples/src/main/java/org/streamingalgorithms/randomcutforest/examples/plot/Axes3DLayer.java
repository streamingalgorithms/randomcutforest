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
 * World axes under a perspective camera.
 *
 * <p>
 * Worth drawing for more than orientation. Every extent box in the scene is
 * axis-aligned by construction, because the cuts are, so these three lines are
 * the only directions the estimator can express an elongation along. A box that
 * looks isotropic next to a visibly filamentary structure is usually a
 * structure whose tangent is running diagonal to all three.
 */
public class Axes3DLayer implements Layer {

    private final Box3DLayer.Projector projector;
    private final double length;
    private final Color color;

    public Axes3DLayer(Box3DLayer.Projector projector, double length, Color color) {
        this.projector = projector;
        this.length = length;
        this.color = color;
    }

    private void segment(Graphics2D g, Plot2D.Viewport vp, double[] a, double[] b) {
        double[] pa = projector.project(a[0], a[1], a[2]);
        double[] pb = projector.project(b[0], b[1], b[2]);
        if (pa == null || pb == null) {
            return;
        }
        g.drawLine((int) Math.round(vp.px(pa[0])), (int) Math.round(vp.py(pa[1])), (int) Math.round(vp.px(pb[0])),
                (int) Math.round(vp.py(pb[1])));
    }

    @Override
    public void draw(Graphics2D g, Plot2D.Viewport vp) {
        g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        String[] names = { "x", "y", "z" };

        for (int axis = 0; axis < 3; axis++) {
            double[] pos = new double[3];
            double[] neg = new double[3];
            pos[axis] = length;
            neg[axis] = -length;

            g.setColor(new Color(color.getRed(), color.getGreen(), color.getBlue(), 70));
            g.setStroke(new BasicStroke(1.0f, BasicStroke.CAP_BUTT, BasicStroke.JOIN_MITER, 10f, new float[] { 4f, 4f },
                    0f));
            segment(g, vp, neg, new double[3]);

            g.setColor(color);
            g.setStroke(new BasicStroke(1.6f));
            segment(g, vp, new double[3], pos);

            double[] tip = projector.project(pos[0], pos[1], pos[2]);
            if (tip != null) {
                g.setFont(g.getFont().deriveFont(Font.BOLD, 12f));
                g.drawString(names[axis], (int) Math.round(vp.px(tip[0])) + 4, (int) Math.round(vp.py(tip[1])) - 3);
            }
        }
    }
}