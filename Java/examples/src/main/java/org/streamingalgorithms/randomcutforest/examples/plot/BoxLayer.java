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
 * Extent boxes, carrying two independent channels.
 *
 * <ul>
 * <li><b>Outline</b> — anisotropy. Amber when the neighbourhood is isotropic,
 * deep red as the axis extents diverge.</li>
 * <li><b>Fill</b> — agreement between the Chhabra-Jensen exponent and the
 * exponent the density formula assumes. Saturated yellow when they match,
 * fading to no fill at all when they do not. An empty box is therefore a box
 * whose shape you can read but whose density you should not trust: the volume
 * is being computed as length raised to a power the measure does not obey
 * there.</li>
 * </ul>
 *
 * <p>
 * The box is asymmetric by construction — the four half-lengths are independent
 * — so the rectangle is not centred on the query point. That offset is the
 * odd-parity part of the neighbourhood, the component no symmetric density
 * tensor can carry, so it is left visible rather than averaged away. A small
 * grey dot marks the true query point.
 *
 * <p>
 * {@link Layers#polyline} already draws a filled translucent closed shape, so a
 * single box needs no new class. This exists to batch a grid of them into one
 * Layer with per-box colour, which matters at 64 glyphs a frame.
 */
public class BoxLayer implements Layer {

    /**
     * Fill ramp: pale straw at the low end of the exponent window, deep gold at the
     * high.
     */
    private static final Color LOW = new Color(253, 243, 200);
    private static final Color HIGH = new Color(214, 158, 24);

    private final double[][] origins; // query points, data coords
    private final double[][] boxes; // {high_x, high_y, low_x, low_y}, data units
    private final double[] anisotropy; // 0..1, amber -> deep red outline
    private final double[] exponent; // normalized 0..1, or negative if not computable
    private final float maxFillAlpha;

    public BoxLayer(double[][] origins, double[][] boxes, double[] anisotropy, double[] exponent, float maxFillAlpha) {
        this.origins = origins;
        this.boxes = boxes;
        this.anisotropy = anisotropy;
        this.exponent = exponent;
        this.maxFillAlpha = maxFillAlpha;
    }

    /** Amber (255,176,59) to deep red (198,40,28). */
    public static Color outlineColor(double t) {
        double u = Math.max(0.0, Math.min(1.0, t));
        return new Color((int) Math.round(255 + u * (198 - 255)), (int) Math.round(176 + u * (40 - 176)),
                (int) Math.round(59 + u * (28 - 59)));
    }

    /**
     * Fill for a normalized exponent in [0, 1]; a negative value means the exponent
     * was not computable and the box is left unfilled.
     *
     * <p>
     * Opacity is constant and the ramp carries the signal. Driving opacity instead
     * only works when the quantity spans its full range; the measured exponent
     * clusters tightly, so an opacity ramp collapses every box to nearly empty and
     * the channel stops saying anything.
     */
    public static Color fillColor(double normalized, float alpha) {
        if (normalized < 0) {
            return new Color(0, 0, 0, 0);
        }
        double u = Math.max(0.0, Math.min(1.0, normalized));
        return new Color((int) Math.round(LOW.getRed() + u * (HIGH.getRed() - LOW.getRed())),
                (int) Math.round(LOW.getGreen() + u * (HIGH.getGreen() - LOW.getGreen())),
                (int) Math.round(LOW.getBlue() + u * (HIGH.getBlue() - LOW.getBlue())), Math.round(alpha * 255));
    }

    @Override
    public void draw(Graphics2D g, Plot2D.Viewport vp) {
        g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

        for (int i = 0; i < origins.length; i++) {
            double[] o = origins[i];
            double[] b = boxes[i];

            // py inverts, so the high-y corner maps to the smaller screen y
            double x0 = vp.px(o[0] - b[2]);
            double x1 = vp.px(o[0] + b[0]);
            double y0 = vp.py(o[1] + b[1]);
            double y1 = vp.py(o[1] - b[3]);

            Path2D.Double rect = new Path2D.Double();
            rect.moveTo(x0, y0);
            rect.lineTo(x1, y0);
            rect.lineTo(x1, y1);
            rect.lineTo(x0, y1);
            rect.closePath();

            Color fill = fillColor(exponent[i], maxFillAlpha);
            if (fill.getAlpha() > 0) {
                g.setColor(fill);
                g.fill(rect);
            }
            g.setColor(outlineColor(anisotropy[i]));
            g.setStroke(new BasicStroke(1.4f));
            g.draw(rect);

            g.setColor(new Color(60, 60, 60, 170));
            g.fillOval((int) Math.round(vp.px(o[0])) - 1, (int) Math.round(vp.py(o[1])) - 1, 3, 3);
        }
    }
}
