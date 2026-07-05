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
import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.List;

import javax.swing.JFrame;
import javax.swing.JPanel;

public final class Plot2D extends JPanel {

    private final double xmin, xmax, ymin, ymax;
    private final int margin = 38;
    private volatile List<Layer> layers = new ArrayList<>();

    private Plot2D(double xmin, double xmax, double ymin, double ymax) {
        this.xmin = xmin;
        this.xmax = xmax;
        this.ymin = ymin;
        this.ymax = ymax;
        setBackground(Color.WHITE);
    }

    // ---- square (symmetric) constructors used by the rotating examples ----
    public static Plot2D open(String title, double range, int sizePx) {
        return openRect(title, -range, range, -range, range, sizePx, sizePx);
    }

    public static Plot2D offscreen(double range) {
        return new Plot2D(-range, range, -range, range);
    }

    // ---- rectangular constructors for time-series / dashboards ----
    public static Plot2D openRect(String title, double xmin, double xmax, double ymin, double ymax, int w, int h) {
        Plot2D panel = new Plot2D(xmin, xmax, ymin, ymax);
        JFrame frame = new JFrame(title);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.getContentPane().add(panel);
        frame.setSize(w, h + 22);
        frame.setLocationRelativeTo(null);
        frame.setVisible(true);
        return panel;
    }

    public static Plot2D offscreenRect(double xmin, double xmax, double ymin, double ymax) {
        return new Plot2D(xmin, xmax, ymin, ymax);
    }

    public void render(List<Layer> scene) {
        this.layers = scene;
        repaint();
    }

    public BufferedImage renderImage(int w, int h, List<Layer> scene) {
        BufferedImage img = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB);
        Graphics2D g2 = img.createGraphics();
        paintScene(g2, w, h, scene);
        g2.dispose();
        return img;
    }

    public final class Viewport {
        private final int w, h;

        Viewport(int w, int h) {
            this.w = w;
            this.h = h;
        }

        public double px(double x) {
            return margin + (x - xmin) / (xmax - xmin) * (w - 2.0 * margin);
        }

        public double py(double y) {
            return margin + (ymax - y) / (ymax - ymin) * (h - 2.0 * margin);
        }

        public double xmin() {
            return xmin;
        }

        public double xmax() {
            return xmax;
        }

        public double ymin() {
            return ymin;
        }

        public double ymax() {
            return ymax;
        }

        public int width() {
            return w;
        }

        public int height() {
            return h;
        }
    }

    private void paintScene(Graphics2D g2, int w, int h, List<Layer> scene) {
        g2.setColor(Color.WHITE);
        g2.fillRect(0, 0, w, h);
        g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        Viewport vp = new Viewport(w, h);

        g2.setColor(new Color(238, 238, 238));
        double sx = niceStep((xmax - xmin) / 10), sy = niceStep((ymax - ymin) / 10);
        for (double gx = Math.ceil(xmin / sx) * sx; gx <= xmax; gx += sx) {
            g2.drawLine((int) vp.px(gx), (int) vp.py(ymax), (int) vp.px(gx), (int) vp.py(ymin));
        }
        for (double gy = Math.ceil(ymin / sy) * sy; gy <= ymax; gy += sy) {
            g2.drawLine((int) vp.px(xmin), (int) vp.py(gy), (int) vp.px(xmax), (int) vp.py(gy));
        }
        g2.setColor(new Color(205, 205, 205));
        g2.drawRect((int) vp.px(xmin), (int) vp.py(ymax), (int) (vp.px(xmax) - vp.px(xmin)),
                (int) (vp.py(ymin) - vp.py(ymax)));

        for (Layer layer : scene) {
            layer.draw(g2, vp);
        }
    }

    private static double niceStep(double rough) {
        double pow = Math.pow(10, Math.floor(Math.log10(rough)));
        double f = rough / pow;
        double nice = f < 1.5 ? 1 : f < 3 ? 2 : f < 7 ? 5 : 10;
        return nice * pow;
    }

    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);
        paintScene((Graphics2D) g, getWidth(), getHeight(), layers);
    }
}
