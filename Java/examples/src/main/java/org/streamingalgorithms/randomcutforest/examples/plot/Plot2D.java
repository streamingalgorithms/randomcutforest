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
import java.awt.event.ActionEvent;
import java.awt.event.KeyEvent;
import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.List;

import javax.swing.AbstractAction;
import javax.swing.JFrame;
import javax.swing.JPanel;
import javax.swing.KeyStroke;

public final class Plot2D extends JPanel {

    private final double xmin, xmax, ymin, ymax;
    private final int margin = 38;
    private volatile List<Layer> layers = new ArrayList<>();

    /**
     * Playback control.
     *
     * <p>
     * The render loop runs on the caller's thread while key events arrive on the
     * EDT, so this is a wait/notify rather than a flag the loop polls: a paused
     * loop blocks in {@link #awaitResume()} and burns no CPU, and the EDT stays
     * free to repaint — which is the entire point, since a frozen frame you cannot
     * redraw is not much use.
     */
    private final Object playLock = new Object();
    private boolean paused;
    private int pendingSteps;

    private Plot2D(double xmin, double xmax, double ymin, double ymax) {
        this.xmin = xmin;
        this.xmax = xmax;
        this.ymin = ymin;
        this.ymax = ymax;
        setBackground(Color.WHITE);
        installPlaybackKeys();
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

    // ------------------------------------------------------------------
    // playback
    // ------------------------------------------------------------------

    /**
     * SPACE toggles pause, RIGHT advances a single frame while paused.
     *
     * <p>
     * Deliberately InputMap/ActionMap rather than a KeyListener: a JPanel is not
     * focusable by default, so a KeyListener attached here would never fire, and
     * would fail silently rather than throwing. WHEN_IN_FOCUSED_WINDOW dispatches
     * at the frame level and needs no focus handling at all.
     *
     * <p>
     * The offscreen constructors install these too. That is harmless — the binding
     * cannot fire without a window — and it means a headless run calling
     * awaitResume() simply returns immediately.
     */
    private void installPlaybackKeys() {
        bind(KeyEvent.VK_SPACE, "rcf.togglePause", this::togglePause);
        bind(KeyEvent.VK_RIGHT, "rcf.stepOnce", this::stepOnce);
    }

    private void bind(int keyCode, String name, Runnable action) {
        getInputMap(WHEN_IN_FOCUSED_WINDOW).put(KeyStroke.getKeyStroke(keyCode, 0), name);
        getActionMap().put(name, new AbstractAction() {
            @Override
            public void actionPerformed(ActionEvent e) {
                action.run();
            }
        });
    }

    /**
     * Register an extra key, for example-specific toggles. Repaints afterwards so a
     * toggle takes effect while paused, when the render loop is blocked in
     * awaitResume and will not rebuild the scene on its own.
     */
    public void bindKey(int keyCode, String name, Runnable action) {
        bind(keyCode, name, () -> {
            action.run();
            repaint();
        });
    }

    public void togglePause() {
        synchronized (playLock) {
            paused = !paused;
            pendingSteps = 0;
            playLock.notifyAll();
        }
        repaint();
    }

    /** Release exactly one frame while paused; ignored while running. */
    public void stepOnce() {
        synchronized (playLock) {
            if (paused) {
                pendingSteps++;
                playLock.notifyAll();
            }
        }
    }

    public boolean isPaused() {
        synchronized (playLock) {
            return paused;
        }
    }

    /**
     * Blocks while paused. Call once per frame from the render loop, never from the
     * EDT: blocking the EDT would freeze the repaint that makes pausing useful in
     * the first place.
     *
     * <p>
     * Where this sits relative to a GIF writer decides what gets recorded. After
     * writeFrame, pausing leaves the recording continuous; before it, the GIF holds
     * only the frames you watched.
     */
    public void awaitResume() {
        synchronized (playLock) {
            while (paused && pendingSteps == 0) {
                try {
                    playLock.wait();
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    return;
                }
            }
            if (paused && pendingSteps > 0) {
                pendingSteps--;
            }
        }
    }

    // ------------------------------------------------------------------

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
        g2.setClip(margin, margin, w - 2 * margin, h - 2 * margin);
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
        Graphics2D g2 = (Graphics2D) g;
        paintScene(g2, getWidth(), getHeight(), layers);
        if (isPaused()) {
            // paintScene leaves the plot area clipped; the hint lives in the margin
            g2.setClip(null);
            g2.setColor(new Color(40, 40, 40, 200));
            g2.setFont(g2.getFont().deriveFont(Font.BOLD, 12f));
            g2.drawString("PAUSED   space = resume, right arrow = step", margin, margin - 12);
        }
    }
}