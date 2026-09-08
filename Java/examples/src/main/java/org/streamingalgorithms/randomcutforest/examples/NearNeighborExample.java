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

package org.streamingalgorithms.randomcutforest.examples;

import static java.lang.Math.PI;
import static org.streamingalgorithms.randomcutforest.examples.datasets.Fan.rotateClockWise;

import java.awt.*;
import java.awt.event.KeyEvent;
import java.awt.geom.Path2D;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;

import org.streamingalgorithms.randomcutforest.RandomCutForest;
import org.streamingalgorithms.randomcutforest.examples.datasets.Yinyang;
import org.streamingalgorithms.randomcutforest.examples.plot.Contour;
import org.streamingalgorithms.randomcutforest.examples.plot.GifWriter;
import org.streamingalgorithms.randomcutforest.examples.plot.Layer;
import org.streamingalgorithms.randomcutforest.examples.plot.Layers;
import org.streamingalgorithms.randomcutforest.examples.plot.Plot2D;
import org.streamingalgorithms.randomcutforest.returntypes.AnisotropicDensityOutput;

/**
 * Dynamic near neighbour, with cut, passage and stop boxes drawn around the
 * query.
 */
public class NearNeighborExample implements Example {

    /**
     * Density isolines. Unlike HalfImpact the data rotates every frame, so the
     * field cannot be precomputed and every contour costs a grid of density
     * queries. At The anisotropic walk costs roughly ten times a plain density
     * query, since it buffers the path and takes four crossings, so the grid
     * dropped to 36 and the refresh to every third frame when the field moved onto
     * it. At roughly 0.2 ms each a 48 by 48 grid is about half a second a frame, so
     * the grid is coarse and FIELD_EVERY lets the field be refreshed less often
     * than the probe moves. The data turns 2 degrees a frame, so reusing a field
     * for three frames lags it by six degrees.
     */
    private static final int FIELD_GRID = 36;
    private static final int FIELD_EVERY = 3;
    private static final int CONTOURS = 5;
    private static final double CONTOUR_LO = 0.45;
    private static final double CONTOUR_HI = 0.97;
    private static final Color ISO = new Color(214, 148, 34);
    /**
     * The half crossing, the minimal object; same kind of thing as the nn-square.
     */
    private static final Color CUT_BOX = new Color(150, 60, 170);
    private static final Color PASSAGE_BOX = new Color(30, 145, 120);
    private static final Color KNN_BALL = new Color(40, 120, 190);
    private static final Color STOP_BOX = new Color(214, 118, 34);

    /** Topo state for the recording, kept apart from the live toggle. */
    private static final boolean GIF_TOPO = true;

    /** Crossing widths in octaves that map to a full and an empty fill. */
    private static final double WIDTH_SHARP = 0.5;
    private static final double WIDTH_FLAT = 3.0;

    public static void main(String[] args) throws Exception {
        new NearNeighborExample().run();
    }

    @Override
    public String command() {
        return "near_neighbor";
    }

    @Override
    public String description() {
        return "Example of dynamic near neighbor computation";
    }

    @Override
    public void run() throws Exception {
        int newDimensions = 2;
        long randomSeed = 123;
        double range = 1.6;

        RandomCutForest newForest = RandomCutForest.builder().numberOfTrees(100).sampleSize(256)
                .dimensions(newDimensions).randomSeed(randomSeed).timeDecay(1.0 / 800).centerOfMassEnabled(true)
                .storeSequenceIndexesEnabled(true).build();

        boolean printFile = false;
        boolean livePlot = true;
        boolean saveGif = true;
        boolean showBox = true;
        int frameDelayMs = 12;
        int gifSizePx = 700;
        int gifDelayMs = 50;
        int gifEvery = 1;
        int reportEvery = 30;

        BufferedWriter file = printFile ? new BufferedWriter(new FileWriter("dynamic_near_neighbor_example")) : null;

        float[][] data = Yinyang.generate(1000);
        float[] queryPoint = new float[] { 0.5f, 0.6f };

        Plot2D plot = livePlot ? Plot2D.open("Dynamic Near Neighbor + Cut / Passage / Stop Boxes", range, 820)
                : Plot2D.offscreen(range);
        GifWriter gif = saveGif ? new GifWriter(new File("dynamic_near_neighbor.gif"), gifDelayMs, true) : null;

        // Gated from inside the layer, so a repaint alone honours the toggle while
        // paused. The recording gets its own list rather than a flag flipped around
        // renderImage: plot.render calls repaint, which is asynchronous, so the EDT
        // could paint inside such a window and the field would flicker back on.
        java.util.concurrent.atomic.AtomicBoolean topo = new java.util.concurrent.atomic.AtomicBoolean(true);
        List<Layer> rawField = new ArrayList<>();
        List<Layer> field = new ArrayList<>();
        plot.bindKey(KeyEvent.VK_T, "nn.topo", () -> topo.set(!topo.get()));
        System.out.println("keys: space pauses, right arrow steps, t toggles the isolines");

        Instant start = Instant.now();
        long fieldNanos = 0;
        long queryNanos = 0;
        long boxNanos = 0;
        int frame = 0;
        int flat = 0;

        for (int degree = 0; degree < 360; degree += 2) {
            float[][] bg = new float[data.length][2];
            int n = 0;
            for (var datum : data) {
                float[] transformed = rotateClockWise(datum, -2 * PI * degree / 360);
                bg[n][0] = transformed[0];
                bg[n][1] = transformed[1];
                n++;
                if (printFile) {
                    file.append(transformed[0] + " " + transformed[1] + "\n");
                }
                newForest.update(transformed);
            }
            if (printFile) {
                file.append("\n");
                file.append("\n");
            }

            if (frame % FIELD_EVERY == 0) {
                Instant f0 = Instant.now();
                rawField = Contour.isolines(
                        (x, y) -> newForest.getAnisotropicDensity(new float[] { (float) x, (float) y })
                                .passageDensity(0.001),
                        -range * 0.95, 2 * range * 0.95, FIELD_GRID, CONTOURS, CONTOUR_LO, CONTOUR_HI, ISO);
                field = new ArrayList<>();
                for (Layer inner : rawField) {
                    field.add((g, vp) -> {
                        if (topo.get()) {
                            inner.draw(g, vp);
                        }
                    });
                }
                fieldNanos += Duration.between(f0, Instant.now()).toNanos();
            }

            float[] movingQuery = rotateClockWise(queryPoint, -3 * PI * degree / 360);
            Instant q0 = Instant.now();
            float[] neighbor = newForest.getNearNeighborsInSample(movingQuery, 1).get(0).point;
            queryNanos += Duration.between(q0, Instant.now()).toNanos();

            double nnDistance = Math.hypot(neighbor[0] - movingQuery[0], neighbor[1] - movingQuery[1]);

            // Three boxes at the same query point; diagnostics below use the cut box.
            double[] box = null;
            double[] passageBox = null;
            double[] stopBox = null;
            double anisotropy = 1.0;
            if (showBox) {
                Instant b0 = Instant.now();
                AnisotropicDensityOutput out = newForest.getAnisotropicDensity(movingQuery);
                boxNanos += Duration.between(b0, Instant.now()).toNanos();
                if (out.isReliable()) {
                    box = out.cutBox();
                    passageBox = out.passageBox();
                    stopBox = out.stopBox();
                    anisotropy = out.getAnisotropy();
                } else {
                    flat++; // no single scale here; drawing a box would assert one
                }
            }

            // Direct data-against-inference check. The box is what the forest concluded;
            // these are the points actually in it. Cheap -- the rotated frame is already
            // in hand -- and it is the only line in the example that touches the raw data
            // to judge the model rather than to feed it.
            int inBox = 0, inIso = 0;
            // Exact k-th nearest distance over the whole frame, k = sqrt(dataSize).
            // Exact and unsampled on purpose: the classical estimator is defined on all
            // the data, and taking it from a 256-point sample would inflate the radius
            // and hand the comparison an easy win. This is the baseline the mass box has
            // to beat on its own terms.
            int kNN = (int) Math.round(Math.sqrt(data.length));
            double[] d2 = new double[bg.length];
            for (int i = 0; i < bg.length; i++) {
                double dx = bg[i][0] - movingQuery[0], dy = bg[i][1] - movingQuery[1];
                d2[i] = dx * dx + dy * dy;
            }
            java.util.Arrays.sort(d2);
            double kRadius = Math.sqrt(d2[Math.min(kNN - 1, d2.length - 1)]);
            double kDensity = (kRadius > 0) ? kNN / (Math.PI * kRadius * kRadius) : 0;
            double isoHalf = 0;
            boolean nnInBox = false;
            if (box != null) {
                isoHalf = nnDistance;

                // high extends toward minus, low toward plus
                double xLo = movingQuery[0] - box[0], xHi = movingQuery[0] + box[2];
                double yLo = movingQuery[1] - box[1], yHi = movingQuery[1] + box[3];
                for (float[] q : bg) {
                    if (q[0] >= xLo && q[0] <= xHi && q[1] >= yLo && q[1] <= yHi) {
                        inBox++;
                    }
                    if (Math.abs(q[0] - movingQuery[0]) <= isoHalf && Math.abs(q[1] - movingQuery[1]) <= isoHalf) {
                        inIso++;
                    }
                }
                nnInBox = neighbor[0] >= xLo && neighbor[0] <= xHi && neighbor[1] >= yLo && neighbor[1] <= yHi;
            }

            if (printFile) {
                file.append(movingQuery[0] + " " + movingQuery[1] + " " + (neighbor[0] - movingQuery[0]) + " "
                        + (neighbor[1] - movingQuery[1]) + "\n");
                file.append("\n");
                file.append("\n");
            }

            List<Layer> body = new ArrayList<>();
            body.add(Layers.dots(bg, new Color(120, 120, 120), 2.2));
            // A circle, not a square: the classical k-NN region is a ball, and drawing
            // it as a square would quietly change its volume by 4/pi and flatter the
            // comparison.
            body.add(circleLayer(movingQuery[0], movingQuery[1], kRadius, KNN_BALL));
            if (box != null) {
                // Translucent fills preserve the nested regions. Draw the soft,
                // outline-free stop box first, then passage and cut on top.
                body.add(boxFill(movingQuery[0], movingQuery[1], stopBox, STOP_BOX, 35, false));
                body.add(boxFill(movingQuery[0], movingQuery[1], passageBox, PASSAGE_BOX, 40, true));
                body.add(boxFill(movingQuery[0], movingQuery[1], box, CUT_BOX, 50, true));
            }
            body.add(Layers.arrows(new double[][] { { movingQuery[0], movingQuery[1] } },
                    new double[][] { { neighbor[0] - movingQuery[0], neighbor[1] - movingQuery[1] } }, Layers.color(1),
                    2.0f));
            body.add(Layers.dots(new float[][] { { neighbor[0], neighbor[1] } }, Layers.color(0), 5));
            body.add(Layers.dots(new float[][] { { movingQuery[0], movingQuery[1] } }, Color.BLACK, 4));
            // read out along the bottom of the plot, in data coordinates
            // Reported, not judged. The nearest neighbour falling outside the box is
            // not a failure: the box is a typical scale, the median over trees of where
            // the separation probability crosses one half, while the nearest neighbour
            // is an extreme statistic -- the closest of a thousand points. In a concave
            // notch the neighbour sits on a thin side while the box runs long elsewhere,
            // and the two are simply answering different questions.
            // Magenta when the nearest neighbour lands outside, deliberately not red.
            // It marks a divergence, not an error: the box is a typical scale -- the
            // median over trees of where the separation probability crosses one half --
            // while the nearest neighbour is an extreme statistic, the closest of a
            // thousand points. The frames where they disagree are the ones worth
            // stopping on, because they are where the two quantities visibly stop being
            // interchangeable.
            double areaBox = (box == null) ? 0 : (box[0] + box[2]) * (box[1] + box[3]);
            double areaIso = 4 * isoHalf * isoHalf;
            double densBox = (areaBox > 0) ? inBox / areaBox : 0;
            double densIso = (areaIso > 0) ? inIso / areaIso : 0;
            String readout = (box == null) ? "no single scale here"
                    // meanVol is the volume getMassDensity actually divides by: the mean
                    // of the per-tree volumes. areaBox is the area of the box drawn, the
                    // product of the mean widths. Positively correlated widths make the
                    // first the larger by Jensen, so a big gap means the picture and the
                    // number disagree about how much space the estimate covers.
                    : String.format(
                            "cut box %d pts / %.3f = %.0f      k-NN ball %d pts / %.3f = %.0f" + "   aspect %.2f",
                            inBox, areaBox, densBox, kNN, Math.PI * kRadius * kRadius, kDensity, anisotropy);
            body.add(Layers.label(-range * 0.92, -range * 0.90, readout,
                    (box == null || nnInBox) ? new Color(60, 60, 60) : new Color(176, 32, 160)));
            body.add(Layers.legend(
                    new String[] { "data", "approx NN, k = 1 (forest)", "cut box", "passage box",
                            "stop box (shading only)", "exact k-NN ball, k = sqrt(n)", "density isolines" },
                    new Color[] { new Color(120, 120, 120), Layers.color(0), CUT_BOX, PASSAGE_BOX, STOP_BOX, KNN_BALL,
                            ISO },
                    new Layers.Swatch[] { Layers.Swatch.DOTS, Layers.Swatch.DOTS, Layers.Swatch.BOX, Layers.Swatch.BOX,
                            Layers.Swatch.BOX, Layers.Swatch.BOX, Layers.Swatch.LINE }));

            List<Layer> scene = new ArrayList<>(field);
            scene.addAll(body);

            if (livePlot) {
                plot.render(scene);
            }
            if (saveGif && frame % gifEvery == 0) {
                List<Layer> gifScene = new ArrayList<>(GIF_TOPO ? rawField : List.of());
                gifScene.addAll(body);
                gif.writeFrame(plot.renderImage(gifSizePx, gifSizePx, gifScene));
            }
            if (livePlot) {
                // Space freezes the frame, right arrow steps one at a time. Placed after
                // writeFrame so pausing to look at something does not punch a hole in the
                // recording: the GIF stays continuous whatever you do on screen. Blocks
                // this thread only, never the EDT, so the frozen frame keeps repainting.
                plot.awaitResume();
                if (frameDelayMs > 0) {
                    Thread.sleep(frameDelayMs);
                }
            }

            if (++frame % reportEvery == 0) {
                // Extent along the direction the nearest neighbour actually lies, not a
                // mean over all four faces. When the query sits outside the cloud only
                // the two faces pointing at the data are active and the other two are
                // near zero, so a four-face mean is halved exactly when the query is
                // outside -- which is what made extent/nn read below one out there and
                // above one inside. Picking the face the neighbour is on compares like
                // with like: high_i extends toward minus i, low_i toward plus i.
                // Distance from the query to the box wall along the neighbour's own
                // direction: the ray hits whichever wall comes first, so this is a MIN
                // over axes, not a projection-weighted blend of the two extents. The
                // blend overestimates badly on an anisotropic box -- it averages in the
                // long axis while the short one is what actually cuts the ray off, which
                // is how reach/nn read above 1 on frames where the neighbour was
                // demonstrably outside the box.
                double reach = 0;
                if (box != null) {
                    double dx = neighbor[0] - movingQuery[0], dy = neighbor[1] - movingQuery[1];
                    double norm = Math.hypot(dx, dy);
                    if (norm > 0) {
                        double ux = dx / norm, uy = dy / norm;
                        double tx = (ux == 0) ? Double.MAX_VALUE : ((ux > 0) ? box[2] : box[0]) / Math.abs(ux);
                        double ty = (uy == 0) ? Double.MAX_VALUE : ((uy > 0) ? box[3] : box[1]) / Math.abs(uy);
                        reach = Math.min(tx, ty);
                    }
                }
                System.out.printf(
                        "[%3d deg] total %d ms | query %.3f ms/frame | box %.3f ms/frame "
                                + "| field %.1f ms | nn %.4f | reach/nn %.2f | aspect %.2f | box %d pts %.0f/area "
                                + "| nn-sq %d pts %.0f/area | x%.2f denser | flat %d%n",
                        degree, Duration.between(start, Instant.now()).toMillis(), queryNanos / 1e6 / frame,
                        boxNanos / 1e6 / frame, fieldNanos / 1e6 / frame, nnDistance,
                        (nnDistance > 0) ? reach / nnDistance : 0, anisotropy, inBox, densBox, inIso, densIso,
                        (densIso > 0) ? densBox / densIso : 0.0, flat);
            }
        }

        if (gif != null) {
            gif.close();
            System.out.println("wrote dynamic_near_neighbor.gif");
        }
        if (file != null) {
            file.close();
        }
    }

    /** box is {high_x, high_y, low_x, low_y}; high extends toward minus. */
    private static void addRect(Path2D.Double path, Plot2D.Viewport vp, double cx, double cy, double[] b) {
        double x0 = vp.px(cx - b[0]), x1 = vp.px(cx + b[2]);
        double y0 = vp.py(cy + b[3]), y1 = vp.py(cy - b[1]);
        path.moveTo(x0, y0);
        path.lineTo(x1, y0);
        path.lineTo(x1, y1);
        path.lineTo(x0, y1);
        path.closePath();
    }

    /**
     * A circle at true scale; the classical k-NN region is a ball, not a square.
     */
    private static Layer circleLayer(double cx, double cy, double r, Color color) {
        return (g, vp) -> {
            g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
            double x0 = vp.px(cx - r), x1 = vp.px(cx + r);
            double y0 = vp.py(cy + r), y1 = vp.py(cy - r);
            g.setColor(new Color(color.getRed(), color.getGreen(), color.getBlue(), 40));
            g.fillOval((int) Math.round(x0), (int) Math.round(y0), (int) Math.round(x1 - x0),
                    (int) Math.round(y1 - y0));
            g.setColor(color);
            g.setStroke(new BasicStroke(1.4f, BasicStroke.CAP_BUTT, BasicStroke.JOIN_MITER, 10f, new float[] { 5f, 4f },
                    0f));
            g.drawOval((int) Math.round(x0), (int) Math.round(y0), (int) Math.round(x1 - x0),
                    (int) Math.round(y1 - y0));
        };
    }

    /**
     * A translucent box at true scale, optionally outlined; high extends toward
     * minus.
     */
    private static Layer boxFill(double cx, double cy, double[] b, Color color, int alpha, boolean outline) {
        return (g, vp) -> {
            Graphics2D copy = (Graphics2D) g.create();
            try {
                copy.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
                Path2D.Double r = new Path2D.Double();
                addRect(r, vp, cx, cy, b);
                copy.setColor(new Color(color.getRed(), color.getGreen(), color.getBlue(), alpha));
                copy.fill(r);
                if (outline) {
                    copy.setColor(color);
                    copy.setStroke(new BasicStroke(1.4f));
                    copy.draw(r);
                }
            } finally {
                copy.dispose();
            }
        };
    }
}
