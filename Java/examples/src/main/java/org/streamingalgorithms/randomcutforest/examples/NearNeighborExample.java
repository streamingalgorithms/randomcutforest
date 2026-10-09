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
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

import org.streamingalgorithms.randomcutforest.RandomCutForest;
import org.streamingalgorithms.randomcutforest.examples.datasets.Yinyang;
import org.streamingalgorithms.randomcutforest.examples.plot.*;
import org.streamingalgorithms.randomcutforest.returntypes.AnisotropicLocalGeometry;
import org.streamingalgorithms.randomcutforest.returntypes.Neighbor;

/**
 * Dynamic near neighbour, with the gap, cut, passage and stop boxes drawn
 * around the query over a shaded density field.
 *
 * <p>
 * The four boxes nest: GAP is the empty margin between the query and the points
 * it merges with, and CUT is that margin plus those points' own spread, so GAP
 * is contained in CUT is contained in PASSAGE is contained in STOP. The gap box
 * is therefore invisible whenever the query sits inside the data, which is the
 * correct reading rather than a missing layer.
 *
 * <p>
 * Keys: <b>space</b> pauses and <b>right arrow</b> steps, <b>d</b> shows or
 * hides the data, <b>t</b> the isolines drawn through the cut, and <b>c</b>
 * cycles the boxes: all four nested, then GAP, CUT, PASSAGE and STOP alone,
 * then none. The readout along the bottom stays on the cut box whatever is
 * drawn.
 */
public class NearNeighborExample implements Example {

    /**
     * Isolines drawn through the cut, from every LINE_STRIDE-th point of a frame,
     * asked before it is inserted; see {@link CutIsolines}. That is one walk per
     * reference point, about what the old grid of density probes cost every third
     * frame, so the lines now follow every frame.
     */
    private static final int LINE_STRIDE = 2;
    private static final int FIELD_GRID = 120;
    private static final Color ISO = new Color(214, 148, 34);
    /**
     * The half crossing, the minimal object; same kind of thing as the nn-square.
     */
    private static final Color CUT_BOX = new Color(150, 60, 170);
    private static final Color PASSAGE_BOX = new Color(30, 145, 120);
    private static final Color KNN_BALL = new Color(40, 120, 190);
    private static final Color STOP_BOX = new Color(214, 118, 34);
    /**
     * The void box: the same red as the neighbour cloud, because it is the same
     * statement about isolation seen from the measure instead of from the data.
     */
    private static final Color GAP_BOX = new Color(214, 39, 40);

    /** What the c key steps through, then none; index 0 draws all four nested. */
    private static final String[] BOXES = { "ALL", "GAP", "CUT", "PASSAGE", "STOP" };

    /**
     * Shading. Cool and neutral so it does not compete with the warm isolines drawn
     * over it, and floored: painting from band BAND_FLOOR up starts inside the line
     * that holds three quarters of the mass, which is where the density actually
     * is.
     */
    private static final Color BAND = new Color(70, 90, 120);
    private static final int BAND_FLOOR = 2;
    /**
     * The whole returned neighbour list, not just the closest. Translucent so
     * overlap reads as concentration; same hue as the winner dot, Layers.color(0).
     */
    private static final Color NN_CLOUD = new Color(214, 39, 40, 90);

    /**
     * Distance threshold handed to getNearNeighborsInSample. It is a threshold and
     * not a k: every tree contributes its traversal leaf, and leaves farther than
     * this are dropped. Sorted ascending by distance, so element 0 is the closest.
     */
    private static final double NN_THRESHOLD = 1.0;
    /**
     * Radius floor and scale for the vote-weighted cloud; area encodes the vote.
     */
    private static final double NN_MIN_R = 1.5;
    private static final double NN_SCALE_R = 5.0;
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

        AtomicBoolean topo = new AtomicBoolean(true);
        plot.bindKey(KeyEvent.VK_T, "nn.isolines", () -> topo.set(!topo.get()));
        AtomicBoolean showData = new AtomicBoolean(true);
        plot.bindKey(KeyEvent.VK_D, "nn.data", () -> showData.set(!showData.get()));
        // BOXES in order, then none (-1); the boxes and the legend read it when drawing
        AtomicInteger boxes = new AtomicInteger(0);
        plot.bindKey(KeyEvent.VK_C, "nn.boxes",
                () -> boxes.set(boxes.get() == BOXES.length - 1 ? -1 : boxes.get() + 1));
        System.out.println("keys: space pauses, right arrow steps, d toggles the data, t the isolines, "
                + "c cycles the boxes (ALL/GAP/CUT/PASSAGE/STOP/none)");
        KeysLayer keys = new KeysLayer(KeysLayer.Corner.TOP_LEFT).add("space", "pause", null)
                .add("\u2192", "step", null).add("d", "data", KeysLayer.onOff(showData::get))
                .add("t", "isolines through the cut", KeysLayer.onOff(topo::get))
                .add("c", "boxes", () -> boxes.get() < 0 ? "off" : BOXES[boxes.get()]);

        Instant start = Instant.now();
        long fieldNanos = 0;
        long queryNanos = 0;
        long boxNanos = 0;
        int frame = 0;
        int flat = 0;

        for (int degree = 0; degree < 360; degree += 2) {
            float[][] bg = new float[data.length][2];
            CutIsolines lines = new CutIsolines(-range * 0.95, 2 * range * 0.95, FIELD_GRID, ISO).bands(BAND,
                    BAND_FLOOR);
            int n = 0;
            for (var datum : data) {
                float[] transformed = rotateClockWise(datum, -2 * PI * degree / 360);
                bg[n][0] = transformed[0];
                bg[n][1] = transformed[1];
                if (n % LINE_STRIDE == 0) {
                    Instant f0 = Instant.now();
                    lines.add(transformed, newForest.getAnisotropicGeometry(transformed).cutSpread());
                    fieldNanos += Duration.between(f0, Instant.now()).toNanos();
                }
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

            // Gated from inside the layer, so a repaint alone honours the toggle while
            // paused. The recording gets its own list rather than a flag flipped around
            // renderImage: plot.render calls repaint, which is asynchronous, so the EDT
            // could paint inside such a window and the field would flicker back on.
            List<Layer> rawField = List.of(lines);
            List<Layer> field = List.of(Layers.when(topo::get, lines));

            float[] movingQuery = rotateClockWise(queryPoint, -3 * PI * degree / 360);

            // The forest returns the whole list, one entry per distinct traversal leaf,
            // merged across trees and sorted ascending by distance. The closest drives
            // the arrow and every diagnostic below; the rest are drawn as the cloud.
            Instant q0 = Instant.now();
            List<Neighbor> neighbors = newForest.getNearNeighborsInSample(movingQuery, NN_THRESHOLD);
            if (neighbors.isEmpty()) {
                // Nothing within the threshold: fall back to the unbounded overload
                // rather than silently dropping the frame.
                neighbors = newForest.getNearNeighborsInSample(movingQuery);
            }
            queryNanos += Duration.between(q0, Instant.now()).toNanos();

            // Defensive only: the list is non-empty once the forest is output-ready,
            // which it is after the first frame's thousand updates.
            float[] neighbor = neighbors.isEmpty() ? movingQuery : neighbors.get(0).point;

            // Neighbor.count is the number of trees that landed on that point, summed
            // by the collector when duplicates merge. Normalised by the largest so the
            // encoding does not change when numberOfTrees does.
            double[][] nnXy = new double[neighbors.size()][2];
            double[] nnWeight = new double[neighbors.size()];
            double nnMax = 1;
            int nnVotes = 0;
            for (Neighbor nb : neighbors) {
                nnMax = Math.max(nnMax, nb.count);
                nnVotes += nb.count;
            }
            for (int i = 0; i < neighbors.size(); i++) {
                Neighbor nb = neighbors.get(i);
                nnXy[i][0] = nb.point[0];
                nnXy[i][1] = nb.point[1];
                nnWeight[i] = nb.count / nnMax;
            }

            double nnDistance = Math.hypot(neighbor[0] - movingQuery[0], neighbor[1] - movingQuery[1]);

            // Three boxes at the same query point; diagnostics below use the cut box.
            double[] box = null;
            double[] gapBox = null;
            double[] passageBox = null;
            double[] stopBox = null;
            double anisotropy = 1.0;
            if (showBox) {
                Instant b0 = Instant.now();
                AnisotropicLocalGeometry out = newForest.getAnisotropicGeometry(movingQuery);
                boxNanos += Duration.between(b0, Instant.now()).toNanos();
                if (out.isReliable()) {
                    box = out.cutBox();
                    gapBox = out.gapBox();
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
            body.add(Layers.when(showData::get, Layers.dots(bg, new Color(120, 120, 120), 2.2)));
            // A circle, not a square: the classical k-NN region is a ball, and drawing
            // it as a square would quietly change its volume by 4/pi and flatter the
            // comparison.
            body.add(circleLayer(movingQuery[0], movingQuery[1], kRadius, KNN_BALL));
            if (box != null) {
                // Translucent fills only. The outlines were fighting the isolines,
                // and the nesting already reads from the stacked alpha. Drawn
                // outermost first so every box stays visible: STOP, PASSAGE, CUT,
                // then the gap on top.
                // Each one gated by the c key, read when drawing.
                body.add(Layers.when(() -> shows(boxes.get(), 4),
                        boxFill(movingQuery[0], movingQuery[1], stopBox, STOP_BOX, 35, false)));
                body.add(Layers.when(() -> shows(boxes.get(), 3),
                        boxFill(movingQuery[0], movingQuery[1], passageBox, PASSAGE_BOX, 40, false)));
                body.add(Layers.when(() -> shows(boxes.get(), 2),
                        boxFill(movingQuery[0], movingQuery[1], box, CUT_BOX, 50, false)));
                // One-sided by construction -- only the faces pointing at the data
                // carry a gap -- and identically zero when the query is interior.
                // A zero-area rectangle there is the statistic working.
                body.add(Layers.when(() -> shows(boxes.get(), 1),
                        boxFill(movingQuery[0], movingQuery[1], gapBox, GAP_BOX, 90, false)));
            }
            // The cloud of leaves the traversals actually reached, above the box fills
            // and below the arrow and the winner. Area encodes the vote, so the shape of
            // the cloud is the forest's implicit kernel at this query. These are not
            // exact nearest neighbours: each tree returns where its random cuts sent the
            // query, which is the point of drawing them.
            body.add(Layers.weightedDots(nnXy, nnWeight, NN_CLOUD, NN_MIN_R, NN_SCALE_R));
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
            body.add((g, vp) -> legend(boxes.get()).draw(g, vp));

            List<Layer> scene = new ArrayList<>(field);
            scene.addAll(body);
            scene.add(keys);

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
                                + "| nn-sq %d pts %.0f/area | x%.2f denser | leaves %d/%d votes | flat %d%n",
                        degree, Duration.between(start, Instant.now()).toMillis(), queryNanos / 1e6 / frame,
                        boxNanos / 1e6 / frame, fieldNanos / 1e6 / frame, nnDistance,
                        (nnDistance > 0) ? reach / nnDistance : 0, anisotropy, inBox, densBox, inIso, densIso,
                        (densIso > 0) ? densBox / densIso : 0.0, neighbors.size(), nnVotes, flat);
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

    /** Whether the c key's choice draws the box at this index of BOXES. */
    private static boolean shows(int selected, int box) {
        return selected == 0 || selected == box;
    }

    /**
     * The key; the box rows follow the c key, so no row names a box that is not
     * drawn.
     */
    private static Layer legend(int selected) {
        List<String> labels = new ArrayList<>(List.of("data", "near neighbors, area = tree votes", "closest (forest)"));
        List<Color> colors = new ArrayList<>(List.of(new Color(120, 120, 120), NN_CLOUD, Layers.color(0)));
        String[] names = { null, "gap box (void, empty when inside)", "cut box", "passage box",
                "stop box (shading only)" };
        Color[] boxColors = { null, GAP_BOX, CUT_BOX, PASSAGE_BOX, STOP_BOX };
        for (int b = 1; b < names.length; b++) {
            if (shows(selected, b)) {
                labels.add(names[b]);
                colors.add(boxColors[b]);
            }
        }
        labels.add("exact k-NN ball, k = sqrt(n)");
        colors.add(KNN_BALL);
        labels.add("isolines through the cut");
        colors.add(ISO);
        Layers.Swatch[] swatches = new Layers.Swatch[labels.size()];
        for (int i = 0; i < swatches.length; i++) {
            swatches[i] = (i < 3) ? Layers.Swatch.DOTS
                    : (i == swatches.length - 1) ? Layers.Swatch.LINE : Layers.Swatch.BOX;
        }
        return Layers.legend(labels.toArray(new String[0]), colors.toArray(new Color[0]), swatches);
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