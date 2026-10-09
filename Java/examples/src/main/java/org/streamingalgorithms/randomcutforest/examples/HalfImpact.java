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

import java.awt.*;
import java.awt.event.KeyEvent;
import java.io.File;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.IntSupplier;
import java.util.stream.Collectors;

import org.streamingalgorithms.randomcutforest.IVisitorFactory;
import org.streamingalgorithms.randomcutforest.RandomCutForest;
import org.streamingalgorithms.randomcutforest.examples.plot.CutIsolines;
import org.streamingalgorithms.randomcutforest.examples.plot.GifWriter;
import org.streamingalgorithms.randomcutforest.examples.plot.KeysLayer;
import org.streamingalgorithms.randomcutforest.examples.plot.Layer;
import org.streamingalgorithms.randomcutforest.examples.plot.Layers;
import org.streamingalgorithms.randomcutforest.examples.plot.Plot2D;
import org.streamingalgorithms.randomcutforest.interpolation.AnisotropicLocalVisitor;
import org.streamingalgorithms.randomcutforest.returntypes.AnisotropicLocalGeometry;
import org.streamingalgorithms.randomcutforest.returntypes.InterpolationMeasure;
import org.streamingalgorithms.randomcutforest.returntypes.Neighbor;

/**
 * A ray swept through an annulus, with the <em>gap</em> box drawn against an
 * isotropic nearest-neighbour square at every step.
 *
 * <p>
 * The gap box is the empty margin between the probe and the points it merges
 * with -- the cut box with the merged node's own spread subtracted -- so it is
 * the void and nothing else. That makes it the right object for this picture:
 * the nn-square asserts the void is a disc of radius nn, and the gap box says
 * what the void actually is. Where the two disagree is the anisotropy, measured
 * rather than assumed.
 *
 * <p>
 * The box collapses to nothing while the probe is inside the wall, which is
 * correct and is the frame that tells you the statistic is not merely a
 * rescaled distance.
 *
 * <p>
 * Two headings are swept because {@code getAnisotropy} is an axis-aligned
 * reading: a ridge at 45 degrees inflates every axis equally and reports as
 * isotropic. The ray-frame aspect below is measured in the ray's own frame and
 * does not have that blind spot, so the two columns disagreeing is the point
 * rather than a defect.
 *
 * <p>
 * Keys: <b>space</b> pauses and <b>right arrow</b> steps, <b>d</b> shows or
 * hides the ring, <b>t</b> the isolines drawn through the cut, <b>c</b> cycles
 * the box drawn along the ray (GAP, CUT, PASSAGE, STOP, then off) and <b>e</b>
 * shows or hides the tree ensemble behind it. The printed tables stay on the
 * gap box whatever is drawn.
 */
public class HalfImpact implements Example {

    private static final double RING_RADIUS = 1.0;
    private static final int RING_POINTS = 2000;

    /**
     * Radial thickness of the annulus, used by the animated sweep. The wall runs
     * from RING_RADIUS - w/2 to RING_RADIUS + w/2, and the point count is held
     * fixed, so a thicker wall is the same material spread wider rather than more
     * material.
     */
    private static final double RING_THICKNESS = 0.25;

    /** Thicknesses visited by the numeric study before the animation. */
    private static final double[] THICKNESS_SWEEP = { 0.01, 0.03, 0.08, 0.16, 0.32, 0.60 };

    /** Impact parameters, as fractions of the ring radius. */
    private static final double[] IMPACTS = { 0.0, 0.5, 0.92 };
    /** Ray headings. Zero rides a coordinate axis; 45 rides the diagonal. */
    private static final double[] ANGLES = { 0.0, 45.0 };
    private static final Color[] TRACK = { new Color(196, 62, 40), new Color(28, 122, 168), new Color(52, 140, 62) };

    /** The naive baseline drawn alongside: a centred square of half-width nn. */
    private static final Color NN_BOX = new Color(90, 90, 105);

    /**
     * The leaves the forest's traversals actually reach from the probe. Violet so
     * it collides with none of the three TRACK colours, since the track drawn on
     * any given pass changes.
     */
    private static final Color LEAF_CLOUD = new Color(126, 68, 178, 95);
    /** Radius floor and scale for the leaf cloud; area encodes the vote. */
    private static final double LEAF_MIN_R = 1.4;
    private static final double LEAF_SCALE_R = 4.0;

    /** Isolines: grid resolution and arrow lattice; see {@link CutIsolines}. */
    private static final int FIELD_GRID = 130;
    private static final int ARROW_GRID = 15;
    private static final double ARROW_LENGTH = 0.13;
    private static final Color ISO = new Color(214, 148, 34);
    private static final Color GRAD = new Color(120, 100, 70);

    /**
     * Shading under the isolines. Cool and neutral so it does not compete with the
     * warm lines drawn over it, and floored: painting from BAND_FLOOR up starts
     * inside the line that holds three quarters of the mass, which puts the ink on
     * the wall. It costs nothing beyond the grid the lines are drawn from.
     */
    private static final Color BAND = new Color(70, 90, 120);
    private static final int BAND_FLOOR = 2;

    /** Steps between printed rows; 89 steps per pass is more than anyone reads. */
    private static final int ROW_EVERY = 8;

    /**
     * Topo state used for the recording, held apart from the live toggle. The gate
     * is evaluated when a layer draws, and renderImage draws, so without this the t
     * key would cut the field in and out of the GIF mid-run. Same reasoning that
     * put awaitResume after writeFrame: what is on screen is for looking at, the
     * recording should come out continuous either way.
     */
    private static final boolean GIF_TOPO = true;

    private static final double START_X = -2.2;
    private static final double END_X = 2.2;
    private static final int STEPS = 88;

    // ------------------------------------------------------------------
    // Additions: the ensemble behind the box, toggled live with e.
    // ------------------------------------------------------------------

    /**
     * Draw every tree's box as a faint outline, the lead from the probe to the mean
     * box centre, and the one-standard-deviation ellipse of the box centres. The
     * averaged box is the mean of this cloud; the cloud is the swing.
     */
    private static final boolean GIF_SWING = true;
    private static final Color LEAD = new Color(30, 30, 30);

    /**
     * The boxes the c key steps through, in the order of the averaged geometry's
     * accessors.
     */
    private static final String[] KINDS = { "GAP", "CUT", "PASSAGE", "STOP" };

    public static void main(String[] args) throws Exception {
        new HalfImpact().run();
    }

    @Override
    public String command() {
        return "half_impact";
    }

    @Override
    public String description() {
        return "cut box along a ray through a ring, by impact parameter";
    }

    @Override
    public void run() throws Exception {
        long randomSeed = 17;
        double range = 2.5;

        boolean livePlot = true;
        boolean saveGif = true;
        int frameDelayMs = 30;
        int gifSizePx = 760;

        RandomCutForest forest = RandomCutForest.builder().numberOfTrees(100).sampleSize(256).dimensions(2)
                .randomSeed(randomSeed).timeDecay(1.0 / (0.8 * RING_POINTS)).centerOfMassEnabled(true).build();

        penetrationStudy(randomSeed);

        float[][] ring = annulus(RING_THICKNESS, randomSeed);

        Plot2D plot = livePlot ? Plot2D.open("Half Impact - cut box along a ray", range, 860) : Plot2D.offscreen(range);
        GifWriter gif = saveGif ? new GifWriter(new File("half_impact.gif"), 60, true) : null;

        // The ring never moves, so the isolines are fixed for the whole run and are
        // drawn once. Only the probe moves. The forest is primed first, since the step
        // loop is what normally feeds it.
        for (int warm = 0; warm < 3; warm++) {
            for (float[] q : ring) {
                forest.update(q);
            }
        }
        // Gate each field layer from the inside rather than filtering the scene list.
        // The render loop blocks in awaitResume while paused, so a flag consulted at
        // scene-build time would not respond until the next step; a layer that checks
        // the flag when it draws responds to a bare repaint.
        AtomicBoolean topo = new AtomicBoolean(true);
        List<Layer> raw = densityField(forest, range, randomSeed);
        List<Layer> field = new ArrayList<>();
        for (Layer inner : raw) {
            field.add(Layers.when(topo::get, inner));
        }
        // The recording gets its own list rather than a temporarily flipped flag.
        // plot.render calls repaint, which is asynchronous, so the EDT can paint at
        // any moment -- including inside a window where the flag had been flipped for
        // the GIF. That is what made the isolines reappear a frame after pressing t.
        List<Layer> gifField = GIF_TOPO ? raw : List.of();
        plot.bindKey(KeyEvent.VK_T, "halfimpact.isolines", () -> topo.set(!topo.get()));
        AtomicBoolean showData = new AtomicBoolean(true);
        plot.bindKey(KeyEvent.VK_D, "halfimpact.data", () -> showData.set(!showData.get()));
        // The boxes in KINDS order, then off (-1). Everything that depends on the
        // choice reads it when drawing, so it takes effect on a paused frame.
        AtomicInteger boxKind = new AtomicInteger(0);
        plot.bindKey(KeyEvent.VK_C, "halfimpact.boxkind",
                () -> boxKind.set(boxKind.get() == KINDS.length - 1 ? -1 : boxKind.get() + 1));

        // Additions: the ensemble toggle, gated at draw time like the isolines, and the
        // per-tree factory behind it. The ensemble reuses the forest's own anisotropic
        // visitor, so every faint box is exactly one tree's share of the averaged box
        // drawn below.
        AtomicBoolean swingOn = new AtomicBoolean(true);
        plot.bindKey(KeyEvent.VK_E, "halfimpact.swing", () -> swingOn.set(!swingOn.get()));
        System.out.println("keys: space pauses, right arrow steps, d toggles the ring, t the isolines, "
                + "c cycles the box (GAP/CUT/PASSAGE/STOP/off), e the tree ensemble");
        KeysLayer keys = new KeysLayer(KeysLayer.Corner.TOP_LEFT).add("space", "pause", null)
                .add("\u2192", "step", null).add("d", "ring", KeysLayer.onOff(showData::get))
                .add("t", "isolines through the cut", KeysLayer.onOff(topo::get))
                .add("c", "box along the ray", () -> boxKind.get() < 0 ? "off" : KINDS[boxKind.get()])
                .add("e", "tree ensemble", KeysLayer.onOff(swingOn::get));
        IVisitorFactory<InterpolationMeasure> perTreeBox = AnisotropicLocalVisitor.reusableFactory(1.0, true);

        List<Layer> tracks = new ArrayList<>();
        Instant start = Instant.now();
        double[][] meanAspect = new double[ANGLES.length][IMPACTS.length];

        for (int a = 0; a < ANGLES.length; a++) {
            double theta = Math.toRadians(ANGLES[a]);
            double ux = Math.cos(theta), uy = Math.sin(theta); // along the ray
            double nx = -uy, ny = ux; // across it

            for (int pass = 0; pass < IMPACTS.length; pass++) {
                double b = IMPACTS[pass] * RING_RADIUS;
                Color color = TRACK[pass];
                // One locus at a time: the boxes from the previous ray would otherwise
                // sit on top of this one and the comparison becomes unreadable.
                tracks.clear();
                double aspectSum = 0;
                int aspectCount = 0;
                // Additions: one row of the ensemble per printed step.
                List<String> swingRows = new ArrayList<>();

                System.out.printf("%npass: heading %.0f deg, impact b = %.2f%n", ANGLES[a], b);
                System.out.printf("%8s %9s %9s %9s %9s %9s %9s %9s%n", "t", "along", "across", "ray asp", "gap asp",
                        "cut asp", "nn", "along/2nn");

                for (int step = 0; step <= STEPS; step++) {
                    double t = START_X + (END_X - START_X) * step / STEPS;
                    double px = t * ux + b * nx, py = t * uy + b * ny;
                    float[] probe = new float[] { (float) px, (float) py };

                    // the ring is static, so it is re-fed every step to hold it against decay
                    for (float[] p : ring) {
                        forest.update(p);
                    }

                    AnisotropicLocalGeometry out = forest.getAnisotropicGeometry(probe);
                    // Every box c can draw, off the one walk; the tables read the gap box.
                    double[][] boxes = out.isReliable()
                            ? new double[][] { out.gapBox(), out.cutBox(), out.passageBox(), out.stopBox() }
                            : null;
                    double[] box = (boxes == null) ? null : boxes[0];

                    // Additions: the per-tree boxes behind the averaged ones.
                    Swing swing = Swing.of(forest.traverseForest(probe, perTreeBox, Collectors.toList()));
                    if (step % ROW_EVERY == 0) {
                        double[] gap = swing.moments(0);
                        swingRows.add(String.format("%8.2f %9.4f %9.4f %9d", t, Swing.lead(gap), Swing.spread(gap),
                                swing.boxes.size()));
                    }

                    // The impact frontier. Every tree returns the leaf its own random cuts
                    // routed the probe to, merged across trees by point with Neighbor.count
                    // holding the number of trees that reached it. Drawn as a cloud, this is
                    // which part of the wall the forest considers local to the probe,
                    //
                    // Unbounded overload on purpose. At t = -2.2 the probe stands more than
                    // a radius clear of the ring, and a distance cap would return nothing
                    // exactly where the frontier is most worth seeing.
                    List<Neighbor> leaves = forest.getNearNeighborsInSample(probe);
                    double[][] leafXy = new double[leaves.size()][2];
                    double[] leafWeight = new double[leaves.size()];
                    double leafMax = 1;
                    for (Neighbor nb : leaves) {
                        leafMax = Math.max(leafMax, nb.count);
                    }
                    for (int i = 0; i < leaves.size(); i++) {
                        Neighbor nb = leaves.get(i);
                        leafXy[i][0] = nb.point[0];
                        leafXy[i][1] = nb.point[1];
                        // Normalised, so the encoding does not shift with numberOfTrees.
                        leafWeight[i] = nb.count / leafMax;
                    }

                    // Exact nearest-neighbour distance, taken from the ring itself rather
                    // than from the forest's sample. It is the ground truth the forest's
                    // own near-neighbour query approximates, and using it here avoids the
                    // stale-point artefact that appears when a decayed sample supplies a
                    // neighbour that is no longer in the drawn frame.
                    double nn = Double.MAX_VALUE;
                    for (float[] q : ring) {
                        nn = Math.min(nn, Math.hypot(q[0] - px, q[1] - py));
                    }

                    double along = 0, across = 0;
                    if (box != null) {
                        // Reach along the ray and across it, as ray-box intersections, so
                        // the two headings are measured in their own frames rather than in
                        // x and y. For the axis-aligned pass these reduce to the x and y
                        // extents; for the diagonal pass they do not.
                        along = reach(box, ux, uy) + reach(box, -ux, -uy);
                        across = reach(box, nx, ny) + reach(box, -nx, -ny);
                        tracks.add(Layers.when(() -> boxKind.get() >= 0,
                                boxOutline(px, py, new double[] { nn, nn, nn, nn }, NN_BOX)));
                        for (int k = 0; k < boxes.length; k++) {
                            int drawnKind = k;
                            tracks.add(
                                    Layers.when(() -> boxKind.get() == drawnKind, boxOutline(px, py, boxes[k], color)));
                        }
                        aspectSum += out.getAnisotropy();
                        aspectCount++;

                        if (step % ROW_EVERY == 0) {
                            // Two aspects, deliberately. "gap asp" is the drawn box's
                            // own axis ratio; "cut asp" is getAnisotropy, which reads
                            // the cut box and is axis-aligned. "ray asp" is measured
                            // in the ray's frame, so it is the one that survives the
                            // 45 degree heading.
                            double gw = box[0] + box[2], gh = box[1] + box[3];
                            double gapAsp = (Math.min(gw, gh) > 0) ? Math.max(gw, gh) / Math.min(gw, gh) : 0;
                            double rayAsp = (across > 0) ? along / across : 0;
                            double base = 2 * nn;
                            System.out.printf("%8.2f %9.4f %9.4f %9.2f %9.2f %9.2f %9.4f %9.2f%n", t, along, across,
                                    rayAsp, gapAsp, out.getAnisotropy(), nn, (base > 0) ? along / base : 0);
                        }
                    }

                    List<Layer> body = new ArrayList<>();
                    body.add(Layers.when(showData::get, Layers.dots(ring, new Color(140, 140, 140), 1.6)));
                    body.add(Layers.dashedPolyline(
                            new double[][] { { START_X * ux + b * nx, START_X * uy + b * ny },
                                    { END_X * ux + b * nx, END_X * uy + b * ny } },
                            color, 1.0f, new float[] { 4f, 5f }));
                    // Under the box outlines, so the outlines stay legible where the cloud
                    // is densest, which is exactly where they overlap.
                    body.add(Layers.weightedDots(leafXy, leafWeight, LEAF_CLOUD, LEAF_MIN_R, LEAF_SCALE_R));
                    body.addAll(tracks);
                    body.add(Layers.dots(new float[][] { probe }, Color.BLACK, 4));
                    // The readout and the legend name the box c has selected, so both are
                    // chosen when drawing; one readout per box, and the last for none.
                    String[] readouts = new String[KINDS.length + 1];
                    for (int k = 0; k <= KINDS.length; k++) {
                        readouts[k] = String.format("heading %.0f deg   b = %.2f   t = %+.2f   ", ANGLES[a], b, t)
                                + readout(boxes, (k < KINDS.length) ? k : -1, nn, ux, uy, nx, ny);
                    }
                    body.add((g, vp) -> Layers
                            .label(-range * 0.94, -range * 0.92,
                                    readouts[(boxKind.get() < 0) ? KINDS.length : boxKind.get()], new Color(60, 60, 60))
                            .draw(g, vp));
                    body.add((g, vp) -> legend(boxKind.get(), swingOn.get(), color).draw(g, vp));

                    List<Layer> scene = new ArrayList<>(field);
                    scene.addAll(body);

                    // Additions, drawn over the existing scene.
                    Layer ensemble = swing.layer(px, py, color, boxKind::get);
                    scene.add(Layers.when(swingOn::get, ensemble));
                    scene.add(keys);

                    if (livePlot) {
                        plot.render(scene);
                    }
                    if (saveGif) {
                        List<Layer> gifScene = new ArrayList<>(gifField);
                        gifScene.addAll(body);
                        if (GIF_SWING) {
                            gifScene.add(ensemble);
                        }
                        gif.writeFrame(plot.renderImage(gifSizePx, gifSizePx, gifScene));
                    }
                    if (livePlot) {
                        plot.awaitResume(); // space pauses, right arrow steps
                        if (frameDelayMs > 0) {
                            Thread.sleep(frameDelayMs);
                        }
                    }
                }
                meanAspect[a][pass] = (aspectCount > 0) ? aspectSum / aspectCount : 0;

                // Additions: the ensemble along this ray, on the gap box. lead is the mean
                // offset of the box centres from the probe, swing their spread across trees.
                System.out.printf("%nensemble, heading %.0f deg, b = %.2f%n", ANGLES[a], b);
                System.out.printf("%8s %9s %9s %9s%n", "t", "lead", "swing", "trees");
                for (String row : swingRows) {
                    System.out.println(row);
                }
            }
        }

        System.out.printf("%n%nmean axis aspect over each pass%n%10s", "b");
        for (double ang : ANGLES) {
            System.out.printf("%12s", String.format("%.0f deg", ang));
        }
        System.out.println("      ratio");
        for (int pass = 0; pass < IMPACTS.length; pass++) {
            System.out.printf("%10.2f", IMPACTS[pass] * RING_RADIUS);
            for (int a = 0; a < ANGLES.length; a++) {
                System.out.printf("%12.2f", meanAspect[a][pass]);
            }
            System.out.printf("%11.2f%n", (meanAspect[1][pass] > 0) ? meanAspect[0][pass] / meanAspect[1][pass] : 0);
        }

        if (gif != null) {
            gif.close();
            System.out.println("\nwrote half_impact.gif");
        }

    }

    /**
     * The isolines drawn through the cut, shaded between the levels, plus the
     * directional density as arrows.
     *
     * <p>
     * The isolines' reference is a fresh draw from the same annulus rather than the
     * ring itself: the ring has been fed to the forest over and over, and a sample
     * the forest has not been told about is the honest one to ask. The ring is
     * static, so this is done once.
     */
    private static List<Layer> densityField(RandomCutForest forest, double range, long seed) {
        System.out.println("building the static isolines...");
        double lo = -range * 0.95, span = 2 * range * 0.95;
        CutIsolines lines = new CutIsolines(lo, span, FIELD_GRID, ISO).bands(BAND, BAND_FLOOR);
        for (float[] p : annulus(RING_THICKNESS, seed + 1)) {
            lines.add(p, forest.getAnisotropicGeometry(p).cutSpread());
        }
        List<Layer> out = new ArrayList<>();
        out.add(lines);

        List<double[]> origins = new ArrayList<>(), deltas = new ArrayList<>();
        for (int i = 0; i < ARROW_GRID; i++) {
            for (int j = 0; j < ARROW_GRID; j++) {
                double x = lo + span * (i + 0.5) / ARROW_GRID, y = lo + span * (j + 0.5) / ARROW_GRID;
                // Projected from the 2d faces into the plane by getDensityGradient,
                // whose magnitude is the asymmetry of the neighbourhood rather than a
                // constant. Arrows are therefore SHORT where opposing faces balance,
                // which is honest: there is no gradient there to point along.
                double[] g = forest.getAnisotropicGeometry(new float[] { (float) x, (float) y }).getDensityGradient();
                double n = Math.hypot(g[0], g[1]);
                if (n > 0) {
                    origins.add(new double[] { x, y });
                    deltas.add(new double[] { g[0] * ARROW_LENGTH, g[1] * ARROW_LENGTH });
                }
            }
        }
        out.add(Layers.arrows(origins.toArray(new double[0][]), deltas.toArray(new double[0][]), GRAD, 1.0f));
        return out;
    }

    /**
     * Points spread through an annulus of the given radial thickness, at a fixed
     * count. Sampling is uniform in area (r drawn as sqrt of a uniform on the
     * squared radii) so the wall has no radial density gradient of its own.
     */
    private static float[][] annulus(double thickness, long seed) {
        java.util.Random random = new java.util.Random(seed);
        double inner = RING_RADIUS - thickness / 2, outer = RING_RADIUS + thickness / 2;
        float[][] out = new float[RING_POINTS][2];
        for (int i = 0; i < RING_POINTS; i++) {
            double t = 2 * Math.PI * random.nextDouble();
            double r = Math.sqrt(inner * inner + random.nextDouble() * (outer * outer - inner * inner));
            out[i] = new float[] { (float) (r * Math.cos(t)), (float) (r * Math.sin(t)) };
        }
        return out;
    }

    private static void penetrationStudy(long seed) {
        double probeR = 1.45;
        System.out.printf("penetration study, probe on the axis at r = %.2f, %d points throughout%n", probeR,
                RING_POINTS);
        System.out.printf("%10s %10s %10s %10s %10s %10s%n", "thickness", "inner", "to inner", "reach in", "ratio",
                "aspect");
        for (double w : THICKNESS_SWEEP) {
            RandomCutForest f = RandomCutForest.builder().numberOfTrees(100).sampleSize(256).dimensions(2)
                    .randomSeed(seed).timeDecay(1.0 / (0.8 * RING_POINTS)).build();
            float[][] pts = annulus(w, seed);
            for (int rep = 0; rep < 3; rep++) {
                for (float[] q : pts) {
                    f.update(q);
                }
            }
            AnisotropicLocalGeometry out = f.getAnisotropicGeometry(new float[] { (float) probeR, 0f });
            double inner = RING_RADIUS - w / 2;
            double toInner = probeR - inner;
            // the probe sits at +x, so the inward reach is the high_x face
            double reachIn = out.passageAxis()[0];
            System.out.printf("%10.3f %10.3f %10.3f %10.3f %10.2f %10.2f%n", w, inner, toInner, reachIn,
                    (toInner > 0) ? reachIn / toInner : 0, out.isReliable() ? out.getAnisotropy() : 0);
        }
        System.out.println("ratio > 1 means the box spans the wall and takes in the hollow centre;");
        System.out.println("ratio < 1 means the wall itself terminated it.\n");
    }

    /** Distance from the box centre to its wall along a unit direction. */
    private static double reach(double[] box, double ux, double uy) {
        double tx = (ux == 0) ? Double.MAX_VALUE : ((ux > 0) ? box[2] : box[0]) / Math.abs(ux);
        double ty = (uy == 0) ? Double.MAX_VALUE : ((uy > 0) ? box[3] : box[1]) / Math.abs(uy);
        return Math.min(tx, ty);
    }

    /**
     * The readout along the bottom for the box c has selected, measured in the
     * ray's own frame: reach along the ray and across it, as ray-box intersections,
     * so the diagonal pass is not read in x and y. kind -1 is off.
     */
    private static String readout(double[][] boxes, int kind, double nn, double ux, double uy, double nx, double ny) {
        if (boxes == null) {
            return "no single scale";
        }
        if (kind < 0) {
            return "";
        }
        double[] box = boxes[kind];
        double along = reach(box, ux, uy) + reach(box, -ux, -uy);
        double across = reach(box, nx, ny) + reach(box, -nx, -ny);
        String name = KINDS[kind].toLowerCase();
        if (along + across == 0) {
            return String.format("inside the wall, no %s   nn-square %.3f", name, 2 * nn);
        }
        return String.format("%s %.3f x %.3f  ray aspect %.2f   nn-square %.3f", name, along, across,
                (across > 0) ? along / across : 0, 2 * nn);
    }

    /** The key; its box rows name the box c has selected and go when c is off. */
    private static Layer legend(int kind, boolean ensemble, Color track) {
        List<String> labels = new ArrayList<>(List.of("ring (data)", "isolines through the cut", "directional density",
                "leaves reached (area = tree votes)"));
        List<Color> colors = new ArrayList<>(List.of(new Color(140, 140, 140), ISO, GRAD, LEAF_CLOUD));
        List<Layers.Swatch> swatches = new ArrayList<>(
                List.of(Layers.Swatch.DOTS, Layers.Swatch.LINE, Layers.Swatch.LINE, Layers.Swatch.DOTS));
        if (kind >= 0) {
            labels.add((kind == 0) ? "gap box (measured void)" : KINDS[kind].toLowerCase() + " box");
            colors.add(track);
            swatches.add(Layers.Swatch.BOX);
            labels.add("nn-square (isotropic baseline)");
            colors.add(NN_BOX);
            swatches.add(Layers.Swatch.BOX);
            if (ensemble) {
                labels.add("each tree's box, lead and swing");
                colors.add(LEAD);
                swatches.add(Layers.Swatch.LINE);
            }
        }
        return Layers.legend(labels.toArray(new String[0]), colors.toArray(new Color[0]),
                swatches.toArray(new Layers.Swatch[0]));
    }

    /** One box as a closed outline, drawn at true scale in data units. */
    private static Layer boxOutline(double x, double y, double[] box, Color color) {
        // box is {high_x, high_y, low_x, low_y}; high extends toward minus
        // high extends toward minus, low toward plus
        double x0 = x - box[0], x1 = x + box[2], y0 = y - box[1], y1 = y + box[3];
        return Layers.polyline(new double[][] { { x0, y0 }, { x1, y0 }, { x1, y1 }, { x0, y1 }, { x0, y0 } },
                new Color(color.getRed(), color.getGreen(), color.getBlue(), 90), true, 0, 1.0f);
    }

    // ------------------------------------------------------------------
    // Additions: the ensemble.
    // ------------------------------------------------------------------

    /**
     * The ensemble behind the averaged box: every tree's boxes, centred on the
     * probe exactly as boxOutline draws them, the mean offset of a box's centres
     * from the probe (the lead), and the covariance of those centres (the swing).
     *
     * <p>
     * Far from the data every tree separates the probe high up, against faces that
     * nested boxes share, so the trees agree and the lead is long. Close in,
     * nesting shortens the lead. Where the trees disagree about the level at which
     * the probe is cut off, the centres scatter: that scatter, not the averaged
     * box, is the uncertainty under the ensemble.
     */
    static final class Swing {
        /** Per tree, its boxes in KINDS order. */
        final List<double[][]> boxes = new ArrayList<>();

        static Swing of(List<InterpolationMeasure> perTree) {
            Swing s = new Swing();
            for (InterpolationMeasure m : perTree) {
                AnisotropicLocalGeometry one = new AnisotropicLocalGeometry(m);
                if (one.getScales().getTreeCount() > 0) {
                    s.boxes.add(new double[][] { one.gapBox(), one.cutBox(), one.passageBox(), one.stopBox() });
                }
            }
            return s;
        }

        /** {lead x, lead y, cxx, cxy, cyy} of the centres of one kind of box. */
        double[] moments(int kind) {
            double[] m = new double[5];
            int n = boxes.size();
            if (n == 0) {
                return m;
            }
            for (double[][] tree : boxes) {
                double[] box = tree[kind]; // {high_x, high_y, low_x, low_y}
                m[0] += 0.5 * (box[2] - box[0]) / n;
                m[1] += 0.5 * (box[3] - box[1]) / n;
            }
            for (double[][] tree : boxes) {
                double[] box = tree[kind];
                double dx = 0.5 * (box[2] - box[0]) - m[0];
                double dy = 0.5 * (box[3] - box[1]) - m[1];
                m[2] += dx * dx / n;
                m[3] += dx * dy / n;
                m[4] += dy * dy / n;
            }
            return m;
        }

        static double lead(double[] m) {
            return Math.hypot(m[0], m[1]);
        }

        static double spread(double[] m) {
            return Math.sqrt(Math.max(0.0, m[2] + m[4]));
        }

        /**
         * The cloud of per-tree boxes of the kind c has selected, the lead arrow, and
         * the swing ellipse; nothing while c is off.
         */
        Layer layer(double px, double py, Color color, IntSupplier kind) {
            List<double[][]> frozen = new ArrayList<>(boxes);
            double[][] stats = new double[KINDS.length][];
            for (int k = 0; k < KINDS.length; k++) {
                stats[k] = moments(k);
            }
            Color faint = new Color(color.getRed(), color.getGreen(), color.getBlue(), 34);
            return (g, vp) -> {
                int k = kind.getAsInt();
                if (k < 0) {
                    return;
                }
                Stroke old = g.getStroke();
                g.setStroke(new BasicStroke(0.8f));
                g.setColor(faint);
                for (double[][] tree : frozen) {
                    double[] box = tree[k];
                    double x0 = vp.px(px - box[0]);
                    double x1 = vp.px(px + box[2]);
                    double y0 = vp.py(py + box[3]);
                    double y1 = vp.py(py - box[1]);
                    g.draw(new java.awt.geom.Rectangle2D.Double(x0, y0, Math.max(1.0, x1 - x0),
                            Math.max(1.0, y1 - y0)));
                }
                g.setStroke(old);
                double[] m = stats[k];
                if (lead(m) > 0) {
                    Layers.arrows(new double[][] { { px, py } }, new double[][] { { m[0], m[1] } }, LEAD, 1.8f).draw(g,
                            vp);
                }
                if (spread(m) > 0) {
                    Layers.polyline(ellipse(px + m[0], py + m[1], m), LEAD, true, 0, 1.2f).draw(g, vp);
                }
            };
        }

        /** The one-standard-deviation ellipse of the box centres, around (cx, cy). */
        private static double[][] ellipse(double cx, double cy, double[] m) {
            double tr = m[2] + m[4];
            double det = m[2] * m[4] - m[3] * m[3];
            double disc = Math.sqrt(Math.max(0.0, 0.25 * tr * tr - det));
            double a1 = Math.sqrt(0.5 * tr + disc);
            double a2 = Math.sqrt(Math.max(0.0, 0.5 * tr - disc));
            double angle = 0.5 * Math.atan2(2 * m[3], m[2] - m[4]);
            double ca = Math.cos(angle);
            double sa = Math.sin(angle);
            double[][] pts = new double[49][2];
            for (int k = 0; k <= 48; k++) {
                double th = 2 * Math.PI * k / 48;
                double u = a1 * Math.cos(th);
                double v = a2 * Math.sin(th);
                pts[k][0] = cx + ca * u - sa * v;
                pts[k][1] = cy + sa * u + ca * v;
            }
            return pts;
        }
    }
}
