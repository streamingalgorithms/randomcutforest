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

import org.streamingalgorithms.randomcutforest.RandomCutForest;
import org.streamingalgorithms.randomcutforest.examples.plot.Contour;
import org.streamingalgorithms.randomcutforest.examples.plot.GifWriter;
import org.streamingalgorithms.randomcutforest.examples.plot.Layer;
import org.streamingalgorithms.randomcutforest.examples.plot.Layers;
import org.streamingalgorithms.randomcutforest.examples.plot.Plot2D;
import org.streamingalgorithms.randomcutforest.returntypes.AnisotropicLocalGeometry;
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

    /** Scalar density field: grid resolution, contour count, arrow lattice. */
    private static final int FIELD_GRID = 130;
    private static final int CONTOURS = 6;
    /**
     * Quantile band the contour levels are drawn from. The lowest levels are the
     * risky ones: most of the grid is far field, where the density is small and
     * nearly flat, so a percent or two of Monte-Carlo variation across the trees
     * moves a contour a long way and can close it into a loop. Raising CONTOUR_LO
     * lifts the lowest contour out of that noise floor; more trees lowers the noise
     * itself.
     */
    private static final double CONTOUR_LO = 0.35;
    private static final double CONTOUR_HI = 0.97;
    /**
     * Levels for the isolines. Null derives them from this grid's own quantile band
     * and prints them; paste those values back in here to compare two grids. A
     * quantile is taken over whatever node set was sampled, so derived levels move
     * with the resolution even when the field does not, and two grids deriving
     * their own levels cannot be compared at all.
     */
    private static final double[] FIXED_LEVELS = null;
    private static final int ARROW_GRID = 15;
    private static final double ARROW_LENGTH = 0.13;
    private static final Color ISO = new Color(214, 148, 34);
    private static final Color GRAD = new Color(120, 100, 70);

    /**
     * Shading under the isolines. Cool and neutral so it does not compete with the
     * warm lines drawn over it, and floored: the levels come from a quantile band
     * starting at CONTOUR_LO, so band 1 covers 1 - CONTOUR_LO of the canvas by
     * construction -- here mostly the flat far field. Painting from BAND_FLOOR up
     * puts the ink on the wall, which is what the eye is looking for. The field is
     * static, so this costs nothing beyond the grid already sampled.
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

        // The ring never moves, so the density field is fixed for the whole run and is
        // built once. Only the probe moves. Recomputing it per frame would be 17000
        // queries a frame; this is 17000 once. The forest is primed first, since the
        // step loop is what normally feeds it.
        for (int warm = 0; warm < 3; warm++) {
            for (float[] q : ring) {
                forest.update(q);
            }
        }
        // Gate each field layer from the inside rather than filtering the scene list.
        // The render loop blocks in awaitResume while paused, so a flag consulted at
        // scene-build time would not respond until the next step; a layer that checks
        // the flag when it draws responds to a bare repaint.
        java.util.concurrent.atomic.AtomicBoolean topo = new java.util.concurrent.atomic.AtomicBoolean(true);
        List<Layer> raw = densityField(forest, range);
        List<Layer> field = new ArrayList<>();
        for (Layer inner : raw) {
            field.add((g, vp) -> {
                if (topo.get()) {
                    inner.draw(g, vp);
                }
            });
        }
        // The recording gets its own list rather than a temporarily flipped flag.
        // plot.render calls repaint, which is asynchronous, so the EDT can paint at
        // any moment -- including inside a window where the flag had been flipped for
        // the GIF. That is what made the isolines reappear a frame after pressing t.
        List<Layer> gifField = GIF_TOPO ? raw : List.of();
        plot.bindKey(KeyEvent.VK_T, "halfimpact.topo", () -> topo.set(!topo.get()));
        System.out.println("keys: space pauses, right arrow steps, t toggles the topo map");

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
                    double[] box = out.isReliable() ? out.gapBox() : null;

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
                        tracks.add(boxOutline(px, py, new double[] { nn, nn, nn, nn }, NN_BOX));
                        tracks.add(boxOutline(px, py, box, color));
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
                    body.add(Layers.dots(ring, new Color(140, 140, 140), 1.6));
                    body.add(Layers.dashedPolyline(
                            new double[][] { { START_X * ux + b * nx, START_X * uy + b * ny },
                                    { END_X * ux + b * nx, END_X * uy + b * ny } },
                            color, 1.0f, new float[] { 4f, 5f }));
                    // Under the box outlines, so the outlines stay legible where the cloud
                    // is densest, which is exactly where they overlap.
                    body.add(Layers.weightedDots(leafXy, leafWeight, LEAF_CLOUD, LEAF_MIN_R, LEAF_SCALE_R));
                    body.addAll(tracks);
                    body.add(Layers.dots(new float[][] { probe }, Color.BLACK, 4));
                    body.add(Layers.label(-range * 0.94, -range * 0.92,
                            String.format("heading %.0f deg   b = %.2f   t = %+.2f   %s", ANGLES[a], b, t,
                                    (box == null) ? "no single scale"
                                            : (along + across == 0)
                                                    ? String.format("inside the wall, no gap   nn-square %.3f", 2 * nn)
                                                    : String.format("gap %.3f x %.3f  ray aspect %.2f   nn-square %.3f",
                                                            along, across, (across > 0) ? along / across : 0, 2 * nn)),
                            new Color(60, 60, 60)));
                    body.add(Layers.legend(
                            new String[] { "ring (data)", "density isolines", "directional density",
                                    "leaves reached (area = tree votes)", "gap box (measured void)",
                                    "nn-square (isotropic baseline)" },
                            new Color[] { new Color(140, 140, 140), ISO, GRAD, LEAF_CLOUD, color, NN_BOX },
                            new Layers.Swatch[] { Layers.Swatch.DOTS, Layers.Swatch.LINE, Layers.Swatch.LINE,
                                    Layers.Swatch.DOTS, Layers.Swatch.BOX, Layers.Swatch.BOX }));

                    List<Layer> scene = new ArrayList<>(field);
                    scene.addAll(body);

                    if (livePlot) {
                        plot.render(scene);
                    }
                    if (saveGif) {
                        List<Layer> gifScene = new ArrayList<>(gifField);
                        gifScene.addAll(body);
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
     * The scalar density field as isolines, plus the directional density as arrows.
     *
     */
    private static List<Layer> densityField(RandomCutForest forest, double range) {
        System.out.println("building the static density field...");
        double lo = -range * 0.95, span = 2 * range * 0.95;

        // passageDensity, not the inherited getDensity: this example is named for
        // the half crossing and everything drawn on it should come from that one
        // scale rule. Mixing contours from the all-levels mean with boxes from
        // the crossing would put two different estimators in one picture.
        Contour.Field density = (x, y) -> forest.getAnisotropicGeometry(new float[] { (float) x, (float) y })
                .passageDensity(0.001);

        double[][] rho = Contour.sample(density, lo, span, FIELD_GRID);
        // Levels at quantiles of the sampled field, not evenly spaced. The density
        // spans orders of magnitude between the hollow interior and the wall, so even
        // spacing would stack every contour on the wall.
        double[] levels = (FIXED_LEVELS != null) ? FIXED_LEVELS
                : Contour.quantileLevels(rho, CONTOURS, CONTOUR_LO, CONTOUR_HI);
        if (FIXED_LEVELS == null) {
            System.out.println("levels @ grid " + FIELD_GRID + ": " + java.util.Arrays.toString(levels));
        }

        // Bands from the grid that already chose the levels, so they cost nothing,
        // and they go in first so the isolines and arrows draw over them.
        List<Layer> out = new ArrayList<>();
        out.add(bandLayer(rho, lo, span, levels, BAND));
        out.addAll(Contour.isolines(density, rho, lo, span, levels, ISO));

        // Same cells and levels the isolines walked. The saddle is the cell where four
        // crossings admit two pairings with DIFFERENT topology: one separates the high
        // corners, the other joins them. The textbook asymptotic decider takes the
        // bilinear value at the centre, the mean of the corners, which assumes the
        // field is bilinear inside the cell. It is not: this field comes from
        // axis-aligned boxes and L1 geometry and can be queried anywhere, so Contour
        // measures the centre instead. This counts how often that mattered.
        int[] audit = Contour.saddleAudit(density, rho, lo, span, levels);

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
        System.out.printf(
                "ambiguous saddle cells %d, of which the measured centre disagrees with the "
                        + "bilinear guess in %d (%.1f%%)%n",
                audit[0], audit[1], (audit[0] > 0) ? 100.0 * audit[1] / audit[0] : 0.0);
        System.out.println("a high disagreement rate means the grid does not resolve the field, and any");
        System.out.println("contour topology read off it -- including closed loops -- is unreliable.\n");
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

    /**
     * Fills each grid cell with the band its corner mean falls in. Reuses the grid
     * already sampled to choose the levels, so it costs no density queries.
     */
    private static Layer bandLayer(double[][] v, double lo, double span, double[] levels, Color color) {
        double[] sorted = java.util.Arrays.copyOf(levels, levels.length);
        java.util.Arrays.sort(sorted);
        int grid = v.length;
        double h = span / (grid - 1.0);
        return (g, vp) -> {
            for (int i = 0; i + 1 < grid; i++) {
                for (int j = 0; j + 1 < grid; j++) {
                    double m = 0.25 * (v[i][j] + v[i + 1][j] + v[i][j + 1] + v[i + 1][j + 1]);
                    int band = 0;
                    // A NaN node falls through as band 0 and is skipped, which is
                    // what an off-support cell should do.
                    while (band < sorted.length && m > sorted[band]) {
                        band++;
                    }
                    if (band < BAND_FLOOR) {
                        continue;
                    }
                    int x0 = (int) Math.round(vp.px(lo + h * i));
                    int x1 = (int) Math.round(vp.px(lo + h * (i + 1)));
                    int y0 = (int) Math.round(vp.py(lo + h * (j + 1)));
                    int y1 = (int) Math.round(vp.py(lo + h * j));
                    int span2 = Math.max(1, sorted.length - BAND_FLOOR);
                    g.setColor(new Color(color.getRed(), color.getGreen(), color.getBlue(),
                            9 + 25 * (band - BAND_FLOOR) / span2));
                    // +1 closes the hairline seam that rounding leaves between cells.
                    g.fillRect(x0, y0, x1 - x0 + 1, y1 - y0 + 1);
                }
            }
        };
    }

    /** Distance from the box centre to its wall along a unit direction. */
    private static double reach(double[] box, double ux, double uy) {
        double tx = (ux == 0) ? Double.MAX_VALUE : ((ux > 0) ? box[2] : box[0]) / Math.abs(ux);
        double ty = (uy == 0) ? Double.MAX_VALUE : ((uy > 0) ? box[3] : box[1]) / Math.abs(uy);
        return Math.min(tx, ty);
    }

    /** One box as a closed outline, drawn at true scale in data units. */
    private static Layer boxOutline(double x, double y, double[] box, Color color) {
        // box is {high_x, high_y, low_x, low_y}; high extends toward minus
        // high extends toward minus, low toward plus
        double x0 = x - box[0], x1 = x + box[2], y0 = y - box[1], y1 = y + box[3];
        return Layers.polyline(new double[][] { { x0, y0 }, { x1, y0 }, { x1, y1 }, { x0, y1 }, { x0, y0 } },
                new Color(color.getRed(), color.getGreen(), color.getBlue(), 90), true, 0, 1.0f);
    }
}
