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
import org.streamingalgorithms.randomcutforest.examples.plot.GifWriter;
import org.streamingalgorithms.randomcutforest.examples.plot.Layer;
import org.streamingalgorithms.randomcutforest.examples.plot.Layers;
import org.streamingalgorithms.randomcutforest.examples.plot.Plot2D;
import org.streamingalgorithms.randomcutforest.returntypes.AnisotropicDensityOutput;

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
    private static final int ARROW_GRID = 15;
    private static final double ARROW_LENGTH = 0.13;
    private static final Color ISO = new Color(214, 148, 34);
    private static final Color GRAD = new Color(120, 100, 70);

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
        return "extent box along a ray through a ring, by impact parameter";
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

        java.util.Random random = new java.util.Random(randomSeed);
        float[][] ring = annulus(RING_THICKNESS, randomSeed);

        Plot2D plot = livePlot ? Plot2D.open("Half Impact - extent box along a ray", range, 860)
                : Plot2D.offscreen(range);
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
                System.out.printf("%8s %9s %9s %9s %9s %9s %9s %9s%n", "t", "along", "across", "axis asp", "nn",
                        "area/nn^2", "inflate", "top3");

                for (int step = 0; step <= STEPS; step++) {
                    double t = START_X + (END_X - START_X) * step / STEPS;
                    double px = t * ux + b * nx, py = t * uy + b * ny;
                    float[] probe = new float[] { (float) px, (float) py };

                    // the ring is static, so it is re-fed every step to hold it against decay
                    for (float[] p : ring) {
                        forest.update(p);
                    }

                    AnisotropicDensityOutput out = forest.getAnisotropicDensity(probe);
                    double[] box = out.isReliable() ? out.cutBox() : null;

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
                        // Both regions on the same ruler. The nn-square is centred and
                        // square by construction -- one distance, no direction, no offset,
                        // which is everything a scalar local scale carries. The extent box
                        // is neither centred nor square. Where the neighbourhood is
                        // genuinely round, at the centre of the ring, the two should
                        // coincide; the ring is equidistant in every direction there and
                        // both should read about R.
                        tracks.add(boxOutline(px, py, new double[] { nn, nn, nn, nn }, NN_BOX));
                        tracks.add(boxOutline(px, py, box, color));
                        aspectSum += out.getAnisotropy();
                        aspectCount++;
                    }

                    List<Layer> body = new ArrayList<>();
                    body.add(Layers.dots(ring, new Color(140, 140, 140), 1.6));
                    body.add(Layers.dashedPolyline(
                            new double[][] { { START_X * ux + b * nx, START_X * uy + b * ny },
                                    { END_X * ux + b * nx, END_X * uy + b * ny } },
                            color, 1.0f, new float[] { 4f, 5f }));
                    body.addAll(tracks);
                    body.add(Layers.dots(new float[][] { probe }, Color.BLACK, 4));
                    body.add(Layers.label(
                            -range * 0.94, -range * 0.92, String
                                    .format("heading %.0f deg   b = %.2f   t = %+.2f   %s", ANGLES[a], b, t,
                                            (box == null) ? "no single scale"
                                                    : String.format("extent %.3f x %.3f  aspect %.2f   nn-square %.3f",
                                                            along, across, out.getAnisotropy(), 2 * nn)),
                            new Color(60, 60, 60)));
                    body.add(Layers.legend(
                            new String[] { "ring (data)", "density isolines", "directional density",
                                    "extent box (measured)", "nn-square (isotropic baseline)" },
                            new Color[] { new Color(140, 140, 140), ISO, GRAD, color, NN_BOX },
                            new Layers.Swatch[] { Layers.Swatch.DOTS, Layers.Swatch.LINE, Layers.Swatch.LINE,
                                    Layers.Swatch.BOX, Layers.Swatch.BOX }));

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
     * <p>
     * All three views come from the same walk that produced the extent box, and
     * they are complementary rather than redundant. The density is a scalar,
     * displaced mass over box volume; the extent box is the local metric that
     * supplied that volume; the directional density says where the mass sits
     * relative to the query. So the box explains the number the isolines are
     * drawing.
     *
     * <p>
     * There is a consistency check visible in the picture. The directional density
     * is a gradient direction and the isolines are level sets, so the arrows should
     * cross the contours at right angles. Anywhere they do not is a place where the
     * scalar and directional halves of the same measurement disagree.
     */
    private static List<Layer> densityField(RandomCutForest forest, double range) {
        System.out.println("building the static density field...");
        double lo = -range * 0.95, span = 2 * range * 0.95;
        double[][] rho = new double[FIELD_GRID][FIELD_GRID];
        List<Double> vals = new ArrayList<>();
        for (int i = 0; i < FIELD_GRID; i++) {
            for (int j = 0; j < FIELD_GRID; j++) {
                double x = lo + span * i / (FIELD_GRID - 1.0), y = lo + span * j / (FIELD_GRID - 1.0);
                // passageDensity, not the inherited getDensity: this example is named for
                // the half crossing and everything drawn on it should come from that one
                // scale rule. Mixing contours from the all-levels mean with boxes from
                // the crossing would put two different estimators in one picture.
                rho[i][j] = forest.getAnisotropicDensity(new float[] { (float) x, (float) y }).passageDensity(0.001);
                vals.add(rho[i][j]);
            }
        }
        java.util.Collections.sort(vals);

        List<Layer> out = new ArrayList<>();
        int saddles = 0, disagree = 0;
        // Levels at quantiles of the sampled field, not evenly spaced. The density
        // spans orders of magnitude between the hollow interior and the wall, so even
        // spacing would stack every contour on the wall.
        for (int c = 1; c <= CONTOURS; c++) {
            double level = vals
                    .get((int) ((CONTOUR_LO + (CONTOUR_HI - CONTOUR_LO) * c / (CONTOURS + 1.0)) * (vals.size() - 1)));
            List<double[]> segs = new ArrayList<>();
            double h = span / (FIELD_GRID - 1.0);
            for (int i = 0; i + 1 < FIELD_GRID; i++) {
                for (int j = 0; j + 1 < FIELD_GRID; j++) {
                    double x0 = lo + span * i / (FIELD_GRID - 1.0), y0 = lo + span * j / (FIELD_GRID - 1.0);
                    double a = rho[i][j], b = rho[i + 1][j], cc = rho[i + 1][j + 1], d = rho[i][j + 1];
                    List<double[]> hits = new ArrayList<>();
                    if ((a > level) != (b > level)) {
                        hits.add(new double[] { x0 + h * frac(a, b, level), y0 });
                    }
                    if ((b > level) != (cc > level)) {
                        hits.add(new double[] { x0 + h, y0 + h * frac(b, cc, level) });
                    }
                    if ((d > level) != (cc > level)) {
                        hits.add(new double[] { x0 + h * frac(d, cc, level), y0 + h });
                    }
                    if ((a > level) != (d > level)) {
                        hits.add(new double[] { x0, y0 + h * frac(a, d, level) });
                    }
                    if (hits.size() == 2) {
                        segs.add(new double[] { hits.get(0)[0], hits.get(0)[1], hits.get(1)[0], hits.get(1)[1] });
                    } else if (hits.size() == 4) {
                        // The saddle. Four crossings admit two pairings with DIFFERENT
                        // topology: one separates the high corners, the other joins them.
                        // Picking in collection order is arbitrary, and an arbitrary rule
                        // applied consistently to a symmetric field yields symmetric
                        // artefacts, so the resulting features cannot be told from real
                        // structure by their symmetry alone.
                        //
                        // The textbook fix is the asymptotic decider, which takes the
                        // BILINEAR value at the centre -- the mean of the corners -- and
                        // asks which side of the level it falls on. That assumes the field
                        // is bilinear inside the cell, which is the right assumption when
                        // grid samples are all you have. They are not: this field comes
                        // from axis-aligned boxes and L1 geometry, has no reason to be
                        // bilinear below cell scale, and can be queried anywhere. So the
                        // centre is MEASURED rather than interpolated. The count of cells
                        // where the two disagree is reported at the end, and is a direct
                        // statement about whether the grid resolves the field.
                        double cx = x0 + h / 2, cy = y0 + h / 2;
                        double centre = forest.getAnisotropicDensity(new float[] { (float) cx, (float) cy })
                                .getCutDensity(0.001, 2);
                        double bilinear = 0.25 * (a + b + cc + d);
                        saddles++;
                        if ((centre > level) != (bilinear > level)) {
                            disagree++;
                        }
                        boolean joinsA = (centre > level) == (a > level);
                        int[][] pairs = joinsA ? new int[][] { { 0, 1 }, { 2, 3 } }
                                : new int[][] { { 0, 3 }, { 1, 2 } };
                        for (int[] pr : pairs) {
                            segs.add(new double[] { hits.get(pr[0])[0], hits.get(pr[0])[1], hits.get(pr[1])[0],
                                    hits.get(pr[1])[1] });
                        }
                    }
                }
            }
            out.add(segments(segs, new Color(ISO.getRed(), ISO.getGreen(), ISO.getBlue(), 55 + 130 * c / CONTOURS),
                    1.1f));
        }

        List<double[]> origins = new ArrayList<>(), deltas = new ArrayList<>();
        for (int i = 0; i < ARROW_GRID; i++) {
            for (int j = 0; j < ARROW_GRID; j++) {
                double x = lo + span * (i + 0.5) / ARROW_GRID, y = lo + span * (j + 0.5) / ARROW_GRID;
                // Projected from the 2d faces into the plane by getDensityGradient,
                // whose magnitude is the asymmetry of the neighbourhood rather than a
                // constant. Arrows are therefore SHORT where opposing faces balance,
                // which is honest: there is no gradient there to point along.
                double[] g = forest.getAnisotropicDensity(new float[] { (float) x, (float) y }).getDensityGradient();
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
                saddles, disagree, (saddles > 0) ? 100.0 * disagree / saddles : 0.0);
        System.out.println("a high disagreement rate means the grid does not resolve the field, and any");
        System.out.println("contour topology read off it -- including closed loops -- is unreliable.\n");
        return out;
    }

    private static double frac(double a, double b, double level) {
        return (b == a) ? 0.5 : (level - a) / (b - a);
    }

    /** Many disconnected segments as one Layer; Layer is a functional interface. */
    private static Layer segments(List<double[]> segs, Color color, float stroke) {
        return (g, vp) -> {
            g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
            g.setColor(color);
            g.setStroke(new BasicStroke(stroke));
            for (double[] sg : segs) {
                g.drawLine((int) Math.round(vp.px(sg[0])), (int) Math.round(vp.py(sg[1])),
                        (int) Math.round(vp.px(sg[2])), (int) Math.round(vp.py(sg[3])));
            }
        };
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

    /**
     * Does the extent box reach through the wall, or stop inside it?
     *
     * <p>
     * A thin wall has nothing in it to terminate the box: the nearest mass to an
     * approaching probe is a sliver, and the enlarged bounding box runs from the
     * probe across the sliver and into the hollow interior. Thickening the wall
     * gives the box somewhere to stop, because a patch of the wall already spans a
     * radial range and the box closes on the probe sooner.
     *
     * <p>
     * Reported as a penetration ratio: the box's reach inward from a probe outside,
     * divided by the distance from that probe to the inner edge. Above one means
     * the box has crossed the wall entirely and is describing the empty interior as
     * part of the probe's neighbourhood. The point count is held fixed across
     * thicknesses, so the only thing varying is how widely the same material is
     * spread.
     */
    private static void penetrationStudy(long seed) {
        double probeR = 1.45;
        System.out.printf("penetration study, probe on the axis at r = %.2f, %d points throughout%n", probeR,
                RING_POINTS);
        System.out.printf("%10s %10s %10s %10s %10s %10s%n", "thickness", "inner", "to inner", "reach in", "ratio",
                "aspect");
        for (double w : THICKNESS_SWEEP) {
            RandomCutForest f = RandomCutForest.builder().numberOfTrees(100).sampleSize(256).dimensions(2)
                    .randomSeed(seed).timeDecay(1.0 / (0.8 * RING_POINTS)).centerOfMassEnabled(true).build();
            float[][] pts = annulus(w, seed);
            for (int rep = 0; rep < 3; rep++) {
                for (float[] q : pts) {
                    f.update(q);
                }
            }
            AnisotropicDensityOutput out = f.getAnisotropicDensity(new float[] { (float) probeR, 0f });
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

    /** One box as a closed outline, drawn at true scale in data units. */
    private static Layer boxOutline(double x, double y, double[] box, Color color) {
        // box is {high_x, high_y, low_x, low_y}; high extends toward minus
        // high extends toward minus, low toward plus
        double x0 = x - box[0], x1 = x + box[2], y0 = y - box[1], y1 = y + box[3];
        return Layers.polyline(new double[][] { { x0, y0 }, { x1, y0 }, { x1, y1 }, { x0, y1 }, { x0, y0 } },
                new Color(color.getRed(), color.getGreen(), color.getBlue(), 90), true, 0, 1.0f);
    }
}