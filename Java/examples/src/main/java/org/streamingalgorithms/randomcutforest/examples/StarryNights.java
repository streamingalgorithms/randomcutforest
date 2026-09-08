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
import java.io.File;
import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.streamingalgorithms.randomcutforest.RandomCutForest;
import org.streamingalgorithms.randomcutforest.examples.plot.Arrow3DLayer;
import org.streamingalgorithms.randomcutforest.examples.plot.Axes3DLayer;
import org.streamingalgorithms.randomcutforest.examples.plot.Box3DLayer;
import org.streamingalgorithms.randomcutforest.examples.plot.BoxLayer;
import org.streamingalgorithms.randomcutforest.examples.plot.GifWriter;
import org.streamingalgorithms.randomcutforest.examples.plot.Layer;
import org.streamingalgorithms.randomcutforest.examples.plot.Layers;
import org.streamingalgorithms.randomcutforest.examples.plot.Plot2D;
import org.streamingalgorithms.randomcutforest.returntypes.AnisotropicDensityOutput;
import org.streamingalgorithms.randomcutforest.returntypes.DiVector;

/**
 * Three-dimensional anisotropic density on two moving structures.
 *
 * <p>
 * <b>The data moves; the axes do not.</b> That is the difference from
 * AnisotropicDensityExample, which rotates the frame to measure how much of the
 * reported anisotropy is an artefact of the axis-aligned cut basis. Here the
 * coordinates keep a fixed meaning throughout and the structures translate and
 * rotate through them, which is the situation the estimator is actually built
 * for: shingle lags, sensor channels, anything where the axes mean something.
 *
 * <p>
 * <b>What should happen.</b> Both structures are curves — Archimedean spiral
 * arms, and a (7,2) torus knot — so the support is one-dimensional inside an
 * ambient three. The local exponent should therefore come back near 1, not near
 * 3, and the extent boxes should be long along the local tangent and thin
 * across it. That is a far less forgiving target than the flow example offered:
 * an exponent near 3 would mean the estimator is reporting the ambient
 * dimension and has learned nothing about the support.
 *
 * <p>
 * The knot also stress-tests the anisotropy in a way the spirals do not. Its
 * tangent direction sweeps through every orientation, so wherever the tangent
 * sits near a coordinate axis the box should be sharply elongated, and wherever
 * it sits near a diagonal the axis-aligned box cannot see the elongation and
 * should read closer to isotropic. That failure is expected and is the point:
 * it is the same 45-degree blindness the rotating example measures, made
 * visible as a spatial pattern rather than a time series.
 */
public class StarryNights implements Example {

    private static final int DIMENSIONS = 3;

    private static final int BALL_POINTS = 450;
    /** Everything emitted per frame; the decay horizon is a multiple of this. */
    private static final int POINTS_PER_FRAME = BALL_POINTS + DiscGalaxy.POINTS;
    /** One comet point every this many arrivals, so the two sources interleave. */
    private static final int COMET_CADENCE = POINTS_PER_FRAME / BALL_POINTS;

    /** Camera position in world coordinates, looking at the origin. */
    private static final double[] EYE = { 10.0, 10.0, 5.0 };
    private static final double[] TARGET = { 0.0, 0.0, 0.0 };
    private static final double FOV_DEGREES = 22.0;

    /** Plot half-range in projected units. Larger pans out; the scene shrinks. */
    private static final double VIEW_RANGE = 0.75;

    /**
     * Fraction of the septafoil the ball covers over the whole run. Well below 1,
     * so it drifts rather than races and the trail reads as a comet: the wake is a
     * fixed number of frames of memory, so a slower ball means those frames span
     * less arc and the tail stays compact behind the head.
     */
    private static final double LOOPS = 0.30;

    /**
     * Length of the comet tail in WORLD UNITS, not frames. Everything else -- the
     * frame count of retained trail, the time decay, the span of the trailing
     * probes -- is derived from it at startup, from the measured arc length of the
     * trajectory.
     *
     * <p>
     * Setting it in frames is what went wrong before. Seven frames sounds like a
     * tail, but the ball covers arcLength * LOOPS / frames per frame, about 0.025
     * here, so seven frames spanned 0.175 world units against a ball diameter of
     * 0.22. Seven copies of a ball that has barely moved is not a comet, and the
     * trailing probes were then packed a hundredth apart, which collapsed the glyph
     * scale to nothing. In world units the tail is a length you can compare to the
     * ball, and the failure is visible in the constant itself.
     *
     * <p>
     * Time decay is a horizon in <i>updates</i>, not frames, so it only means
     * anything relative to how many points arrive per frame. DensityExample pairs
     * 2000 points a frame with 1/800, which retains 0.4 of a frame -- the current
     * snapshot and nothing more, correct when every frame is a complete picture. A
     * comet needs the opposite, and setting the two independently is how the probes
     * ended up reaching further back than the retained mass.
     */
    private static final double TAIL_LENGTH = 0.85;

    /** Ball radius. The tail must be several of these long to read as a comet. */
    private static final double BALL_RADIUS = 0.075;

    /**
     * Window the exponent fill is stretched over. Recalibrate from the report line.
     */
    private static final double ALPHA_LO = 0.5;
    private static final double ALPHA_HI = 2.0;

    /**
     * Probes follow the wake rather than a fixed lattice. A lattice over this scene
     * is mostly vacuum: a first pass had 72 of 75 probes empty every frame, so the
     * surviving one or two flickered from place to place and the reported exponent
     * was an average over a single sample. Sampling the trajectory backwards
     * guarantees every probe has mass, and makes the glyphs trace the trail.
     */
    /** Probe budget for the online cover; the radius grows until this holds. */
    private static final int PROBE_BUDGET = 34;
    private static final double PROBE_EXTENT = 1.6;

    private static final int SOURCE_GALAXY = 0;
    private static final int SOURCE_COMET = 1;

    /** Sparse lattice for the directional density field. */
    private static final int FIELD_XY = 4;
    private static final int FIELD_Z = 2;
    private static final double FIELD_EXTENT = 1.15;
    private static final double FIELD_Z_EXTENT = 0.55;
    private static final double ARROW_LENGTH = 0.20;

    /**
     * Draw the directional field at all, and what fraction of lattice cells to
     * keep.
     *
     * <p>
     * The far-field arrows all point at the galaxy, which is true and
     * uninformative: a ring of arrows saying "the mass is in the middle" costs a
     * lot of ink for one bit. Keeping only the densest cells leaves the arrows that
     * sit between the arms and near the comet, where the gradient actually turns.
     * Fraction rather than an absolute threshold so it rescales itself as the scene
     * evolves.
     */
    private static final boolean SHOW_FIELD = false;
    private static final double FIELD_KEEP = 0.45;

    /**
     * Cool for data, warm for inference. Worth being strict about: with the boxes
     * drawn amber and the comet drawn on a heat ramp, the chain of trailing boxes
     * reads as a tail of points and there is no way to tell measurement from model
     * output. Everything the forest was told is blue or teal; everything the forest
     * concluded is amber, red, or grey.
     */
    private static final Color GALAXY = new Color(26, 148, 138);
    private static final Color COMET = new Color(36, 84, 214);

    public static void main(String[] args) throws Exception {
        new StarryNights().run();
    }

    @Override
    public String command() {
        return "starry-nights";
    }

    @Override
    public String description() {
        return "3D anisotropic density on spiral galaxies and a (7,2) torus knot";
    }

    @Override
    public void run() throws Exception {
        long randomSeed = 42;
        int manifoldDimension = 1; // both structures are curves
        double q = 0.001;

        boolean livePlot = true;
        boolean saveGif = true;
        int frames = 240;
        int frameDelayMs = 20;
        int gifSizePx = 760;
        int reportEvery = 20;

        DiscGalaxy disc = new DiscGalaxy(randomSeed);
        SeptafoilBall ball = new SeptafoilBall(BALL_POINTS, BALL_RADIUS, randomSeed);

        // Tail geometry, derived rather than guessed
        double travelPerFrame = SeptafoilBall.arcLength() * LOOPS / frames;
        int tailFrames = Math.max(3, (int) Math.round(TAIL_LENGTH / travelPerFrame));
        System.out.printf("arc %.2f | travel/frame %.4f | tail %d frames = %.2f units " + "(%.1f ball diameters)%n",
                SeptafoilBall.arcLength(), travelPerFrame, tailFrames, tailFrames * travelPerFrame,
                tailFrames * travelPerFrame / (2 * BALL_RADIUS));

        RandomCutForest forest = RandomCutForest.builder().numberOfTrees(50).sampleSize(256).dimensions(DIMENSIONS)
                .randomSeed(randomSeed).timeDecay(1.0 / (POINTS_PER_FRAME * tailFrames)).centerOfMassEnabled(true)
                .build();

        // Probes live exactly as long as the data that justified them: the same window
        // as the forest's decay, so the comet leaves no trail of stale probes behind.
        ProbeNet net = new ProbeNet(PROBE_BUDGET, (long) POINTS_PER_FRAME * tailFrames);

        Camera camera = new Camera(EYE, TARGET, FOV_DEGREES);
        Plot2D plot = livePlot ? Plot2D.open("Starry Nights — 3D extent boxes", VIEW_RANGE, 860)
                : Plot2D.offscreen(VIEW_RANGE);
        GifWriter gif = saveGif ? new GifWriter(new File("starry_nights.gif"), 50, true) : null;

        Instant start = Instant.now();
        long scoreNanos = 0;

        for (int frame = 0; frame < frames; frame++) {
            double t = LOOPS * frame / (double) frames; // ball phase, fraction of the knot
            // The galaxy turns on its own clock. Feeding it the ball phase multiplied
            // TURNS_PER_RUN by LOOPS, so 0.35 turns became 0.105 -- about 38 degrees
            // across the whole run, which reads as stationary.
            double galaxyPhase = frame / (double) frames;
            // the wake spans exactly the arc covered during the retained frames
            double wakeSpan = LOOPS * tailFrames / (double) frames;

            List<float[]> discPoints = new ArrayList<>();
            List<float[]> ballPoints = new ArrayList<>();
            // Phase advances per POINT across the interval, not per frame. Emitting a
            // whole frame at one phase put twenty discrete copies of the arm into the
            // decay window -- ghosts, not a smear -- and no amount of reordering fixes
            // that, because the copies are in the geometry. The emitter also cycles
            // arms and features as it goes, so arrival order carries no spatial
            // pattern and expiry is uniform without anything being shuffled.
            // The only thing that has to differ from DensityExample's loop: the phase
            // advances per POINT, not per frame. DensityExample can batch a frame
            // because its decay retains 0.4 of one, so nothing from a previous angle
            // survives; this example asks for a 20-frame tail, and emitting a frame at
            // a single phase would put 20 frozen copies of the arm in that window.
            for (int i = 0; i < POINTS_PER_FRAME; i++) {
                double u = (frame + i / (double) POINTS_PER_FRAME) / frames;
                float[] p;
                int source;
                if (i % COMET_CADENCE == 0) {
                    p = ball.sample(LOOPS * u);
                    ballPoints.add(p);
                    source = SOURCE_COMET;
                } else {
                    p = disc.sample(u, i);
                    discPoints.add(p);
                    source = SOURCE_GALAXY;
                }
                forest.update(p);
                // Probes are placed in the same pass, from the data itself: a point
                // already within the cover radius of a probe is represented and is
                // ignored, otherwise it becomes one.
                net.offer(p, source);
            }
            net.expire();

            // ---- probe lattice ----
            ProbeSet wake = new ProbeSet();
            ProbeSet arms = new ProbeSet();

            // Evaluated after the pass, never during it: placement is a property of
            // the stream, evaluation is a query against the model the stream built.
            Instant t0 = Instant.now();
            for (int i = 0; i < net.size(); i++) {
                ProbeSet target = (net.source(i) == SOURCE_COMET) ? wake : arms;
                target.probe(forest, net.centre(i));
            }
            List<double[]> probeOrigins = new ArrayList<>(wake.origins);
            probeOrigins.addAll(arms.origins);
            List<double[]> probeBoxes = new ArrayList<>(wake.boxes);
            probeBoxes.addAll(arms.boxes);
            List<Double> shade = new ArrayList<>(wake.shade);
            shade.addAll(arms.shade);
            List<Double> fill = new ArrayList<>(wake.fill);
            fill.addAll(arms.fill);
            int drawn = wake.drawn + arms.drawn;
            int flat = wake.flat + arms.flat;

            // Directional density field, the 3D form of the arrows in DensityExample.
            // The DiVector high component on axis i accumulates when the query point
            // overhangs the box ABOVE on that axis, which means the mass lies below it.
            // So (low - high) points toward the mass. The 2D example computes the same
            // thing; its local variable names read backwards.
            List<double[]> fieldOrigins = new ArrayList<>();
            List<double[]> fieldDeltas = new ArrayList<>();
            if (SHOW_FIELD) {
                List<double[]> candidates = new ArrayList<>(); // {gx, gy, gz, vx, vy, vz, density}
                for (int ix = 0; ix < FIELD_XY; ix++) {
                    for (int iy = 0; iy < FIELD_XY; iy++) {
                        for (int iz = 0; iz < FIELD_Z; iz++) {
                            double gx = lattice(ix, FIELD_XY, FIELD_EXTENT);
                            double gy = lattice(iy, FIELD_XY, FIELD_EXTENT);
                            double gz = lattice(iz, FIELD_Z, FIELD_Z_EXTENT);
                            var density = forest.getSimpleDensity(new float[] { (float) gx, (float) gy, (float) gz });
                            DiVector dir = density.getDirectionalDensity(q, manifoldDimension);
                            double[] v = new double[DIMENSIONS];
                            double norm = 0;
                            for (int d = 0; d < DIMENSIONS; d++) {
                                v[d] = dir.low[d] - dir.high[d];
                                norm += v[d] * v[d];
                            }
                            norm = Math.sqrt(norm);
                            if (!(norm > 0)) {
                                continue;
                            }
                            candidates.add(
                                    new double[] { gx, gy, gz, v[0] * ARROW_LENGTH / norm, v[1] * ARROW_LENGTH / norm,
                                            v[2] * ARROW_LENGTH / norm, density.getDensity(q, manifoldDimension) });
                        }
                    }
                }
                candidates.sort((a, b) -> Double.compare(b[6], a[6]));
                int keep = (int) Math.ceil(FIELD_KEEP * candidates.size());
                for (int i = 0; i < keep && i < candidates.size(); i++) {
                    double[] c = candidates.get(i);
                    fieldOrigins.add(new double[] { c[0], c[1], c[2] });
                    fieldDeltas.add(new double[] { c[3], c[4], c[5] });
                }
            }
            scoreNanos += Duration.between(t0, Instant.now()).toNanos();

            // The cover guarantees probes are at least its radius apart, so that IS the
            // pitch. Nothing is inferred from pairwise distances any more, and a pair
            // that happens to land close together can no longer shrink every glyph.
            double pitch = (net.radius() > 0) ? net.radius() : 2 * PROBE_EXTENT / PROBE_BUDGET;
            double longest = 0;
            for (double[] b : probeBoxes) {
                for (double v : b) {
                    longest = Math.max(longest, v);
                }
            }
            double glyph = (longest > 0) ? 0.9 * pitch / longest : 0.0;
            double[][] boxArray = new double[probeBoxes.size()][];
            for (int i = 0; i < probeBoxes.size(); i++) {
                double[] b = probeBoxes.get(i);
                boxArray[i] = new double[6];
                for (int k = 0; k < 6; k++) {
                    boxArray[i][k] = b[k] * glyph;
                }
            }

            // ---- scene ----
            List<Layer> scene = new ArrayList<>();
            // Current frame only. Drawing the retained window would put the model's own
            // memory on screen dressed as data, which is the one thing the colour split
            // exists to prevent. The boxes SHOULD look larger than the visible points:
            // they are inference over a window the data no longer shows.
            scene.add(Layers.dots(project(camera, discPoints), GALAXY, 1.5));
            scene.add(Layers.dots(project(camera, ballPoints), COMET, 1.6));
            scene.add(new Axes3DLayer(camera::project, PROBE_EXTENT, new Color(90, 90, 110)));
            if (SHOW_FIELD) {
                scene.add(new Arrow3DLayer(fieldOrigins.toArray(new double[0][]), fieldDeltas.toArray(new double[0][]),
                        new Color(70, 70, 70), 1.1f).withProjector(camera::project));
            }
            scene.add(new Box3DLayer(probeOrigins.toArray(new double[0][]), boxArray, toArray(shade), toArray(fill),
                    camera::project, EYE, 0.30f));
            // The arrow row is dropped with the arrows themselves: a legend entry for a
            // layer that is not in the scene is a caption for something the reader
            // cannot find.
            List<String> labels = new ArrayList<>(
                    List.of("DATA  galaxy arms", "DATA  comet, fading = decaying out", "MODEL  3D extent box",
                            "MODEL  anisotropic", String.format("MODEL  alpha %.1f filamentary", ALPHA_LO),
                            String.format("MODEL  alpha %.1f space filling", ALPHA_HI)));
            List<Color> colors = new ArrayList<>(List.of(GALAXY, COMET, BoxLayer.outlineColor(0.0),
                    BoxLayer.outlineColor(1.0), BoxLayer.fillColor(0.0, 1.0f), BoxLayer.fillColor(1.0, 1.0f)));
            List<Layers.Swatch> swatches = new ArrayList<>(List.of(Layers.Swatch.DOTS, Layers.Swatch.DOTS,
                    Layers.Swatch.BOX, Layers.Swatch.BOX, Layers.Swatch.BOX, Layers.Swatch.BOX));
            if (SHOW_FIELD) {
                labels.add("MODEL  toward higher density");
                colors.add(new Color(70, 70, 70));
                swatches.add(Layers.Swatch.LINE);
            }
            scene.add(Layers.legend(labels.toArray(new String[0]), colors.toArray(new Color[0]),
                    swatches.toArray(new Layers.Swatch[0])));

            if (livePlot) {
                plot.render(scene);
            }
            if (saveGif) {
                gif.writeFrame(plot.renderImage(gifSizePx, gifSizePx, scene));
            }
            if (livePlot) {
                plot.awaitResume(); // space pauses, right arrow steps; GIF stays continuous
                if (frameDelayMs > 0) {
                    Thread.sleep(frameDelayMs);
                }
            }

            if ((frame + 1) % reportEvery == 0) {
                // Wake and arms reported separately: pooling them averages a tube around
                // a curve with a disc, which is the one comparison this example exists to
                // make. The wake should sit lower.
                System.out.printf(
                        "[%3d] %d ms | probe %.2f ms/frame | boxes %d (flat %d) "
                                + "| probes %d R=%.3f | comet a=%.3f %s | galaxy a=%.3f %s | mass %.4f%n",
                        frame, Duration.between(start, Instant.now()).toMillis(), scoreNanos / 1e6 / (frame + 1), drawn,
                        flat, net.size(), net.radius(), wake.meanAlpha(), wake.range(), arms.meanAlpha(), arms.range(),
                        0.5 * (wake.meanMass() + arms.meanMass()));
            }
        }

        if (gif != null) {
            gif.close();
            System.out.println("wrote starry_nights.gif");
        }
    }

    /**
     * One probe site's worth of work, with the checks that gate it.
     *
     * <p>
     * The wake and the arms run identical code over different points, so it lives
     * here once. Keeping them in separate instances is deliberate: the comet wake
     * is a tube around a curve and the arms are a disc, and averaging their
     * exponents together reports a number describing neither.
     */
    private static final class ProbeSet {
        final List<double[]> origins = new ArrayList<>();
        final List<double[]> boxes = new ArrayList<>();
        final List<Double> shade = new ArrayList<>();
        final List<Double> fill = new ArrayList<>();

        double alphaSum, massSum;
        double alphaMin = Double.MAX_VALUE, alphaMax = -Double.MAX_VALUE;
        int probes, drawn, flat, fitted;

        boolean probe(RandomCutForest forest, double[] c) {
            probes++;
            AnisotropicDensityOutput out = forest
                    .getAnisotropicDensity(new float[] { (float) c[0], (float) c[1], (float) c[2] });
            massSum += out.getSampleSize();
            if (!out.isReliable()) {
                flat++;
                return false; // flat crossing or masked by duplicates; a box would lie
            }
            drawn++;

            origins.add(c);
            boxes.add(out.passageBox());
            shade.add(Math.min(1.0, log2(out.getAnisotropy()) / 3.0));
            fill.add( -1.0);
            return true;
        }

        double meanAlpha() {
            return (fitted > 0) ? alphaSum / fitted : 0.0;
        }

        /** Fraction of drawn boxes that yielded an exponent at all. */
        double coverage() {
            return (drawn > 0) ? fitted / (double) drawn : 0.0;
        }

        /** Mean first-passage weight; should print 1.0000. */
        double meanMass() {
            return (probes > 0) ? massSum / probes : 0.0;
        }

        String range() {
            return (fitted > 0) ? String.format("[%.2f, %.2f]", alphaMin, alphaMax) : "[-, -]";
        }
    }

    private static double[] toArray(List<Double> list) {
        double[] out = new double[list.size()];
        for (int i = 0; i < out.length; i++) {
            out[i] = list.get(i);
        }
        return out;
    }

    /**
     * Projects world points to plot coordinates, dropping anything behind the
     * camera.
     */
    private static float[][] project(Camera camera, List<float[]> points) {
        List<float[]> out = new ArrayList<>(points.size());
        for (float[] p : points) {
            double[] q = camera.project(p[0], p[1], p[2]);
            if (q != null) {
                out.add(new float[] { (float) q[0], (float) q[1] });
            }
        }
        return out.toArray(new float[0][]);
    }

    private static double lattice(int i, int count, double extent) {
        return (count == 1) ? 0.0 : -extent + 2 * extent * i / (double) (count - 1);
    }

    private static double log2(double x) {
        return (x > 0) ? Math.log(x) / Math.log(2.0) : 0.0;
    }

    // ------------------------------------------------------------------

    /** Pinhole camera with an explicit up axis, looking from EYE at TARGET. */
    static final class Camera {
        private final double[] eye;
        private final double[] s = new double[3], u = new double[3], f = new double[3];
        private final double scale;

        Camera(double[] eye, double[] target, double fovDegrees) {
            this.eye = eye.clone();
            for (int i = 0; i < 3; i++) {
                f[i] = target[i] - eye[i];
            }
            normalize(f);
            double[] up = { 0, 0, 1 };
            cross(f, up, s);
            normalize(s);
            cross(s, f, u);
            normalize(u);
            this.scale = 1.0 / Math.tan(Math.toRadians(fovDegrees) / 2);
        }

        /** Returns plot coordinates, or null when the point is behind the camera. */
        double[] project(double x, double y, double z) {
            double dx = x - eye[0], dy = y - eye[1], dz = z - eye[2];
            double depth = dx * f[0] + dy * f[1] + dz * f[2];
            if (depth <= 1e-6) {
                return null;
            }
            double vx = dx * s[0] + dy * s[1] + dz * s[2];
            double vy = dx * u[0] + dy * u[1] + dz * u[2];
            return new double[] { scale * vx / depth, scale * vy / depth };
        }

        private static void cross(double[] a, double[] b, double[] out) {
            out[0] = a[1] * b[2] - a[2] * b[1];
            out[1] = a[2] * b[0] - a[0] * b[2];
            out[2] = a[0] * b[1] - a[1] * b[0];
        }

        private static void normalize(double[] v) {
            double m = Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
            if (m > 0) {
                v[0] /= m;
                v[1] /= m;
                v[2] /= m;
            }
        }
    }

    /**
     * One galaxy in the x-y plane: three rope-like arms curling outward, each built
     * from discrete clumps rather than a uniformly filled tube.
     *
     * <p>
     * <b>Why a logarithmic spiral.</b> r = R0 * exp(b * theta) has a constant pitch
     * angle, which is what makes real arms look like ropes trailing outward, and r
     * grows without bound so the arms genuinely reach out. The cochleoid r = a *
     * sin(theta)/theta does the opposite: it starts at r = a as theta goes to zero
     * and decays like a/theta, monotonically down to zero at theta = pi, so three
     * copies at 120 degrees give an inward rosette of petals rather than outward
     * arms. Swap {@link #armRadius} if the petal shape is wanted anyway.
     *
     * <p>
     * <b>Why clumps.</b> A uniformly filled tube at this point density is a smear
     * with no visible structure, and a smear gives the density gradient nothing to
     * point at. Discrete knots along the arm read as arms, and give the extent
     * boxes something anisotropic to find: inside a clump the measure is locally
     * three-dimensional, between clumps it is filamentary along the arm.
     */
    static final class DiscGalaxy {
        static final int ARMS = 3;
        private static final double TURNS = 1.15;
        private static final double THETA_MAX = TURNS * 2 * Math.PI;
        private static final double R_INNER = 0.20;
        private static final double R_OUTER = 1.55;
        /** Growth rate fixed by the two radii, so the arm reaches R_OUTER exactly. */
        private static final double B = Math.log(R_OUTER / R_INNER) / THETA_MAX;

        static final int CLUMPS_PER_ARM = 14;
        static final int POINTS_PER_CLUMP = 42;
        private static final double CLUMP_RADIUS = 0.085;
        static final int STRAND_SAMPLES = 220;
        private static final double STRAND_RADIUS = 0.030;
        static final int POINTS = ARMS * (CLUMPS_PER_ARM * POINTS_PER_CLUMP + STRAND_SAMPLES);

        /** Turns of the whole galaxy over the entire run. Slow on purpose. */
        private static final double TURNS_PER_RUN = 0.35;

        private final Random random;

        DiscGalaxy(long seed) {
            this.random = new Random(seed * 31 + 7);
        }

        /** Logarithmic spiral; replace with a * sin(t)/t for a cochleoid. */
        private static double armRadius(double theta) {
            return R_INNER * Math.exp(B * theta);
        }

        private static double spinPhase(double t) {
            return TURNS_PER_RUN * 2 * Math.PI * t;
        }

        /** Centre of clump `k` of arm `a`, in world coordinates. */
        double[] clumpCentre(double t, int arm, int k) {
            return armAt(t, arm, (k + 0.5) / CLUMPS_PER_ARM);
        }

        /**
         * Any point on an arm centreline, at fraction f of the way along the spiral
         * angle. Continuous, so probes can sit between clumps as well as on them.
         */
        double[] armAt(double t, int arm, double f) {
            double theta = THETA_MAX * f;
            double r = armRadius(theta);
            double ang = theta + spinPhase(t) + 2 * Math.PI * arm / ARMS;
            return new double[] { r * Math.cos(ang), r * Math.sin(ang), 0.0 };
        }

        /**
         * The i-th galaxy point of a frame. Stateless: the arrival number picks the arm
         * and the feature, so consecutive arrivals land on different arms and different
         * stations and points expire uniformly across the disc. Nothing needs
         * shuffling, because nothing was ever emitted in spatial blocks.
         */
        float[] sample(double phase, int i) {
            int arm = i % ARMS;
            int k = i / ARMS;
            if (k % 4 == 0) {
                return strandSample(phase, arm, (k / 4) % STRAND_SAMPLES);
            }
            return clumpSample(phase, arm, k % CLUMPS_PER_ARM);
        }

        /** One strand point on the given arm, at strand station `index`. */
        float[] strandSample(double t, int arm, int index) {
            double theta = THETA_MAX * index / (double) STRAND_SAMPLES;
            double r = armRadius(theta);
            double ang = theta + spinPhase(t) + 2 * Math.PI * arm / ARMS;
            double jitter = STRAND_RADIUS * (0.6 + 0.8 * theta / THETA_MAX);
            return new float[] { (float) (r * Math.cos(ang) + jitter * (2 * random.nextDouble() - 1)),
                    (float) (r * Math.sin(ang) + jitter * (2 * random.nextDouble() - 1)),
                    (float) (0.4 * jitter * (2 * random.nextDouble() - 1)) };
        }

        /** One point drawn inside clump `k` of the given arm. */
        float[] clumpSample(double t, int arm, int k) {
            double[] c = clumpCentre(t, arm, k);
            // clumps farther out are looser, as the arm stretches
            double spread = CLUMP_RADIUS * (0.6 + 0.9 * k / (double) CLUMPS_PER_ARM);
            double x, y, z;
            do {
                x = 2 * random.nextDouble() - 1;
                y = 2 * random.nextDouble() - 1;
                z = 2 * random.nextDouble() - 1;
            } while (x * x + y * y + z * z > 1);
            return new float[] { (float) (c[0] + spread * x), (float) (c[1] + spread * y),
                    (float) (c[2] + 0.45 * spread * z) };
        }
    }

    static final class SeptafoilBall {
        private static final double R = 1.0, RR = 0.35;
        private final int samples;
        private final double radius;
        private final Random random;

        SeptafoilBall(int samples, double radius, long seed) {
            this.samples = samples;
            this.radius = radius;
            this.random = new Random(seed);
        }

        /** Numerically integrated length of one full loop, in world units. */
        static double arcLength() {
            double total = 0;
            double[] prev = centre(0);
            int steps = 4000;
            for (int i = 1; i <= steps; i++) {
                double[] p = centre(i / (double) steps);
                total += Math.sqrt((p[0] - prev[0]) * (p[0] - prev[0]) + (p[1] - prev[1]) * (p[1] - prev[1])
                        + (p[2] - prev[2]) * (p[2] - prev[2]));
                prev = p;
            }
            return total;
        }

        /** Centre of the ball at phase t in [0,1). */
        static double[] centre(double t) {
            double u = 2 * Math.PI * t;
            double radial = R + RR * Math.cos(7 * u);
            return new double[] { Math.cos(2 * u) * radial, Math.sin(2 * u) * radial, Math.sin(7 * u) };
        }

        /** One point drawn uniformly in the ball at its position for this phase. */
        float[] sample(double t) {
            double[] c = centre(t);
            // rejection sampling gives a uniform ball; a Gaussian would put a spurious
            // density peak at the centre
            double x, y, z;
            do {
                x = 2 * random.nextDouble() - 1;
                y = 2 * random.nextDouble() - 1;
                z = 2 * random.nextDouble() - 1;
            } while (x * x + y * y + z * z > 1);
            return new float[] { (float) (c[0] + radius * x), (float) (c[1] + radius * y),
                    (float) (c[2] + radius * z) };
        }
    }
}
