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
 *
 */

package org.streamingalgorithms.randomcutforest.examples.summarization;

import static java.lang.Math.PI;
import static org.streamingalgorithms.randomcutforest.examples.datasets.Fan.rotateClockWise;

import java.awt.*;
import java.awt.event.KeyEvent;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

import javax.imageio.ImageIO;
import javax.swing.*;

import org.streamingalgorithms.randomcutforest.RandomCutForest;
import org.streamingalgorithms.randomcutforest.examples.Example;
import org.streamingalgorithms.randomcutforest.examples.datasets.Fan;
import org.streamingalgorithms.randomcutforest.examples.datasets.NormalMixture;
import org.streamingalgorithms.randomcutforest.examples.plot.BoxUnionLayer;
import org.streamingalgorithms.randomcutforest.examples.plot.GifWriter;
import org.streamingalgorithms.randomcutforest.examples.plot.Layer;
import org.streamingalgorithms.randomcutforest.examples.plot.Layers;
import org.streamingalgorithms.randomcutforest.examples.plot.Plot2D;
import org.streamingalgorithms.randomcutforest.summarization.ICluster;
import org.streamingalgorithms.randomcutforest.summarization.Summarizer;
import org.streamingalgorithms.randomcutforest.util.Weighted;

/**
 * Dynamic summarization of a rotating fan, with the measure drawn alongside the
 * clustering.
 *
 * <p>
 * The same forest supplies both, but not the same part of it, and that is what
 * makes the comparison worth drawing. {@code summarize} reaches the point store
 * and runs CURE over the surviving points weighted by their reference counts;
 * It uses the sampler.{@link RandomCutForest#getAnisotropicDensity} is the
 * opposite: it is nothing but tree geometry, a walk up the cuts.
 *
 * <p>
 * So the clustering and the measure are two readings of one sketch that share
 * the sample and share nothing else. A box around a representative is not a
 * restatement of how that representative was chosen. Two dimensions means the
 * boxes are drawn exactly, not projected.
 *
 * <p>
 * Keys: <b>space</b> pauses and <b>right arrow</b> steps one frame, <b>t</b>
 * shows or hides the boxes, <b>c</b> cycles the scale the boxes are read at
 * (CUT, PASSAGE, STOP), <b>k</b> turns centring on and off, and <b>s</b> writes
 * the current frame to a png. All of them work while paused, and both the png
 * and the gif record whatever the toggles say at the moment the frame is
 * written -- the layers consult their gates at draw time, so a recording made
 * while toggling shows the toggling.
 */
public class Summarization implements Example {

    public static void main(String[] args) throws Exception {
        new Summarization().run();
    }

    @Override
    public String command() {
        return "summarization";
    }

    @Override
    public String description() {
        return "Dynamic clustering/summarization";
    }

    /**
     * Scale the c key starts on. The query points are fixed: always the cluster's
     * own members, never its representatives.
     *
     */
    private static final ClusterBoxes.BoxKind DRAWN_KIND = ClusterBoxes.BoxKind.PASSAGE;

    /**
     * Starting state of the k key -- shows the symmetrized box.
     *
     * <p>
     * Press k to see the difference; {@code ClusterBoxes.describe} prints the mean
     * discarded drift on every line that reports a box size, so how much is being
     * given up is never a matter of opinion.
     */
    private static final boolean CENTER_BOXES = false;

    /**
     * Fraction of the boxes actually drawn. The census, the volumes and every other
     * number use all of them.
     */
    private static final double DRAW_FRACTION = 0.25;

    /**
     * Side of the png written by the s key; sized for a page, not for a display.
     */
    private static final int SHOT_PX = 1400;

    /** Frame at which the one-shot box-size diagnostic prints. */
    private static final int FIRST_REPORT_DEGREE = 5;

    /**
     * Points per frame, over all blades, so about 540 each at five blades.
     *
     * <p>
     * Raised from 1350. The blades were thin enough that the cover had visible gaps
     * along them and the unclaimed figure sat near 4%, and at that density it is
     * hard to tell a gap in the measure from a gap in the data. The decay below
     * moves with this, so the reservoir keeps holding the same fraction of a frame.
     *
     */
    private static final int DATA_SIZE = 2700;

    /**
     * Sample size, decay and tree count, kept together because only their ratios
     * matter and setting one without the others is what makes the boxes wrong.
     *
     * <p>
     * The forest is a sketch, and what the boxes can resolve is set by how much of
     * the picture a single tree holds, and that is set by the sample rather than by
     * the data: at a 256 sample over five thin blades, one tree sees about fifty
     * points per blade spread along its length whatever DATA_SIZE is, and the walk
     * stops at the mass crossing at sampleSize^(1/2) = 16 of them -- a third of
     * what the tree knows about that blade, measured on a plot whose half-range is
     * 15. NearNeighborExample gets away with the same 256 because its whole world
     * is 1.6 across and its data is one filled blob, so 256 points resolve it. Here
     * they do not.
     *
     * <p>
     * Raising the sample to hold a whole frame was tried and reverted: it cost
     * several times the update phase and the boxes did not improve, because what
     * limits them here is the blade's geometry within a tree rather than the number
     * of points in the reservoir. 256 stays.
     *
     * <p>
     * Read together: DATA_SIZE sets how densely the cover is queried, TIME_DECAY
     * sets what the clustering is computed from, and sampleSize sets what one tree
     * can resolve. They were entangled here and are not any more.
     *
     */
    private static final int SAMPLE_SIZE = 256;
    private static final double TIME_DECAY = 1.0 / 800;
    private static final int NUMBER_OF_TREES = 100;

    /**
     * Query budget per cluster per frame. Zero means every member, which is the
     * default and the honest one.
     *
     */
    private static final int MEMBER_QUERIES = 0;

    /**
     * One hue per blade, validated rather than chosen by eye.
     *
     * <p>
     * These five clear every pairwise gate against a white surface. There is no
     * sixth hue on purpose. {@code summarize} may return up to
     * {@code 2 * blades + 2} clusters, and a sixth found in a five-blade fan is
     * over-segmentation, usually transient as two blades pass.
     */
    private static final Color[] PALETTE = { new Color(0x2A, 0x78, 0xD6), // blue
            new Color(0xEB, 0x68, 0x34), // orange
            new Color(0x1B, 0xAF, 0x7A), // aqua
            new Color(0x4A, 0x3A, 0xA7), // violet
            new Color(0xC2, 0x18, 0x5B), // crimson
    };

    /** Anything past the palette: over-segmentation, shown as such. */
    private static final Color OVERFLOW = new Color(125, 125, 125);

    private static final Color CLAIMED = new Color(198, 198, 198);
    private static final Color UNCLAIMED = new Color(72, 72, 72);

    private static Color clusterColor(int index) {
        return (index >= 0 && index < PALETTE.length) ? PALETTE[index] : OVERFLOW;
    }

    @Override
    public void run() throws Exception {
        int newDimensions = 2;
        long randomSeed = 123;
        int dataSize = DATA_SIZE;
        int numberOfBlades = 5;
        double range = 15.0;

        RandomCutForest newForest = RandomCutForest.builder().numberOfTrees(NUMBER_OF_TREES).sampleSize(SAMPLE_SIZE)
                .dimensions(newDimensions).randomSeed(randomSeed).timeDecay(TIME_DECAY).build();

        boolean printFile = false; // old gnuplot text dump
        boolean livePlot = true; // on-screen window
        boolean saveGif = true; // animated gif output
        int frameDelayMs = 8; // on-screen pacing
        int gifSizePx = 700;
        int gifDelayMs = 40; // ~25 fps in the file
        int gifEvery = 1; // write every Nth frame to the gif; raise to shrink the file
        int reportEvery = 30;

        BufferedWriter file = printFile ? new BufferedWriter(new FileWriter("dynamic_summarization_example")) : null;

        float[][] data = Fan.getData(dataSize, 0, numberOfBlades);

        Plot2D plot = livePlot ? Plot2D.open("Dynamic Summarization (" + numberOfBlades + " blades)", range, 860)
                : Plot2D.offscreen(range);
        GifWriter gif = saveGif ? new GifWriter(new File("dynamic_summarization.gif"), gifDelayMs, true) : null;

        List<ICluster<float[]>> oldSummary = null;
        int[] oldColors = null;

        // Gated inside the layer, so a repaint alone honours the toggle while paused.
        AtomicBoolean showBoxes = new AtomicBoolean(true);
        plot.bindKey(KeyEvent.VK_T, "summarization.boxes", () -> showBoxes.set(!showBoxes.get()));

        AtomicReference<ClusterBoxes.BoxKind> drawnKind = new AtomicReference<>(DRAWN_KIND);
        plot.bindKey(KeyEvent.VK_C, "summarization.boxkind", () -> {
            ClusterBoxes.BoxKind[] all = ClusterBoxes.BoxKind.values();
            ClusterBoxes.BoxKind next = all[(drawnKind.get().ordinal() + 1) % all.length];
            drawnKind.set(next);
            System.out.println("scale -> " + next + " (" + next.blurb() + ")");
        });

        AtomicBoolean centered = new AtomicBoolean(CENTER_BOXES);
        plot.bindKey(KeyEvent.VK_K, "summarization.center", () -> {
            centered.set(!centered.get());
            System.out.println("boxes -> " + (centered.get()
                    ? "CENTRED on the query (symmetric by construction, " + "widths preserved, positions moved)"
                    : "as the forest returned them (asymmetric)"));
        });

        AtomicReference<List<Layer>> current = new AtomicReference<>(new ArrayList<>());
        AtomicInteger shotIndex = new AtomicInteger();
        plot.bindKey(KeyEvent.VK_S, "summarization.shot", () -> {
            List<Layer> scene = current.get();
            if (scene.isEmpty()) {
                return;
            }
            File out = new File(String.format("summarization_%02d.png", shotIndex.getAndIncrement()));
            try {
                ImageIO.write(plot.renderImage(SHOT_PX, SHOT_PX, scene), "png", out);
                System.out.println("wrote " + out.getName());
            } catch (IOException e) {
                System.out.println("screenshot failed: " + e.getMessage());
            }
        });
        System.out.println("keys: space pauses, right arrow steps, t toggles the boxes, "
                + "c cycles the scale (CUT/PASSAGE/STOP), k toggles centring, s writes a png");
        System.out.printf("drawing %.0f%% of the boxes; every printed figure uses all of them%n", 100 * DRAW_FRACTION);

        int count = 0, sum = 0, over = 0, under = 0;
        Instant start = Instant.now();
        long updateNanos = 0, summarizeNanos = 0, boxNanos = 0;

        for (int degree = 0; degree < 360; degree += 1) {
            float[][] bg = new float[data.length][2];
            int n = 0;

            Instant u0 = Instant.now();
            for (var datum : data) {
                float[] vec = rotateClockWise(datum, -2 * PI * degree / 360);
                bg[n][0] = vec[0];
                bg[n][1] = vec[1];
                n++;
                if (printFile) {
                    file.append(vec[0] + " " + vec[1] + "\n");
                }
                newForest.update(vec);
            }
            updateNanos += Duration.between(u0, Instant.now()).toNanos();
            if (printFile) {
                file.append("\n");
                file.append("\n");
            }

            Instant s0 = Instant.now();
            List<ICluster<float[]>> summary = newForest.summarize(2 * numberOfBlades + 2, 0.05, 5, 0.8,
                    Summarizer::L2distance, oldSummary);
            summarizeNanos += Duration.between(s0, Instant.now()).toNanos();

            // The same forest that produced the clustering also carries the measure
            // around it: one anisotropic box per representative, so each cluster is a
            // union of boxes. In two dimensions the boxes project to themselves, so
            // what is on screen is the set, not a shadow of it.
            Instant b0 = Instant.now();
            ClusterBoxes.BoxKind frameKind = drawnKind.get();
            List<ClusterBoxes.Union> unions = ClusterBoxes.forMembers(newForest, summary, bg, MEMBER_QUERIES, frameKind,
                    centered.get(), ClusterBoxes.DEFAULT_CORE_QUANTILE);
            boxNanos += Duration.between(b0, Instant.now()).toNanos();

            // Say how big they came out, once the forest has seen enough to be
            // worth asking. Boxes too small to see and no boxes at all look the
            // same on screen and are not the same problem.
            if (degree == FIRST_REPORT_DEGREE) {
                System.out.printf(
                        "forest: %d trees, sample %d of %d points/frame, decay 1/%.0f "
                                + "(window %.2f frames), mass target sampleSize^(1/2) = %.0f%n",
                        NUMBER_OF_TREES, SAMPLE_SIZE, dataSize, 1 / TIME_DECAY, 1 / (TIME_DECAY * dataSize),
                        Math.sqrt(SAMPLE_SIZE));

                System.out.printf(
                        "  summarizer clusters the point store: about %.0f distinct points "
                                + "(the decay window, not the sample), max %d clusters allowed%n",
                        1 / TIME_DECAY, 2 * numberOfBlades + 2);

                Map<ClusterBoxes.BoxKind, List<ClusterBoxes.Union>> sweep = ClusterBoxes.forMembersAll(newForest,
                        summary, bg, MEMBER_QUERIES, centered.get(), ClusterBoxes.DEFAULT_CORE_QUANTILE);
                for (ClusterBoxes.BoxKind kind : ClusterBoxes.BoxKind.values()) {
                    System.out.printf("  reps    %-8s %s%n", kind,
                            ClusterBoxes.describe(ClusterBoxes.forClusters(newForest, summary, kind, centered.get())));
                    System.out.printf("  members %-8s %s%n", kind, ClusterBoxes.describe(sweep.get(kind)));
                }
                System.out.println("  plot half-range is " + range + ", so compare the half-widths against that");
                ClusterBoxes.reachReport(sweep, summary);
            }

            int[] hits = ClusterBoxes.hitCounts(unions, bg);
            int[] frameCensus = ClusterBoxes.censusOf(hits, bg.length);

            sum += summary.size();

            int[] reachTotal = ClusterBoxes.reachTotal(ClusterBoxes.reach(unions, summary));
            double boxes = Math.max(1, reachTotal[4]);

            System.out.printf(
                    "%3d  clusters %2d  unclaimed %5.1f%%  contested %4.1f%%  |  %s reach: own %5.1f%%  "
                            + "other-only %4.1f%%  none %4.1f%%%n",
                    degree, summary.size(), 100.0 * frameCensus[2] / bg.length, 100.0 * frameCensus[1] / bg.length,
                    frameKind, 100.0 * reachTotal[ClusterBoxes.REACH_OWN] / boxes,
                    100.0 * reachTotal[ClusterBoxes.REACH_OTHER_ONLY] / boxes,
                    100.0 * reachTotal[ClusterBoxes.REACH_NONE] / boxes);
            if (summary.size() == numberOfBlades) {
                ++count;
            } else if (summary.size() > numberOfBlades) {
                ++over;
            } else {
                ++under;
            }
            int[] colors = align(summary, oldSummary, oldColors);

            // ---- build the scene ----
            List<Layer> scene = new ArrayList<>();

            // behind the data, so the points stay legible through the translucent fill
            List<double[][]> polygons = new ArrayList<>();
            List<Integer> owner = new ArrayList<>();
            Color[] boxPalette = new Color[summary.size()];
            // Drawn from a thinned copy; every figure above came from the full
            // union. drawStride is 1 when DRAW_FRACTION is 1.0.
            int drawStride = (int) Math.max(1, Math.round(1.0 / Math.max(1e-9, DRAW_FRACTION)));
            for (int i = 0; i < summary.size(); i++) {
                boxPalette[i] = clusterColor(colors[i]);
                if (i < unions.size()) {
                    ClusterBoxes.Union shown = ClusterBoxes.everyNth(unions.get(i), drawStride);
                    for (double[][] poly : ClusterBoxes.projectAxes(shown, 0, 1, newDimensions)) {
                        polygons.add(poly);
                        owner.add(i);
                    }
                }
            }
            if (!polygons.isEmpty()) {
                scene.add(new BoxUnionLayer(polygons, owner.stream().mapToInt(Integer::intValue).toArray(), boxPalette,
                        showBoxes::get, 45, 170));
            }

            int uncovered = frameCensus[2];
            float[][] claimed = new float[bg.length - uncovered][];
            float[][] unclaimed = new float[uncovered][];
            int ci = 0;
            int ui = 0;
            for (int i = 0; i < bg.length; i++) {
                if (hits[i] > 0 || uncovered == 0) {
                    claimed[ci++] = bg[i];
                } else {
                    unclaimed[ui++] = bg[i];
                }
            }
            scene.add(Layers.dots(claimed, CLAIMED, 1.6));
            scene.add(Layers.dots(unclaimed, UNCLAIMED, 2.4));
            for (int i = 0; i < summary.size(); i++) {
                double weight = summary.get(i).getWeight();
                List<double[]> rp = new ArrayList<>();
                List<Double> rw = new ArrayList<>();
                for (Weighted<float[]> rep : summary.get(i).getRepresentatives()) {
                    double t = rep.weight / weight;
                    if (t > 0.05) {
                        if (printFile) {
                            file.append(rep.index[0] + " " + rep.index[1] + " " + t + " " + colors[i] + "\n");
                        }
                        rp.add(new double[] { rep.index[0], rep.index[1] });
                        rw.add(t);
                    }
                }
                if (rp.isEmpty()) {
                    continue;
                }
                int[] ord = angleOrder(rp);
                double[][] ring = new double[ord.length][];
                double[] w = new double[ord.length];
                for (int k = 0; k < ord.length; k++) {
                    ring[k] = rp.get(ord[k]);
                    w[k] = rw.get(ord[k]);
                }
                Color col = clusterColor(colors[i]);
                scene.add(Layers.polyline(ring, col, true, 55, 1.6f));
                scene.add(Layers.weightedDots(ring, w, col, 3.0, 13.0));
            }
            if (printFile) {
                file.append("\n");
                file.append("\n");
            }

            // ---- output the scene ----
            if (livePlot) {
                plot.render(scene);
                if (frameDelayMs > 0) {
                    Thread.sleep(frameDelayMs);
                }
            }
            if (saveGif && degree % gifEvery == 0) {
                gif.writeFrame(plot.renderImage(gifSizePx, gifSizePx, scene));
            }

            current.set(scene);
            plot.awaitResume();

            if ((degree + 1) % reportEvery == 0) {
                long ms = Duration.between(start, Instant.now()).toMillis();
                System.out.printf(
                        "[%3d deg] total %d ms | summarize %.1f ms (%.2f ms/frame) | update %.1f ms"
                                + " | boxes %.2f ms/frame%n",
                        degree + 1, ms, summarizeNanos / 1e6, summarizeNanos / 1e6 / (degree + 1), updateNanos / 1e6,
                        boxNanos / 1e6 / (degree + 1));

                int[] census = frameCensus; // already computed for the per-frame line
                double total = Math.max(1, bg.length);
                int connected = 0;
                double cohesion = 0;
                for (int c = 0; c < unions.size(); c++) {
                    if (ClusterBoxes.components(unions.get(c)) == 1) {
                        connected++;
                    }
                    cohesion += ClusterBoxes.cohesion(unions.get(c), degree * 131L + c, 4000);
                }
                System.out.printf(
                        "          census: covered %.1f%%, contested %.1f%%, unclaimed %.1f%%"
                                + " | %d/%d unions connected, mean cohesion %.2f%n",
                        100 * census[0] / total, 100 * census[1] / total, 100 * census[2] / total, connected,
                        unions.size(), unions.isEmpty() ? 0.0 : cohesion / unions.size());
                System.out.println("          members/" + frameKind + ": " + ClusterBoxes.describe(unions));
            }

            if (summary.size() == numberOfBlades) {
                oldSummary = summary;
                oldColors = colors;
            }
        }
        System.out.printf("Exact cluster count: %.2f of frames returned exactly %d clusters, avg %.2f%n",
                Math.round(count / 3.6) * 0.01, numberOfBlades, Math.round(sum / 3.6) * 0.01);
        System.out.printf("  misses: %d frames over-segmented, %d under -- over means every blade was found "
                + "and one was split%n", over, under);

        if (gif != null) {
            gif.close();
            System.out.println("wrote dynamic_summarization.gif");
        }
        if (file != null) {
            file.close();
        }
    }

    private static int[] angleOrder(List<double[]> pts) {
        double cx = 0, cy = 0;
        for (double[] p : pts) {
            cx += p[0];
            cy += p[1];
        }
        final double fcx = cx / pts.size(), fcy = cy / pts.size();
        Integer[] idx = new Integer[pts.size()];
        for (int k = 0; k < idx.length; k++) {
            idx[k] = k;
        }
        Arrays.sort(idx, (a, b) -> Double.compare(Math.atan2(pts.get(a)[1] - fcy, pts.get(a)[0] - fcx),
                Math.atan2(pts.get(b)[1] - fcy, pts.get(b)[0] - fcx)));
        int[] out = new int[idx.length];
        for (int k = 0; k < idx.length; k++) {
            out[k] = idx[k];
        }
        return out;
    }

    private Color colorFor(int c, int max) {
        return Color.getHSBColor((float) c / max, 0.78f, 0.9f);
    }

    public float[][] getData(int dataSize, int seed, int fans) {
        Random prg = new Random(0);
        NormalMixture generator = new NormalMixture(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
        int newDimensions = 2;
        float[][] data = generator.generateData(dataSize, newDimensions, seed).data;

        for (int i = 0; i < dataSize; i++) {
            int nextFan = prg.nextInt(fans);
            // scale, make an ellipse
            data[i][1] *= 1.0 / fans;
            data[i][0] *= 2.0;
            // shift
            data[i][0] += 5.0 + fans / 2;
            data[i] = rotateClockWise(data[i], 2 * PI * nextFan / fans);
        }

        return data;
    }

    /**
     * Carries colours over from the previous frame, as a matching.
     *
     * <p>
     * Taking each cluster's nearest predecessor independently does not give a
     * matching: two clusters can both be nearest to the same predecessor and both
     * inherit its colour, which is how a five-blade fan ends up drawn with two
     * blades in the same blue. Assigning greedily in order of increasing distance
     *
     * <p>
     * A cluster with no colour left over -- a new one, or one whose predecessor was
     * claimed -- takes the lowest index nobody is using.
     */
    int[] align(List<ICluster<float[]>> current, List<ICluster<float[]>> previous, int[] oldColors) {
        int n = current.size();
        int[] assigned = new int[n];
        Arrays.fill(assigned, -1);
        if (previous == null || previous.isEmpty() || oldColors == null) {
            for (int i = 0; i < n; i++) {
                assigned[i] = i;
            }
            return assigned;
        }

        int m = Math.min(previous.size(), oldColors.length);
        List<double[]> pairs = new ArrayList<>(n * m); // {distance, current, previous}
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                pairs.add(new double[] { previous.get(j).distance(current.get(i), Summarizer::L1distance), i, j });
            }
        }
        pairs.sort((a, b) -> Double.compare(a[0], b[0]));

        java.util.Set<Integer> taken = new java.util.HashSet<>();
        for (double[] pair : pairs) {
            int i = (int) pair[1];
            int colour = oldColors[(int) pair[2]];
            if (assigned[i] >= 0 || taken.contains(colour)) {
                continue;
            }
            assigned[i] = colour;
            taken.add(colour);
        }
        int next = 0;
        for (int i = 0; i < n; i++) {
            if (assigned[i] < 0) {
                while (taken.contains(next)) {
                    next++;
                }
                assigned[i] = next;
                taken.add(next);
            }
        }
        return assigned;
    }
}
