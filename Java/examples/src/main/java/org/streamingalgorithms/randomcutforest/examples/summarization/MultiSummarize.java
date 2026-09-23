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

package org.streamingalgorithms.randomcutforest.examples.summarization;

import static java.lang.Math.abs;

import java.awt.*;
import java.awt.event.KeyEvent;
import java.io.File;
import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.EnumMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;

import javax.imageio.ImageIO;

import org.streamingalgorithms.randomcutforest.RandomCutForest;
import org.streamingalgorithms.randomcutforest.examples.Example;
import org.streamingalgorithms.randomcutforest.examples.datasets.Clusters;
import org.streamingalgorithms.randomcutforest.examples.plot.BoxUnionLayer;
import org.streamingalgorithms.randomcutforest.examples.plot.Layer;
import org.streamingalgorithms.randomcutforest.examples.plot.Layers;
import org.streamingalgorithms.randomcutforest.examples.plot.Plot2D;
import org.streamingalgorithms.randomcutforest.summarization.ICluster;
import org.streamingalgorithms.randomcutforest.summarization.Summarizer;
import org.streamingalgorithms.randomcutforest.util.Weighted;

/**
 * centroidal clustering fails in many scenarios; primarily because a single
 * point in combination with a distance metric can only represent a sphere. A
 * reasonable solution is to use multiple well scattered centroids to represent
 * a cluster and has been long in use, see CURE
 * https://en.wikipedia.org/wiki/CURE_algorithm
 *
 * The following example demonstrates the use of a multicentroid clustering; the
 * data corresponds to 2*d clusters in d dimensions (d chosen randomly) such
 * that the clusters almost touch, but remain separable. Note that the knowledge
 * of the true number of clusters is not required -- the clustering is invoked
 * with a maximum of 5*d potential clusters, and yet the example often finds the
 * true 2*d clusters.
 *
 * See {@link ClusterBoxes}. Press <b>t</b> to show or hide the boxes.
 *
 * <p>
 * The step from one centroid to several was motivated by a sphere being too
 * poor a description of a cluster. The step from several centroids to a union
 * of boxes is the same move made once more, and this time the object it
 * produces supports set operations.
 *
 * mvn -pl examples -am install -DskipTests \ java --add-modules
 * jdk.incubator.vector -jar examples/target/*-jar-with-dependencies.jar \
 * rcf_multi_summarize
 */
public class MultiSummarize implements Example {

    public static void main(String[] args) throws Exception {
        new MultiSummarize().run();
    }

    @Override
    public String command() {
        return "multi_summarize";
    }

    @Override
    public String description() {
        return "Example of Multi Summarization";
    }

    @Override
    public void run() throws Exception {
        long seed = new Random().nextLong();
        Random random = new Random(seed);
        int newDimensions = random.nextInt(10) + 3;
        int dataSize = 200000;

        float[][] points = Clusters.getData(dataSize, newDimensions, random.nextInt(), Summarizer::L2distance);

        double epsilon = 0.01;

        Instant t0 = Instant.now();
        List<ICluster<float[]>> summary = Summarizer.multiSummarize(points, 5 * newDimensions, 0.1, true, 5,
                random.nextLong());
        Duration elapsed = Duration.between(t0, Instant.now());

        System.out.println(summary.size() + " clusters for " + newDimensions + " dimensions, seed : " + seed);
        System.out.printf("multiSummarize over %d points (%d dims) took %d ms%n", dataSize, newDimensions,
                elapsed.toMillis());
        double weight = summary.stream().map(e -> e.getWeight()).reduce(Double::sum).get();
        System.out.println(
                "Total weight " + ((float) Math.round(weight * 1000) * 0.001) + " rounding to multiples of " + epsilon);
        System.out.println();

        for (int i = 0; i < summary.size(); i++) {
            double clusterWeight = summary.get(i).getWeight();
            System.out.println(
                    "Cluster " + i + " representatives, weight " + ((float) Math.round(1000 * clusterWeight) * 0.001));
            List<Weighted<float[]>> representatives = summary.get(i).getRepresentatives();
            for (int j = 0; j < representatives.size(); j++) {
                double t = representatives.get(j).weight;
                t = Math.round(1000.0 * t / clusterWeight) * 0.001;
                System.out.print("relative weight " + (float) t + " center (approx)  ");
                printArray(representatives.get(j).index, epsilon);
                System.out.println();
            }
            System.out.println();
        }

        // The forest, at last -- see the class note. It plays no part in finding the
        // clusters; it supplies the measure around the ones already found.
        Instant f0 = Instant.now();
        // timeDecay 0: the data is static, so a uniform reservoir over the whole
        // stream is what is wanted; the default decay would weight the tail of it.
        RandomCutForest forest = RandomCutForest.builder().numberOfTrees(100).sampleSize(256).dimensions(newDimensions)
                .randomSeed(random.nextLong()).centerOfMassEnabled(true).timeDecay(0).build();
        for (float[] point : points) {
            forest.update(point);
        }
        // Two member-sampled families, both built with forMembersAll, which
        // takes one forest traversal per query point and reads all three boxes
        // off it: cutBox, passageBox and stopBox come from the same walk, so
        // asking for them one at a time would pay three walks for what one walk
        // already produced. The drawn family is thinned to COVER_DRAW_QUERIES so
        // the tour stays legible; the reference family is the full budget and is
        // what every printed number below is measured on.
        Map<ClusterBoxes.BoxKind, List<ClusterBoxes.Union>> drawByKind = ClusterBoxes.forMembersAll(forest, summary,
                points, COVER_DRAW_QUERIES, null, CENTER_BOXES, ClusterBoxes.DEFAULT_CORE_QUANTILE);
        Map<ClusterBoxes.BoxKind, List<ClusterBoxes.Union>> refByKind = ClusterBoxes.forMembersAll(forest, summary,
                points, MEMBER_QUERIES, null,CENTER_BOXES, ClusterBoxes.DEFAULT_CORE_QUANTILE);
        List<ClusterBoxes.Union> reference = refByKind.get(DRAWN_KIND);
        System.out.printf("forest over %d points, boxes for %d clusters: %d ms%n", dataSize, summary.size(),
                Duration.between(f0, Instant.now()).toMillis());

        // The boxes at the representatives are still printed, and are still
        // worth printing here in a way they are not in the two-dimensional fan
        // of Summarization. A first-passage box collapses when the query is deep
        // inside its own cluster, because the walk halts at the first node whose
        // bounding box already contains it; in the fan a tree holds fifty points
        // of a blade strung along a line and its middle is genuinely interior.
        // Here a tree holds 256 points over 2d clusters in three to twelve
        // dimensions, so a cluster contributes about twenty points to a tree and
        // nothing is deeply interior -- with that few points in that many
        // dimensions essentially every sampled point is on the hull of its own
        // neighbourhood, the walk runs for several levels from a centroid, and
        // the box is substantial.
        //
        // But that is a statement about extent and it was being read as one
        // about coverage. Drawn as a union it covered 13% of the data, which is
        // a fact about placing five boxes in eleven dimensions and not about the
        // clustering, and the c key used to offer it as the rival to the cover.
        // It no longer does: the query points are fixed at the members and c
        // moves the scale instead. The rows below are the extents as extents.
        System.out.println();
        List<ClusterBoxes.Union> byKind = new ArrayList<>();
        for (ClusterBoxes.BoxKind kind : ClusterBoxes.BoxKind.values()) {
            List<ClusterBoxes.Union> at = ClusterBoxes.forClusters(forest, summary, kind, null,CENTER_BOXES);
            byKind.add(at.isEmpty() ? null : at.get(0));
            System.out.printf("  reps    %-8s %s%n", kind, ClusterBoxes.describe(at));
        }
        for (ClusterBoxes.BoxKind kind : ClusterBoxes.BoxKind.values()) {
            System.out.printf("  members %-8s %s%n", kind, ClusterBoxes.describe(refByKind.get(kind)));
        }

        // The three sizes above are only comparable if they were read from one
        // walk, which is the claim this checks; see ClusterBoxes.nestingViolations.
        if (byKind.get(0) != null) {
            int bad = ClusterBoxes.nestingViolations(byKind.get(0), byKind.get(1), byKind.get(2));
            System.out.printf("  nesting CUT <= PASSAGE <= STOP on cluster 0: %s%n",
                    bad == 0 ? "holds" : bad + " violations");
        }

        // The statistic that is not label-invariant. Everything else printed
        // here -- coverage, volume, overlap -- is a property of the box family
        // and the query budget: permute which union owns which boxes and none of
        // it moves, so a deliberately wrong clustering with the same k scores
        // about the same. reach asks whether a member's own first-passage box
        // contains a representative of the cluster that member was assigned to.
        // It is per point, it is local, and nothing forces a representative into
        // a member's box, so there is no tautology in it.
        ClusterBoxes.reachReport(refByKind, summary);

        // Volumes as the side of the cube of equal volume: a raw volume at these
        // dimensions is unreadable, and equivalent sides are comparable across
        // dimensions in a way volumes are not. Read across the row and the three
        // scales are a ladder on the same query points; PASSAGE over CUT is how
        // much of the neighbourhood lies between the median separation and the
        // 5% tail, and STOP over PASSAGE is what the mass rule was truncating.
        System.out.println();
        System.out.printf("Union volume at %d dimensions, as the side of the cube of equal volume%n", newDimensions);
        System.out.println("                      members, %d queries/cluster".formatted(MEMBER_QUERIES));
        System.out.println("  cluster   boxes      CUT      PASSAGE         STOP     STOP/CUT");
        long volSeed = random.nextLong();
        for (int c = 0; c < reference.size(); c++) {
            double[] side = new double[ClusterBoxes.BoxKind.values().length];
            for (int k = 0; k < side.length; k++) {
                ClusterBoxes.Union un = refByKind.get(ClusterBoxes.BoxKind.values()[k]).get(c);
                double lv = ClusterBoxes.logUnionVolume(un, volSeed + 7919L * k + c, ClusterBoxes.DEFAULT_SAMPLES);
                side[k] = Double.isInfinite(lv) ? 0.0 : Math.exp(lv / newDimensions);
            }
            System.out.printf("  %7d  %6d  %11.4f  %11.4f  %11.4f  %11.2f%n", c, reference.get(c).size(), side[0],
                    side[1], side[2], (side[0] > 0) ? side[2] / side[0] : Double.NaN);
        }

        // Separation as measures, on the reference family. P[x in U_col | x ~
        // Unif(U_row)] is not symmetric and
        // that asymmetry is the point: a small union sitting inside a large one reads
        // 1.0 one way and a small
        // fraction the other. Parts > 1 says the summarizer grouped pieces the measure
        // keeps apart.
        System.out.println();
        System.out.printf("Structure and separation at %s, %d queries/cluster%n", DRAWN_KIND, MEMBER_QUERIES);
        System.out.println("  cluster  boxes  parts  cohesion   worst overlap with");
        double[][] overlap = ClusterBoxes.overlapMatrix(reference, random.nextLong(), PAIR_SAMPLES);
        for (int c = 0; c < reference.size(); c++) {
            int worst = -1;
            for (int b = 0; b < reference.size(); b++) {
                if (b != c && (worst < 0 || overlap[c][b] > overlap[c][worst])) {
                    worst = b;
                }
            }
            System.out.printf("  %7d  %5d  %5d  %8.3f   %s%n", c, reference.get(c).size(),
                    ClusterBoxes.components(reference.get(c)),
                    ClusterBoxes.cohesion(reference.get(c), random.nextLong(), PAIR_SAMPLES),
                    (worst < 0) ? "-" : String.format("%d at %.3f", worst, overlap[c][worst]));
        }

        // The census on the reference family, against a stride of the data: the
        // reference carries thousands of boxes and the census is quadratic in
        // the two.
        float[][] censusSample = stride(points, CENSUS_POINTS);
        Map<ClusterBoxes.BoxKind, int[]> censusByKind = new EnumMap<>(ClusterBoxes.BoxKind.class);
        double total = Math.max(1, censusSample.length);
        System.out.println();
        System.out.printf("Census on the reference cover (%d queries/cluster, %d points)%n", MEMBER_QUERIES,
                censusSample.length);
        for (ClusterBoxes.BoxKind kind : ClusterBoxes.BoxKind.values()) {
            int[] cen = ClusterBoxes.census(refByKind.get(kind), censusSample);
            censusByKind.put(kind, cen);
            System.out.printf("  %-8s covered %.1f%%, two or more %.1f%%, none %.1f%%%n", kind, 100 * cen[0] / total,
                    100 * cen[1] / total, 100 * cen[2] / total);
        }

        // Whether that shortfall is the budget's or the data's. Sub-unions are
        // strides of a stride, so this is the same cover read at a sequence of
        // budgets and costs no extra work in the forest. A curve still climbing
        // at the right-hand end means the number above is an artefact of
        // MEMBER_QUERIES and should not be quoted; a flat one means the points
        // really are outside, which at these dimensions is the expected answer
        // -- the clusters are Gaussian, so they have no boundary, and most of a
        // Gaussian's mass in d dimensions sits in a shell that boxes placed at a
        // one-percent sample of the members will not reach.
        System.out.println();
        System.out.println("Does the cover saturate? census vs query budget, same boxes read at a coarser stride");
        System.out.println("  per cluster   covered    unclaimed");
        for (int div = 16; div >= 1; div /= 2) {
            List<ClusterBoxes.Union> thinned = new ArrayList<>();
            int boxes = 0;
            for (ClusterBoxes.Union un : reference) {
                ClusterBoxes.Union t = ClusterBoxes.everyNth(un, div);
                thinned.add(t);
                boxes += t.size();
            }
            int[] c = ClusterBoxes.census(thinned, censusSample);
            System.out.printf("  %11d   %6.1f%%   %9.1f%%%n", boxes / Math.max(1, reference.size()), 100 * c[0] / total,
                    100 * c[2] / total);
        }
        System.out.println("  read the right-hand end: still falling means the budget, flat means the data");

        boolean showProjection = true;
        if (showProjection) {
            // plotProjection(points, summary, newDimensions);
            plotTourLive(points, summary, drawByKind, censusByKind, newDimensions, random.nextLong());
            // plotTour(points, summary, newDimensions, random.nextLong());
        }
    }

    /**
     * Scale the c key starts on. The query points are the cluster's own members in
     * every case; only the separation probability the box is read at moves.
     *
     */
    private static final ClusterBoxes.BoxKind DRAWN_KIND = ClusterBoxes.BoxKind.PASSAGE;

    /**
     * Off here, unlike in {@link Summarization}. Centring is a concession to a
     * rotating frame and there is no rotating frame in this example.
     */
    private static final boolean CENTER_BOXES = false;

    /**
     * Query budget per cluster for the member-sampled reference. A uniform stride,
     * not a selection -- see {@code ClusterBoxes.subsample} for why that
     * distinction is the whole point. Full membership is out of reach here: 200k
     * points against up to 5d clusters is a traversal per point.
     *
     */
    private static final int MEMBER_QUERIES = 240;

    private static final int COVER_DRAW_QUERIES = 24;

    /** Points used for the reference census; the census is quadratic. */
    private static final int CENSUS_POINTS = 5000;

    /**
     * Karp-Luby budget for the pairwise overlap matrices, which are quadratic in k.
     */
    private static final int PAIR_SAMPLES = 2000;

    /** Uniform stride of the data, for the census only. */
    private static float[][] stride(float[][] points, int cap) {
        if (points.length <= cap) {
            return points;
        }
        int step = (points.length + cap - 1) / cap;
        int n = (points.length + step - 1) / step;
        float[][] out = new float[n][];
        for (int i = 0, j = 0; i < points.length && j < n; i += step, j++) {
            out[j] = points[i];
        }
        return out;
    }

    private void plotProjection(float[][] points, List<ICluster<float[]>> summary, int dim) throws Exception {
        // 1. pick the two highest-variance coordinates from a subsample
        int s = Math.min(points.length, 5000);
        double[] sum = new double[dim], sumSq = new double[dim];
        for (int i = 0; i < s; i++) {
            for (int j = 0; j < dim; j++) {
                sum[j] += points[i][j];
                sumSq[j] += points[i][j] * points[i][j];
            }
        }
        int ax = 0, ay = 1;
        double v0 = -1, v1 = -1;
        for (int j = 0; j < dim; j++) {
            double var = sumSq[j] / s - (sum[j] / s) * (sum[j] / s);
            if (var > v0) {
                v1 = v0;
                ay = ax;
                v0 = var;
                ax = j;
            } else if (var > v1) {
                v1 = var;
                ay = j;
            }
        }

        // 2. subsample, assign each point to its nearest representative's cluster,
        // project to (ax, ay)
        int k = summary.size();
        int stride = Math.max(1, points.length / 4000);
        List<List<float[]>> byCluster = new ArrayList<>();
        for (int c = 0; c < k; c++) {
            byCluster.add(new ArrayList<>());
        }
        for (int i = 0; i < points.length; i += stride) {
            int best = nearestCluster(points[i], summary);
            byCluster.get(best).add(new float[] { points[i][ax], points[i][ay] });
        }

        // 3. range from the projected extent
        double maxAbs = 1e-6;
        for (List<float[]> g : byCluster) {
            for (float[] p : g) {
                maxAbs = Math.max(maxAbs, Math.max(Math.abs(p[0]), Math.abs(p[1])));
            }
        }
        double range = maxAbs * 1.1;

        // 4. scene: points colored by cluster, representatives as big outlined dots
        Plot2D plot = Plot2D.open("RCF Multi-Summarize  (dims " + ax + " vs " + ay + ",  k=" + k + ")", range, 820);
        List<Layer> scene = new ArrayList<>();
        for (int c = 0; c < k; c++) {
            if (!byCluster.get(c).isEmpty()) {
                scene.add(Layers.dots(byCluster.get(c).toArray(new float[0][]), Layers.color(c), 1.8));
            }
        }
        for (int c = 0; c < k; c++) {
            List<double[]> reps = new ArrayList<>();
            for (Weighted<float[]> r : summary.get(c).getRepresentatives()) {
                reps.add(new double[] { r.index[ax], r.index[ay] });
            }
            double[] ones = new double[reps.size()];
            java.util.Arrays.fill(ones, 1.0);
            scene.add(Layers.weightedDots(reps.toArray(new double[0][]), ones, Layers.color(c), 5.0, 0.0));
        }
        plot.render(scene);

        ImageIO.write(plot.renderImage(800, 800, scene), "png", new File("rcf_multi_summarize.png"));
        System.out.println("wrote rcf_multi_summarize.png (projection onto dims " + ax + ", " + ay + ")");
    }

    private static int nearestCluster(float[] p, List<ICluster<float[]>> summary) {
        int best = 0;
        double bd = Double.MAX_VALUE;
        for (int c = 0; c < summary.size(); c++) {
            for (Weighted<float[]> r : summary.get(c).getRepresentatives()) {
                double d = 0;
                for (int j = 0; j < p.length; j++) {
                    double e = p[j] - r.index[j];
                    d += e * e;
                }
                if (d < bd) {
                    bd = d;
                    best = c;
                }
            }
        }
        return best;
    }

    void printArray(float[] values, double epsilon) {
        System.out.print(" [");
        if (abs(values[0]) < epsilon) {
            System.out.print("0");
        } else {
            if (epsilon <= 0) {
                System.out.print(values[0]);
            } else {
                long t = (int) Math.round(values[0] / epsilon);
                System.out.print(t * epsilon);
            }
        }
        for (int i = 1; i < values.length; i++) {
            if (abs(values[i]) < epsilon) {
                System.out.print(", 0");
            } else {
                if (epsilon <= 0) {
                    System.out.print(", " + values[i]);
                } else {
                    long t = Math.round(values[i] / epsilon);
                    System.out.print(", " + t * epsilon);
                }
            }
        }
        System.out.print("]");
    }

    private static double[] randUnit(int dim, Random rnd) {
        double[] x = new double[dim];
        double s = 0;
        for (int i = 0; i < dim; i++) {
            x[i] = rnd.nextGaussian();
            s += x[i] * x[i];
        }
        s = Math.sqrt(s);
        for (int i = 0; i < dim; i++)
            x[i] /= s;
        return x;
    }

    // make u unit, then v orthogonal to u and unit (Gram–Schmidt)
    private static void orthonormalize(double[] u, double[] v) {
        double nu = 0;
        for (double x : u)
            nu += x * x;
        nu = Math.sqrt(nu);
        for (int i = 0; i < u.length; i++)
            u[i] /= nu;
        double d = 0;
        for (int i = 0; i < u.length; i++)
            d += u[i] * v[i];
        for (int i = 0; i < v.length; i++)
            v[i] -= d * u[i];
        double nv = 0;
        for (double x : v)
            nv += x * x;
        nv = Math.sqrt(nv);
        for (int i = 0; i < v.length; i++)
            v[i] /= nv;
    }

    private static double dot(double[] u, float[] p) {
        double s = 0;
        for (int i = 0; i < u.length; i++)
            s += u[i] * p[i];
        return s;
    }

    private void plotTourLive(float[][] points, List<ICluster<float[]>> summary,
            Map<ClusterBoxes.BoxKind, List<ClusterBoxes.Union>> drawByKind,
            Map<ClusterBoxes.BoxKind, int[]> censusByKind, int dim, long seed) {
        int k = summary.size();
        // Same stride the startup census uses, so the c key's figure and the
        // reference figure are taken against the same points and differ only in
        // the boxes.
        float[][] censusSample = stride(points, CENSUS_POINTS);

        // subsample + assign ONCE (full-dim; assignment never changes, only coords do)
        int stride = Math.max(1, points.length / 4000);
        List<float[]> smp = new ArrayList<>();
        List<Integer> asg = new ArrayList<>();
        for (int i = 0; i < points.length; i += stride) {
            smp.add(points[i]);
            asg.add(nearestCluster(points[i], summary));
        }
        float[][] sample = smp.toArray(new float[0][]);
        int[] assign = asg.stream().mapToInt(Integer::intValue).toArray();
        int[][] groups = groupByCluster(assign, k);

        List<float[]> reps = new ArrayList<>();
        List<Integer> repCluster = new ArrayList<>();
        for (int c = 0; c < k; c++)
            for (Weighted<float[]> r : summary.get(c).getRepresentatives()) {
                reps.add(r.index);
                repCluster.add(c);
            }

        // weighted centroid of each cluster (full-dim), computed once
        double[][] centroid = new double[k][dim];
        for (int c = 0; c < k; c++) {
            double wsum = 0;
            for (Weighted<float[]> r : summary.get(c).getRepresentatives()) {
                for (int j = 0; j < dim; j++)
                    centroid[c][j] += r.weight * r.index[j];
                wsum += r.weight;
            }
            if (wsum > 0)
                for (int j = 0; j < dim; j++)
                    centroid[c][j] /= wsum;
        }

        // dominant axis + sign for each cluster
        int[] axis = new int[k], sign = new int[k];
        for (int c = 0; c < k; c++) {
            int best = 0;
            for (int j = 1; j < dim; j++)
                if (Math.abs(centroid[c][j]) > Math.abs(centroid[c][best]))
                    best = j;
            axis[c] = best;
            sign[c] = centroid[c][best] >= 0 ? +1 : -1;
        }

// pair index per axis, in axis order of first appearance (stable)
        Integer[] order = new Integer[k];
        for (int c = 0; c < k; c++)
            order[c] = c;
        java.util.Arrays.sort(order,
                (a, b) -> axis[a] != axis[b] ? Integer.compare(axis[a], axis[b]) : Integer.compare(sign[b], sign[a])); // +
        // before
        // -
        java.util.Map<Integer, Integer> pairOf = new java.util.LinkedHashMap<>();
        for (Integer c : order)
            pairOf.putIfAbsent(axis[c], pairOf.size());
        int nPairs = pairOf.size();

// pair shares a hue; + is saturated, - is pale -> "same family, opposite ends"
        Color[] clusterColor = new Color[k];
        for (int c = 0; c < k; c++) {
            float hue = pairOf.get(axis[c]) / (float) nPairs;
            float sat = sign[c] > 0 ? 0.85f : 0.40f;
            float bri = sign[c] > 0 ? 0.85f : 0.98f;
            clusterColor[c] = Color.getHSBColor(hue, sat, bri);
        }

        Random rnd = new Random(seed);
        double maxAbs = 1e-6; // fixed range so the view doesn't jump
        for (int t = 0; t < 32; t++) {
            double[] w = randUnit(dim, rnd);
            for (float[] p : sample)
                maxAbs = Math.max(maxAbs, Math.abs(dot(w, p)));
        }
        Plot2D plot = Plot2D.open("RCF grand tour  (" + dim + " dims, k=" + k + ")", maxAbs * 1.05, 820);

        // Gated inside the layer rather than around the scene build, so a bare
        // repaint honours the key; see BoxUnionLayer.
        AtomicBoolean showBoxes = new AtomicBoolean(true);
        plot.bindKey(KeyEvent.VK_T, "multisummarize.boxes", () -> showBoxes.set(!showBoxes.get()));

        AtomicReference<ClusterBoxes.BoxKind> kind = new AtomicReference<>(DRAWN_KIND);
        AtomicReference<List<ClusterBoxes.Union>> drawn = new AtomicReference<>(drawByKind.get(DRAWN_KIND));
        plot.bindKey(KeyEvent.VK_C, "multisummarize.boxkind", () -> {
            ClusterBoxes.BoxKind[] all = ClusterBoxes.BoxKind.values();
            ClusterBoxes.BoxKind next = all[(kind.get().ordinal() + 1) % all.length];
            kind.set(next);
            drawn.set(drawByKind.get(next));
            int boxes = drawn.get().stream().mapToInt(ClusterBoxes.Union::size).sum();

            int[] c = ClusterBoxes.census(drawn.get(), censusSample);
            double n = Math.max(1, censusSample.length);
            System.out.printf("scale -> %-8s (%s)%n", next, next.blurb());
            System.out.printf("        drawn %d boxes, %d/cluster: covered %.1f%%, two or more %.1f%%, none %.1f%%%n",
                    boxes, boxes / Math.max(1, drawn.get().size()), 100 * c[0] / n, 100 * c[1] / n, 100 * c[2] / n);
            int[] ref = censusByKind.get(next);
            if (ref != null) {
                System.out.printf("        reference at %d/cluster covers %.1f%% (the difference is the draw "
                        + "budget, not the measure)%n", MEMBER_QUERIES, 100 * ref[0] / n);
            }
            // reach on the boxes being drawn, so the caption on the picture is
            // the statistic the picture can be wrong about. Coverage cannot: it
            // does not know which union owns which boxes.
            int[] t = ClusterBoxes.reachTotal(ClusterBoxes.reach(drawn.get(), summary));
            double nb = Math.max(1, t[4]);
            System.out.printf("        reach: own %.1f%% (ambiguous %.1f%%), other-only %.1f%%, none %.1f%%%n",
                    100 * t[ClusterBoxes.REACH_OWN] / nb, 100 * t[ClusterBoxes.REACH_AMBIGUOUS] / nb,
                    100 * t[ClusterBoxes.REACH_OTHER_ONLY] / nb, 100 * t[ClusterBoxes.REACH_NONE] / nb);
        });
        System.out.println("keys: space pauses, right arrow steps, t toggles the density boxes, "
                + "c cycles the scale (CUT/PASSAGE/STOP) and prints reach for it");

        Thread tour = new Thread(() -> {
            double[] u = randUnit(dim, rnd), v = randUnit(dim, rnd);
            orthonormalize(u, v);
            int stepsPerLeg = 40;
            boolean rotateV = true;
            try {
                while (true) {
                    double[] moving = rotateV ? v : u;
                    double[] w = freshDir(dim, u, v, rnd); // ⟂ to both u and v
                    for (int s = 0; s <= stepsPerLeg; s++) {
                        double th = (Math.PI / 2) * s / stepsPerLeg;
                        double cs = Math.cos(th), sn = Math.sin(th);
                        double[] mov = new double[dim];
                        for (int i = 0; i < dim; i++)
                            mov[i] = cs * moving[i] + sn * w[i];
                        double[] uu = rotateV ? u : mov;
                        double[] vv = rotateV ? mov : v;
                        plot.render(buildScene(sample, groups, reps, repCluster, centroid, clusterColor, k, uu, vv,
                                drawn.get(), showBoxes));
                        plot.awaitResume();
                        Thread.sleep(25);
                    }
                    if (rotateV)
                        v = w;
                    else
                        u = w; // commit the folded-in axis
                    orthonormalize(u, v);
                    rotateV = !rotateV; // alternate which axis rotates
                }
            } catch (InterruptedException ignored) {
            }
        }, "grand-tour");
        tour.setDaemon(true);
        tour.start();
    }

    private static double[] freshDir(int dim, double[] u, double[] v, Random rnd) {
        double[] w = new double[dim];
        for (int i = 0; i < dim; i++)
            w[i] = rnd.nextGaussian();
        double du = 0, dv = 0;
        for (int i = 0; i < dim; i++) {
            du += w[i] * u[i];
            dv += w[i] * v[i];
        }
        double n = 0;
        for (int i = 0; i < dim; i++) {
            w[i] -= du * u[i] + dv * v[i];
            n += w[i] * w[i];
        }
        n = Math.sqrt(n);
        for (int i = 0; i < dim; i++)
            w[i] /= n;
        return w;
    }

    private List<Layer> buildScene(float[][] sample, int[][] groups, List<float[]> reps, List<Integer> repCluster,
            double[][] centroid, Color[] clusterColor, int k, double[] u, double[] v, List<ClusterBoxes.Union> unions,
            AtomicBoolean showBoxes) {
        List<Layer> scene = new ArrayList<>();

        // First, so the boxes sit behind the data. The shadow of an axis-parallel box
        // under an arbitrary projection is a zonogon, not a rectangle -- drawing a
        // rectangle would understate it and make the clusters look better separated
        // than they are.
        if (unions != null && !unions.isEmpty()) {
            List<double[][]> polygons = new ArrayList<>();
            List<Integer> owner = new ArrayList<>();
            for (int c = 0; c < Math.min(k, unions.size()); c++) {
                for (double[][] poly : ClusterBoxes.project(unions.get(c), u, v)) {
                    polygons.add(poly);
                    owner.add(c);
                }
            }
            if (!polygons.isEmpty()) {
                scene.add(new BoxUnionLayer(polygons, owner.stream().mapToInt(Integer::intValue).toArray(),
                        clusterColor, showBoxes::get, 26, 90));
            }
        }

        for (int c = 0; c < k; c++) {
            int[] g = groups[c];
            if (g.length == 0)
                continue;
            float[][] xy = new float[g.length][];
            for (int m = 0; m < g.length; m++)
                xy[m] = new float[] { (float) dot(u, sample[g[m]]), (float) dot(v, sample[g[m]]) };
            scene.add(Layers.dots(xy, clusterColor[c], 1.6));
        }
        for (int c = 0; c < k; c++) {
            double cx = 0, cy = 0;
            for (int j = 0; j < u.length; j++) {
                cx += u[j] * centroid[c][j];
                cy += v[j] * centroid[c][j];
            }
            scene.add(Layers.arrows(new double[][] { { 0, 0 } }, new double[][] { { cx, cy } }, clusterColor[c], 2.0f));
        }
        for (int c = 0; c < k; c++) {
            List<double[]> rr = new ArrayList<>();
            for (int r = 0; r < reps.size(); r++)
                if (repCluster.get(r) == c)
                    rr.add(new double[] { dot(u, reps.get(r)), dot(v, reps.get(r)) });
            double[] ones = new double[rr.size()];
            java.util.Arrays.fill(ones, 1.0);
            scene.add(Layers.weightedDots(rr.toArray(new double[0][]), ones, clusterColor[c], 5.0, 0.0));
        }
        return scene;
    }

    private static int[][] groupByCluster(int[] assign, int k) {
        int[] cnt = new int[k];
        for (int a : assign)
            cnt[a]++;
        int[][] groups = new int[k][];
        for (int c = 0; c < k; c++)
            groups[c] = new int[cnt[c]];
        int[] w = new int[k];
        for (int q = 0; q < assign.length; q++)
            groups[assign[q]][w[assign[q]]++] = q;
        return groups;
    }

}