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
import java.io.File;
import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import javax.imageio.ImageIO;

import org.streamingalgorithms.randomcutforest.examples.Example;
import org.streamingalgorithms.randomcutforest.examples.datasets.Clusters;
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
 */
/*
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

        boolean showProjection = true;
        if (showProjection) {
            // plotProjection(points, summary, newDimensions);
            plotTourLive(points, summary, newDimensions, random.nextLong());
            // plotTour(points, summary, newDimensions, random.nextLong());
        }
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

    private void plotTourLive(float[][] points, List<ICluster<float[]>> summary, int dim, long seed) {
        int k = summary.size();

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
                        plot.render(buildScene(sample, groups, reps, repCluster, centroid, clusterColor, k, uu, vv));
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
            double[][] centroid, Color[] clusterColor, int k, double[] u, double[] v) {
        List<Layer> scene = new ArrayList<>();
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
