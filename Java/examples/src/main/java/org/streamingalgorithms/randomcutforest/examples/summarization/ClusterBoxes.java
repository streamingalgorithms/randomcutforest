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

import java.util.ArrayList;
import java.util.Arrays;
import java.util.EnumMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import org.streamingalgorithms.randomcutforest.RandomCutForest;
import org.streamingalgorithms.randomcutforest.returntypes.AnisotropicDensityOutput;
import org.streamingalgorithms.randomcutforest.summarization.ICluster;
import org.streamingalgorithms.randomcutforest.util.Weighted;

/**
 * A cluster as a <i>union of density boxes</i>, and the set operations that
 * union supports.
 *
 * <p>
 * A representative plus a radius is a sphere; what the forest knows around a
 * point is an anisotropic box, read off the first-passage distribution of the
 * random cuts ({@link RandomCutForest#getAnisotropicDensity}). So a cluster is
 * U_c = union over query points q of B(q), a measurable set: volume says how
 * much space it claims, intersection with another cluster's union says whether
 * the two are separated as measures, and a census says what fraction of the
 * data no union claims -- the one number a convex hull can never report.
 *
 * <p>
 * Nothing here returns a raw volume; they overflow in d dimensions long before
 * they stop being meaningful. Sizes come back as a log volume or as the side of
 * the cube of equal volume, and overlap comes back as a probability.
 */
public final class ClusterBoxes {

    private ClusterBoxes() {
    }

    /** Default Monte-Carlo budget; the estimators are O(samples * boxes * d). */
    public static final int DEFAULT_SAMPLES = 20000;

    /**
     * Where along the walk to read the box.
     *
     * <p>
     * Each level of a leaf-to-root walk carries a local cut probability q; the
     * cumulative quantity is the survival S = prod (1 - q) over the levels above,
     * and P[separated] = 1 - S. The three boxes are the geometry where that
     * probability crosses a fixed value, so a lower value crosses higher up and
     * gives a larger box:
     *
     * <pre>
     *   CUT      P[separated] = 0.50, capped at the mass level
     *   PASSAGE  P[separated] = 0.05, capped at the mass level
     *   STOP     P[separated] = 0.05, uncapped
     * </pre>
     *
     * These are separation probabilities, not quantiles and not local cut
     * probabilities. PASSAGE and STOP are read at the same 0.05 and differ only in
     * the mass rule, so STOP is PASSAGE with the truncation lifted and the gap
     * between them is a readout of local mass rather than a second scale. CUT is
     * contained in PASSAGE is contained in STOP on every tree, since all three are
     * read from one buffered path.
     */
    public enum BoxKind {
        /** P[separated] = 0.50, mass-bounded. The margin scale. */
        CUT("P[sep] 0.50, mass-bounded -- the margin scale"),
        /** P[separated] = 0.05, mass-bounded. The cover scale. */
        PASSAGE("P[sep] 0.05, mass-bounded -- the cover scale"),
        /** P[separated] = 0.05 with the mass rule lifted. */
        STOP("P[sep] 0.05, unbounded -- PASSAGE without the mass rule");

        private final String blurb;

        BoxKind(String blurb) {
            this.blurb = blurb;
        }

        /** One line for a log, so a key press says what it changed. */
        public String blurb() {
            return blurb;
        }
    }

    // ------------------------------------------------------------------
    // the union
    // ------------------------------------------------------------------

    /**
     * One cluster's union of boxes, stored as corners rather than as the
     * query-relative DiVector layout the forest returns, because every operation
     * below wants corners.
     */
    public static final class Union {
        /** Query points the boxes were read at, m x d. */
        public final double[][] origin;
        /** Lower corners, m x d. */
        public final double[][] lo;
        /** Upper corners, m x d. */
        public final double[][] hi;
        /** Natural log of each box's volume over its axes of positive width. */
        public final double[] logVolume;
        /** Axes of positive width, per box. */
        public final int[] active;

        /**
         * Bounding box of the whole union, as an exact reject for {@link #contains}.
         */
        private final double[] bbLo;
        private final double[] bbHi;

        /** Whether these boxes were symmetrised about their query points. */
        public final boolean centered;

        /**
         * Mean over boxes and axes of {@code |plus - minus| / (plus + minus)}, the
         * one-sided extents as the forest returned them, before any centring.
         */
        public final double meanDrift;

        Union(List<double[]> origins, List<double[]> los, List<double[]> his) {
            this(origins, los, his, false, 0.0);
        }

        Union(List<double[]> origins, List<double[]> los, List<double[]> his, boolean centered, double meanDrift) {
            this.centered = centered;
            this.meanDrift = meanDrift;
            int m = origins.size();
            origin = origins.toArray(new double[0][]);
            lo = los.toArray(new double[0][]);
            hi = his.toArray(new double[0][]);
            logVolume = new double[m];
            active = new int[m];
            int d = (m > 0) ? lo[0].length : 0;
            bbLo = new double[d];
            bbHi = new double[d];
            Arrays.fill(bbLo, Double.POSITIVE_INFINITY);
            Arrays.fill(bbHi, Double.NEGATIVE_INFINITY);
            for (int j = 0; j < m; j++) {
                double acc = 0.0;
                int a = 0;
                for (int i = 0; i < lo[j].length; i++) {
                    double w = hi[j][i] - lo[j][i];
                    if (w > 0.0) {
                        acc += Math.log(w);
                        a++;
                    }
                    bbLo[i] = Math.min(bbLo[i], lo[j][i]);
                    bbHi[i] = Math.max(bbHi[i], hi[j][i]);
                }
                active[j] = a;
                logVolume[j] = (a > 0) ? acc : Double.NEGATIVE_INFINITY;
            }
        }

        private boolean outsideBounds(double[] x) {
            for (int i = 0; i < bbLo.length; i++) {
                if (x[i] < bbLo[i] || x[i] > bbHi[i]) {
                    return true;
                }
            }
            return false;
        }

        private boolean outsideBounds(float[] x) {
            for (int i = 0; i < bbLo.length; i++) {
                if (x[i] < bbLo[i] || x[i] > bbHi[i]) {
                    return true;
                }
            }
            return false;
        }

        public int size() {
            return lo.length;
        }

        public boolean isEmpty() {
            return lo.length == 0;
        }

        public boolean contains(double[] x) {
            if (lo.length == 0 || outsideBounds(x)) {
                return false;
            }
            for (int j = 0; j < lo.length; j++) {
                if (inBox(x, j)) {
                    return true;
                }
            }
            return false;
        }

        public boolean contains(float[] x) {
            if (lo.length == 0 || outsideBounds(x)) {
                return false;
            }
            for (int j = 0; j < lo.length; j++) {
                if (inBox(x, j)) {
                    return true;
                }
            }
            return false;
        }

        boolean inBox(double[] x, int j) {
            for (int i = 0; i < x.length; i++) {
                if (x[i] < lo[j][i] || x[i] > hi[j][i]) {
                    return false;
                }
            }
            return true;
        }

        boolean inBox(float[] x, int j) {
            for (int i = 0; i < x.length; i++) {
                if (x[i] < lo[j][i] || x[i] > hi[j][i]) {
                    return false;
                }
            }
            return true;
        }
    }

    /**
     * Reads one box per representative of each cluster and collects them into a
     * union.
     */
    public static List<Union> forClusters(RandomCutForest forest, List<ICluster<float[]>> summary, BoxKind kind,
            boolean centered) {
        List<Union> out = new ArrayList<>();
        for (ICluster<float[]> cluster : summary) {
            List<float[]> queries = new ArrayList<>();
            for (Weighted<float[]> rep : cluster.getRepresentatives()) {
                queries.add(rep.index);
            }
            out.add(forQueries(forest, queries, kind, centered));
        }
        return out;
    }

    /**
     * A union of the boxes at an arbitrary set of query points.
     */

    private static Union forQueries(RandomCutForest forest, List<float[]> queries, BoxKind kind, boolean centered) {
        List<double[]> origins = new ArrayList<>();
        List<double[]> los = new ArrayList<>();
        List<double[]> his = new ArrayList<>();
        double driftSum = 0.0;
        int driftTerms = 0;
        for (float[] query : queries) {
            AnisotropicDensityOutput density = forest.getAnisotropicDensity(query);
            if (!density.isReliable()) {
                continue;
            }
            double[] box = boxOf(density, kind);
            int d = query.length;
            double[] o = new double[d];
            double[] l = new double[d];
            double[] h = new double[d];
            for (int i = 0; i < d; i++) {
                o[i] = query[i];
                // DiVector layout: box[i] is the extent toward MINUS i,
                // box[i + d] the extent toward PLUS i. Getting this backwards
                // puts the box on the far side of the query.
                double width = box[i] + box[i + d];
                if (width > 0.0) {
                    driftSum += Math.abs(box[i + d] - box[i]) / width;
                    driftTerms++;
                }
                if (centered) {
                    // Half of the sum, so the width on this axis is exactly what
                    // the forest reported and only the position moves. Never the
                    // max of the two, which would inflate every box.
                    double half = 0.5 * width;
                    l[i] = o[i] - half;
                    h[i] = o[i] + half;
                } else {
                    l[i] = o[i] - box[i];
                    h[i] = o[i] + box[i + d];
                }
            }
            origins.add(o);
            los.add(l);
            his.add(h);
        }
        return new Union(origins, los, his, centered, (driftTerms == 0) ? 0.0 : driftSum / driftTerms);
    }

    /**
     * All three boxes at one set of queries, from one traversal each.
     */
    private static Map<BoxKind, Union> forQueriesAll(RandomCutForest forest, List<float[]> queries, boolean centered) {
        BoxKind[] kinds = BoxKind.values();
        List<List<double[]>> origins = new ArrayList<>();
        List<List<double[]>> los = new ArrayList<>();
        List<List<double[]>> his = new ArrayList<>();
        for (int k = 0; k < kinds.length; k++) {
            origins.add(new ArrayList<>());
            los.add(new ArrayList<>());
            his.add(new ArrayList<>());
        }
        double driftSum = 0.0;
        int driftTerms = 0;
        for (float[] query : queries) {
            AnisotropicDensityOutput density = forest.getAnisotropicDensity(query);
            if (!density.isReliable()) {
                continue;
            }
            int d = query.length;
            for (int k = 0; k < kinds.length; k++) {
                double[] box = boxOf(density, kinds[k]);
                double[] o = new double[d];
                double[] l = new double[d];
                double[] h = new double[d];
                for (int i = 0; i < d; i++) {
                    o[i] = query[i];
                    double width = box[i] + box[i + d];
                    if (k == 0 && width > 0.0) {
                        driftSum += Math.abs(box[i + d] - box[i]) / width;
                        driftTerms++;
                    }
                    if (centered) {
                        double half = 0.5 * width;
                        l[i] = o[i] - half;
                        h[i] = o[i] + half;
                    } else {
                        l[i] = o[i] - box[i];
                        h[i] = o[i] + box[i + d];
                    }
                }
                origins.get(k).add(o);
                los.get(k).add(l);
                his.get(k).add(h);
            }
        }
        double drift = (driftTerms == 0) ? 0.0 : driftSum / driftTerms;
        Map<BoxKind, Union> out = new EnumMap<>(BoxKind.class);
        for (int k = 0; k < kinds.length; k++) {
            out.put(kinds[k], new Union(origins.get(k), los.get(k), his.get(k), centered, drift));
        }
        return out;
    }

    /**
     * A union built from every {@code stride}-th box of another.
     */
    public static Union everyNth(Union u, int stride) {
        List<double[]> o = new ArrayList<>();
        List<double[]> l = new ArrayList<>();
        List<double[]> h = new ArrayList<>();
        for (int j = 0; j < u.size(); j += Math.max(1, stride)) {
            o.add(u.origin[j]);
            l.add(u.lo[j]);
            h.add(u.hi[j]);
        }
        return new Union(o, l, h, u.centered, u.meanDrift);
    }

    /**
     * Counts violations of {@code CUT} &sube; {@code PASSAGE} &sube; {@code STOP}
     * over three unions read at the same query points, as (box, axis, side)
     * triples.
     */
    public static int nestingViolations(Union cut, Union passage, Union stop) {
        if (cut.size() != passage.size() || passage.size() != stop.size()) {
            return 0;
        }
        int bad = 0;
        for (int j = 0; j < cut.size(); j++) {
            for (int i = 0; i < cut.lo[j].length; i++) {
                if (cut.lo[j][i] < passage.lo[j][i] || passage.lo[j][i] < stop.lo[j][i]) {
                    bad++;
                }
                if (cut.hi[j][i] > passage.hi[j][i] || passage.hi[j][i] > stop.hi[j][i]) {
                    bad++;
                }
            }
        }
        return bad;
    }

    /**
     * Unions built from the cluster's own members rather than its centroids.
     *
     * <p>
     * A box at a centroid degenerates: the walk halts at the first node whose
     * bounding box already contains the query, which for an interior point is two
     * or three levels above the leaf. So centroid boxes describe the spacing
     * between nearby points, not the cluster, and they are not a cover however high
     * the level is set. A cover wants many query points spread through the cluster
     * -- interior boxes are small but tile, rim boxes reach outward.
     *
     * @param perCluster query budget per cluster, one forest traversal each; 0
     *                   means every member
     */
    public static List<Union> forMembers(RandomCutForest forest, List<ICluster<float[]>> summary, float[][] points,
            int perCluster, BoxKind kind, boolean centered, double coreQuantile) {
        List<List<float[]>> queries = memberQueries(summary, points, perCluster, coreQuantile);
        List<Union> out = new ArrayList<>();
        for (List<float[]> q : queries) {
            out.add(forQueries(forest, q, kind, centered));
        }
        return out;
    }

    /**
     * All three kinds at the members of every cluster, from one traversal per query
     * point.
     */
    public static Map<BoxKind, List<Union>> forMembersAll(RandomCutForest forest, List<ICluster<float[]>> summary,
            float[][] points, int perCluster, boolean centered, double coreQuantile) {
        List<List<float[]>> queries = memberQueries(summary, points, perCluster, coreQuantile);
        Map<BoxKind, List<Union>> out = new EnumMap<>(BoxKind.class);
        for (BoxKind kind : BoxKind.values()) {
            out.put(kind, new ArrayList<>());
        }
        for (List<float[]> q : queries) {
            Map<BoxKind, Union> byKind = forQueriesAll(forest, q, centered);
            for (BoxKind kind : BoxKind.values()) {
                out.get(kind).add(byKind.get(kind));
            }
        }
        return out;
    }

    /**
     * The query points {@link #forMembers} would use, per cluster, without touching
     * the forest.
     */
    private static List<List<float[]>> memberQueries(List<ICluster<float[]>> summary, float[][] points, int perCluster,
            double coreQuantile) {
        int k = summary.size();
        List<List<float[]>> members = new ArrayList<>();
        List<List<Double>> spread = new ArrayList<>();
        for (int c = 0; c < k; c++) {
            members.add(new ArrayList<>());
            spread.add(new ArrayList<>());
        }
        for (float[] p : points) {
            int best = nearestCluster(p, summary);
            if (best >= 0) {
                members.get(best).add(p);
                spread.get(best).add(distanceToCluster(p, summary.get(best)));
            }
        }
        List<List<float[]>> out = new ArrayList<>();
        for (int c = 0; c < k; c++) {
            List<float[]> all = members.get(c);
            List<Double> distance = spread.get(c);
            List<float[]> queries = new ArrayList<>();
            if (!all.isEmpty()) {
                double cutoff = quantile(distance, coreQuantile);
                // Filter first, then stride over what survives in its original
                // order: striding a distance-sorted list would pile every query
                // into the middle of the cluster, which is the one place the box
                // has nothing to say.
                List<float[]> core = new ArrayList<>(all.size());
                for (int i = 0; i < all.size(); i++) {
                    if (distance.get(i) <= cutoff) {
                        core.add(all.get(i));
                    }
                }
                if (core.isEmpty()) {
                    core = all;
                }
                queries = subsample(core, perCluster);
            }
            out.add(queries);
        }
        return out;
    }

    /**
     * Fraction of a cluster's members used as query points, nearest first.
     *
     * <p>
     * Nearest-representative assignment hands every point to some cluster,
     * including strays that belong to none. Their boxes are <em>large</em> -- a
     * point alone in a void is separated by the first cut that comes near it --
     * which is the right answer about a void and the wrong one for a cover, since
     * one stray adds a box the size of the empty region. Cutting the far tail is
     * therefore what makes the census readable, and it is also why an uncovered
     * point is a statement about the budget rather than about density.
     */
    public static final double DEFAULT_CORE_QUANTILE = 0.95;

    private static double distanceToCluster(float[] p, ICluster<float[]> cluster) {
        double best = Double.MAX_VALUE;
        for (Weighted<float[]> rep : cluster.getRepresentatives()) {
            double acc = 0;
            for (int i = 0; i < p.length; i++) {
                double e = p[i] - rep.index[i];
                acc += e * e;
            }
            best = Math.min(best, acc);
        }
        return Math.sqrt(best);
    }

    private static double quantile(List<Double> values, double q) {
        if (values.isEmpty()) {
            return Double.MAX_VALUE;
        }
        double[] sorted = new double[values.size()];
        for (int i = 0; i < sorted.length; i++) {
            sorted[i] = values.get(i);
        }
        Arrays.sort(sorted);
        int index = (int) Math.floor(Math.max(0.0, Math.min(1.0, q)) * (sorted.length - 1));
        return sorted[index];
    }

    /**
     * A uniform subsample of the members, or all of them when {@code budget} is
     * zero or less.
     */
    private static List<float[]> subsample(List<float[]> points, int budget) {
        if (budget <= 0 || budget >= points.size()) {
            return points;
        }
        List<float[]> out = new ArrayList<>(budget);
        int stride = Math.max(1, points.size() / budget);
        for (int i = 0; i < points.size() && out.size() < budget; i += stride) {
            out.add(points.get(i));
        }
        return out;
    }

    /** Index of the cluster owning the representative closest to p, or -1. */
    public static int nearestCluster(float[] p, List<ICluster<float[]>> summary) {
        int best = -1;
        double bestDistance = Double.MAX_VALUE;
        for (int c = 0; c < summary.size(); c++) {
            for (Weighted<float[]> rep : summary.get(c).getRepresentatives()) {
                double acc = 0;
                for (int i = 0; i < p.length; i++) {
                    double e = p[i] - rep.index[i];
                    acc += e * e;
                }
                if (acc < bestDistance) {
                    bestDistance = acc;
                    best = c;
                }
            }
        }
        return best;
    }

    /**
     * Why centring is the default here, when {@code BoxLayer} argues for keeping
     * the drift.
     */
    private static double[] boxOf(AnisotropicDensityOutput density, BoxKind kind) {
        switch (kind) {
        case CUT:
            return density.cutBox();
        case STOP:
            return density.stopBox();
        case PASSAGE:
        default:
            return density.passageBox();
        }
    }

    // ------------------------------------------------------------------
    // measure of a union
    // ------------------------------------------------------------------

    /**
     * Karp-Luby estimate of log(volume of the union).
     *
     * <p>
     * Sample a box with probability proportional to its volume, sample a point
     * uniformly in it, and accept only if no earlier box contains the point. The
     * acceptance rate is at least 1/m, so the estimator is unbiased and its
     * relative error does not depend on the dimension. Carried in logs throughout,
     * since the sum of box volumes overflows well before the answer stops being
     * meaningful.
     */
    public static double logUnionVolume(Union u, long seed, int samples) {
        if (u.isEmpty()) {
            return Double.NEGATIVE_INFINITY;
        }
        double logTotal = logSumExp(u.logVolume);
        if (Double.isInfinite(logTotal)) {
            return Double.NEGATIVE_INFINITY;
        }
        Random random = new Random(seed);
        double[] cdf = normalizedCdf(u.logVolume, logTotal);
        int accepted = 0;
        double[] x = new double[u.lo[0].length];
        for (int s = 0; s < samples; s++) {
            int j = pick(cdf, random.nextDouble());
            sampleInBox(u, j, random, x);
            if (firstContaining(u, x) == j) {
                accepted++;
            }
        }
        return (accepted == 0) ? Double.NEGATIVE_INFINITY : logTotal + Math.log(accepted / (double) samples);
    }

    /**
     * The fraction of {@code a}'s union that also lies in {@code b}'s.
     */
    public static double overlapFraction(Union a, Union b, long seed, int samples) {
        if (a.isEmpty() || b.isEmpty()) {
            return 0.0;
        }
        double logTotal = logSumExp(a.logVolume);
        if (Double.isInfinite(logTotal)) {
            return Double.NaN;
        }
        Random random = new Random(seed);
        double[] cdf = normalizedCdf(a.logVolume, logTotal);
        int drawn = 0;
        int shared = 0;
        double[] x = new double[a.lo[0].length];
        for (int s = 0; s < samples; s++) {
            int j = pick(cdf, random.nextDouble());
            sampleInBox(a, j, random, x);
            if (firstContaining(a, x) != j) {
                continue; // counted through its lowest-indexed box instead
            }
            drawn++;
            if (b.contains(x)) {
                shared++;
            }
        }
        return (drawn == 0) ? Double.NaN : shared / (double) drawn;
    }

    /**
     * How much of a cluster's own union is claimed by more than one of its boxes.
     */
    public static double cohesion(Union u, long seed, int samples) {
        if (u.size() < 2) {
            return 0.0;
        }
        double logTotal = logSumExp(u.logVolume);
        if (Double.isInfinite(logTotal)) {
            return 0.0;
        }
        Random random = new Random(seed);
        double[] cdf = normalizedCdf(u.logVolume, logTotal);
        int drawn = 0;
        int multiple = 0;
        double[] x = new double[u.lo[0].length];
        for (int s = 0; s < samples; s++) {
            int j = pick(cdf, random.nextDouble());
            sampleInBox(u, j, random, x);
            if (firstContaining(u, x) != j) {
                continue; // this point belongs to an earlier box's share
            }
            drawn++;
            int hits = 0;
            for (int b = 0; b < u.size() && hits < 2; b++) {
                if (u.inBox(x, b)) {
                    hits++;
                }
            }
            if (hits > 1) {
                multiple++;
            }
        }
        return (drawn == 0) ? 0.0 : multiple / (double) drawn;
    }

    /**
     * Connected components of a union, two boxes being adjacent when they
     * intersect.
     */
    public static int components(Union u) {
        int m = u.size();
        if (m == 0) {
            return 0;
        }
        int[] parent = new int[m];
        for (int i = 0; i < m; i++) {
            parent[i] = i;
        }
        for (int i = 0; i < m; i++) {
            for (int j = i + 1; j < m; j++) {
                if (intersects(u, i, j)) {
                    union(parent, i, j);
                }
            }
        }
        int roots = 0;
        for (int i = 0; i < m; i++) {
            if (find(parent, i) == i) {
                roots++;
            }
        }
        return roots;
    }

    private static boolean intersects(Union u, int a, int b) {
        for (int i = 0; i < u.lo[a].length; i++) {
            if (u.lo[a][i] > u.hi[b][i] || u.lo[b][i] > u.hi[a][i]) {
                return false;
            }
        }
        return true;
    }

    private static int find(int[] parent, int i) {
        while (parent[i] != i) {
            parent[i] = parent[parent[i]];
            i = parent[i];
        }
        return i;
    }

    private static void union(int[] parent, int a, int b) {
        int ra = find(parent, a);
        int rb = find(parent, b);
        if (ra != rb) {
            parent[rb] = ra;
        }
    }

    /**
     * How the data falls across the unions: {covered by at least one, claimed by
     * two or more, claimed by none}.
     *
     * <p>
     * Containment is closed, so a point exactly on a face counts as inside. All
     * three figures are label-invariant: permute which union owns which boxes and
     * none of them moves, so they grade the box family and the query budget rather
     * than the clustering. {@link #reach} is the one that does not.
     */
    public static int[] census(List<Union> unions, float[][] points) {
        return censusOf(hitCounts(unions, points), points.length);
    }

    /**
     * How many unions contain each point.
     */
    public static int[] hitCounts(List<Union> unions, float[][] points) {
        int[] hits = new int[points.length];
        for (int i = 0; i < points.length; i++) {
            int count = 0;
            for (Union u : unions) {
                if (u.contains(points[i])) {
                    count++;
                }
            }
            hits[i] = count;
        }
        return hits;
    }

    /** {covered, claimed by two or more, claimed by none} from hit counts. */
    public static int[] censusOf(int[] hits, int total) {
        int covered = 0;
        int ambiguous = 0;
        for (int h : hits) {
            if (h > 0) {
                covered++;
            }
            if (h > 1) {
                ambiguous++;
            }
        }
        return new int[] { covered, ambiguous, total - covered };
    }

    // ------------------------------------------------------------------
    // projection: the shadow of a box is a zonogon
    // ------------------------------------------------------------------

    /**
     * The exact outline of an axis-parallel box seen through a linear projection
     * onto {@code (u, v)}: a zonogon, not a rectangle. Built as the Minkowski sum
     * of the projected edge vectors, sorted by angle, so it is the set on screen
     * rather than a bounding box of it.
     */
    public static double[][] projectBox(double[] lo, double[] hi, double[] u, double[] v) {
        int d = lo.length;
        double cx = 0.0;
        double cy = 0.0;
        List<double[]> gen = new ArrayList<>(d);
        for (int i = 0; i < d; i++) {
            double c = 0.5 * (hi[i] + lo[i]);
            double half = 0.5 * (hi[i] - lo[i]);
            cx += u[i] * c;
            cy += v[i] * c;
            if (half <= 0.0) {
                continue;
            }
            double gx = u[i] * half;
            double gy = v[i] * half;
            if (gy < 0 || (gy == 0 && gx < 0)) { // fold into the upper half plane
                gx = -gx;
                gy = -gy;
            }
            if (gx != 0.0 || gy != 0.0) {
                gen.add(new double[] { gx, gy });
            }
        }
        if (gen.isEmpty()) {
            return new double[][] { { cx, cy } };
        }
        gen.sort((p, q) -> Double.compare(Math.atan2(p[1], p[0]), Math.atan2(q[1], q[0])));

        double sx = 0.0;
        double sy = 0.0;
        for (double[] g : gen) {
            sx += g[0];
            sy += g[1];
        }
        int n = gen.size();
        double[][] poly = new double[2 * n][2];
        double x = cx - sx;
        double y = cy - sy;
        for (int i = 0; i < n; i++) {
            x += 2 * gen.get(i)[0];
            y += 2 * gen.get(i)[1];
            poly[i][0] = x;
            poly[i][1] = y;
        }
        for (int i = 0; i < n; i++) { // central symmetry closes the ring
            poly[n + i][0] = 2 * cx - poly[i][0];
            poly[n + i][1] = 2 * cy - poly[i][1];
        }
        return poly;
    }

    /** Every box of a union, projected. */
    public static List<double[][]> project(Union u, double[] a, double[] b) {
        List<double[][]> out = new ArrayList<>(u.size());
        for (int j = 0; j < u.size(); j++) {
            out.add(projectBox(u.lo[j], u.hi[j], a, b));
        }
        return out;
    }

    /** Axis-aligned convenience for a 2-D plot: project onto coordinates ax, ay. */
    public static List<double[][]> projectAxes(Union u, int ax, int ay, int dimensions) {
        double[] a = new double[dimensions];
        double[] b = new double[dimensions];
        a[ax] = 1.0;
        b[ay] = 1.0;
        return project(u, a, b);
    }

    // ------------------------------------------------------------------
    // reach: does a member's own neighbourhood contain its claimed centre
    // ------------------------------------------------------------------

    /**
     * Indices into the array {@link #reach} returns.
     *
     * <p>
     * Does a member's own first-passage box contain a representative <em>of the
     * cluster that member was assigned to</em>. Unlike the census this is not
     * label-invariant, so it is the one statistic here that a wrong clustering can
     * fail; it is per point, and it has no tautology -- a query is trivially inside
     * its own box, but nothing makes a representative land there. A member that
     * reaches some other cluster's representative and not its own is a
     * misassignment witness. Read across {@link BoxKind} it is a calibration: the
     * separation probability at which a member first reaches its claimed centre.
     */
    public static final int REACH_OWN = 0;
    public static final int REACH_OTHER_ONLY = 1;
    public static final int REACH_NONE = 2;
    /** Subset of REACH_OWN that also contains a foreign representative. */
    public static final int REACH_AMBIGUOUS = 3;

    /**
     * Flattened representatives of every cluster, with the cluster each belongs to.
     */
    private static float[][] repsOf(List<ICluster<float[]>> summary, int[][] ownerOut) {
        List<float[]> reps = new ArrayList<>();
        List<Integer> owner = new ArrayList<>();
        for (int c = 0; c < summary.size(); c++) {
            for (Weighted<float[]> rep : summary.get(c).getRepresentatives()) {
                reps.add(rep.index);
                owner.add(c);
            }
        }
        ownerOut[0] = owner.stream().mapToInt(Integer::intValue).toArray();
        return reps.toArray(new float[0][]);
    }

    /**
     * Per-cluster reach counts, one row per union, four columns in the order of the
     * REACH_ constants. Union {@code c} holds the boxes of cluster {@code c}'s
     * members, which is what {@link #forMembers} produces. Costs no forest queries.
     */
    public static int[][] reach(List<Union> unions, List<ICluster<float[]>> summary) {
        int[][] ownerBox = new int[1][];
        float[][] reps = repsOf(summary, ownerBox);
        int[] owner = ownerBox[0];
        int[][] out = new int[unions.size()][4];
        for (int c = 0; c < unions.size(); c++) {
            Union u = unions.get(c);
            for (int j = 0; j < u.size(); j++) {
                boolean own = false;
                boolean other = false;
                for (int r = 0; r < reps.length; r++) {
                    if (u.inBox(reps[r], j)) {
                        if (owner[r] == c) {
                            own = true;
                        } else {
                            other = true;
                        }
                        if (own && other) {
                            break;
                        }
                    }
                }
                if (own) {
                    out[c][REACH_OWN]++;
                    if (other) {
                        out[c][REACH_AMBIGUOUS]++;
                    }
                } else if (other) {
                    out[c][REACH_OTHER_ONLY]++;
                } else {
                    out[c][REACH_NONE]++;
                }
            }
        }
        return out;
    }

    /** Column sums of {@link #reach}, plus the box count as a fifth entry. */
    public static int[] reachTotal(int[][] perCluster) {
        int[] t = new int[5];
        for (int[] row : perCluster) {
            for (int i = 0; i < 4; i++) {
                t[i] += row[i];
            }
            t[4] += row[REACH_OWN] + row[REACH_OTHER_ONLY] + row[REACH_NONE];
        }
        return t;
    }

    /**
     * The reach sweep: the same member boxes read at every {@link BoxKind}.
     */
    public static void reachReport(Map<BoxKind, List<Union>> byKind, List<ICluster<float[]>> summary) {
        System.out.println();
        System.out.println("Reach: does a member's own box hold a representative of its own cluster?");
        System.out.println("  P[sep]  kind      boxes    own   ambiguous   other only     none");
        for (BoxKind kind : BoxKind.values()) {
            List<Union> unions = byKind.get(kind);
            if (unions == null) {
                continue;
            }
            int[] t = reachTotal(reach(unions, summary));
            double n = Math.max(1, t[4]);
            System.out.printf("  %6s  %-8s %6d  %5.1f%%   %7.1f%%     %7.1f%%  %6.1f%%%n",
                    kind == BoxKind.CUT ? "0.50" : kind == BoxKind.PASSAGE ? "0.05" : "stop", kind, t[4],
                    100 * t[REACH_OWN] / n, 100 * t[REACH_AMBIGUOUS] / n, 100 * t[REACH_OTHER_ONLY] / n,
                    100 * t[REACH_NONE] / n);
        }
        System.out.println("  own rises with the box, so the kind at which it saturates is the scale");
        System.out.println("  at which the assignment is supported; other-only is a misassignment witness");
    }

    /** Every ordered pair of {@link #overlapFraction}; the diagonal is NaN. */
    public static double[][] overlapMatrix(List<Union> unions, long seed, int samples) {
        int k = unions.size();
        double[][] overlap = new double[k][k];
        for (int a = 0; a < k; a++) {
            for (int b = 0; b < k; b++) {
                overlap[a][b] = (a == b) ? Double.NaN
                        : overlapFraction(unions.get(a), unions.get(b), seed + 31L * a + b, samples);
            }
        }
        return overlap;
    }

    /**
     * One line saying how big the boxes actually came out.
     */
    public static String describe(List<Union> unions) {
        int boxes = 0;
        int flatAxes = 0;
        int axes = 0;
        double sumHalf = 0.0;
        double minHalf = Double.MAX_VALUE;
        double maxHalf = 0.0;
        for (Union u : unions) {
            for (int j = 0; j < u.size(); j++) {
                boxes++;
                for (int i = 0; i < u.lo[j].length; i++) {
                    double half = 0.5 * (u.hi[j][i] - u.lo[j][i]);
                    axes++;
                    if (half <= 0.0) {
                        flatAxes++;
                    } else {
                        sumHalf += half;
                        minHalf = Math.min(minHalf, half);
                        maxHalf = Math.max(maxHalf, half);
                    }
                }
            }
        }
        if (boxes == 0) {
            return "no boxes at all: every representative returned an unreliable density";
        }
        int live = axes - flatAxes;
        if (live == 0) {
            return String.format("%d boxes, every axis flat: the walk recorded no extent", boxes);
        }
        // The centring state belongs on every line that reports a box size,
        // because a symmetric box is not what the forest returned and a reader
        // looking at symmetric rectangles is entitled to be told so rather than
        // to go and find the constant.
        boolean anyCentered = false;
        double drift = 0.0;
        for (Union u : unions) {
            anyCentered |= u.centered;
            drift = Math.max(drift, u.meanDrift);
        }
        return String.format(
                "%d boxes over %d unions, half-width min %.4f mean %.4f max %.4f, %d/%d axes flat, %s "
                        + "(discarded drift %.3f)",
                boxes, unions.size(), minHalf, sumHalf / live, maxHalf, flatAxes, axes,
                anyCentered ? "CENTRED on the query, so symmetric by construction" : "as returned, asymmetric", drift);
    }

    /** Points used for the mass and census figures; the rest add no precision. */

    // ------------------------------------------------------------------
    // internals
    // ------------------------------------------------------------------

    private static double logSumExp(double[] logs) {
        double max = Double.NEGATIVE_INFINITY;
        for (double l : logs) {
            max = Math.max(max, l);
        }
        if (Double.isInfinite(max)) {
            return Double.NEGATIVE_INFINITY;
        }
        double acc = 0.0;
        for (double l : logs) {
            acc += Math.exp(l - max);
        }
        return max + Math.log(acc);
    }

    private static double[] normalizedCdf(double[] logs, double logTotal) {
        double[] cdf = new double[logs.length];
        double acc = 0.0;
        for (int i = 0; i < logs.length; i++) {
            acc += Math.exp(logs[i] - logTotal);
            cdf[i] = acc;
        }
        cdf[logs.length - 1] = 1.0;
        return cdf;
    }

    private static int pick(double[] cdf, double uniform) {
        int i = Arrays.binarySearch(cdf, uniform);
        if (i < 0) {
            i = -i - 1;
        }
        return Math.min(i, cdf.length - 1);
    }

    private static void sampleInBox(Union u, int j, Random random, double[] out) {
        for (int i = 0; i < out.length; i++) {
            double w = u.hi[j][i] - u.lo[j][i];
            out[i] = (w > 0.0) ? u.lo[j][i] + random.nextDouble() * w : u.lo[j][i];
        }
    }

    private static int firstContaining(Union u, double[] x) {
        for (int j = 0; j < u.size(); j++) {
            if (u.inBox(x, j)) {
                return j;
            }
        }
        return -1;
    }
}