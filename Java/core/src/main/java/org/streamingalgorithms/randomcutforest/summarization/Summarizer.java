/*
 * Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

package org.streamingalgorithms.randomcutforest.summarization;

import static java.lang.Math.max;
import static org.streamingalgorithms.randomcutforest.CommonUtils.checkArgument;
import static org.streamingalgorithms.randomcutforest.util.Weighted.createSample;
import static org.streamingalgorithms.randomcutforest.util.Weighted.prefixPick;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.function.ToDoubleBiFunction;
import java.util.stream.IntStream;

import org.streamingalgorithms.randomcutforest.returntypes.SampleSummary;
import org.streamingalgorithms.randomcutforest.tree.VectorSupport;
import org.streamingalgorithms.randomcutforest.util.Weighted;

public class Summarizer {

    /**
     * a factor that controls weight assignment for soft clustering; this is the
     * multiple of the minimum distance and should be greater or equal 1.
     */
    public static double WEIGHT_ALLOCATION_THRESHOLD = 1.25;

    /**
     * the following determines the ratio between the sum of the (average) radius
     * and the separation between centers for a merge; ratio greater than 1 means
     * significant overlap a ratio of 0 means merge closest pairs without
     * consideration of separartion
     *
     **/
    public static double DEFAULT_SEPARATION_RATIO_FOR_MERGE = 0.8;

    public static int PHASE2_THRESHOLD = 2;

    public static int LENGTH_BOUND = 1000;

    public static double L1distance(float[] a, float[] b) {
        return VectorSupport.L1distance(a, b);
    }

    public static double L2distance(float[] a, float[] b) {
        return VectorSupport.L2distance(a, b);
    }

    public static double LInfinitydistance(float[] a, float[] b) {
        return VectorSupport.LInfinitydistance(a, b);
    }

    /**
     * a function that reassigns points to clusters
     *
     * @param sampledPoints   a list of sampled points with weights
     * @param clusters        a list of current clusters, because random access to
     *                        the elements is necessary
     * @param distance        a distance function
     * @param parallelEnabled a flag enabling limited parallelism; only during
     *                        cluster by cluster recomputation. Using parallel mode
     *                        during the assignment of points does not seem to help
     */
    @SuppressWarnings({ "unchecked", "rawtypes" })
    public static <R> void assignAndRecompute(List<Weighted<Integer>> sampledPoints, Function<Integer, R> getPoint,
            List<ICluster<R>> clusters, ToDoubleBiFunction<R, R> distance, boolean parallelEnabled) {
        checkArgument(clusters.size() > 0, " cannot be empty list of clusters");
        ICluster<R>[] byId = clusters.toArray(new ICluster[0]);
        int[] order = new int[byId.length];
        for (int k = 0; k < order.length; k++) {
            order[k] = k;
        }
        Workspace workspace = Workspace.acquire();
        try {
            workspace.ensure(byId.length);
            assign(sampledPoints, getPoint, byId, order, byId.length, distance, parallelEnabled, workspace.distances);
        } finally {
            workspace.release();
        }
    }

    /**
     * The assignment step over the live clusters byId[order[0..count)], in that
     * order. Each point is fetched once and its distances go into a reused array;
     * the arithmetic is that of the list version above.
     */
    static <R> void assign(List<Weighted<Integer>> sampledPoints, Function<Integer, R> getPoint, ICluster<R>[] byId,
            int[] order, int count, ToDoubleBiFunction<R, R> distance, boolean parallelEnabled, double[] dist) {
        checkArgument(count > 0, " cannot be empty list of clusters");
        checkArgument(sampledPoints.size() > 0, " cannot be empty list of points");

        for (int k = 0; k < count; k++) {
            byId[order[k]].reset();
        }

        for (Weighted<Integer> point : sampledPoints) {
            if (point.weight > 0) {
                R coordinates = getPoint.apply(point.index);
                Arrays.fill(dist, 0, count, Double.MAX_VALUE);
                double minDist = Double.MAX_VALUE;
                int minDistNbr = -1;
                for (int i = 0; i < count; i++) {
                    // will check for negative distances
                    dist[i] = byId[order[i]].distance(coordinates, distance);
                    if (minDist > dist[i]) {
                        minDist = dist[i];
                        minDistNbr = i;
                    }
                    if (minDist == 0) {
                        break;
                    }
                }

                if (minDist == 0) {
                    byId[order[minDistNbr]].addPoint(point.index, point.weight, 0, coordinates, distance);
                } else {
                    double sum = 0;
                    for (int i = 0; i < count; i++) {
                        if (dist[i] <= WEIGHT_ALLOCATION_THRESHOLD * minDist) {
                            sum += minDist / dist[i]; // setting up harmonic mean
                        }
                    }
                    for (int i = 0; i < count; i++) {
                        if (dist[i] <= WEIGHT_ALLOCATION_THRESHOLD * minDist) {
                            // harmonic mean
                            byId[order[i]].addPoint(point.index, (float) (point.weight * minDist / (dist[i] * sum)),
                                    dist[i], coordinates, distance);
                        }
                    }
                }
            }
        }

        if (parallelEnabled) {
            IntStream.range(0, count).parallel().forEach(k -> byId[order[k]].recompute(getPoint, true, distance));
        } else {
            for (int k = 0; k < count; k++) {
                byId[order[k]].recompute(getPoint, true, distance);
            }
        }
    }

    /**
     * Scratch kept per thread and reused across calls, so that summarizing a stream
     * once per frame allocates little: the cluster distances, the order of the live
     * clusters, and the per-point distances.
     *
     * <p>
     * A call can begin on a thread whose workspace is in use: with parallelEnabled
     * the recompute runs as a parallel stream, and a fork-join worker that waits
     * for it may run another task meanwhile, which can be another summarization.
     * Such a call gets a workspace of its own instead of the thread's.
     */
    static final class Workspace {
        final ClusterDistanceCache cache = new ClusterDistanceCache();
        int[] order = new int[0];
        int[] orderScratch = new int[0];
        double[] keys = new double[0];
        double[] keyScratch = new double[0];
        double[] distances = new double[0];
        double[] extents = new double[0];
        final Batch batch = new Batch();
        private boolean busy;

        /** The thread's workspace, or a fresh one if a call on this thread holds it. */
        static Workspace acquire() {
            Workspace w = WORKSPACE.get();
            if (w.busy) {
                w = new Workspace();
            }
            w.busy = true;
            return w;
        }

        void release() {
            cache.release();
            busy = false;
        }

        void ensure(int n) {
            if (order.length < n) {
                order = new int[n];
                orderScratch = new int[n];
                keys = new double[n];
                keyScratch = new double[n];
                distances = new double[n];
                extents = new double[n];
            }
        }
    }

    private static final ThreadLocal<Workspace> WORKSPACE = ThreadLocal.withInitial(Workspace::new);

    /**
     * The assignment step in blocks of points, for MultiCenter clusters under a
     * BatchDistance in a dimension where its blocks are exact. The points are
     * fetched once per call into one array, coordinate by coordinate; at each
     * assignment the live clusters' representatives are laid out in one array, and
     * every block of points is measured against each cluster at once, which gives
     * each point's distance to the cluster (the nearest representative, shrunk
     * toward the first) and its closest representative. The decisions that follow
     * are those of assign, line for line, and the updates those of
     * MultiCenter.addPoint and recompute, so the clustering is the same bit for
     * bit; only the work is arranged differently.
     */
    static final class Batch {
        static final int BLOCK = 64;

        private BatchDistance distance;
        private int d;
        private int n;
        private float[] points = new float[0];
        private int[] index = new int[0];
        private float[] weight = new float[0];
        private float[] reps = new float[0];
        private int[] start = new int[1];
        private double[] first = new double[0];
        private double[] best = new double[0];
        private int[] arg = new int[0];
        private final double[] work = new double[BatchDistance.WORK_PER_POINT * BLOCK];
        private double[] dist = new double[0];
        private MultiCenter[] live = new MultiCenter[0];

        /**
         * True, with the points fetched, when the blocks apply: a BatchDistance, every
         * cluster exactly a MultiCenter, and a dimension where the blocks are exact.
         */
        <R> boolean prepare(List<Weighted<Integer>> refs, Function<Integer, R> getPoint,
                ToDoubleBiFunction<R, R> distanceFunction, ICluster<R>[] byId) {
            if (!(distanceFunction instanceof BatchDistance) || byId.length == 0) {
                return false;
            }
            for (ICluster<R> c : byId) {
                if (c.getClass() != MultiCenter.class) {
                    return false;
                }
            }
            distance = (BatchDistance) distanceFunction;
            d = ((MultiCenter) byId[0]).getRepresentatives().get(0).index.length;
            if (!distance.exactAt(d)) {
                return false;
            }
            int count = 0;
            for (Weighted<Integer> point : refs) {
                if (point.weight > 0) {
                    count++;
                }
            }
            n = count;
            if (points.length < d * n) {
                points = new float[d * n];
            }
            if (index.length < n) {
                index = new int[n];
                weight = new float[n];
            }
            int p = 0;
            for (Weighted<Integer> point : refs) {
                if (point.weight > 0) {
                    float[] x = (float[]) getPoint.apply(point.index);
                    for (int j = 0; j < d; j++) {
                        points[j * n + p] = x[j];
                    }
                    index[p] = point.index;
                    weight[p++] = point.weight;
                }
            }
            return true;
        }

        /** assign(refs, getPoint, byId, order, count, distance, ...), in blocks. */
        <R> void assign(ICluster<R>[] byId, int[] order, int count) {
            checkArgument(count > 0, " cannot be empty list of clusters");
            checkArgument(n > 0, " cannot be empty list of points");
            if (live.length < count) {
                live = new MultiCenter[count];
                start = new int[count + 1];
                dist = new double[count];
            }
            if (best.length < count * BLOCK) {
                first = new double[count * BLOCK];
                best = new double[count * BLOCK];
                arg = new int[count * BLOCK];
            }
            int m = 0;
            for (int k = 0; k < count; k++) {
                live[k] = (MultiCenter) byId[order[k]];
                live[k].reset();
                start[k] = m;
                for (Weighted<float[]> r : live[k].getRepresentatives()) {
                    if (reps.length < (m + 1) * d) {
                        reps = Arrays.copyOf(reps, Math.max(2 * reps.length, (m + 1) * d));
                    }
                    System.arraycopy(r.index, 0, reps, m * d, d);
                    m++;
                }
            }
            start[count] = m;

            for (int from = 0; from < n; from += BLOCK) {
                int to = Math.min(n, from + BLOCK);
                for (int k = 0; k < count; k++) {
                    distance.nearest(d, points, n, from, to, reps, start[k], start[k + 1], first, best, arg, k * BLOCK,
                            work);
                }
                for (int p = from; p < to; p++) {
                    int q = p - from;
                    double minDist = Double.MAX_VALUE;
                    int minDistNbr = -1;
                    for (int i = 0; i < count; i++) {
                        // GenericMultiCenter.distance(point): the nearest representative,
                        // shrunk toward the first
                        double shrinkage = live[i].shrinkage;
                        dist[i] = (1 - shrinkage) * best[i * BLOCK + q] + shrinkage * first[i * BLOCK + q];
                        if (minDist > dist[i]) {
                            minDist = dist[i];
                            minDistNbr = i;
                        }
                        if (minDist == 0) {
                            break;
                        }
                    }
                    if (minDist == 0) {
                        live[minDistNbr].addPointAt(index[p], weight[p], 0, arg[minDistNbr * BLOCK + q]);
                    } else {
                        double sum = 0;
                        for (int i = 0; i < count; i++) {
                            if (dist[i] <= WEIGHT_ALLOCATION_THRESHOLD * minDist) {
                                sum += minDist / dist[i]; // setting up harmonic mean
                            }
                        }
                        for (int i = 0; i < count; i++) {
                            if (dist[i] <= WEIGHT_ALLOCATION_THRESHOLD * minDist) {
                                // harmonic mean
                                live[i].addPointAt(index[p], (float) (weight[p] * minDist / (dist[i] * sum)), dist[i],
                                        arg[i * BLOCK + q]);
                            }
                        }
                    }
                }
            }
            for (int k = 0; k < count; k++) {
                live[k].recomputeAssigned();
            }
            Arrays.fill(live, 0, count, null);
        }
    }

    /**
     * Stable sort of order[0..count) by non-decreasing weight -- the order
     * List.sort(Comparator.comparingDouble(ICluster::getWeight)) gives -- then
     * drops the clusters of zero weight from the front. Returns the new count.
     */
    static <R> int sortAndDropEmpty(ICluster<R>[] byId, int[] order, int count, Workspace w) {
        double[] keys = w.keys;
        for (int k = 0; k < count; k++) {
            keys[k] = byId[order[k]].getWeight();
        }
        int[] srcO = order, dstO = w.orderScratch;
        double[] srcK = keys, dstK = w.keyScratch;
        for (int width = 1; width < count; width *= 2) {
            for (int lo = 0; lo < count; lo += 2 * width) {
                int mid = Math.min(lo + width, count), hi = Math.min(lo + 2 * width, count);
                int i = lo, j = mid, k = lo;
                while (i < mid && j < hi) {
                    if (Double.compare(srcK[j], srcK[i]) < 0) {
                        dstO[k] = srcO[j];
                        dstK[k++] = srcK[j++];
                    } else {
                        dstO[k] = srcO[i];
                        dstK[k++] = srcK[i++];
                    }
                }
                while (i < mid) {
                    dstO[k] = srcO[i];
                    dstK[k++] = srcK[i++];
                }
                while (j < hi) {
                    dstO[k] = srcO[j];
                    dstK[k++] = srcK[j++];
                }
            }
            int[] tO = srcO;
            srcO = dstO;
            dstO = tO;
            double[] tK = srcK;
            srcK = dstK;
            dstK = tK;
        }
        if (srcO != order) {
            System.arraycopy(srcO, 0, order, 0, count);
            System.arraycopy(srcK, 0, keys, 0, count);
        }
        int drop = 0;
        while (drop < count && keys[drop] == 0.0) {
            drop++;
        }
        if (drop > 0) {
            for (int k = 0; k < drop; k++) {
                w.cache.invalidate(order[k], order, count);
            }
            System.arraycopy(order, drop, order, 0, count - drop);
        }
        return count - drop;
    }

    /** Removes position p from order[0..count); returns the new count. */
    private static int removeAt(int[] order, int count, int p) {
        System.arraycopy(order, p + 1, order, p, count - p - 1);
        return count - 1;
    }

    private static <R> boolean anyMovesRepresentatives(ICluster<R>[] byId, int[] order, int count) {
        for (int k = 0; k < count; k++) {
            if (byId[order[k]].recomputeMovesRepresentatives()) {
                return true;
            }
        }
        return false;
    }

    /**
     * The core subroutine for iterative clustering used herein. The clustering
     * algorithm borrows from CURE https://en.wikipedia.org/wiki/CURE_algorithm,
     * which used sampling as a tradeoff of representationa accuracy versus
     * algorithmic efficiency. Note however random sampling can also perform
     * denoising and reduce space as a filtering mechanism. Note that hierarchical
     * iterative merging strategies can be proven to not degrade clustering
     * https://en.wikipedia.org/wiki/Data_stream_clustering with the benefit of
     * small space. The algorithm herein proceeds in three phases, where the first
     * phase corresponds from the initial seeding to about twice the maximum number
     * of clusters one wishes to consider. The second phase corresponds to reducing
     * that number to the maximum allowable number. The third phase corresponds to
     * continuing the clustering as long as the conditions are similar to the end of
     * phase two, thereby enabling us to use a rough estimate for the maximum
     * allowed. By default, recomputation of the cluster makes sense in phases 2 and
     * 3 -- however can be enabled for phase 1 as well, thereby enabling the regular
     * K-Means algorithm to be expressed in the below algorithm as well. The
     * algorithm can also express Minimum Spanning Tree based clustering with
     * repeated merging of closest pair (which is a capability derived from CURE)
     *
     * The primary reason for the number of parameters is the ability to invoke this
     * function without creating a copy of the points (or equivalent large objects),
     * and hence the functions as parameters
     *
     * @param maxAllowed           number of maximum clusters one is interested in
     * @param initial              an initial number of sampled centers to start
     *                             from
     * @param stopAt               a hard lower bound on the number of clusters
     * @param refs                 a (possibly sampled) list of references with
     *                             weight
     * @param getPoint             a function which retrives the point/object given
     *                             an index in the refs
     * @param distance             a distance function
     * @param clusterInitializer   a function that creates a cluster given an object
     *                             aand a weight
     * @param seed                 a random seed
     * @param parallelEnabled      enabling parallel computation in the first phase
     *                             when points are assigned to different sampled
     *                             centers; and the centers are possibly adjusted
     * @param phase2GlobalReassign a flag that determines if the points would be
     *                             reassigned when the clusters fall below 1.2 *
     *                             maxAllowed -- this serves as a denoising.
     * @param overlapParameter     a parameter that controls the ordering of the
     *                             merges as well as the stopping condition of the
     *                             merges
     * @param previousClustering   a possibly null list of clusters seen previously,
     *                             used as zero weight seeds to smoothen the
     *                             continuous clustering
     * @param <R>                  type of object being clustered
     * @return a list of clusters
     */
    public static <R> List<ICluster<R>> iterativeClustering(int maxAllowed, int initial, int stopAt,
            List<Weighted<Integer>> refs, Function<Integer, R> getPoint, ToDoubleBiFunction<R, R> distance,
            BiFunction<R, Float, ICluster<R>> clusterInitializer, long seed, boolean parallelEnabled,
            boolean phase2GlobalReassign, double overlapParameter, List<ICluster<R>> previousClustering) {

        checkArgument(refs.size() > 0, "empty list, nothing to do");
        checkArgument(stopAt > 0, "has to stop at 1 cluster");
        checkArgument(stopAt <= maxAllowed, "cannot stop before achieving the limit");

        Random rng = new Random(seed);
        double sampledSum = refs.stream().map(e -> {
            checkArgument(Double.isFinite(e.weight), " weights have to be finite");
            checkArgument(e.weight >= 0.0, () -> "negative weights are not meaningful" + e.weight);
            return (double) e.weight;
        }).reduce(0.0, Double::sum);
        checkArgument(sampledSum > 0, " total weight has to be positive");
        ArrayList<ICluster<R>> seeds = new ArrayList<>();
        if (refs.size() < 10 * (initial + 5)) {
            for (Weighted<Integer> point : refs) {
                seeds.add(clusterInitializer.apply(getPoint.apply(point.index), 0f));
            }
        } else {
            for (int k = 0; k < 2 * (initial + 5); k++) {
                double wt = rng.nextDouble() * sampledSum;
                Weighted<Integer> picked = prefixPick(refs, wt);
                seeds.add(clusterInitializer.apply(getPoint.apply(picked.index), 0f));
            }
        }
        if (previousClustering != null) {
            for (ICluster<R> previousCluster : previousClustering) {
                List<Weighted<R>> representatives = previousCluster.getRepresentatives();
                for (Weighted<R> representative : representatives) {
                    seeds.add(clusterInitializer.apply(representative.index, 0f));
                }
            }
        }

        // The clusters keep their ids, their positions in seeds; the live ones are
        // order[0..count), sorted by weight as the list of centers used to be.
        @SuppressWarnings({ "unchecked", "rawtypes" })
        ICluster<R>[] byId = seeds.toArray(new ICluster[0]);
        int n = byId.length;
        Workspace workspace = Workspace.acquire();
        try {
            workspace.ensure(n);
            int[] order = workspace.order;
            for (int k = 0; k < n; k++) {
                order[k] = k;
            }
            int count = n;
            ClusterDistanceCache cache = workspace.cache;
            cache.reset(n);
            double[] extents = workspace.extents;
            Batch batch = workspace.batch.prepare(refs, getPoint, distance, byId) ? workspace.batch : null;

            if (batch != null) {
                batch.assign(byId, order, count);
            } else {
                assign(refs, getPoint, byId, order, count, distance, parallelEnabled, workspace.distances);
            }
            // assignment would change weights, sorting in non-decreasing order
            count = sortAndDropEmpty(byId, order, count, workspace);

            double phase3Distance = 0;
            double runningPhase3Distance = 0;
            boolean keepReducingCenters = (count > maxAllowed);

            while (keepReducingCenters) {
                double measure = 0;
                double measureDist = Double.MAX_VALUE;
                int lower = 0;
                int firstOfMerge = lower;
                int secondOfMerge = lower + 1;// will be reset before exiting the loop
                boolean foundMerge = false;
                double minDist = Double.MAX_VALUE;
                for (int k = 0; k < count; k++) {
                    extents[k] = byId[order[k]].extentMeasure();
                }

                while (lower < count - 1 && !foundMerge) {
                    // we will keep searching
                    int minNbr = -1;
                    int a = order[lower];
                    for (int j = lower + 1; j < count; j++) {
                        // a distance is computed once and kept until one of the two clusters
                        // changes its representatives
                        int b = order[j];
                        double dist = cache.get(a, b);
                        if (Double.isNaN(dist)) {
                            dist = byId[a].distance(byId[b], distance);
                            cache.put(a, b, dist);
                        }
                        if (dist == 0) {
                            foundMerge = true;
                            firstOfMerge = lower;
                            secondOfMerge = minNbr = j;
                            minDist = measureDist = 0.0;
                            break;
                        } else {
                            if (minDist > dist) {
                                minNbr = j;
                                minDist = dist;
                            }

                            double temp = (extents[lower] + extents[j] + phase3Distance) / dist;
                            if (temp > overlapParameter && measure < temp) {
                                firstOfMerge = lower;
                                secondOfMerge = j;
                                measure = temp;
                                measureDist = dist;
                            }
                        }
                    }
                    if (lower == 0 && !foundMerge) {
                        measureDist = minDist;
                        // this is set assuming we may be interested in merging the minimum weight
                        // cluster which corresponds to lower == 0
                        secondOfMerge = minNbr;
                    }
                    ++lower;
                }

                int inital = count;

                if (inital > maxAllowed || foundMerge || (inital > stopAt && measure > overlapParameter)) {
                    int first = order[firstOfMerge];
                    int second = order[secondOfMerge];
                    byId[second].absorb(byId[first], distance);
                    if (phase2GlobalReassign && count <= PHASE2_THRESHOLD * maxAllowed + 1) {
                        count = removeAt(order, count, firstOfMerge);
                        if (batch != null) {
                            batch.assign(byId, order, count);
                        } else {
                            assign(refs, getPoint, byId, order, count, distance, parallelEnabled, workspace.distances);
                        }
                        if (anyMovesRepresentatives(byId, order, count)) {
                            cache.invalidateAll();
                        }
                    } else {
                        byId[second].recompute(getPoint, false, distance);
                        count = removeAt(order, count, firstOfMerge);
                    }
                    // the absorbing cluster has new representatives; the absorbed one is gone
                    cache.invalidate(second, order, count);
                    cache.invalidate(first, order, count);
                    count = sortAndDropEmpty(byId, order, count, workspace);
                    if (inital < 1.2 * maxAllowed + 1) {
                        // phase 3 kicks in; but this will execute at most once
                        // note that measureDist can be 0 as well
                        runningPhase3Distance = max(runningPhase3Distance, measureDist);
                        if (inital > maxAllowed && count <= maxAllowed) {
                            phase3Distance = runningPhase3Distance;
                        }
                    }
                } else {
                    keepReducingCenters = false;
                }
            }
            ArrayList<ICluster<R>> centers = new ArrayList<>(count);
            for (int k = 0; k < count; k++) {
                centers.add(byId[order[k]]);
            }
            // sort in decreasing weight
            centers.sort((o1, o2) -> Double.compare(o2.getWeight(), o1.getWeight()));
            return centers;
        } finally {
            workspace.release();
        }
    }

    /**
     * the following function returns a summary of the input points
     *
     * @param points               points with associated weights
     * @param maxAllowed           the maximum number of clusters/summary points
     * @param initial              the initial number of clusters/summary points,
     *                             chosen at random
     * @param stopAt               a hard lower bound on the number of clusters
     * @param phase2GlobalReassign a flag that performs global reassignments when
     *                             the number of clusters is in the range
     *                             [maxAllowed, ceil(1.2*maxAllowed)]
     * @param overlapParameter     a control for merging clusters
     * @param distance             a distance function for the points, that
     *                             determines the order of the reverse delete
     *                             however the EM like step uses L1 measure (to be
     *                             robust to noise)
     * @param clusterInitializer   a function that creates the cluster type given a
     *                             single object and a weight
     * @param seed                 a random seed for controlling the randomness
     * @param parallelEnabled      flag enabling (limited) parallelism
     * @param previousClustering   any previous clustering that can be used as zero
     *                             weight seeds to ensure smoothness
     * @return a clustering of the input points (Note: the median returned is an
     *         approximate median; exact computation is unlikely to be critical for
     *         true applications of summarization)
     */
    public static <R> List<ICluster<R>> summarize(List<Weighted<R>> points, int maxAllowed, int initial, int stopAt,
            boolean phase2GlobalReassign, double overlapParameter, ToDoubleBiFunction<R, R> distance,
            BiFunction<R, Float, ICluster<R>> clusterInitializer, long seed, boolean parallelEnabled,
            List<ICluster<R>> previousClustering) {
        checkArgument(maxAllowed < 100, "are you sure you want more elements in the summary?");
        checkArgument(maxAllowed <= initial, "initial parameter should be at least maximum allowed in final result");

        double totalWeight = points.stream().map(e -> {
            checkArgument(Double.isFinite(e.weight), " weights have to be finite");
            checkArgument(e.weight >= 0.0, () -> "negative weights are not meaningful" + e.weight);
            return (double) e.weight;
        }).reduce(0.0, Double::sum);
        checkArgument(totalWeight > 0, " total weight has to be positive");
        Random rng = new Random(seed);
        // the following list is explicity copied and sorted for potential efficiency
        List<Weighted<R>> sampledPoints = createSample(points, rng.nextLong(), 5 * LENGTH_BOUND, 0.005, 1.0);

        ArrayList<Weighted<Integer>> refs = new ArrayList<>();
        for (int i = 0; i < sampledPoints.size(); i++) {
            refs.add(new Weighted<>(i, sampledPoints.get(i).weight));
        }

        Function<Integer, R> getPoint = (i) -> sampledPoints.get(i).index;

        return iterativeClustering(maxAllowed, initial, stopAt, refs, getPoint, distance, clusterInitializer,
                rng.nextLong(), parallelEnabled, phase2GlobalReassign, overlapParameter, previousClustering);
    }

    // same as above, specific for single centroid clustering of float[]
    // with an explicit stopping condition as well as a global reassignment option
    public static List<ICluster<float[]>> singleCentroidSummarize(List<Weighted<float[]>> points, int maxAllowed,
            int initial, int stopAt, boolean phase2GlobalReassign, ToDoubleBiFunction<float[], float[]> distance,
            long seed, boolean parallelEnabled, List<ICluster<float[]>> previousClustering) {
        return summarize(points, maxAllowed, initial, stopAt, phase2GlobalReassign, DEFAULT_SEPARATION_RATIO_FOR_MERGE,
                distance, Center::initialize, seed, parallelEnabled, previousClustering);
    }

    /**
     * the following function returns a summary of the input points
     *
     * @param points          points with associated weights
     * @param maxAllowed      the maximum number of clusters/summary points
     * @param initial         the initial number of clusters/summary points, chosen
     *                        at random
     * @param phase1reassign  (this parameter is ignored in the current version, but
     *                        the signature is unchanged for convenience)
     * @param distance        a distance function for the points, that determines
     *                        the order of the reverse delete however the EM like
     *                        step uses L1 measure (to be robust to noise)
     * @param seed            a random seed for controlling the randomness
     * @param parallelEnabled flag enabling (limited) parallelism
     * @return a summary of the input points (Note: the median returned is an
     *         approximate median; exact computation is unlikely to be critical for
     *         true applications of summarization)
     */
    public static SampleSummary summarize(List<Weighted<float[]>> points, int maxAllowed, int initial,
            boolean phase1reassign, ToDoubleBiFunction<float[], float[]> distance, long seed, boolean parallelEnabled,
            int numberOfReps, double shrinkage) {
        checkArgument(maxAllowed < 100, "are you sure you want more elements in the summary?");
        checkArgument(maxAllowed <= initial, "initial parameter should be at least maximum allowed in final result");

        double totalWeight = points.stream().map(e -> {
            checkArgument(Double.isFinite(e.weight), " weights have to be finite");
            checkArgument(e.weight >= 0.0, () -> "negative weights are not meaningful" + e.weight);
            return (double) e.weight;
        }).reduce(0.0, Double::sum);
        checkArgument(totalWeight > 0, " total weight has to be positive");

        Random rng = new Random(seed);
        // the following list is explicity copied and sorted for potential efficiency
        List<Weighted<float[]>> sampledPoints = createSample(points, rng.nextLong(), 5 * LENGTH_BOUND, 0.005, 1.0);

        List<ICluster<float[]>> centers = (numberOfReps == 1)
                ? summarize(sampledPoints, maxAllowed, initial, 1, true, DEFAULT_SEPARATION_RATIO_FOR_MERGE, distance,
                        Center::initialize, seed, parallelEnabled, null)
                : multiSummarizeWeighted(sampledPoints, maxAllowed, initial, 1, false,
                        DEFAULT_SEPARATION_RATIO_FOR_MERGE, distance, seed, parallelEnabled, shrinkage, numberOfReps);

        int num = centers.stream().mapToInt(x -> x.getRepresentatives().size()).sum();
        float[][] pointList = new float[num][];
        float[] likelihood = new float[num];
        float[] measure = new float[num];

        int dimensions = centers.get(0).primaryRepresentative(distance).length;
        int count = 0;
        for (int i = 0; i < centers.size(); i++) {
            for (Weighted<float[]> rep : centers.get(i).getRepresentatives()) {
                pointList[count] = Arrays.copyOf(rep.index, dimensions);
                likelihood[count] = (float) (rep.weight / totalWeight);
                measure[count++] = (float) centers.get(i).averageRadius();
            }
        }

        return new SampleSummary(sampledPoints, pointList, likelihood, measure);
    }

    public static SampleSummary summarize(List<Weighted<float[]>> points, int maxAllowed, int initial,
            boolean phase1reassign, ToDoubleBiFunction<float[], float[]> distance, long seed, boolean parallelEnabled) {
        return summarize(points, maxAllowed, initial, phase1reassign, distance, seed, parallelEnabled, 1, 0);
    }

    /**
     * Same as previous over a flat collection of unweighted float[]
     *
     * @param points          points represented by float[][]
     * @param maxAllowed      maximum number of clusters in output
     * @param initial         initial number of points to seed; a control parameter
     *                        that serves both as a denoiser, as well as as a
     *                        facilitator of coninuity (large numbers would
     *                        correspond to MST like clustering)
     * @param reassignPerStep unusued in current version
     * @param distance        distance metric over float []
     * @param seed            random seed
     * @param parallelEnabled flag enabling (limited) parallelism
     * @return a list of centers with weights
     */
    public static SampleSummary summarize(float[][] points, int maxAllowed, int initial, boolean reassignPerStep,
            ToDoubleBiFunction<float[], float[]> distance, long seed, Boolean parallelEnabled) {
        ArrayList<Weighted<float[]>> weighted = new ArrayList<>();
        for (float[] point : points) {
            weighted.add(new Weighted<>(point, 1.0f));
        }
        return summarize(weighted, maxAllowed, initial, reassignPerStep, distance, seed, parallelEnabled);
    }

    /**
     * same as before with common cases filled in, used in analysis of
     * ConditionalSamples
     *
     * @param points          points in ProjectedPoint{}
     * @param maxAllowed      maximum number of groups/clusters
     * @param initial         a parameter controlling the initialization
     * @param reassignPerStep if reassignment is to be performed each step
     * @param seed            random seed
     * @return a summarization
     */
    public static SampleSummary l2summarize(List<Weighted<float[]>> points, int maxAllowed, int initial,
            boolean reassignPerStep, long seed) {
        return summarize(points, maxAllowed, initial, reassignPerStep, Summarizer::L2distance, seed, false);
    }

    /**
     * Same as above, with the most common use cases filled in
     *
     * @param points     points in float[][], each of weight 1.0
     * @param maxAllowed maximum number of clusters one is interested in
     * @param seed       random seed
     * @return a summarization
     */
    public static SampleSummary l2summarize(float[][] points, int maxAllowed, long seed) {
        return summarize(points, maxAllowed, 4 * maxAllowed, false, Summarizer::L2distance, seed, false);
    }

    /**
     *
     * @param points                  points represented by R[]
     * @param maxAllowed              maximum number of clusters in output
     * @param initial                 initial number of points to seed; a control
     *                                parameter that serves both as a denoiser, as
     *                                well as as a facilitator of coninuity (large
     *                                numbers would correspond to MST like
     *                                clustering)
     * @param phase2GlobalReassign    a boolean determining global reassignment
     * @param overlapParameter        a parameter controlling order of mergers
     * @param distance                distance metric over float []
     * @param seed                    random seed
     * @param parallelEnabled         flag enabling (limited) parallelism
     * @param shrinkage               a parameter that morphs from centroidal
     *                                behavior (=1) to robust Minimum Spanning Tree
     *                                (=0)
     * @param numberOfRepresentatives the number of representatives ina multicentrod
     *                                representation used to cluster potentially
     *                                non-spherical shapes
     * @return a list of centers with weights
     */
    public static <R> List<ICluster<R>> multiSummarize(List<R> points, int maxAllowed, int initial, int stopAt,
            boolean phase2GlobalReassign, double overlapParameter, ToDoubleBiFunction<R, R> distance, long seed,
            Boolean parallelEnabled, double shrinkage, int numberOfRepresentatives) {

        ArrayList<Weighted<R>> weighted = new ArrayList<>();
        for (R point : points) {
            weighted.add(new Weighted<>(point, 1.0f));
        }
        return multiSummarizeWeighted(weighted, maxAllowed, initial, stopAt, phase2GlobalReassign, overlapParameter,
                distance, seed, parallelEnabled, shrinkage, numberOfRepresentatives);
    }

    public static <R> List<ICluster<R>> multiSummarizeWeighted(List<Weighted<R>> points, int maxAllowed, int initial,
            int stopAt, boolean phase2GlobalReassign, double overlapParameter, ToDoubleBiFunction<R, R> distance,
            long seed, boolean parallelEnabled, double shrinkage, int numberOfRepresentatives) {
        BiFunction<R, Float, ICluster<R>> clusterInitializer = (a, b) -> GenericMultiCenter.initialize(a, b, shrinkage,
                numberOfRepresentatives);
        return summarize(points, maxAllowed, initial, stopAt, phase2GlobalReassign, overlapParameter, distance,
                clusterInitializer, seed, parallelEnabled, null);
    }

    // same as above, different input
    public static <R> List<ICluster<R>> multiSummarize(R[] points, int maxAllowed, int initial, int stopAt,
            boolean phase2GlobalReassign, double overlapParameter, ToDoubleBiFunction<R, R> distance, long seed,
            Boolean parallelEnabled, double shrinkage, int numberOfRepresentatives) {

        ArrayList<Weighted<R>> weighted = new ArrayList<>();
        for (R point : points) {
            weighted.add(new Weighted<>(point, 1.0f));
        }
        BiFunction<R, Float, ICluster<R>> clusterInitializer = (a, b) -> GenericMultiCenter.initialize(a, b, shrinkage,
                numberOfRepresentatives);
        return summarize(weighted, maxAllowed, initial, stopAt, phase2GlobalReassign, overlapParameter, distance,
                clusterInitializer, seed, parallelEnabled, null);
    }

    // same as above, with multicenter instead of generic
    public static List<ICluster<float[]>> multiSummarize(float[][] points, int maxAllowed, double shrinkage,
            boolean parallelEnabled, int numberOfRepresentatives, long seed) {

        ArrayList<Weighted<float[]>> weighted = new ArrayList<>();
        for (float[] point : points) {
            weighted.add(new Weighted<>(point, 1.0f));
        }
        return multiSummarizeWeighted(weighted, maxAllowed, shrinkage, parallelEnabled, numberOfRepresentatives, seed);
    }

    public static List<ICluster<float[]>> multiSummarizeWeighted(List<Weighted<float[]>> points, int maxAllowed,
            double shrinkage, boolean parallelEnabled, int numberOfRepresentatives, long seed) {
        BiFunction<float[], Float, ICluster<float[]>> clusterInitializer = (a, b) -> MultiCenter.initialize(a, b,
                shrinkage, numberOfRepresentatives);
        return summarize(points, maxAllowed, 4 * maxAllowed, 1, true, DEFAULT_SEPARATION_RATIO_FOR_MERGE,
                Summarizer::L2distance, clusterInitializer, seed, parallelEnabled, null);
    }
}