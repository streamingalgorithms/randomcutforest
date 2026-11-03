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

package org.streamingalgorithms.randomcutforest.summarization;

import java.util.Arrays;

import it.unimi.dsi.fastutil.longs.Long2DoubleOpenHashMap;

/**
 * Distances between clusters during iterative clustering, keyed by the ordered
 * pair of cluster ids, so a distance is computed once and reused until one of
 * its two clusters changes its representatives. Ordered, because
 * ICluster.distance(other) is not promised to be symmetric.
 *
 * <p>
 * Up to DENSE_LIMIT pairs the cache is a plain array, reused across calls. Past
 * that it is an open hash map holding only the pairs actually asked for, and it
 * stops storing at SPARSE_LIMIT entries: past that bound a distance is simply
 * recomputed, so the space stays bounded whatever the input. NaN marks a pair
 * not known.
 */
final class ClusterDistanceCache {

    /** Ordered pairs held in the dense array: 2^21 doubles, 16 MB. */
    static final int DENSE_LIMIT = 1 << 21;

    /** Entries held in the sparse map before it stops storing. */
    static final int SPARSE_LIMIT = 1 << 20;

    /** A dense array larger than this is not kept for the next call. */
    static final int RETAIN_LIMIT = 1 << 18;

    private double[] dense = new double[0];
    private Long2DoubleOpenHashMap sparse;
    private int n;
    private boolean isDense;

    /** Prepares for ids 0..n-1, all unknown. */
    void reset(int n) {
        this.n = n;
        long pairs = (long) n * n;
        isDense = pairs <= DENSE_LIMIT;
        if (isDense) {
            if (dense.length < pairs) {
                dense = new double[(int) pairs];
            }
            Arrays.fill(dense, 0, (int) pairs, Double.NaN);
        } else {
            if (sparse == null) {
                sparse = new Long2DoubleOpenHashMap();
                sparse.defaultReturnValue(Double.NaN);
            }
            sparse.clear();
        }
    }

    /** Lets go of a large dense array or a sparse map after a call. */
    void release() {
        if (dense.length > RETAIN_LIMIT) {
            dense = new double[0];
        }
        if (sparse != null && sparse.size() > RETAIN_LIMIT) {
            sparse = null;
        }
    }

    private static long key(int a, int b) {
        return ((long) a << 32) | (b & 0xffffffffL);
    }

    /** The cached distance from a to b, or NaN. */
    double get(int a, int b) {
        return isDense ? dense[a * n + b] : sparse.get(key(a, b));
    }

    void put(int a, int b, double distance) {
        if (isDense) {
            dense[a * n + b] = distance;
        } else if (sparse.size() < SPARSE_LIMIT) {
            sparse.put(key(a, b), distance);
        }
    }

    /**
     * Forgets every distance to and from cluster a; alive lists the ids that can
     * still be asked about, alive[0..count).
     */
    void invalidate(int a, int[] alive, int count) {
        if (isDense) {
            int row = a * n;
            for (int k = 0; k < count; k++) {
                int b = alive[k];
                dense[row + b] = Double.NaN;
                dense[b * n + a] = Double.NaN;
            }
        } else {
            for (int k = 0; k < count; k++) {
                int b = alive[k];
                sparse.remove(key(a, b));
                sparse.remove(key(b, a));
            }
        }
    }

    /** Forgets everything. */
    void invalidateAll() {
        if (isDense) {
            Arrays.fill(dense, 0, n * n, Double.NaN);
        } else {
            sparse.clear();
        }
    }
}
