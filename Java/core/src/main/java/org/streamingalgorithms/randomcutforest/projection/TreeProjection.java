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

package org.streamingalgorithms.randomcutforest.projection;

import static org.streamingalgorithms.randomcutforest.CommonUtils.checkArgument;

/**
 * One tree restricted to a set of points: the vertices at which the points'
 * paths part, and the leaves they reach, in post-order, children before
 * parents. Filled by {@link TreeProjector}; allocated once by the caller for a
 * capacity and reused, so a projection allocates nothing.
 *
 * <p>
 * The points of a node are {@code order[lo[v], hi[v])}, positions into the
 * caller's array of points, so a node's points are contiguous and a node holds
 * its children's ranges side by side. Two points join at the deepest node whose
 * range holds both, the vertex of the stored tree at which their paths part.
 * Every node keeps the stored vertex's box as the tree holds it, the float
 * bound [max, -min] bit for bit, with its mass and its depth in the stored
 * tree; widths, ranges and affinity weights are the reader's, in double. A leaf
 * is a stored leaf, and may hold several points.
 *
 * <p>
 * Upward passes run over the nodes in index order, downward passes in reverse;
 * {@link #root()} is the last node.
 */
public final class TreeProjection {

    public final int capacity;
    public final int dimensions;
    /** The points, permuted so that each node's are contiguous. */
    public final int[] order;
    /** Children; -1 at a leaf. */
    public final int[] left;
    public final int[] right;
    /** The node's points: order[lo, hi). */
    public final int[] lo;
    public final int[] hi;
    /** Mass and depth of the stored vertex. */
    public final int[] mass;
    public final int[] depth;
    /**
     * At a leaf, the point-store index of the stored point; -1 at a vertex where
     * paths part.
     */
    public final int[] storedPoint;
    /** Per node, 2 * dimensions floats: the stored box [max, -min]. */
    public final float[] bounds;

    int nodes;

    public TreeProjection(int capacity, int dimensions) {
        checkArgument(capacity > 0, "capacity must be positive");
        checkArgument(dimensions > 0, "dimensions must be positive");
        this.capacity = capacity;
        this.dimensions = dimensions;
        int maxNodes = 2 * capacity - 1;
        order = new int[capacity];
        left = new int[maxNodes];
        right = new int[maxNodes];
        lo = new int[maxNodes];
        hi = new int[maxNodes];
        mass = new int[maxNodes];
        depth = new int[maxNodes];
        storedPoint = new int[maxNodes];
        bounds = new float[maxNodes * 2 * dimensions];
    }

    /** Nodes filled by the last projection. */
    public int nodes() {
        return nodes;
    }

    /** The node holding every point; -1 when no point was projected. */
    public int root() {
        return nodes - 1;
    }

    public boolean isLeaf(int node) {
        return left[node] < 0;
    }

    /** Width of the stored box of a node on coordinate j: max - min, in double. */
    public double width(int node, int j) {
        int base = node * 2 * dimensions;
        return (double) bounds[base + j] + bounds[base + dimensions + j];
    }

    /** Sum of the widths of a node's stored box. */
    public double rangeSum(int node) {
        double s = 0;
        for (int j = 0; j < dimensions; j++) {
            s += width(node, j);
        }
        return s;
    }

    void clear() {
        nodes = 0;
    }

    int addLeaf(int from, int to, int nodeMass, int nodeDepth, float[] point, int pointIndex) {
        int v = nodes++;
        left[v] = right[v] = -1;
        lo[v] = from;
        hi[v] = to;
        mass[v] = nodeMass;
        depth[v] = nodeDepth;
        storedPoint[v] = pointIndex;
        int base = v * 2 * dimensions;
        for (int j = 0; j < dimensions; j++) {
            bounds[base + j] = point[j];
            bounds[base + dimensions + j] = -point[j];
        }
        return v;
    }

    int addVertex(int leftChild, int rightChild, int from, int to, int nodeMass, int nodeDepth, float[] values,
            int offset) {
        int v = nodes++;
        left[v] = leftChild;
        right[v] = rightChild;
        lo[v] = from;
        hi[v] = to;
        mass[v] = nodeMass;
        depth[v] = nodeDepth;
        storedPoint[v] = -1;
        System.arraycopy(values, offset, bounds, v * 2 * dimensions, 2 * dimensions);
        return v;
    }
}
