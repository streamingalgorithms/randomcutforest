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
package org.streamingalgorithms.randomcutforest.interpolation;

import static org.streamingalgorithms.randomcutforest.CommonUtils.checkArgument;
import static org.streamingalgorithms.randomcutforest.CommonUtils.checkNotNull;

import org.streamingalgorithms.randomcutforest.tree.ITree;

/**
 * A reference set that was routed through some trees of a forest beforehand, as
 * sums an {@link EscapeVisitor} reads while it walks a query through the same
 * trees. Whoever routes the reference set builds this; the forest must not be
 * updated between that and the reads.
 *
 * <p>
 * Per tree, slots follow the tree from its root at slot 0: the two children of
 * a slot (-1 at a leaf), the affinity phi of two points that part at the slot,
 * and the reference node below it, or -1 when no reference point reaches the
 * slot. sums[t][node * columns + c] is column c summed over the reference
 * points below that node. A column is any weight on the reference points: a
 * class indicator, a label, an eigenvector, a row of an inverse.
 *
 * <p>
 * The row of a query against the columns is then, per tree, the sum over the
 * reference points of affinity(query, point) times the point's weights. With
 * routing, the affinity to the points that left the query's path at a slot is
 * the slot's phi, and one to the points of its leaf. With insertion, the query
 * is first cut off at each slot with the probability a random cut of the box
 * with the query separates it; cut off, its affinity to the points below is
 * exp(-(range with the query) / h), kept at least the phi of the slot above. A
 * query that is a stored point of the tree is never cut off.
 */
public final class ReferenceSums {

    /** Which child of a tree's slot a point goes to. */
    public interface Side {
        boolean left(int tree, int slot, float[] point);
    }

    final int columns;
    final double h;
    final boolean insertion;
    final ITree<?, ?>[] tree;
    final int[][] left, right, node;
    final double[][] phi, sums;
    final Side side;

    /**
     * @param columns   weight columns per reference point
     * @param h         the scale of the affinity, used by insertion
     * @param insertion true for the affinities of an insertion, false for routing
     * @param tree      the trees the reference set was routed through
     */
    public ReferenceSums(int columns, double h, boolean insertion, ITree<?, ?>[] tree, int[][] left, int[][] right,
            int[][] node, double[][] phi, double[][] sums, Side side) {
        checkArgument(columns > 0, "columns must be positive");
        checkArgument(!insertion || h > 0, "insertion needs a positive scale");
        checkNotNull(side, "side must not be null");
        int n = tree.length;
        checkArgument(left.length == n && right.length == n && node.length == n && phi.length == n && sums.length == n,
                "one entry per tree");
        this.columns = columns;
        this.h = h;
        this.insertion = insertion;
        this.tree = tree;
        this.left = left;
        this.right = right;
        this.node = node;
        this.phi = phi;
        this.sums = sums;
        this.side = side;
    }

    public int getColumns() {
        return columns;
    }

    public int getTreeCount() {
        return tree.length;
    }

    /** The index of a tree among those the reference set went through, or -1. */
    int indexOf(ITree<?, ?> candidate) {
        for (int t = 0; t < tree.length; t++) {
            if (tree[t] == candidate) {
                return t;
            }
        }
        return -1;
    }
}