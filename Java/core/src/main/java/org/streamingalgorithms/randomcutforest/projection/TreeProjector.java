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
import static org.streamingalgorithms.randomcutforest.CommonUtils.checkNotNull;

import java.util.Arrays;

import org.streamingalgorithms.randomcutforest.IRFMultiVisitor;
import org.streamingalgorithms.randomcutforest.MultiVisitor;
import org.streamingalgorithms.randomcutforest.executor.ITraversable;
import org.streamingalgorithms.randomcutforest.tree.ArrayBox;
import org.streamingalgorithms.randomcutforest.tree.INodeView;
import org.streamingalgorithms.randomcutforest.tree.ITree;
import org.streamingalgorithms.randomcutforest.tree.NodeView;

/**
 * Routes a set of points through one tree at once and records the tree
 * restricted to them as a {@link TreeProjection}: one walk over the union of
 * their paths, which visits each vertex of that union once.
 *
 * <p>
 * It is a multi-visitor under the existing contract. At each vertex,
 * {@link #trigger} partitions the walk's points by the vertex's cut, in place,
 * with the tree's own rule (left when the coordinate is at most the cut). When
 * both sides hold points it returns true, and the tree visits the left subtree
 * and then the right. Otherwise it returns false and the tree follows the cut
 * of the traversal point; the visitor has just written the cut coordinate of
 * one of its points there, and every point of a walk that does not part goes
 * the same way, so the tree follows the points. The traversal point is the
 * projector's own scratch array.
 *
 * <p>
 * Nothing is allocated once the projector is warm. Forks are frames of a pool
 * shared by the root and its forks: the traversal is depth-first and each fork
 * is combined before its parent forks again, so at most depth frames are live
 * and they are taken and returned last in first out. The pending splits are a
 * stack of the same discipline. The projection is the caller's, sized once.
 *
 * <p>
 * Boxes are read from the tree, through the view, only at the vertices where
 * paths part; at a leaf the stored point is the box.
 */
public final class TreeProjector implements IRFMultiVisitor<TreeProjection> {

    /** What the root and its forks share. */
    private static final class Shared {
        final float[] steer;
        float[][] points;
        TreeProjection out;
        // pending splits, last in first out: the walk's range [lo, hi) and where it
        // parts
        int[] splitLo = new int[64];
        int[] splitMid = new int[64];
        int[] splitHi = new int[64];
        int splits;
        TreeProjector[] frames = new TreeProjector[16];
        int top;
        NodeView view;

        Shared(int dimensions) {
            steer = new float[dimensions];
        }
    }

    private final Shared shared;
    private int lo;
    private int hi;
    /**
     * The projection node of what this walk has built so far; -1 before its leaf.
     */
    private int current = -1;
    /** At a vertex where the walk parted, the right side's node, set by combine. */
    private int rightPart = -1;

    public TreeProjector(int dimensions) {
        checkArgument(dimensions > 0, "dimensions must be positive");
        this.shared = new Shared(dimensions);
    }

    private TreeProjector(Shared shared) {
        this.shared = shared;
    }

    /**
     * Projects points[0], ..., points[n - 1] through the tree into out, which is
     * cleared first. The points are read, not retained past the call.
     */
    public TreeProjection project(ITraversable tree, float[][] points, int n, TreeProjection out) {
        checkNotNull(tree, "tree must not be null");
        checkNotNull(points, "points must not be null");
        checkNotNull(out, "projection must not be null");
        checkArgument(n >= 0 && n <= points.length && n <= out.capacity, "more points than the projection holds");
        checkArgument(out.dimensions == shared.steer.length, "dimensions must match");
        out.clear();
        if (n == 0) {
            return out;
        }
        shared.points = points;
        shared.out = out;
        shared.splits = 0;
        shared.top = 0;
        for (int i = 0; i < n; i++) {
            out.order[i] = i;
        }
        lo = 0;
        hi = n;
        current = -1;
        rightPart = -1;
        shared.view = tree.reusableTraverseMulti(shared.steer, this, shared.view);
        shared.points = null;
        return out;
    }

    @Override
    public void prepare(ITree<?, ?> tree, float[] rawPoint) {
        // armed by project(), which owns the range and the scratch point
    }

    @Override
    public boolean trigger(INodeView node) {
        int dim = node.getCutDimension();
        double cut = node.getCutValue();
        float[][] points = shared.points;
        int[] order = shared.out.order;
        // partition [lo, hi) in place: the points the tree sends left first
        int i = lo, j = hi - 1;
        while (i <= j) {
            if (points[order[i]][dim] <= cut) {
                i++;
            } else {
                int t = order[i];
                order[i] = order[j];
                order[j] = t;
                j--;
            }
        }
        int mid = i;
        if (mid > lo && mid < hi) {
            push(lo, mid, hi);
            hi = mid;
            return true;
        }
        // one side only: put the walk's cut coordinate where the tree will read it
        shared.steer[dim] = points[order[lo]][dim];
        return false;
    }

    private void push(int from, int mid, int to) {
        Shared s = shared;
        if (s.splits == s.splitLo.length) {
            s.splitLo = Arrays.copyOf(s.splitLo, 2 * s.splits);
            s.splitMid = Arrays.copyOf(s.splitMid, 2 * s.splits);
            s.splitHi = Arrays.copyOf(s.splitHi, 2 * s.splits);
        }
        s.splitLo[s.splits] = from;
        s.splitMid[s.splits] = mid;
        s.splitHi[s.splits] = to;
        s.splits++;
    }

    @Override
    public MultiVisitor<TreeProjection> newPartialCopy() {
        Shared s = shared;
        if (s.top == s.frames.length) {
            s.frames = Arrays.copyOf(s.frames, 2 * s.top);
        }
        TreeProjector fork = s.frames[s.top];
        if (fork == null) {
            fork = new TreeProjector(s);
            s.frames[s.top] = fork;
        }
        s.top++;
        // the right side of the split this walk has just finished the left of
        int k = s.splits - 1;
        fork.lo = s.splitMid[k];
        fork.hi = s.splitHi[k];
        fork.current = -1;
        fork.rightPart = -1;
        return fork;
    }

    @Override
    public void combine(MultiVisitor<TreeProjection> other) {
        TreeProjector fork = (TreeProjector) other;
        Shared s = shared;
        int k = --s.splits;
        lo = s.splitLo[k];
        hi = s.splitHi[k];
        rightPart = fork.current;
        if (s.top > 0 && s.frames[s.top - 1] == fork) {
            s.top--;
        }
    }

    @Override
    public void acceptLeaf(INodeView leafNode, int depthOfNode) {
        current = shared.out.addLeaf(lo, hi, leafNode.getMass(), depthOfNode, leafNode.getLeafPoint(),
                leafNode.getLeafPointIndex());
    }

    @Override
    public void accept(INodeView node, int depthOfNode) {
        if (rightPart < 0) {
            return; // the walk did not part here: nothing to record
        }
        ArrayBox box = (ArrayBox) node.getBoundingBox();
        current = shared.out.addVertex(current, rightPart, lo, hi, node.getMass(), depthOfNode, box.values, box.offset);
        rightPart = -1;
    }

    @Override
    public TreeProjection getResult() {
        return shared.out;
    }
}
