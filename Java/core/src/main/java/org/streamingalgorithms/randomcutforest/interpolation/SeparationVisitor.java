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

import java.util.Arrays;

import org.streamingalgorithms.randomcutforest.RFVisitor;
import org.streamingalgorithms.randomcutforest.tree.ArrayBox;
import org.streamingalgorithms.randomcutforest.tree.INodeView;
import org.streamingalgorithms.randomcutforest.tree.VectorSupport;

/**
 * The walk of a query up its path, recorded as a transcript that readers
 * consume when the tree is done. See the package documentation for the reasons
 * behind its choices.
 *
 * <p>
 * Each level of the transcript, leaf first, holds what the tree stores and the
 * one number the walk decides with:
 * <ul>
 * <li>the data box without the query, as the tree's float bound [max, -min]
 * copied bit for bit (at a leaf, [p, -p]; above a leaf the query duplicates,
 * the shadow box of the vertex without that leaf);</li>
 * <li>the vertex's mass;</li>
 * <li>the query's own cut probability q, the chance that one cut of the box
 * holding the query separates the query, computed by the float gap kernels of
 * {@link VectorSupport}.</li>
 * </ul>
 * Readers compute reaches and gaps from the bound and the query [q, -q] in
 * double, where the difference of two floats is exact, so the transcript adds
 * no rounding of its own.
 *
 * <p>
 * The walk ends where {@link #levelRecorded} says the last box any reader needs
 * has been recorded. From then on {@code isConverged()} is true, the tree stops
 * materializing boxes, and {@link #aboveWalk} sees each remaining ancestor
 * without one. A reader's own stop is an index into the transcript, not the end
 * of the walk.
 */
public abstract class SeparationVisitor<R> extends RFVisitor<R> {

    private static final int INITIAL_CAPACITY = 32;

    protected final int dim;
    protected final int len;

    /** [q, -q], as the tree projects the query; set at the leaf. */
    private final float[] query;
    private final float[] gap; // scratch for the kernels
    private float[] bounds; // leaf first, len per level
    private double[] cut;
    private double[] mass;
    private int levels;
    private boolean pointEqualsLeaf;

    protected SeparationVisitor(int dimension) {
        this.dim = dimension;
        this.len = 2 * dimension;
        this.query = new float[len];
        this.gap = new float[len];
        this.bounds = new float[INITIAL_CAPACITY * len];
        this.cut = new double[INITIAL_CAPACITY];
        this.mass = new double[INITIAL_CAPACITY];
    }

    /**
     * Called once per recorded level, leaf first, with that level's index. Return
     * true once every box a reader needs is in the transcript; the walk then ends.
     */
    protected abstract boolean levelRecorded(int level);

    /** Each ancestor above the end of the walk, whose box is not materialized. */
    protected void aboveWalk(INodeView node, int depthOfNode) {
    }

    @Override
    protected void reset() {
        levels = 0;
        pointInsideBox = false;
        shadowBoxActive = false;
        pointEqualsLeaf = false;
    }

    @Override
    public void acceptLeaf(INodeView leafNode, int depthOfNode) {
        float[] leaf = leafNode.getLeafPoint();
        float[] expanded = leafNode.expanded();
        System.arraycopy(expanded, 0, query, 0, len);
        double s = VectorSupport.signedGapInto(expanded, 0, +1f, leaf, 0, gap, 0, dim)
                + VectorSupport.signedGapInto(expanded, dim, -1f, leaf, 0, gap, dim, dim);
        ensureCapacity();
        int base = levels * len;
        for (int i = 0; i < dim; i++) {
            bounds[base + i] = leaf[i];
            bounds[base + dim + i] = -leaf[i];
        }
        // A leaf box has zero range, so any gap separates the query surely. With
        // no gap the query is the leaf's point; the walk goes on with the shadow box.
        pointEqualsLeaf = !(s > 0);
        record(pointEqualsLeaf ? 0.0 : 1.0, leafNode.getMass());
    }

    @Override
    public void accept(INodeView node, int depthOfNode) {
        if (pointInsideBox) {
            aboveWalk(node, depthOfNode);
            return;
        }
        ArrayBox small;
        double s;
        if (pointEqualsLeaf) {
            // The vertex without the query's leaf; the vertex's own box is that
            // shadow with the query added, so the gap is the query's beyond it.
            small = growShadow((ArrayBox) node.getSiblingBoundingBox());
            ArrayBox large = (ArrayBox) node.getBoundingBox();
            s = VectorSupport.gapInto(large.values, large.offset, small.values, small.offset, gap, 0, len);
        } else {
            small = (ArrayBox) node.getBoundingBox();
            s = VectorSupport.gapInto(node.expanded(), 0, small.values, small.offset, gap, 0, len);
        }
        double range = small.getRangeSum() + s;
        ensureCapacity();
        System.arraycopy(small.values, small.offset, bounds, levels * len, len);
        record((range == 0.0) ? 0.0 : s / range, node.getMass());
    }

    private void record(double q, double m) {
        cut[levels] = q;
        mass[levels] = m;
        levels++;
        if (levelRecorded(levels - 1)) {
            pointInsideBox = true;
        }
    }

    private void ensureCapacity() {
        if (levels == cut.length) {
            int capacity = 2 * cut.length;
            bounds = Arrays.copyOf(bounds, capacity * len);
            cut = Arrays.copyOf(cut, capacity);
            mass = Arrays.copyOf(mass, capacity);
        }
    }

    // ------------------------------------------------------------------
    // reading the transcript
    // ------------------------------------------------------------------

    /** Levels recorded, the leaf being level 0. */
    protected final int levels() {
        return levels;
    }

    protected final double cutProbability(int level) {
        return cut[level];
    }

    protected final double mass(int level) {
        return mass[level];
    }

    /** True when the query is a point stored at its leaf. */
    protected final boolean queryIsStored() {
        return pointEqualsLeaf;
    }

    /** How far face k of the box with the query lies beyond the query. */
    protected final double reach(int level, int k) {
        return Math.max(0.0, (double) bounds[level * len + k] - query[k]);
    }

    /** How far the query lies beyond face k of the data box. */
    protected final double gap(int level, int k) {
        return Math.max(0.0, (double) query[k] - bounds[level * len + k]);
    }

    /** Reach and gap at every face of one level, written at the offset. */
    protected final void readLevel(int level, double[] reach, double[] gap, int offset) {
        int base = level * len;
        for (int k = 0; k < len; k++) {
            double slack = (double) bounds[base + k] - query[k];
            reach[offset + k] = Math.max(0.0, slack);
            gap[offset + k] = Math.max(0.0, -slack);
        }
    }

    /**
     * Descending from {@code top} toward the leaf, the deepest level at which the
     * query is still unseparated with probability at least 1 - p, counting only the
     * levels from {@code top} down.
     */
    protected final int quantileLevel(int top, double p) {
        double survival = 1.0;
        int found = top;
        for (int level = top; level >= 0; level--) {
            if (1.0 - survival <= p) {
                found = level;
            }
            survival *= 1.0 - cut[level];
        }
        return found;
    }
}
