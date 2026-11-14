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

import static org.streamingalgorithms.randomcutforest.CommonUtils.checkNotNull;

import java.util.Arrays;

import org.streamingalgorithms.randomcutforest.IRFVisitor;
import org.streamingalgorithms.randomcutforest.IVisitorFactory;
import org.streamingalgorithms.randomcutforest.Visitor;
import org.streamingalgorithms.randomcutforest.returntypes.Escape;
import org.streamingalgorithms.randomcutforest.returntypes.LocalGeometryConfig;
import org.streamingalgorithms.randomcutforest.tree.ITree;

/**
 * Reads an {@link Escape} from the separation transcript of a
 * {@link SeparationVisitor} walked to the root: codisplacement, range escape,
 * the exits by face, and the cut and passage boxes under the rules of a
 * {@link LocalGeometryConfig}. With {@link ReferenceSums} it reads, on the same
 * walk, the query's row against a reference set routed through the trees.
 *
 * <p>
 * The walk goes to the root because codisplacement needs every ancestor's mass
 * and the exits need the root box. When the query is a stored point of the
 * tree, the boxes are those of the tree without that point's leaf, as
 * everywhere in the separation walk.
 *
 * <p>
 * Reusable, foldable and reusable across queries: one visitor adds each tree
 * into its reading as it is walked, so a forest, or a batch of points, costs
 * one reading per point.
 */
public class EscapeVisitor extends SeparationVisitor<Escape> {

    private final LocalGeometryConfig config;
    private final ReferenceSums sums; // null: no reference set
    private final Escape folded;
    // this tree's reading
    private final double[] faceExit, cutGap;
    private double codisplacement, rootCodisplacement, rangeEscape, logPassageMass;
    private boolean read;
    // the walk down the reference sums of this tree
    private final double[] row;
    private int tree = -1, slot;
    private float[] point;
    private double survive, cutOff, parentPhi;

    public EscapeVisitor(int dimension, LocalGeometryConfig config) {
        this(dimension, config, null);
    }

    public EscapeVisitor(int dimension, LocalGeometryConfig config, ReferenceSums sums) {
        super(dimension);
        checkNotNull(config, "config must not be null");
        this.config = config;
        this.sums = sums;
        this.folded = new Escape(dimension);
        this.faceExit = new double[len];
        this.cutGap = new double[len];
        this.row = (sums == null) ? null : new double[sums.columns];
    }

    @Override
    protected boolean levelRecorded(int level) {
        return false; // the whole path: every ancestor's mass and the root box
    }

    @Override
    protected void reset() {
        super.reset();
        read = false;
    }

    @Override
    protected void onTree(ITree<?, ?> walked, float[] rawPoint) {
        if (sums != null) {
            tree = sums.indexOf(walked);
            point = rawPoint;
        }
    }

    private double rangeWithQuery(int level) {
        double r = 0;
        for (int k = 0; k < len; k++) {
            r += reach(level, k);
        }
        return r;
    }

    private int firstWithMass(double target) {
        for (int k = 0; k < levels(); k++) {
            if (mass(k) >= target) {
                return k;
            }
        }
        return -1;
    }

    /** This tree's reading, once per walk. Returns false for an empty tree. */
    private boolean read() {
        int top = levels() - 1;
        if (read || top < 0) {
            return top >= 0;
        }
        read = true;
        Arrays.fill(faceExit, 0.0);
        boolean rows = sums != null && tree >= 0;
        if (rows) {
            Arrays.fill(row, 0.0);
            slot = 0;
            survive = 1.0;
            cutOff = 0.0;
            parentPhi = 0.0;
        }
        // root first: s is the probability of arriving unseparated
        double s = 1, range = 0, colluder = 0;
        codisplacement = 0;
        rootCodisplacement = 0;
        for (int k = top; k >= 0; k--) {
            if (k < top) { // the subtree of this vertex, with the query, against its sibling
                colluder = Math.max(colluder, (mass(k + 1) - mass(k)) / (mass(k) + 1));
            }
            double separate = s * cutProbability(k);
            double withQuery = rangeWithQuery(k), gapSum = 0;
            range += separate * withQuery;
            double ratio = Math.max(mass(k), colluder);
            codisplacement += separate * ratio;
            rootCodisplacement += separate * Math.sqrt(ratio);
            for (int f = 0; f < len; f++) {
                gapSum += gap(k, f);
            }
            if (separate > 0 && gapSum > 0) {
                for (int f = 0; f < len; f++) {
                    double g = gap(k, f);
                    if (g > 0) { // the face in proportion to its gap; out before back
                        faceExit[f] += separate * (g / gapSum) * g / (g + reach(top, f));
                    }
                }
            }
            if (rows && slot >= 0) {
                referenceLevel(gapSum, withQuery);
            }
            s -= separate;
        }
        double root = rangeWithQuery(top);
        rangeEscape = (root > 0) ? range / root : 0.0;

        // the cut and passage boxes, by the configuration's rules
        int stop = top;
        for (int k = 1; k < levels(); k++) {
            if (cutProbability(k) <= config.stopProbability()) {
                stop = k;
                break;
            }
        }
        int cap = firstWithMass(config.capMass(treeMass));
        int cutAt = quantileLevel(stop, config.cutQuantile());
        int passageAt = quantileLevel(stop, config.passageQuantile());
        if (cap >= 0) {
            cutAt = Math.min(cutAt, cap);
            passageAt = Math.min(passageAt, cap);
        }
        for (int f = 0; f < len; f++) {
            cutGap[f] = gap(cutAt, f);
        }
        logPassageMass = Math.log(mass(passageAt));
        return true;
    }

    /** One level of the path, root first, against the reference sums. */
    private void referenceLevel(double gapSum, double range) {
        int[] node = sums.node[tree];
        if (node[slot] < 0) {
            slot = -1;
            return;
        }
        double p = (!sums.insertion || queryIsStored() || !(range > 0)) ? 0.0 : gapSum / range;
        if (p > 0) {
            cutOff += survive * p * Math.max(Math.exp(-range / sums.h), parentPhi);
        }
        int l = sums.left[tree][slot];
        if (l < 0) { // not cut off, the query sits with the leaf's points
            addSums(cutOff + survive * (1.0 - p), node[slot]);
            slot = -1;
            return;
        }
        int r = sums.right[tree][slot];
        boolean toLeft = sums.side.left(tree, slot, point);
        int other = toLeft ? r : l;
        double here = sums.phi[tree][slot];
        if (node[other] >= 0) {
            addSums(cutOff + survive * (1.0 - p) * here, node[other]);
        }
        survive *= 1.0 - p;
        parentPhi = here;
        slot = toLeft ? l : r;
    }

    private void addSums(double f, int node) {
        double[] from = sums.sums[tree];
        int m = sums.columns, base = node * m;
        for (int c = 0; c < m; c++) {
            row[c] += f * from[base + c];
        }
    }

    private void addTo(Escape target) {
        if (read()) {
            target.addTree(codisplacement, rootCodisplacement, rangeEscape, logPassageMass, faceExit, cutGap);
            if (sums != null && tree >= 0) {
                target.addRow(row, sums.columns);
            }
        }
    }

    /** This tree alone, detached: a fresh result each call. */
    @Override
    public Escape getResult() {
        Escape one = new Escape(dim);
        addTo(one);
        return one;
    }

    @Override
    public void foldOut() {
        addTo(folded);
    }

    /** The trees folded so far, detached. */
    @Override
    public Escape getFoldResult() {
        return folded.copy();
    }

    @Override
    public void resetAcrossQueries(float[] point) {
        reset();
        folded.clear();
    }

    @Override
    public boolean absorb(IRFVisitor<Escape> other) {
        if (!(other instanceof EscapeVisitor)) {
            return false;
        }
        folded.add(((EscapeVisitor) other).folded);
        return true;
    }

    /**
     * Reusable, foldable, reusable across queries. A forest reduces its per-tree or
     * per-thread results with {@link Escape#merge}.
     */
    public static IVisitorFactory<Escape> factory(LocalGeometryConfig config) {
        return factory(config, null);
    }

    /** As above; with sums, each reading also carries the row against them. */
    public static IVisitorFactory<Escape> factory(LocalGeometryConfig config, ReferenceSums sums) {
        checkNotNull(config, "config must not be null");
        return new IVisitorFactory<Escape>() {
            @Override
            public boolean isReusable() {
                return true;
            }

            @Override
            public boolean isReusableAcrossQueries() {
                return true;
            }

            @Override
            public boolean isFoldable() {
                return true;
            }

            @Override
            public IRFVisitor<Escape> newReusableVisitor(float[] point) {
                return new EscapeVisitor(point.length, config, sums);
            }

            @Override
            public Visitor<Escape> newVisitor(ITree<?, ?> tree, float[] point) {
                EscapeVisitor visitor = new EscapeVisitor(tree.projectToTree(point).length, config, sums);
                visitor.prepare(tree, point);
                return visitor;
            }
        };
    }
}