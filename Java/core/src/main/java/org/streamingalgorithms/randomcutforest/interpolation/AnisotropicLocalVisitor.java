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

import org.streamingalgorithms.randomcutforest.IRFVisitor;
import org.streamingalgorithms.randomcutforest.IVisitorFactory;
import org.streamingalgorithms.randomcutforest.Visitor;
import org.streamingalgorithms.randomcutforest.returntypes.AnisotropicLocalGeometry;
import org.streamingalgorithms.randomcutforest.returntypes.LocalGeometryConfig;
import org.streamingalgorithms.randomcutforest.tree.ITree;

/**
 * Reads, in each tree, the three boxes of an {@link AnisotropicLocalGeometry}
 * from the transcript of a {@link SeparationVisitor}.
 *
 * <pre>
 *   cut box      quantile cutQuantile of the separation law, under the cap
 *   passage box  quantile passageQuantile, under the cap
 *   reading box  the first ancestor holding the reading mass
 * </pre>
 *
 * Both quantiles are read from the stop down: the stop is the first vertex
 * above the leaf whose own cut probability is at most stopProbability, and the
 * separation the law accumulates above it is not counted. The cap is the first
 * ancestor holding the cap mass. Since the reading mass is at least the cap
 * mass, cut box &sube; passage box &sube; reading box on every tree.
 *
 * <p>
 * The walk ends at the later of the stop and the reading box; the cap never
 * ends it. Above the reading box no box is materialized.
 */
public class AnisotropicLocalVisitor extends SeparationVisitor<AnisotropicLocalGeometry> {

    private static final int CUT = AnisotropicLocalGeometry.CUT;
    private static final int PASSAGE = AnisotropicLocalGeometry.PASSAGE;
    private static final int READING = AnisotropicLocalGeometry.READING;
    private static final int BOXES = AnisotropicLocalGeometry.BOXES;

    private final LocalGeometryConfig config;

    // per tree, found as the levels are recorded
    private double capMass;
    private double readingMass;
    private int stopLevel;
    private int capLevel;
    private int readingLevel;

    // per tree, filled by flush(); preallocated so a tree costs no allocation
    private final double[] reach;
    private final double[] gap;
    private final double[] boxMass = new double[BOXES];
    private final double[] boxCut = new double[BOXES];
    private final boolean[] capped = new boolean[BOXES];
    private boolean flushed;

    private final AnisotropicLocalGeometry.Accumulator folded;

    public AnisotropicLocalVisitor(int dimension, LocalGeometryConfig config) {
        super(dimension);
        checkNotNull(config, "config must not be null");
        this.config = config;
        this.reach = new double[BOXES * len];
        this.gap = new double[BOXES * len];
        this.folded = new AnisotropicLocalGeometry.Accumulator(dimension, config);
        reset();
    }

    public AnisotropicLocalVisitor(int dimension, int treeMass, LocalGeometryConfig config) {
        this(dimension, config);
        this.treeMass = treeMass;
    }

    @Override
    protected void reset() {
        super.reset();
        stopLevel = capLevel = readingLevel = -1;
        flushed = false;
    }

    @Override
    protected boolean levelRecorded(int level) {
        if (level == 0) {
            capMass = config.capMass(treeMass);
            readingMass = config.readingMass(treeMass);
        }
        double m = mass(level);
        if (capLevel < 0 && m >= capMass) {
            capLevel = level;
        }
        if (readingLevel < 0 && m >= readingMass) {
            readingLevel = level;
        }
        if (stopLevel < 0 && level > 0 && cutProbability(level) <= config.stopProbability()) {
            stopLevel = level;
        }
        return stopLevel >= 0 && readingLevel >= 0;
    }

    private void flush() {
        if (flushed) {
            return;
        }
        flushed = true;
        int top = levels() - 1;
        int stop = (stopLevel >= 0) ? stopLevel : top;
        int cutAt = quantileLevel(stop, config.cutQuantile());
        int passageAt = quantileLevel(stop, config.passageQuantile());
        capped[CUT] = capLevel >= 0 && capLevel < cutAt;
        capped[PASSAGE] = capLevel >= 0 && capLevel < passageAt;
        if (capped[CUT]) {
            cutAt = capLevel;
        }
        if (capped[PASSAGE]) {
            passageAt = capLevel;
        }
        // a root lighter than the reading mass gives the root, and says so
        capped[READING] = readingLevel < 0;
        int readingAt = capped[READING] ? top : readingLevel;
        read(CUT, cutAt);
        read(PASSAGE, passageAt);
        read(READING, readingAt);
    }

    private void read(int box, int level) {
        readLevel(level, reach, gap, box * len);
        boxMass[box] = mass(level);
        boxCut[box] = cutProbability(level);
    }

    @Override
    public void foldOut() {
        flush();
        if (levels() > 0) {
            folded.addTree(treeMass, reach, gap, boxMass, boxCut, capped);
        }
    }

    /** This tree alone, detached: a fresh result each call. */
    @Override
    public AnisotropicLocalGeometry getResult() {
        flush();
        AnisotropicLocalGeometry.Accumulator one = new AnisotropicLocalGeometry.Accumulator(dim, config);
        if (levels() > 0) {
            one.addTree(treeMass, reach, gap, boxMass, boxCut, capped);
        }
        return one.toGeometry();
    }

    @Override
    public AnisotropicLocalGeometry getFoldResult() {
        return folded.toGeometry();
    }

    @Override
    public void resetAcrossQueries(float[] point) {
        reset();
        folded.clear();
    }

    /**
     * Reusable, foldable, reusable across queries. A forest reduces its per-tree or
     * per-thread results with {@link AnisotropicLocalGeometry#merge}.
     */
    public static IVisitorFactory<AnisotropicLocalGeometry> factory(LocalGeometryConfig config) {
        checkNotNull(config, "config must not be null");
        return new IVisitorFactory<AnisotropicLocalGeometry>() {
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
            public IRFVisitor<AnisotropicLocalGeometry> newReusableVisitor(float[] point) {
                return new AnisotropicLocalVisitor(point.length, config);
            }

            @Override
            public Visitor<AnisotropicLocalGeometry> newVisitor(ITree<?, ?> tree, float[] point) {
                return new AnisotropicLocalVisitor(tree.projectToTree(point).length, tree.getMass(), config);
            }
        };
    }
}