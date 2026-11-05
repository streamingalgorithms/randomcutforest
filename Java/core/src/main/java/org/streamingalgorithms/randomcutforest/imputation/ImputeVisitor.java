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
 *
 * The following file is substantially modified from the original which had
 * this notice.
 *
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

package org.streamingalgorithms.randomcutforest.imputation;

import static org.streamingalgorithms.randomcutforest.CommonUtils.checkArgument;

import java.util.Arrays;
import java.util.Random;

import org.streamingalgorithms.randomcutforest.CommonUtils;
import org.streamingalgorithms.randomcutforest.IMultiVisitorFactory;
import org.streamingalgorithms.randomcutforest.IRFMultiVisitor;
import org.streamingalgorithms.randomcutforest.MultiVisitor;
import org.streamingalgorithms.randomcutforest.returntypes.ConditionalTreeSample;
import org.streamingalgorithms.randomcutforest.tree.*;

/**
 * A MultiVisitor which imputes missing values in a point. Reuse-capable: one
 * root instance is re-armed per tree via {@link #prepare} on the reusable
 * collector path.
 *
 * <p>
 * <b>Forks are pooled.</b> The traversal is depth-first and every fork is
 * combined before its parent forks again, last in first out, so at most depth
 * forks are live at once. {@link #newPartialCopy} takes the next frame of a
 * pool the root and its forks share, re-armed exactly as a fresh copy would be,
 * and {@link #combine} returns it. The pool grows to the deepest nesting of the
 * first trees and allocates nothing after.
 *
 * <p>
 * <b>Forks share the working buffer.</b> Above a fork, every leaf below it lies
 * inside each ancestor's box, so whichever branch's imputed values the buffer
 * holds, the missing coordinates add no gap there: the cut probabilities of the
 * ancestors depend on the known coordinates only, which the branches share.
 *
 * <p>
 * <b>A root can serve many queries.</b> {@link #rearm} puts a root in the state
 * a new root built from the same arguments would have, in its own buffers, so a
 * caller that keeps one root (RandomCutForest on its sequential path) allocates
 * nothing per query for it; the pool of forks is kept as well.
 */

public class ImputeVisitor implements IRFMultiVisitor<ConditionalTreeSample> {

    public static double DEFAULT_INIT_VALUE = Double.MAX_VALUE;

    protected final boolean[] missing;
    protected float[] queryPoint;
    protected float[] expandedPoint;
    protected final float[] queryPointOriginal;
    protected double anomalyRank;
    protected double distance;
    protected double centrality;
    protected long randomSeed;
    protected Random rng;
    protected double randomRank;
    protected boolean converged;
    protected int pointIndex;
    protected int[] dimensionsUsed;
    protected ArrayBox box; // first-internal-node sentinel ONLY; never retained
    /** The missing coordinates, as indices: the first missingCount entries. */
    protected final int[] missingIndex;
    protected int missingCount;
    /** The forks of this root, shared by the root and every fork it makes. */
    protected final Forks forks;

    /** Fork frames, reused last in first out; see the class comment. */
    static final class Forks {
        ImputeVisitor[] frames = new ImputeVisitor[16];
        int top;
    }

    public ImputeVisitor(float[] liftedPoint, float[] queryPoint, int[] liftedMissingIndexes, int[] missingIndexes,
            double centrality, long randomSeed) {
        checkArgument(centrality >= 0, " cannoit be negative ");
        checkArgument(centrality <= 1.0, " cannot be more than 1.0");
        this.queryPoint = Arrays.copyOf(queryPoint, queryPoint.length);
        this.queryPointOriginal = Arrays.copyOf(queryPoint, queryPoint.length);
        this.expandedPoint = new float[2 * queryPoint.length];
        VectorSupport.expandInto(queryPoint, 0, expandedPoint, 0, queryPoint.length); // allocation
        this.missing = new boolean[queryPoint.length];
        this.centrality = centrality;
        this.randomSeed = randomSeed;
        this.dimensionsUsed = new int[queryPoint.length];
        this.rng = new Random(randomSeed);
        if (missingIndexes == null) {
            missingIndexes = new int[0];
        }

        for (int i = 0; i < missingIndexes.length; i++) {
            checkArgument(0 <= missingIndexes[i], "Missing value indexes cannot be negative");
            checkArgument(missingIndexes[i] < queryPoint.length,
                    "Missing value indexes must be less than query length");
            missing[missingIndexes[i]] = true;
        }
        this.missingIndex = new int[queryPoint.length];
        this.missingCount = indexMissing(missing, missingIndex);
        this.forks = new Forks();

        anomalyRank = DEFAULT_INIT_VALUE;
        distance = DEFAULT_INIT_VALUE;
    }

    public ImputeVisitor(float[] queryPoint, int numberOfMissingIndices, int[] missingIndexes) {
        this(queryPoint, Arrays.copyOf(queryPoint, queryPoint.length),
                Arrays.copyOf(missingIndexes, Math.min(numberOfMissingIndices, missingIndexes.length)),
                Arrays.copyOf(missingIndexes, Math.min(numberOfMissingIndices, missingIndexes.length)), 1.0, 0L);
    }

    /** Writes the indices of the missing flags into index; returns their number. */
    private static int indexMissing(boolean[] missing, int[] index) {
        int count = 0;
        for (int i = 0; i < missing.length; i++) {
            if (missing[i]) {
                index[count++] = i;
            }
        }
        return count;
    }

    /**
     * Re-arms this root for another query of the same dimension: afterwards it is
     * what
     * {@code new ImputeVisitor(point, point, missing, missing, centrality, 0L)}
     * would be, the root the reusable factory makes, but in this root's buffers.
     * The random stream restarts from the same seed, so a query's result does not
     * depend on the queries before it. Only a root is re-armed this way; its forks
     * follow it through {@link #newPartialCopy}.
     */
    public ImputeVisitor rearm(float[] point, int[] missingIndexes, double centrality) {
        checkArgument(centrality >= 0, " cannoit be negative ");
        checkArgument(centrality <= 1.0, " cannot be more than 1.0");
        checkArgument(point.length == queryPointOriginal.length, "incorrect dimension");
        System.arraycopy(point, 0, queryPointOriginal, 0, point.length);
        System.arraycopy(point, 0, queryPoint, 0, point.length);
        VectorSupport.expandInto(queryPoint, 0, expandedPoint, 0, queryPoint.length);
        Arrays.fill(missing, false);
        if (missingIndexes != null) {
            for (int i : missingIndexes) {
                checkArgument(0 <= i, "Missing value indexes cannot be negative");
                checkArgument(i < queryPoint.length, "Missing value indexes must be less than query length");
                missing[i] = true;
            }
        }
        missingCount = indexMissing(missing, missingIndex);
        this.centrality = centrality;
        this.randomSeed = 0L;
        rng.setSeed(0L);
        Arrays.fill(dimensionsUsed, 0);
        forks.top = 0;
        anomalyRank = DEFAULT_INIT_VALUE;
        distance = DEFAULT_INIT_VALUE;
        randomRank = 0.0;
        converged = false;
        pointIndex = 0;
        box = null;
        return this;
    }

    // ---- reuse: re-arm this ROOT visitor for the next tree ----

    @Override
    public void prepare(ITree<?, ?> tree, float[] rawPoint) {
        // projectToTree is identity for the single tree class -> restore constant.
        System.arraycopy(queryPointOriginal, 0, queryPoint, 0, queryPoint.length);
        Arrays.fill(dimensionsUsed, 0);
        box = null; // sentinel reset (load-bearing)
        forks.top = 0; // a tree starts with no live forks
        anomalyRank = DEFAULT_INIT_VALUE;
        distance = DEFAULT_INIT_VALUE;
        converged = false;
        pointIndex = 0;
        randomSeed = tree.getRandomSeed(); // match legacy: factory seeded per tree
    }

    public void accept(final INodeView node, final int depthOfNode) {
        // note querypoint now has values filled -- either expand or we do not use simd
        double p = ((ArrayBox) node.getBoundingBox()).probabilityOfCut(expandedPoint, null, null);
        converged = (p == 0);
        if (p <= 0)
            return;
        anomalyRank = p * scoreUnseen(depthOfNode, node.getMass()) + (1 - p) * anomalyRank;
    }

    @Override
    public void acceptLeaf(final INodeView leafNode, final int depthOfNode) {
        float[] leafPoint = leafNode.getLeafPoint();
        pointIndex = leafNode.getLeafPointIndex();
        // the missing coordinates take the leaf's values, so they add nothing to the
        // L1 distance over all coordinates, which is then one SIMD kernel
        for (int k = 0; k < missingCount; k++) {
            int i = missingIndex[k];
            queryPoint[i] = leafPoint[i];
        }
        double distance = VectorSupport.L1distance(queryPoint, leafPoint);
        VectorSupport.expandInto(queryPoint, 0, expandedPoint, 0, queryPoint.length);
        if (centrality < 1.0) {
            // Random rng = new Random(randomSeed);
            randomSeed = rng.nextLong();
            randomRank = rng.nextDouble();
        }

        this.distance = distance;
        if (distance <= 0) {
            converged = true;
            if (depthOfNode == 0) {
                anomalyRank = 0;
            } else {
                anomalyRank = scoreSeen(depthOfNode, leafNode.getMass());
            }
        } else {
            anomalyRank = scoreUnseen(depthOfNode, leafNode.getMass());
        }
    }

    /**
     * @return the imputed point. Note the pointstore ref is all what is usually
     *         needed
     */
    public ConditionalTreeSample getResult() {
        return new ConditionalTreeSample(pointIndex, null, distance, null); // leafPoint filled at lift
    }

    @Override
    public boolean trigger(final INodeView node) {
        int index = node.getCutDimension();
        ++dimensionsUsed[index];
        return missing[index];
    }

    protected double getAnomalyRank() {
        return anomalyRank;
    }

    /** Whether coordinate i is missing in the query this root is armed for. */
    public boolean isMissing(int i) {
        return missing[i];
    }

    /**
     * The point-store index of the leaf this tree's walk settled on, what
     * {@link #getResult} wraps; read after the tree, without allocating.
     */
    public int getLeafPointIndex() {
        return pointIndex;
    }

    protected double getDistance() {
        return distance;
    }

    @Override
    public MultiVisitor<ConditionalTreeSample> newPartialCopy() {
        if (forks.top == forks.frames.length) {
            forks.frames = Arrays.copyOf(forks.frames, 2 * forks.top);
        }
        ImputeVisitor fork = forks.frames[forks.top];
        if (fork == null) {
            fork = new ImputeVisitor(this);
            forks.frames[forks.top] = fork;
        } else {
            fork.armFrom(this);
        }
        forks.top++;
        return fork;
    }

    ImputeVisitor(ImputeVisitor original) {
        this.queryPointOriginal = original.queryPointOriginal; // immutable; forks never call prepare
        this.missing = original.missing; // immutable after construction
        this.missingIndex = original.missingIndex;
        this.forks = original.forks;
        armFrom(original);
    }

    /**
     * Re-arms a fork from the visitor forking: the state a fresh copy would have.
     * The buffers are shared, not copied (see the class comment); the random stream
     * advances once per fork, as it always has.
     */
    private void armFrom(ImputeVisitor original) {
        this.queryPoint = original.queryPoint;
        this.expandedPoint = original.expandedPoint;
        this.dimensionsUsed = original.dimensionsUsed; // write-only (trigger++), never read
        this.missingCount = original.missingCount;
        this.centrality = original.centrality;
        this.randomSeed = original.rng.nextLong(); // in case we needed it
        this.rng = original.rng;
        anomalyRank = DEFAULT_INIT_VALUE;
        distance = DEFAULT_INIT_VALUE;
        randomRank = 0.0;
        converged = false;
        pointIndex = 0;
        box = null;
    }

    double adjustedRank() {
        return (1 - centrality) * randomRank + centrality * anomalyRank;
    }

    protected boolean updateCombine(ImputeVisitor other) {
        return other.adjustedRank() < adjustedRank();
    }

    @Override
    public void combine(MultiVisitor<ConditionalTreeSample> other) {
        ImputeVisitor visitor = (ImputeVisitor) other;
        if (updateCombine(visitor)) {
            updateFrom(visitor);
        }
        // the fork is done: back to the pool, last in first out
        if (forks.top > 0 && forks.frames[forks.top - 1] == visitor) {
            forks.top--;
        }
    }

    protected void updateFrom(ImputeVisitor visitor) {
        System.arraycopy(visitor.queryPoint, 0, queryPoint, 0, queryPoint.length);
        pointIndex = visitor.pointIndex;
        anomalyRank = visitor.anomalyRank;
        box = visitor.box; // sentinel state follows the winning branch (unchanged)
        converged = visitor.converged;
        distance = visitor.distance;
    }

    protected double scoreSeen(int depth, int mass) {
        return CommonUtils.defaultScoreSeenFunction(depth, mass);
    }

    protected double scoreUnseen(int depth, int mass) {
        return CommonUtils.defaultScoreUnseenFunction(depth, mass);
    }

    @Override
    public boolean isConverged() {
        return converged;
    }

    public static IMultiVisitorFactory<ConditionalTreeSample> factory(int[] liftedIndices, double centrality) {
        return new IMultiVisitorFactory<>() {
            @Override
            public MultiVisitor<ConditionalTreeSample> newVisitor(ITree<?, ?> tree, float[] y) {
                return new ImputeVisitor(y, tree.projectToTree(y), liftedIndices,
                        tree.projectMissingIndices(liftedIndices), centrality, tree.getRandomSeed());
            }

            @Override
            public boolean isReusable() {
                return true;
            }

            @Override
            public IRFMultiVisitor<ConditionalTreeSample> newReusableMultiVisitor(float[] y) {
                return new ImputeVisitor(y, y, liftedIndices, liftedIndices, centrality, 0L);
            }
        };
    }
}