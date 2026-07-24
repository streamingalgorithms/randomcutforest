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
 * collector path; forks still allocate via {@link #newPartialCopy} (a later,
 * independent cut).
 *
 *
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

        anomalyRank = DEFAULT_INIT_VALUE;
        distance = DEFAULT_INIT_VALUE;
    }

    public ImputeVisitor(float[] queryPoint, int numberOfMissingIndices, int[] missingIndexes) {
        this(queryPoint, Arrays.copyOf(queryPoint, queryPoint.length),
                Arrays.copyOf(missingIndexes, Math.min(numberOfMissingIndices, missingIndexes.length)),
                Arrays.copyOf(missingIndexes, Math.min(numberOfMissingIndices, missingIndexes.length)), 1.0, 0L);
    }

    // ---- reuse: re-arm this ROOT visitor for the next tree ----

    @Override
    public void prepare(ITree<?, ?> tree, float[] rawPoint) {
        // projectToTree is identity for the single tree class -> restore constant.
        System.arraycopy(queryPointOriginal, 0, queryPoint, 0, queryPoint.length);
        Arrays.fill(dimensionsUsed, 0);
        box = null; // sentinel reset (load-bearing)
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
        double distance = 0;
        for (int i = 0; i < queryPoint.length; i++) {
            if (missing[i]) {
                queryPoint[i] = leafPoint[i];
            } else {
                double t = (queryPoint[i] - leafPoint[i]);
                distance += Math.abs(t);
            }
        }
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

    protected double getDistance() {
        return distance;
    }

    @Override
    public MultiVisitor<ConditionalTreeSample> newPartialCopy() {
        return new ImputeVisitor(this);
    }

    ImputeVisitor(ImputeVisitor original) {
        // this.queryPoint = Arrays.copyOf(original.queryPoint,
        // original.queryPoint.length); // the ONE it must own
        this.queryPoint = original.queryPoint;
        this.queryPointOriginal = original.queryPointOriginal; // immutable; forks never call prepare
        this.expandedPoint = original.expandedPoint;
        this.missing = original.missing; // immutable after construction
        this.dimensionsUsed = original.dimensionsUsed; // write-only (trigger++), never read
        this.centrality = original.centrality;
        this.randomSeed = original.rng.nextLong(); // in case we needed it
        this.rng = original.rng;
        anomalyRank = DEFAULT_INIT_VALUE;
        distance = DEFAULT_INIT_VALUE;
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