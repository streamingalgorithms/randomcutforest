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
 * This file is substantially modified from the file which had the following notice.
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

package org.streamingalgorithms.randomcutforest.tree;

import static java.lang.Math.*;
import static org.streamingalgorithms.randomcutforest.CommonUtils.checkArgument;
import static org.streamingalgorithms.randomcutforest.CommonUtils.checkNotNull;
import static org.streamingalgorithms.randomcutforest.CommonUtils.checkState;
import static org.streamingalgorithms.randomcutforest.sampler.CompactSampler.SEQUENCE_INDEX_NA;
import static org.streamingalgorithms.randomcutforest.tree.NodeStore.DEFAULT_STORE_PARENT;
import static org.streamingalgorithms.randomcutforest.tree.NodeStore.Null;
import static org.streamingalgorithms.randomcutforest.tree.VectorSupport.gapAttribution;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Optional;
import java.util.Random;

import org.streamingalgorithms.randomcutforest.*;
import org.streamingalgorithms.randomcutforest.config.Config;
import org.streamingalgorithms.randomcutforest.store.IPointStoreView;

import it.unimi.dsi.fastutil.ints.Int2IntMap;
import it.unimi.dsi.fastutil.ints.Int2IntOpenHashMap;
import it.unimi.dsi.fastutil.objects.ObjectIterator;
import lombok.Getter;

/**
 *
 * The offsets are encoded as follows: an offset greater or equal maxSize
 * corresponds to a leaf node of offset (offset - maxSize) otherwise the offset
 * corresponds to an internal node
 *
 */
public class RandomCutTree implements ITree<Integer, float[]> {

    /**
     * The index value used to represent the absence of a node. For example, when
     * the tree is created the root node index will be NULL. After a point is added
     * and a root node is created, the root node's parent will be NULL, and so on.
     */

    private Random testRandom;
    @Getter
    protected boolean storeSequenceIndexesEnabled;
    @Getter
    protected boolean centerOfMassEnabled;
    @Getter
    private long randomSeed;
    protected int root;
    protected IPointStoreView<float[]> pointStoreView;
    @Getter
    protected int numberOfLeaves;
    @Getter
    protected NodeStore nodeStore;
    @Getter
    protected double boundingBoxCacheFraction;
    @Getter
    protected int outputAfter;
    @Getter
    protected int dimension;
    protected final Int2IntOpenHashMap leafMass;
    protected double[] rangeSumData;
    protected float[] boundingBoxData;
    protected float[] pointSum;
    protected HashMap<Integer, List<Long>> sequenceMap;
    protected int basicCache;
    protected int limit;
    protected Int2IntOpenHashMap smallMap;
    protected Int2IntOpenHashMap spareMap; // double buffer; dRV set once at construction

    public static int MAX_BASIC_CACHE = Long.SIZE;

    // the following is a separation oracle that takes a bounding box and generates
    // a cut
    @FunctionalInterface
    public interface SeparationOracle {
        double computeGap(IBoundingBoxView box, double[] target);
    }

    protected RandomCutTree(Builder<?> builder) {
        pointStoreView = builder.pointStoreView;
        numberOfLeaves = builder.capacity;
        randomSeed = builder.randomSeed;
        testRandom = builder.random;
        outputAfter = builder.outputAfter.orElse(max(1, numberOfLeaves / 4));
        dimension = (builder.dimension != 0) ? builder.dimension : pointStoreView.getDimensions();
        nodeStore = (builder.nodeStore != null) ? builder.nodeStore
                : NodeStore.nodeStore(numberOfLeaves - 1, dimension, pointStoreView.getCapacity() + numberOfLeaves - 1L,
                        builder.storeParent);

        this.boundingBoxCacheFraction = builder.boundingBoxCacheFraction;
        this.storeSequenceIndexesEnabled = builder.storeSequenceIndexesEnabled;
        this.centerOfMassEnabled = builder.centerOfMassEnabled;
        this.root = builder.root;
        leafMass = new Int2IntOpenHashMap();
        leafMass.defaultReturnValue(0);
        if (this.centerOfMassEnabled) {
            pointSum = new float[(numberOfLeaves - 1) * dimension];
        }
        if (this.storeSequenceIndexesEnabled) {
            sequenceMap = new HashMap<>();
        }
        resizeCache(boundingBoxCacheFraction);
    }

    @Override
    public <T> void setConfig(String name, T value, Class<T> clazz) {
        if (Config.BOUNDING_BOX_CACHE_FRACTION.equals(name)) {
            checkArgument(Double.class.isAssignableFrom(clazz),
                    () -> String.format("Setting '%s' must be a double value", name));
            setBoundingBoxCacheFraction((Double) value);
        } else {
            throw new IllegalArgumentException("Unsupported configuration setting: " + name);
        }
    }

    @Override
    public <T> T getConfig(String name, Class<T> clazz) {
        checkNotNull(clazz, "clazz must not be null");
        if (Config.BOUNDING_BOX_CACHE_FRACTION.equals(name)) {
            checkArgument(clazz.isAssignableFrom(Double.class),
                    () -> String.format("Setting '%s' must be a double value", name));
            return clazz.cast(boundingBoxCacheFraction);
        } else {
            throw new IllegalArgumentException("Unsupported configuration setting: " + name);
        }
    }

    // dynamically change the fraction of the new nodes which caches their bounding
    // boxes. However in prior releases, it was not made explicit that any value
    // other
    // than 0 or 1 would have changed state (i.e., writes, the essence of caching!)
    // 0 would mean less space usage, but slower throughput
    // 1 would imply larger space but better throughput
    // Note that switching the values is also a change of state
    // These two values {0,1} should not change state
    // We recommend considering 0.001 that provides a minimal data dependent
    // caching (which is one of the reasons for caching) and yet uses little space
    public void setBoundingBoxCacheFraction(double fraction) {
        checkArgument(0 <= fraction && fraction <= 1, "incorrect parameter");
        boundingBoxCacheFraction = fraction;
        resizeCache(fraction);
    }

    /**
     * Return a new {@link Cut}, which is chosen uniformly over the space of
     * possible cuts for a bounding box. Either the bounding box is non-trivial (not
     * a single point) or its union with a point has to be non-trivial -- otherwise
     * there is no cut. The point can be null, so that this single cut routine and
     * its logic can be used everywhere.
     *
     * @param factor between 0 and 1, (in infinite precision) indicates the cut
     * @param point  the point whose union is taken with the box (can be null)
     * @param box    A bounding box that we want to find a random cut for.
     * @param oracle A separation oracle; which can be null in which case we will
     *               use Max - Min in that coordinate, widened to double; the
     *               invariant must be that iff the float gap is non-zero then the
     *               gap computed must be non-zero (and nextAfter is defined)
     * @param output Cut corresponding to a random cut in the bounding box.
     */
    protected void randomCut(double factor, float[] point, ArrayBox box, double[] oracle, boolean union, Cut output) {
        checkArgument(box != null, "box cannot be null, can be a point");
        double range = 0.0;
        int dimension = box.getDimensions();
        if (union) {
            // point cannot be null
            // oracle has to be null -- because the oracle has no union
            for (int i = 0; i < dimension; i++) {
                range += max(point[i], box.values[i + box.offset])
                        + max(-point[i], box.values[i + box.offset + dimension]);
            }
        } else {
            if (oracle == null) {
                range = box.rangeSum;
            } else {
                checkArgument(oracle.length == dimension, "incorrect probability specification");
                for (int i = 0; i < dimension; i++) {
                    checkArgument(oracle[i] >= 0, "cannot be negative");
                    range += oracle[i];
                }
            }
        }

        checkArgument(range > 0, () -> " the union is a single point " + Arrays.toString(point)
                + "or the box is inappropriate, box" + box.toString() + "factor =" + factor);

        double breakPoint = factor * range;

        for (int i = 0; i < dimension; i++) {
            double gap = (union)
                    ? max(point[i], box.values[i + box.offset]) + max(-point[i], box.values[i + box.offset + dimension])
                    : box.values[i + box.offset] + box.values[i + box.offset + dimension];
            // the invariant we must maintain is that if the float values
            // are not the same then the gap must be non-zeor
            // double gap = maxValue - minValue;
            if (breakPoint <= gap && gap > 0) {
                float minValue = box.getMinValue(i);
                float maxValue = box.getMaxValue(i);
                if (union) {
                    minValue = min(minValue, point[i]);
                    maxValue = max(maxValue, point[i]);
                }
                float cutValue = (float) (minValue + breakPoint);

                // Random cuts have to take a value in the half-open interval [minValue,
                // maxValue) to ensure that a
                // Node has a valid left child and right child.
                if (cutValue >= maxValue) {
                    cutValue = Math.nextAfter((float) maxValue, minValue);
                }

                output.set(i, cutValue);
                return;
            }
            breakPoint -= gap;
        }
        cleanup(factor, box, point, output);
    }

    /**
     * Cleanup fires for anomalous distributions -- it ignores all ranges and
     * performs a nontrivial cut in the first dimension it can cut in, starting
     * evenly from 0 or the box dimension-1
     * 
     * @param factor random seed to control/reproduce behavior
     * @param box    the pathological box in question
     * @param point  can be null, non-null points are unioned
     * @param output the output cut
     */
    protected void cleanup(double factor, ArrayBox box, float[] point, Cut output) {

        // if we are here then factor is likely almost 1 and we have floating point
        // issues. we will randomize between the first and the last non-zero ranges
        // and choose the same cutValue as using nextAfter above -- we will use the
        // factor as a seed and not be optimizing this (either in execution or code)
        // to ensure easier debugging. this should be an anomaly - no pun intended.

        // this is an allocation, but this should be rare
        ArrayBox merged = (point != null) ? box.copy().addPoint(point) : box;
        Random rng = new Random((long) (factor * Long.MAX_VALUE / 2));
        if (rng.nextDouble() < 0.5) {
            for (int i = 0; i < merged.getDimensions(); i++) {
                float minValue = (float) merged.getMinValue(i);
                float maxValue = (float) merged.getMaxValue(i);
                if (maxValue > minValue) {
                    double cutValue = Math.nextAfter((float) maxValue, minValue);
                    output.set(i, cutValue);
                    return;
                }
            }
        } else {
            for (int i = merged.getDimensions() - 1; i >= 0; i--) {
                float minValue = (float) merged.getMinValue(i);
                float maxValue = (float) merged.getMaxValue(i);
                if (maxValue > minValue) {
                    double cutValue = Math.nextAfter((float) maxValue, minValue);
                    output.set(i, cutValue);
                    return;
                }
            }
        }

        // flag
        checkArgument(false, () -> "The break point did not lie inside the expected range; factor " + factor
                + ", point " + Arrays.toString(point) + " box " + box.toString());
    }

    // the following are just for testing and not to disturb the 100s of tests
    protected Cut randomCut(double factor, float[] point, BoundingBox box) {
        Cut output = new Cut(0, 0);
        randomCut(factor, point, new ArrayBox(box.minValues, box.maxValues), null, true, output);
        return output;
    }

    protected void randomCut(double factor, ArrayBox box, Cut output) {
        randomCut(factor, null, box, null, false, output);
    }

    /**
     * the following function adds a point to the tree
     * 
     * @param pointIndex    the number corresponding to the point
     * @param sequenceIndex sequence index of the point
     * @return the value of the point index where the point was added; this is
     *         pointIndex if there are no duplicates; otherwise it is the value of
     *         the point being duplicated.
     */
    public Integer addPoint(Integer pointIndex, long sequenceIndex, UpdateHelper<Integer> helper) {
        if (helper == null) {
            helper = new UpdateHelper<Integer>(dimension, numberOfLeaves);
        } else {
            helper.resize(numberOfLeaves); // should be a no-op
        }
        if (root == Null) {
            root = nodeStore.convertToLeaf(pointIndex);
            addLeaf(pointIndex, sequenceIndex);
            return pointIndex;
        }

        // updates are single threaded
        pointStoreView.getNumericVectorInto(pointIndex, helper.pointScratch);
        // the following is redundant, but left as a placeholder as
        // a starting place for any subsequent development
        float[] point = projectToTree(helper.pointScratch);
        int depth = nodeStore.getPathInto(root, point, helper.nodeScratch);
        int leafIdx = depth - 1;
        int leafNode = helper.nodeScratch[leafIdx];
        int leafPointIndex = getPointIndex(leafNode);

        if (pointStoreView.isEqual(leafPointIndex, helper.pointScratch)) {
            increaseLeafMass(leafNode);
            manageAncestorsAdd(leafIdx, helper.nodeScratch, point, true);
            addLeaf(leafPointIndex, sequenceIndex);
            return leafPointIndex;
        }
        pointStoreView.setAsSlice(leafPointIndex, helper.boxScratch.values, helper.boxScratch.offset);
        helper.boxScratch.rangeSum = 0;
        helper.updateBox.copyFrom(helper.boxScratch); // make a copy

        int savedIdx = leafIdx, savedDim = Integer.MAX_VALUE;
        float savedCutValue = 0f;
        Random rng;
        if (testRandom == null) {
            rng = new Random(randomSeed);
            randomSeed = rng.nextLong();
        } else
            rng = testRandom;

        int i = leafIdx;
        while (true) {
            double factor = rng.nextDouble();
            randomCut(factor, point, helper.boxScratch, null, true, helper.cutScratch);
            int dim = helper.cutScratch.getDimension();
            float value = (float) helper.cutScratch.getValue();
            if ((point[dim] <= value && value < helper.boxScratch.getMinValue(dim))
                    || (point[dim] > value && value >= helper.boxScratch.getMaxValue(dim))) {
                savedCutValue = value;
                savedDim = dim;
                savedIdx = i;
                helper.updateBox.copyFrom(helper.boxScratch);
                // discard prior (lower in the tree) update
            }
            if (helper.boxScratch.contains(point) || i == 0)
                break;
            growArrayBox(helper.boxScratch, pointStoreView,
                    nodeStore.getSibling(helper.nodeScratch[i], helper.nodeScratch[i - 1]));
            --i;
        }

        int savedNode = helper.nodeScratch[savedIdx];
        int savedParent = (savedIdx == 0) ? Null : helper.nodeScratch[savedIdx - 1];
        int childMassIfLeaf = isLeaf(savedNode) ? getLeafMass(savedNode) : 0;
        int mergedNode = nodeStore.addNode(savedParent, point, sequenceIndex, pointIndex, savedNode, childMassIfLeaf,
                savedDim, savedCutValue);
        addLeaf(pointIndex, sequenceIndex);
        int idx = translate(mergedNode);
        if (idx != Integer.MAX_VALUE) {
            copyArrayBoxToData(idx, helper.updateBox);
            rangeSumData[idx] = ArrayBox.addPointInPlace(boundingBoxData, 2 * idx * dimension, dimension, point, 0);
        }
        manageAncestorsAdd(savedIdx, helper.nodeScratch, point, false); // nodeScratch[0..savedIdx)
        if (pointSum != null)
            recomputePointSum(mergedNode);
        if (savedParent == Null)
            root = mergedNode;
        return pointIndex;

    }

    public void makeTree(int size, int[] indexList, int[] outputList, int[] pointList, long[] sequenceIndex,
            SeparationOracle oracle, long seed, double[] attributionScratch, UpdateHelper helper) {
        // this function allows a public call, which may be useful someday
        // that day is today ....
        if (size > 0 && size <= numberOfLeaves) {
            checkArgument(indexList.length <= size, "incorrect input buffer");
            checkArgument(outputList.length <= size, "incorrect output buffer");
            checkArgument(sequenceIndex == null || (pointList.length == sequenceIndex.length), "mismatched input");

            Random ring = new Random(seed);

            root = makeTreeInt(indexList, outputList, 0, size, pointList, 0, oracle, ring, attributionScratch, helper);
            // the cuts are specififed; now build tree
            // note the contents of the indexList will be permuted.
            for (int i = 0; i < size; i++) {
                long seq = (sequenceIndex != null) ? sequenceIndex[indexList[i]] : SEQUENCE_INDEX_NA;
                checkArgument(outputList[i] == addPointToPartialTree(pointList[indexList[i]], seq, helper),
                        "error in construction");
            }
            if (root != Null) {
                validateAndReconstruct(root, false, true, centerOfMassEnabled);
            }
        } else {
            root = Null;
        }
    }

    // the following takes an int[] chosen, which initially corresponds to entries
    // in pointlist
    // the function modifies the subarray [start,end), with the node position
    // firstFree
    // it uses thr scratch to perform a stable reordering;
    // at the end it returns that node position (internal or leaf) and reorders
    // chosen[]
    // the outputList is populated such that chosen[i] correponds to the inserted
    // point (which is revealed only after insertion)
    int makeTreeInt(int[] chosen, int[] output, int start, int end, int[] pointList, int firstFree,
            SeparationOracle vecBuild, Random ring, double[] attributionScratch, UpdateHelper helper) {

        if (end - start == 0)
            return Null;
        pointStoreView.getNumericVectorInto(pointList[chosen[start]], helper.pointScratch);
        helper.boxScratch.fromPoint(helper.pointScratch);

        // output is not written yet
        for (int i = start + 1; i < end; i++) {
            output[i] = pointList[chosen[i]];

        }
        helper.boxScratch.rangeSum = pointStoreView.addToSlice(helper.boxScratch.values, 0, output, start + 1, end);
        // pointStoreView.getNumericVectorInto(pointList[chosen[i]],
        // helper.pointScratch);
        // helper.boxScratch.addPoint(helper.pointScratch);
        // we will overwrite output now
        if (helper.boxScratch.getRangeSum() <= 0) {
            int rep = pointList[chosen[start]];
            // all of these points are now the same as rep
            // note chosen is not changed,
            for (int i = start; i < end; i++) {
                output[i] = rep;
            }
            return rep + nodeStore.getCapacity() + 1;
        }

        double factor = ring.nextDouble();
        if (vecBuild != null) {
            vecBuild.computeGap(helper.boxScratch, attributionScratch);
            randomCut(factor, null, helper.boxScratch, attributionScratch, false, helper.cutScratch);
        } else {
            randomCut(factor, null, helper.boxScratch, null, false, helper.cutScratch); // box-range path
        }

        float cutValue = (float) helper.cutScratch.getValue();
        int cutDimension = helper.cutScratch.getDimension();

        // stable in-place partition of [start, end):
        // lefts compacted into [start, mid); rights buffered in order, then written
        // into [mid, end)
        int write = start;
        int r = 0;
        for (int j = start; j < end; j++) {
            int v = chosen[j];
            if (pointStoreView.leftOf(pointList[v], cutValue, cutDimension)) {
                chosen[write++] = v;
            } else {
                helper.nodeScratch[r++] = v;
            }
        }
        int mid = write;
        for (int k = 0; k < r; k++) {
            chosen[mid + k] = helper.nodeScratch[k];
        }
        // scratch is no longer useful
        int idx = translate(firstFree);
        if (idx != Integer.MAX_VALUE) {
            copyArrayBoxToData(idx, helper.boxScratch);
        }
        // box and cut can be resued subsequently
        int leftCount = mid - start;
        int leftIndex = makeTreeInt(chosen, output, start, mid, pointList, firstFree + 1, vecBuild, ring,
                attributionScratch, helper);
        int rightIndex = makeTreeInt(chosen, output, mid, end, pointList, firstFree + leftCount, vecBuild, ring,
                attributionScratch, helper);
        nodeStore.addRecord(firstFree, min(leftIndex, numberOfLeaves - 1), min(rightIndex, numberOfLeaves - 1),
                cutValue, cutDimension);
        return firstFree;
    }

    private void manageAncestorsAdd(int count, int[] nodeScratch, float[] point, boolean boxesUnchanged) {
        boolean resolved = boxesUnchanged;
        for (int i = count - 1; i >= 0; i--) {
            int node = nodeScratch[i];
            nodeStore.increaseMassOfInternalNode(node);
            if (pointSum != null) {
                recomputePointSum(node);
            }
            if (!resolved) {
                resolved = growBoxOrResolve(node, point);
            }
        }
    }

    /**
     * Bottom-up box maintenance for an insertion. Children of index are already
     * current, so the only change to this subtree is the one new point: box_new =
     * box_old ∪ {point}.
     *
     * @return true iff box_old already contained point -- this box is unchanged,
     *         and by nesting so is every ancestor's, so the caller can stop doing
     *         box work.
     */
    private boolean growBoxOrResolve(int index, float[] point) {
        int idx = translate(index);
        if (idx == Integer.MAX_VALUE) {
            return false; // uncached: nothing to store, nothing proven
        }
        int base = 2 * idx * dimension;
        if (rangeSumData[idx] == 0.0) { // stale slot: contents are meaningless
            rangeSumData[idx] = buildInto(boundingBoxData, base, index, pointStoreView);
            return false; // rebuilt box trivially contains point; proves nothing above
        }
        if (containsPoint(base, point)) {
            return true;
        }
        rangeSumData[idx] = ArrayBox.addPointInPlace(boundingBoxData, base, dimension, point, 0);
        return false;
    }

    /** non-strict; values[0,d) are maxima, values[d,2d) are negated minima */
    private boolean containsPoint(int base, float[] point) {
        for (int i = 0; i < dimension; i++) {
            if (point[i] > boundingBoxData[base + i] || -point[i] > boundingBoxData[base + i + dimension]) {
                return false;
            }
        }
        return true;
    }

    /**
     * the following function is used to rebuild the tree structure. This function
     * does not create mass, auxiliary arrays, which should be performed using
     * validateAndReconstruct()
     * 
     * @param pointIndex    index of point (in point store)
     * @param sequenceIndex sequence index (stored in sampler)
     */
    public int addPointToPartialTree(int pointIndex, long sequenceIndex, UpdateHelper<?> helper) {
        checkArgument(root != Null, " a null root is not a partial tree");

        // Single-threaded scratchpad reuse (Excellent for performance!)
        pointStoreView.getNumericVectorInto(pointIndex, helper.pointScratch);
        float[] point = projectToTree(helper.pointScratch);
        // checkArgument(point.length == dimension, () -> " incorrect projection at
        // index " + pointIndex);

        // --- ALLOCATION-FREE WALK ---
        int parentNode = Null;
        int leafNode = root;

        while (isInternal(leafNode)) {
            parentNode = leafNode;
            if (nodeStore.leftOf(leafNode, point)) {
                leafNode = nodeStore.getLeftIndex(leafNode);
            } else {
                leafNode = nodeStore.getRightIndex(leafNode);
            }
        }
        // -----------------------------

        if (!isLeaf(leafNode)) {
            if (parentNode == Null) {
                root = nodeStore.convertToLeaf(pointIndex);
            } else {
                nodeStore.assignInPartialTree(parentNode, point, nodeStore.convertToLeaf(pointIndex));
                addLeaf(pointIndex, sequenceIndex);
            }
            return pointIndex;
        }

        // Standard leaf insertion path
        int leafPointIndex = getPointIndex(leafNode);
        checkArgument(pointStoreView.isEqual(leafPointIndex, helper.pointScratch),
                () -> "incorrect state on adding " + pointIndex);

        increaseLeafMass(leafNode);
        addLeaf(leafPointIndex, sequenceIndex);
        return leafPointIndex;
    }

    public Integer deletePoint(Integer pointIndex, long sequenceIndex, UpdateHelper helper) {
        checkArgument(root != Null, " deleting from an empty tree");
        if (helper == null) {
            helper = new UpdateHelper(dimension, numberOfLeaves);
        } else {
            helper.resize(numberOfLeaves); // should be a no-op
        }
        pointStoreView.getNumericVectorInto(pointIndex, helper.pointScratch);
        float[] point = projectToTree(helper.pointScratch);
        // --- PASS 1: descend, record ancestors, NO mutation ---
        int depth = 0;
        int current = root;
        while (!isLeaf(current)) {
            helper.nodeScratch[depth++] = current;
            current = nodeStore.leftOf(current, point) ? nodeStore.getLeftIndex(current)
                    : nodeStore.getRightIndex(current);
        }
        int leafNode = current;
        int leafPointIndex = getPointIndex(leafNode);
        checkArgument(leafPointIndex == pointIndex,
                () -> " deleting wrong node " + leafPointIndex + " instead of " + pointIndex);

        // ---- point of no return: validation passed, begin mutating ----
        removeLeaf(leafPointIndex, sequenceIndex);

        if (decreaseLeafMass(leafNode) != 0) {
            manageAncestors(depth, helper.nodeScratch, point, true); // duplicates remain: update all ancestors
            return leafPointIndex;
        }
        if (depth == 0) { // leaf was the root
            root = Null;
            return leafPointIndex;
        }

        int parent = helper.nodeScratch[depth - 1];
        int sibling = (nodeStore.getLeftIndex(parent) == leafNode) ? nodeStore.getRightIndex(parent)
                : nodeStore.getLeftIndex(parent);
        if (depth == 1) { // parent was the root
            root = sibling;
            nodeStore.detachAsRoot(sibling);
        } else {
            nodeStore.replaceParentBySibling(helper.nodeScratch[depth - 2], parent, leafNode);
        }

        nodeStore.deleteInternalNode(parent);
        if (pointSum != null) {
            invalidatePointSum(parent);
        }
        int idx = translate(parent);
        if (idx != Integer.MAX_VALUE) {
            rangeSumData[idx] = 0.0;
        }

        manageAncestors(depth - 1, helper.nodeScratch, point, false); // skip the deleted parent
        return leafPointIndex;
    }

    // processes pathScratch[count-1] down to pathNodes[0] (bottom-up)
    private void manageAncestors(int count, int[] nodeScratch, float[] point, boolean duplicate) {
        boolean resolved = duplicate;
        for (int i = count - 1; i >= 0; i--) {
            int node = nodeScratch[i];
            nodeStore.decreaseMassOfInternalNode(node);
            if (pointSum != null) {
                recomputePointSum(node);
            }
            if (!resolved) {
                resolved = checkContainsAndRebuildBox(node, point, pointStoreView);
            }
        }
    }

    //// leaf, nonleaf representations

    protected boolean isLeaf(int index) {
        // note that numberOfLeaves - 1 corresponds to an unspefied leaf in partial tree
        // 0 .. numberOfLeaves - 2 corresponds to internal nodes
        return index >= numberOfLeaves;
    }

    protected boolean isInternal(int index) {
        // note that numberOfLeaves - 1 corresponds to an unspefied leaf in partial tree
        // 0 .. numberOfLeaves - 2 corresponds to internal nodes
        return index < numberOfLeaves - 1 && index >= 0;
    }

    protected int getPointIndex(int index) {
        return nodeStore.translateIndex(index);
    }

    protected int getLeftChild(int index) {
        // checkArgument(isInternal(index), "incorrect call to get left Index ");
        return nodeStore.getLeftIndex(index);
    }

    protected int getRightChild(int index) {
        // checkArgument(isInternal(index), () -> "incorrect call to get right child " +
        // index);
        return nodeStore.getRightIndex(index);
    }

    protected int getCutDimension(int index) {
        // checkArgument(isInternal(index), () -> "incorrect call to get cut dimension "
        // + index);
        return nodeStore.getCutDimension(index);
    }

    protected double getCutValue(int index) {
        // checkArgument(isInternal(index), () -> "incorrect call to get cut value " +
        // index);
        return nodeStore.getCutValue(index);
    }

    ///// mass assignments; separating leafs and internal nodes

    protected int getMass(int index) {
        return (isLeaf(index)) ? getLeafMass(index) : nodeStore.getMass(index);
    }

    protected int getLeafMass(int index) {
        int y = (index - numberOfLeaves);
        int value = leafMass.get(y);
        return (value != 0) ? value + 1 : 1;
    }

    protected void increaseLeafMass(int index) {
        int y = (index - numberOfLeaves);
        leafMass.addTo(y, 1);
    }

    protected int decreaseLeafMass(int index) {
        int y = (index - numberOfLeaves);
        int value = leafMass.remove(y);
        if (value != 0) {
            if (value > 1) {
                leafMass.put(y, (value - 1));
                return value;
            } else {
                return 1;
            }
        } else {
            return 0;
        }
    }

    @Override
    public int getMass() {
        return root == Null ? 0 : isLeaf(root) ? getLeafMass(root) : nodeStore.getMass(root);
    }

    /////// Bounding box

    public void resizeCache(double fraction) {
        basicCache = min((int) (2 * log(numberOfLeaves) / log(2.0)), MAX_BASIC_CACHE);
        limit = min(basicCache + (int) Math.floor(fraction * (numberOfLeaves - 1)), numberOfLeaves - 1);
        if (limit == numberOfLeaves - 1 || fraction == 0) {
            // fracion is a hard signal
            basicCache = 0;
            smallMap = null;
            spareMap = null;
        } else {
            if (smallMap == null) {
                smallMap = new Int2IntOpenHashMap(basicCache);
                smallMap.defaultReturnValue(Integer.MAX_VALUE);
            }
            if (spareMap == null) {
                spareMap = new Int2IntOpenHashMap(basicCache);
                spareMap.defaultReturnValue(Integer.MAX_VALUE);
            }
            smallMap.clear();
            spareMap.clear();
        }
        if (fraction > 0) {
            rangeSumData = new double[limit];
            boundingBoxData = new float[limit * 2 * dimension];
        } else {
            limit = 0;
            rangeSumData = null;
            boundingBoxData = null;
        }
        boundingBoxCacheFraction = fraction;
    }

    protected int translate(int index) {
        if (rangeSumData == null)
            return Integer.MAX_VALUE;
        if (basicCache + index < limit) {
            // should be the case for fraction = 1
            return basicCache + index;
        }
        if (smallMap != null) {
            int v = smallMap.get(index);
            return v;
        }
        return Integer.MAX_VALUE;
    }

    void installCache(int[] nodeScratch, int count, int depth) {
        if (smallMap == null)
            return;

        Int2IntOpenHashMap temp = spareMap;
        long hash = 0L; // this is why MAX_BASIC_CACHE is Long.SIZE

        for (int i = 0; i < count + depth; i++) {
            int node = (i < count) ? nodeScratch[i] : nodeScratch[numberOfLeaves - 1 - i + count];
            if (node + basicCache < limit)
                continue;
            int slot = smallMap.get(node);
            if (slot != Integer.MAX_VALUE) {
                temp.put(node, slot);
                // carry valid box at its slot
                // maybe covered elsewhere but does not matter
            } else if (smallMap.size() < basicCache) {
                int fresh = smallMap.size(); // dense invariant => next free slot
                rangeSumData[fresh] = 0.0; // invalidate -> rebuild on first read
                smallMap.put(node, fresh); // advance size() AND mark handled for B
                temp.put(node, fresh);
            }
        }

        ObjectIterator<Int2IntMap.Entry> it = smallMap.int2IntEntrySet().fastIterator();

        for (int i = 0; i < count + depth; i++) {
            int node = (i < count) ? nodeScratch[i] : nodeScratch[numberOfLeaves - 1 - i + count];
            if (temp.containsKey(node) || node + basicCache < limit) {
                continue; // already carried or fresh in phase 1
            }
            int slot = Integer.MAX_VALUE;
            while (it.hasNext()) {
                Int2IntMap.Entry e = it.next();
                int key = e.getIntKey();
                if (!temp.containsKey(key)) { // stale entry -> reclaim its slot
                    slot = e.getIntValue();
                    hash |= 1L << slot; // remember: victim's slot was taken
                    rangeSumData[slot] = 0.0;
                    temp.put(node, slot);
                    break;
                }
            }
            if (slot == Integer.MAX_VALUE) {
                break; // smallMap exhausted of stale entries: working set exceeds capacity
            }
        }

        it = smallMap.int2IntEntrySet().fastIterator();
        while (it.hasNext()) {
            Int2IntMap.Entry e = it.next();
            int key = e.getIntKey();
            boolean covered = (hash & (1L << e.getIntValue())) != 0;
            if (!temp.containsKey(key) && !covered && temp.size() < basicCache) {
                temp.put(key, e.getIntValue()); // valid box, carry as-is (backfills density)
            }
        }

        smallMap.clear();
        spareMap = smallMap;
        smallMap = temp;
    }

    int addLeafPoints(int[] leafArray, int node, int firstEmpty) {
        if (isLeaf(node)) {
            leafArray[firstEmpty++] = getPointIndex(node);
            return firstEmpty;
        }
        return addLeafPoints(leafArray, getRightChild(node), addLeafPoints(leafArray, getLeftChild(node), firstEmpty));
    }

    void copyArrayBoxToData(int idx, ArrayBox box) {
        int base = 2 * idx * dimension;
        System.arraycopy(box.values, box.offset, boundingBoxData, base, 2 * dimension);
        rangeSumData[idx] = box.getRangeSum();
    }

    /**
     * Overwrite data[base .. base+2*dim) with index's box. Warms the cache slot if
     * index has one. Returns rangeSum.
     */
    double fillInto(float[] data, int base, int index, IPointStoreView<float[]> psv) {
        if (isLeaf(index)) {
            psv.setAsSlice(getPointIndex(index), data, base);
            return 0;
        }
        checkArgument(isInternal(index), () -> " incomplete state " + index);
        int idx = translate(index);
        if (idx == Integer.MAX_VALUE) {
            return buildInto(data, base, index, psv); // uncached: straight into destination
        }
        int slot = 2 * idx * dimension;
        if (rangeSumData[idx] == 0) {
            rangeSumData[idx] = buildInto(boundingBoxData, slot, index, psv);
        }
        if (data != boundingBoxData || base != slot) {
            System.arraycopy(boundingBoxData, slot, data, base, 2 * dimension);
        }
        return rangeSumData[idx];
    }

    /**
     * Rebuild index's box from its CHILDREN into data[base..). Never consults
     * index's own slot.
     */
    double buildInto(float[] data, int base, int index, IPointStoreView<float[]> psv) {
        fillInto(data, base, nodeStore.getLeftIndex(index), psv);
        return growInto(data, base, nodeStore.getRightIndex(index), psv);
    }

    /** Union sibling's box into data[base..). Returns updated rangeSum. */
    double growInto(float[] data, int base, int sibling, IPointStoreView<float[]> psv) {
        if (isLeaf(sibling)) {
            return psv.addToSlice(getPointIndex(sibling), data, base);
        }
        checkArgument(isInternal(sibling), () -> " incomplete state " + sibling);
        int sidx = translate(sibling);
        if (sidx != Integer.MAX_VALUE) {
            int slot = 2 * sidx * dimension;
            if (rangeSumData[sidx] == 0) {
                rangeSumData[sidx] = buildInto(boundingBoxData, slot, sibling, psv);
            }
            return VectorSupport.addSlice(data, base, dimension, boundingBoxData, slot);
        }
        growInto(data, base, nodeStore.getLeftIndex(sibling), psv);
        return growInto(data, base, nodeStore.getRightIndex(sibling), psv);
    }

    void fillArrayBox(int index, ArrayBox aBox) {
        aBox.rangeSum = fillInto(aBox.values, aBox.offset, index, pointStoreView);
    }

    void growArrayBox(ArrayBox aBox, IPointStoreView<float[]> psv, int sibling) {
        checkArgument(aBox != null, "box cannot be null");
        aBox.rangeSum = growInto(aBox.values, aBox.offset, sibling, psv);
    }

    public ArrayBox getArrayBox(int index) {
        ArrayBox aBox = new ArrayBox(dimension);
        fillArrayBox(index, aBox);
        return aBox;
    }

    boolean checkContainsAndRebuildBox(int index, float[] point, IPointStoreView<float[]> psv) {
        int idx = translate(index);
        if (idx == Integer.MAX_VALUE) {
            return false;
        }
        int base = 2 * idx * dimension;
        if (rangeSumData[idx] == 0.0) {
            rangeSumData[idx] = buildInto(boundingBoxData, base, index, psv);
            return false;
        }
        boolean inside = true;
        for (int i = 0; i < dimension && inside; i++) {
            inside = (point[i] < boundingBoxData[base + i]) && (-point[i] < boundingBoxData[base + i + dimension]);
        }
        if (!inside) {
            rangeSumData[idx] = buildInto(boundingBoxData, base, index, psv);
            return false;
        }
        return true;
    }

    // the following is over 2*d space
    public double probabilityOfCutExpanded(int node, float[] expandedPoint, ArrayBox otherBox, float[] components,
            ArrayBox reusableBox) {
        int nodeIdx = translate(node);
        if (nodeIdx != Integer.MAX_VALUE && rangeSumData[nodeIdx] != 0) {
            int base = 2 * nodeIdx * dimension;
            return gapAttribution(boundingBoxData, base, dimension, rangeSumData[nodeIdx], expandedPoint, 0,
                    components);
        } else if (otherBox != null) {
            return otherBox.probabilityOfCut(expandedPoint, components);
        } else {
            fillArrayBox(node, reusableBox);
            return reusableBox.probabilityOfCut(expandedPoint, components);
        }
    }

    /*
     * public double probabilityFromCache(int node, float[] expandedPoint, float[]
     * components, int bs, long[] mask) { int nodeIdx = translate(node); if (nodeIdx
     * == Integer.MAX_VALUE || rangeSumData[nodeIdx] == 0) { return Double.NaN; //
     * miss; no explanation offered } int base = 2 * nodeIdx * dimension; return
     * VectorSupport.gapAttributionPruned(boundingBoxData, base, dimension,
     * rangeSumData[nodeIdx], expandedPoint, 0, components, bs, mask); }
     */
    /// additional information at nodes

    public float[] getPointSum(int index) {
        checkArgument(centerOfMassEnabled, " enable center of mass");
        if (isLeaf(index)) {
            float[] point = projectToTree(pointStoreView.getNumericVector(getPointIndex(index)));
            checkArgument(point.length == dimension, () -> " incorrect projection");
            int mass = getMass(index);
            for (int i = 0; i < point.length; i++) {
                point[i] *= mass;
            }
            return point;
        } else {
            return Arrays.copyOfRange(pointSum, index * dimension, (index + 1) * dimension);
        }
    }

    public void invalidatePointSum(int index) {
        for (int i = 0; i < dimension; i++) {
            pointSum[index * dimension + i] = 0;
        }
    }

    public void recomputePointSum(int index) {
        float[] left = getPointSum(nodeStore.getLeftIndex(index));
        float[] right = getPointSum(nodeStore.getRightIndex(index));
        for (int i = 0; i < dimension; i++) {
            pointSum[index * dimension + i] = left[i] + right[i];
        }
    }

    public HashMap<Long, Integer> getSequenceMap(int index) {
        HashMap<Long, Integer> hashMap = new HashMap<>();
        List<Long> list = getSequenceList(index);
        for (Long e : list) {
            hashMap.merge(e, 1, Integer::sum);
        }
        return hashMap;
    }

    public List<Long> getSequenceList(int index) {
        return sequenceMap.get(index);
    }

    protected void addLeaf(int pointIndex, long sequenceIndex) {
        if (storeSequenceIndexesEnabled) {
            List<Long> leafList = sequenceMap.remove(pointIndex);
            if (leafList == null) {
                leafList = new ArrayList<>(1);
            }
            leafList.add(sequenceIndex);
            sequenceMap.put(pointIndex, leafList);
        }
    }

    public void removeLeaf(int leafPointIndex, long sequenceIndex) {
        if (storeSequenceIndexesEnabled) {
            List<Long> leafList = sequenceMap.remove(leafPointIndex);
            checkArgument(leafList != null, " leaf index not found in tree");
            checkArgument(leafList.remove(sequenceIndex), " sequence index not found in leaf");
            if (!leafList.isEmpty()) {
                sequenceMap.put(leafPointIndex, leafList);
            }
        }
    }

    //// validations

    public void validateAndReconstruct() {
        if (root != Null) {
            validateAndReconstruct(root, true, true, centerOfMassEnabled);
        }
    }

    /**
     * This function is supposed to validate the integrity of the tree and rebuild
     * internal data structures. At this moment the only internal structure is the
     * pointsum.
     * 
     * @param index the node of a tree
     * @return a bounding box of the points
     */
    public ArrayBox validateAndReconstruct(int index, boolean retainBoxes, boolean rebuildMass, boolean recompute) {
        if (isLeaf(index)) {
            return (retainBoxes) ? getArrayBox(index) : null;
        } else {
            checkState(isInternal(index), "illegal state");
            int left = getLeftChild(index);
            int right = getRightChild(index);
            ArrayBox leftBox = validateAndReconstruct(left, retainBoxes, rebuildMass, recompute);
            ArrayBox rightBox = validateAndReconstruct(right, retainBoxes, rebuildMass, recompute);
            if (retainBoxes && (leftBox.getMaxValue(getCutDimension(index)) > getCutValue(index)
                    || rightBox.getMinValue(getCutDimension(index)) <= getCutValue(index))) {
                throw new IllegalStateException(" incorrect bounding state at index " + index + " cut value "
                        + getCutValue(index) + "cut dimension " + getCutDimension(index) + " left Box "
                        + leftBox.toString() + " right box " + rightBox.toString());
            }
            if (recompute) {
                recomputePointSum(index);
            }
            if (rebuildMass) {
                // post-order fold: same pass, no extra traversal, no up-the-path replay
                nodeStore.setMassOfInternalNode(index, getMass(left) + getMass(right));
            }
            if (retainBoxes) {
                rightBox.addBox(leftBox);
                int idx = translate(index);
                if (idx != Integer.MAX_VALUE) { // always add irrespective of rangesum
                    copyArrayBoxToData(idx, rightBox);
                }
                return rightBox;
            }
            return null;
        }
    }

    //// traversals

    /**
     * Starting from the root, traverse the canonical path to a leaf node and visit
     * the nodes along the path. The canonical path is determined by the input
     * point: at each interior node, we select the child node by comparing the
     * node's {@link Cut} to the corresponding coordinate value in the input point.
     * The method recursively traverses to the leaf node first and then invokes the
     * visitor on each node in reverse order. That is, if the path to the leaf node
     * determined by the input point is root, node1, node2, ..., node(N-1), nodeN,
     * leaf; then we will first invoke visitor::acceptLeaf on the leaf node, and
     * then we will invoke visitor::accept on the remaining nodes in the following
     * order: nodeN, node(N-1), ..., node2, node1, and root. The reusableTraverse
     * earns it place by reusing the same visitor -- instead of a factory method and
     * thereby saves on object allocation -- note that if a forest has 50 trees
     * (seems obvious, why else is it a forest and not a spiney) then this is 50x
     * less GC.
     *
     * @param point          A point which determines the traversal path from the
     *                       root to a leaf node.
     * @param visitorFactory A visitor that will be invoked for each node on the
     *                       path.
     * @param <R>            The return type of the Visitor.
     * @return the value of {@link Visitor#getResult()}} after the traversal.
     */
    @Override
    public <R> R traverse(float[] point, IVisitorFactory<R> visitorFactory) {
        checkArgument(root != Null, "this tree doesn't contain any nodes");
        checkNotNull(point, "point must not be null");
        checkNotNull(visitorFactory, "visitor must not be null");
        Visitor<R> visitor = visitorFactory.newVisitor(this, point);
        NodeView currentNodeView = new NodeView(this, pointStoreView, root, point);
        traversePathToLeafAndVisitNodes(point, visitor, currentNodeView, root, 0, 0);
        return visitorFactory.liftResult(this, visitor.getResult());
    }

    public <R> NodeView reusableTraverse(float[] point, IRFVisitor<R> v, NodeView view) {
        v.prepare(this, point); // self-arm: I supply my own projection + mass
        if (view == null)
            view = new NodeView(this, pointStoreView, root, point);
        else
            view.rearm(this, root);
        traversePathToLeafAndVisitNodes(point, v, view, root, 0, 0);
        // result extracted from visitor; one can do v.getResult()
        // the viewing tower (not the view) passed along
        return view;
    }

    protected <R> void traversePathToLeafAndVisitNodes(float[] point, Visitor<R> visitor, NodeView currentNodeView,
            int node, int depthOfNode, int count) {
        if (isLeaf(node)) {
            currentNodeView.setCurrentNode(node, getPointIndex(node), true);
            visitor.acceptLeaf(currentNodeView, depthOfNode);
            // have all the nodes in the path in nodeScratch[0,depthOfnode)
            // will have nulled fields for boundingboxFraction=0
            installCache(currentNodeView.nodeScratch, count, depthOfNode);
        } else {
            checkState(isInternal(node), " incomplete state ");
            currentNodeView.nodeScratch[numberOfLeaves - 1 - depthOfNode] = node;
            int right = nodeStore.getRightIndex(node);
            int left = nodeStore.getLeftIndex(node);
            if (nodeStore.toLeft(point, node)) {
                if (isInternal(right)) {
                    currentNodeView.nodeScratch[count++] = right;
                }
                traversePathToLeafAndVisitNodes(point, visitor, currentNodeView, left, depthOfNode + 1, count);
                currentNodeView.updateToParent(node, right, !visitor.isConverged());
            } else {
                if (isInternal(left)) {
                    // overwrite
                    currentNodeView.nodeScratch[count++] = left;
                }
                traversePathToLeafAndVisitNodes(point, visitor, currentNodeView, right, depthOfNode + 1, count);
                currentNodeView.updateToParent(node, left, !visitor.isConverged());
            }
            visitor.accept(currentNodeView, depthOfNode);
        }
    }

    /**
     * This is a traversal method which follows the standard traversal path (defined
     * in {@link #traverse(float[], IVisitorFactory)}) but at Node in checks to see
     * whether the visitor should split. If a split is triggered, then independent
     * copies of the visitor are sent down each branch of the tree and then merged
     * before propagating the result.
     *
     * @param point          A point which determines the traversal path from the
     *                       root to a leaf node.
     * @param visitorFactory A visitor that will be invoked for each node on the
     *                       path.
     * @param <R>            The return type of the Visitor.
     * @return the value of {@link Visitor#getResult()}} after the traversal.
     */

    @Override
    public <R> R traverseMulti(float[] point, IMultiVisitorFactory<R> visitorFactory) {
        checkArgument(root != Null, "this tree doesn't contain any nodes");
        checkNotNull(point, "point must not be null");
        checkNotNull(visitorFactory, "visitor must not be null");
        MultiVisitor<R> visitor = visitorFactory.newVisitor(this, point);
        NodeView currentNodeView = new NodeView(this, pointStoreView, root, point);
        traverseTreeMulti(point, visitor, currentNodeView, root, 0);
        return visitor.getResult(); // liftResult was identity -> gone
    }

    public <R> NodeView reusableTraverseMulti(float[] point, IRFMultiVisitor<R> v, NodeView view) {
        v.prepare(this, point);
        if (view == null)
            view = new NodeView(this, pointStoreView, root, point);
        else
            view.rearm(this, root);
        traverseTreeMulti(point, v, view, root, 0);
        return view; // result pulled off the slot by the caller
    }

    protected <R> void traverseTreeMulti(float[] point, MultiVisitor<R> visitor, NodeView currentNodeView, int node,
            int depthOfNode) {
        if (isLeaf(node)) {
            currentNodeView.setCurrentNode(node, getPointIndex(node), false);
            visitor.acceptLeaf(currentNodeView, depthOfNode);
        } else {
            checkState(isInternal(node), " incomplete state");
            currentNodeView.setCurrentNodeOnly(node);
            if (visitor.trigger(currentNodeView)) {
                traverseTreeMulti(point, visitor, currentNodeView, nodeStore.getLeftIndex(node), depthOfNode + 1);
                MultiVisitor<R> newVisitor = visitor.newPartialCopy();
                currentNodeView.setCurrentNodeOnly(nodeStore.getRightIndex(node));
                traverseTreeMulti(point, newVisitor, currentNodeView, nodeStore.getRightIndex(node), depthOfNode + 1);
                currentNodeView.updateToParent(node, nodeStore.getLeftIndex(node), false);
                visitor.combine(newVisitor);
            } else if (nodeStore.toLeft(point, node)) {
                traverseTreeMulti(point, visitor, currentNodeView, nodeStore.getLeftIndex(node), depthOfNode + 1);
                currentNodeView.updateToParent(node, nodeStore.getRightIndex(node), false);
            } else {
                traverseTreeMulti(point, visitor, currentNodeView, nodeStore.getRightIndex(node), depthOfNode + 1);
                currentNodeView.updateToParent(node, nodeStore.getLeftIndex(node), false);
            }
            visitor.accept(currentNodeView, depthOfNode);
        }
    }

    /**
     *
     * @return the root of the tree
     */
    public Integer getRoot() {
        return (int) root;
    }

    @Override
    public boolean isOutputReady() {
        return getMass() >= outputAfter;
    }

    public float[] projectToTree(float[] point) {
        // there is only one tree in the repo and this defensive copy
        // is perhaps not required and increasing GC
        // there is a risk of mutation, but the library is mature
        return point;
        // return Arrays.copyOf(point, point.length);
    }

    public float[] liftFromTree(float[] result) {
        // there is only one tree in the repo and this defensive copy
        // is perhaps not required and increasing GC
        // there is a risk of mutation, but the library is mature
        return result;
        // return Arrays.copyOf(result, result.length);
    }

    public double[] liftFromTree(double[] result) {
        // there is only one tree in the repo and this defensive copy
        // is perhaps not required and increasing GC
        // there is a risk of mutation, but the library is mature
        return result;
        // return Arrays.copyOf(result, result.length);
    }

    public int[] projectMissingIndices(int[] list) {
        return Arrays.copyOf(list, list.length);
    }

    public static class Builder<T extends Builder<T>> {
        protected boolean storeSequenceIndexesEnabled = RandomCutForest.DEFAULT_STORE_SEQUENCE_INDEXES_ENABLED;
        protected boolean centerOfMassEnabled = RandomCutForest.DEFAULT_CENTER_OF_MASS_ENABLED;
        protected double boundingBoxCacheFraction = RandomCutForest.DEFAULT_BOUNDING_BOX_CACHE_FRACTION;
        protected long randomSeed = new Random().nextLong();
        protected Random random = null;
        protected int capacity = RandomCutForest.DEFAULT_SAMPLE_SIZE;
        protected Optional<Integer> outputAfter = Optional.empty();
        protected int dimension;
        protected IPointStoreView<float[]> pointStoreView;
        protected NodeStore nodeStore;
        protected int root = Null;
        protected boolean storeParent = DEFAULT_STORE_PARENT;

        public T capacity(int capacity) {
            this.capacity = capacity;
            return (T) this;
        }

        public T boundingBoxCacheFraction(double boundingBoxCacheFraction) {
            this.boundingBoxCacheFraction = boundingBoxCacheFraction;
            return (T) this;
        }

        public T pointStoreView(IPointStoreView<float[]> pointStoreView) {
            this.pointStoreView = pointStoreView;
            return (T) this;
        }

        public T nodeStore(NodeStore nodeStore) {
            this.nodeStore = nodeStore;
            return (T) this;
        }

        public T randomSeed(long randomSeed) {
            this.randomSeed = randomSeed;
            return (T) this;
        }

        public T random(Random random) {
            this.random = random;
            return (T) this;
        }

        public T outputAfter(int outputAfter) {
            this.outputAfter = Optional.of(outputAfter);
            return (T) this;
        }

        public T dimension(int dimension) {
            this.dimension = dimension;
            return (T) this;
        }

        public T setRoot(int root) {
            this.root = root;
            return (T) this;
        }

        public T storeParent(boolean storeParent) {
            this.storeParent = storeParent;
            return (T) this;
        }

        public T storeSequenceIndexesEnabled(boolean storeSequenceIndexesEnabled) {
            this.storeSequenceIndexesEnabled = storeSequenceIndexesEnabled;
            return (T) this;
        }

        public T centerOfMassEnabled(boolean centerOfMassEnabled) {
            this.centerOfMassEnabled = centerOfMassEnabled;
            return (T) this;
        }

        public RandomCutTree build() {
            return new RandomCutTree(this);
        }
    }

    public static Builder builder() {
        return new Builder();
    }
}
