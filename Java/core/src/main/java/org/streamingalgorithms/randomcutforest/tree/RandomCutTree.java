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

import static java.lang.Math.max;
import static java.lang.Math.min;
import static org.streamingalgorithms.randomcutforest.CommonUtils.checkArgument;
import static org.streamingalgorithms.randomcutforest.CommonUtils.checkNotNull;
import static org.streamingalgorithms.randomcutforest.CommonUtils.checkState;
import static org.streamingalgorithms.randomcutforest.sampler.CompactSampler.SEQUENCE_INDEX_NA;
import static org.streamingalgorithms.randomcutforest.tree.ArrayBox.gapAttribution;
import static org.streamingalgorithms.randomcutforest.tree.NodeStore.DEFAULT_STORE_PARENT;
import static org.streamingalgorithms.randomcutforest.tree.NodeStore.Null;
import static org.streamingalgorithms.randomcutforest.tree.NodeStore.nodeStore;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Optional;
import java.util.Random;
import java.util.Stack;

import org.streamingalgorithms.randomcutforest.IMultiVisitorFactory;
import org.streamingalgorithms.randomcutforest.IVisitorFactory;
import org.streamingalgorithms.randomcutforest.MultiVisitor;
import org.streamingalgorithms.randomcutforest.RandomCutForest;
import org.streamingalgorithms.randomcutforest.Visitor;
import org.streamingalgorithms.randomcutforest.config.Config;
import org.streamingalgorithms.randomcutforest.store.IPointStoreView;

import it.unimi.dsi.fastutil.ints.Int2IntOpenHashMap;

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
    protected boolean storeSequenceIndexesEnabled;
    protected boolean centerOfMassEnabled;
    private long randomSeed;
    protected int root;
    protected IPointStoreView<float[]> pointStoreView;
    protected int numberOfLeaves;
    protected NodeStore nodeStore;
    protected double boundingBoxCacheFraction;
    protected int outputAfter;
    protected int dimension;
    protected final Int2IntOpenHashMap leafMass;
    protected double[] rangeSumData;
    protected float[] boundingBoxData;
    protected float[] pointSum;
    protected HashMap<Integer, List<Long>> sequenceMap;
    protected final float[] pointScratch;
    protected final int[] nodeScratch;
    protected final double[] attributionScratch;
    protected final Cut cutScratch;
    protected final ArrayBox boxScratch;

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
        pointScratch = new float[dimension];
        nodeScratch = new int[numberOfLeaves]; // cannot be -1
        attributionScratch = new double[dimension];
        cutScratch = new Cut(0, 0);
        if (this.centerOfMassEnabled) {
            pointSum = new float[(numberOfLeaves - 1) * dimension];
        }
        if (this.storeSequenceIndexesEnabled) {
            sequenceMap = new HashMap<>();
        }
        resizeCache(boundingBoxCacheFraction);
        boxScratch = new ArrayBox(dimension);
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
    // boxes
    // 0 would mean less space usage, but slower throughput
    // 1 would imply larger space but better throughput
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
                attributionScratch[i] = max(point[i], box.values[i + box.offset])
                        + max(-point[i], box.values[i + box.offset + dimension]);
                range += attributionScratch[i];
            }
        } else {
            if (oracle == null) {
                for (int i = 0; i < dimension; i++) {
                    attributionScratch[i] = box.values[i + box.offset] + box.values[i + box.offset + dimension];
                }
                range = box.rangeSum;
            } else {
                checkArgument(oracle.length == dimension, "incorrect probability specification");
                for (int i = 0; i < dimension; i++) {
                    checkArgument(oracle[i] >= 0, "cannot be negative");
                    attributionScratch[i] = oracle[i];
                    range += attributionScratch[i];
                }
            }
        }

        checkArgument(range > 0, () -> " the union is a single point " + Arrays.toString(point)
                + "or the box is inappropriate, box" + box.toString() + "factor =" + factor);

        double breakPoint = factor * range;

        for (int i = 0; i < dimension; i++) {
            double gap = attributionScratch[i];

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
     * @return
     */
    private void cleanup(double factor, ArrayBox box, float[] point, Cut output) {

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

        throw new IllegalStateException("The break point did not lie inside the expected range; factor " + factor
                + ", point " + Arrays.toString(point) + " box " + box.toString());
    }

    // the following is just for testing and not to disturb the 100s of tests
    protected Cut randomCut(double factor, float[] point, BoundingBox box) {
        Cut output = new Cut(0, 0);
        randomCut(factor, point, new ArrayBox(box), null, true, output);
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
    public Integer addPoint(Integer pointIndex, long sequenceIndex) {

        if (root == Null) {
            root = convertToLeaf(pointIndex);
            addLeaf(pointIndex, sequenceIndex);
            return pointIndex;
        } else {
            // updates are single threaded
            pointStoreView.getNumericVectorInto(pointIndex, pointScratch);
            float[] point = projectToTree(pointScratch);
            // point,pointScratch are exact locations now
            checkArgument(point.length == dimension, () -> " mismatch in dimensions for " + pointIndex);
            Stack<int[]> pathToRoot = nodeStore.getPath(root, point, false);
            int[] first = pathToRoot.pop();
            int leafNode = first[0];
            int savedParent = (pathToRoot.size() == 0) ? Null : pathToRoot.lastElement()[0];
            int leafSavedSibling = first[1];
            int sibling = leafSavedSibling;
            int leafPointIndex = getPointIndex(leafNode);
            float[] oldPoint = projectToTree(pointStoreView.getNumericVector(leafPointIndex));
            checkArgument(oldPoint.length == dimension, () -> " mismatch in dimensions for " + pointIndex);

            Stack<int[]> parentPath = new Stack<>();

            if (Arrays.equals(point, oldPoint)) {
                increaseLeafMass(leafNode);
                manageAncestorsAdd(pathToRoot, point);
                addLeaf(leafPointIndex, sequenceIndex);
                return leafPointIndex;
            } else {
                int node = leafNode;
                int savedNode = node;
                int parent = savedParent;
                float savedCutValue = (float) 0.0;
                boxScratch.fromPoint(oldPoint);
                ArrayBox savedBox = boxScratch.copy();
                int savedDim = Integer.MAX_VALUE;
                Random rng;
                if (testRandom == null) {
                    rng = new Random(randomSeed);
                    randomSeed = rng.nextLong();
                } else {
                    rng = testRandom;
                }
                while (true) {
                    double factor = rng.nextDouble();
                    randomCut(factor, point, boxScratch, null, true, cutScratch);
                    int dim = cutScratch.getDimension();
                    float value = (float) cutScratch.getValue();

                    boolean separation = ((point[dim] <= value && value < boxScratch.getMinValue(dim)
                            || point[dim] > value && value >= boxScratch.getMaxValue(dim)));

                    if (separation) {
                        savedCutValue = value;
                        savedDim = dim;
                        savedParent = parent;
                        savedNode = node;
                        savedBox.copyFrom(boxScratch);
                        parentPath.clear();
                    } else {
                        parentPath.push(new int[] { node, sibling });
                    }

                    if (boxScratch.contains(point) || parent == Null) {
                        break;
                    } else {
                        growArrayBox(boxScratch, pointStoreView, parent, sibling);
                        int[] next = pathToRoot.pop();
                        node = next[0];
                        sibling = next[1];
                        if (pathToRoot.size() != 0) {
                            parent = pathToRoot.lastElement()[0];
                        } else {
                            parent = Null;
                        }
                    }
                }
                if (savedParent != Null) {
                    while (!parentPath.isEmpty()) {
                        pathToRoot.push(parentPath.pop());
                    }
                }

                int childMassIfLeaf = isLeaf(savedNode) ? getLeafMass(savedNode) : 0;
                int mergedNode = nodeStore.addNode(pathToRoot, point, sequenceIndex, pointIndex, savedNode,
                        childMassIfLeaf, savedDim, savedCutValue);
                addLeaf(pointIndex, sequenceIndex);
                int idx = translate(mergedNode);
                if (idx != Integer.MAX_VALUE) { // always add irrespective of rangesum
                    copyArrayBoxToData(idx, savedBox);
                    rangeSumData[idx] = ArrayBox.addPointInPlace(boundingBoxData, 2 * idx * dimension, dimension, point,
                            0);
                }
                manageAncestorsAdd(pathToRoot, point);
                if (pointSum != null) {
                    recomputePointSum(mergedNode);
                }
                if (savedParent == Null) {
                    root = mergedNode;
                }
            }
            return pointIndex;
        }
    }

    public void makeTree(int size, int[] indexList, int[] outputList, int[] pointList, long[] sequenceIndex,
            SeparationOracle oracle, long seed) {
        // this function allows a public call, which may be useful someday
        // that day is today ....
        if (size > 0 && size <= numberOfLeaves) {
            checkArgument(indexList.length <= size, "incorrect input buffer");
            checkArgument(outputList.length <= size, "incorrect output buffer");
            checkArgument(sequenceIndex == null || (pointList.length == sequenceIndex.length), "mismatched input");

            Random ring = new Random(seed);
            Cut scratchCut = new Cut(0, 0);
            root = makeTreeInt(indexList, outputList, 0, size, pointList, 0, oracle, nodeScratch, scratchCut, ring);
            // the cuts are specififed; now build tree
            // note the contents of the indexList will be permuted.
            for (int i = 0; i < size; i++) {
                long seq = (sequenceIndex != null) ? sequenceIndex[indexList[i]] : SEQUENCE_INDEX_NA;
                checkArgument(outputList[i] == addPointToPartialTree(pointList[indexList[i]], seq),
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
            SeparationOracle vecBuild, int[] scratch, Cut cutScratch, Random ring) {

        if (end - start == 0)
            return Null;
        pointStoreView.getNumericVectorInto(pointList[chosen[start]], pointScratch);
        boxScratch.fromPoint(pointScratch);

        for (int i = start + 1; i < end; i++) {
            pointStoreView.getNumericVectorInto(pointList[chosen[i]], pointScratch);
            boxScratch.addPoint(pointScratch);
        }

        if (boxScratch.getRangeSum() <= 0) {
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
            vecBuild.computeGap(boxScratch, attributionScratch);
            randomCut(factor, null, boxScratch, attributionScratch, false, cutScratch);
        } else {
            randomCut(factor, null, boxScratch, null, false, cutScratch); // box-range path
        }

        float cutValue = (float) cutScratch.getValue();
        int cutDimension = cutScratch.getDimension();
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
                scratch[r++] = v;
            }
        }
        int mid = write;
        for (int k = 0; k < r; k++) {
            chosen[mid + k] = scratch[k];
        }
        int idx = translate(firstFree);
        if (idx != Integer.MAX_VALUE) {
            copyArrayBoxToData(idx, boxScratch);
        }
        int leftCount = mid - start;
        int leftIndex = makeTreeInt(chosen, output, start, mid, pointList, firstFree + 1, vecBuild, scratch, cutScratch,
                ring);
        int rightIndex = makeTreeInt(chosen, output, mid, end, pointList, firstFree + leftCount, vecBuild, scratch,
                cutScratch, ring);
        nodeStore.addRecord(firstFree, min(leftIndex, numberOfLeaves - 1), min(rightIndex, numberOfLeaves - 1),
                cutValue, cutDimension);
        return firstFree;
    }

    protected void manageAncestorsAdd(Stack<int[]> path, float[] point) {
        boolean resolved = false;
        while (!path.isEmpty()) {
            int index = path.pop()[0];
            nodeStore.increaseMassOfInternalNode(index);
            if (pointSum != null) {
                recomputePointSum(index);
            }
            if (boundingBoxCacheFraction > 0.0) {
                int idx = translate(index);
                if (idx != Integer.MAX_VALUE) {
                    rangeSumData[idx] = 0;
                    ArrayBox inPlace = new ArrayBox(boundingBoxData, 2 * idx * dimension, dimension, 0);
                    reconstructArrayBox(inPlace, index, pointStoreView);
                    // double priorSum = rangeSumData[idx];
                    rangeSumData[idx] = ArrayBox.addPointInPlace(boundingBoxData, 2 * idx * dimension, dimension, point,
                            0);
                    // checkContainsAndRebuildBox(index, point, pointStoreView);
                    // addPointInPlace(index, point);
                }
            }
        }
    }

    /**
     * the following function is used to rebuild the tree structure. This function
     * does not create mass, auxiliary arrays, which should be performed using
     * validateAndReconstruct()
     * 
     * @param pointIndex    index of point (in point store)
     * @param sequenceIndex sequence index (stored in sampler)
     */
    public int addPointToPartialTree(int pointIndex, long sequenceIndex) {
        checkArgument(root != Null, " a null root is not a partial tree");

        // Single-threaded scratchpad reuse (Excellent for performance!)
        pointStoreView.getNumericVectorInto(pointIndex, pointScratch);
        float[] point = projectToTree(pointScratch);
        checkArgument(point.length == dimension, () -> " incorrect projection at index " + pointIndex);

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
                root = convertToLeaf(pointIndex);
            } else {
                nodeStore.assignInPartialTree(parentNode, point, convertToLeaf(pointIndex));
                addLeaf(pointIndex, sequenceIndex);
            }
            return pointIndex;
        }

        // Standard leaf insertion path
        int leafPointIndex = getPointIndex(leafNode);
        checkArgument(pointStoreView.isEqual(leafPointIndex, pointScratch),
                () -> "incorrect state on adding " + pointIndex);

        increaseLeafMass(leafNode);
        addLeaf(leafPointIndex, sequenceIndex);
        return leafPointIndex;
    }

    public Integer deletePoint(Integer pointIndex, long sequenceIndex) {
        checkArgument(root != Null, " deleting from an empty tree");
        pointStoreView.getNumericVectorInto(pointIndex, pointScratch);
        float[] point = projectToTree(pointScratch);
        checkArgument(point.length == dimension, () -> " incorrect projection at index " + pointIndex);

        // --- PASS 1: descend, record ancestors, NO mutation ---
        int depth = 0;
        int current = root;
        while (!isLeaf(current)) {
            nodeScratch[depth++] = current;
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
            manageAncestors(depth, point); // duplicates remain: update all ancestors
            return leafPointIndex;
        }
        if (depth == 0) { // leaf was the root
            root = Null;
            return leafPointIndex;
        }

        int parent = nodeScratch[depth - 1];
        int sibling = (nodeStore.getLeftIndex(parent) == leafNode) ? nodeStore.getRightIndex(parent)
                : nodeStore.getLeftIndex(parent);
        if (depth == 1) { // parent was the root
            root = sibling;
        } else {
            nodeStore.replaceParentBySibling(nodeScratch[depth - 2], parent, leafNode);
        }
        nodeStore.deleteInternalNode(parent);
        if (pointSum != null) {
            invalidatePointSum(parent);
        }
        int idx = translate(parent);
        if (idx != Integer.MAX_VALUE) {
            rangeSumData[idx] = 0.0;
        }

        manageAncestors(depth - 1, point); // skip the deleted parent
        return leafPointIndex;
    }

    // processes pathScratch[count-1] down to pathNodes[0] (bottom-up)
    private void manageAncestors(int count, float[] point) {
        boolean resolved = false;
        for (int i = count - 1; i >= 0; i--) {
            int node = nodeScratch[i];
            nodeStore.decreaseMassOfInternalNode(node);
            if (pointSum != null) {
                recomputePointSum(node);
            }
            if (boundingBoxCacheFraction > 0.0 && !resolved) {
                resolved = checkContainsAndRebuildBox(node, point, pointStoreView);
            }
        }
    }

    //// leaf, nonleaf representations

    public boolean isLeaf(int index) {
        // note that numberOfLeaves - 1 corresponds to an unspefied leaf in partial tree
        // 0 .. numberOfLeaves - 2 corresponds to internal nodes
        return index >= numberOfLeaves;
    }

    public boolean isInternal(int index) {
        // note that numberOfLeaves - 1 corresponds to an unspefied leaf in partial tree
        // 0 .. numberOfLeaves - 2 corresponds to internal nodes
        return index < numberOfLeaves - 1 && index >= 0;
    }

    public int convertToLeaf(int pointIndex) {
        return pointIndex + numberOfLeaves;
    }

    public int getPointIndex(int index) {
        checkArgument(index >= numberOfLeaves, () -> " does not have a point associated " + index);
        return index - numberOfLeaves;
    }

    public int getLeftChild(int index) {
        checkArgument(isInternal(index), () -> "incorrect call to get left Index " + index);
        return nodeStore.getLeftIndex(index);
    }

    public int getRightChild(int index) {
        checkArgument(isInternal(index), () -> "incorrect call to get right child " + index);
        return nodeStore.getRightIndex(index);
    }

    public int getCutDimension(int index) {
        checkArgument(isInternal(index), () -> "incorrect call to get cut dimension " + index);
        return nodeStore.getCutDimension(index);
    }

    public double getCutValue(int index) {
        checkArgument(isInternal(index), () -> "incorrect call to get cut value " + index);
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
        if (fraction == 0) {
            rangeSumData = null;
            boundingBoxData = null;
        } else {
            int limit = (int) Math.floor(fraction * (numberOfLeaves - 1));
            rangeSumData = (rangeSumData == null) ? new double[limit] : Arrays.copyOf(rangeSumData, limit);
            boundingBoxData = (boundingBoxData == null) ? new float[limit * 2 * dimension]
                    : Arrays.copyOf(boundingBoxData, limit * 2 * dimension);
        }
        boundingBoxCacheFraction = fraction;
    }

    protected int translate(int index) {
        if (rangeSumData == null || rangeSumData.length <= index) {
            return Integer.MAX_VALUE;
        } else {
            return index;
        }
    }

    void copyArrayBoxToData(int idx, ArrayBox box) {
        int base = 2 * idx * dimension;
        System.arraycopy(box.values, box.offset, boundingBoxData, base, 2 * dimension);
        rangeSumData[idx] = box.getRangeSum();
    }

    // mainly for tests
    protected BoundingBox getBox(int index) {
        BoundingBox answer = new BoundingBox(dimension);
        answer.setAs(getArrayBox(index));
        return answer;
    }

    public ArrayBox getArrayBox(int index) {
        ArrayBox aBox = new ArrayBox(dimension);
        if (isLeaf(index)) {
            pointStoreView.setAsArrayBox(getPointIndex(index), aBox);
        } else {
            checkState(isInternal(index), " incomplete state");
            int idx = translate(index);
            if (idx != Integer.MAX_VALUE) {
                if (rangeSumData[idx] == 0) {
                    // object creation
                    ArrayBox inPlace = new ArrayBox(boundingBoxData, 2 * idx * dimension, dimension, 0);
                    reconstructArrayBox(inPlace, index, pointStoreView);
                    rangeSumData[idx] = inPlace.getRangeSum();
                }
                return aBox.copyFromSlice(boundingBoxData, 2 * idx * dimension, rangeSumData[idx]);
            }
            reconstructArrayBox(aBox, index, pointStoreView);
            // no need to save rangesum
        }
        return aBox;
    }

    void reconstructArrayBox(ArrayBox aBox, int index, IPointStoreView<float[]> pointStoreView) {
        if (isLeaf(index)) {
            pointStoreView.setAsArrayBox(getPointIndex(index), aBox);
            return;
        }
        int idx = translate(index);
        if (idx != Integer.MAX_VALUE && rangeSumData[idx] != 0) {
            aBox.copyFromSlice(boundingBoxData, 2 * idx * dimension, rangeSumData[idx]);
            return;
        }
        // rebuild THIS node's box from its children into aBox
        reconstructArrayBox(aBox, nodeStore.getLeftIndex(index), pointStoreView);
        growArrayBox(aBox, pointStoreView, index, nodeStore.getRightIndex(index));
    }

    boolean checkContainsAndRebuildBox(int index, float[] point, IPointStoreView<float[]> pointStoreView) {
        int idx = translate(index);
        if (idx != Integer.MAX_VALUE) {
            int base = 2 * idx * dimension;
            boolean inside = true;
            for (int i = 0; i < dimension && inside; i++) {
                inside = (point[i] < boundingBoxData[base + i]) && (-point[i] < boundingBoxData[base + i + dimension]);
            }
            if (!inside) {
                rangeSumData[idx] = 0; // force rebuild
                // object creation
                ArrayBox inPlace = new ArrayBox(boundingBoxData, base, dimension, 0);
                reconstructArrayBox(inPlace, index, pointStoreView);
                rangeSumData[idx] = inPlace.getRangeSum();
                return false;
            }
            return true;
        }
        return false;
    }

    void growArrayBox(ArrayBox aBox, IPointStoreView<float[]> pointStoreView, int node, int sibling) {
        if (isLeaf(sibling)) {
            // project-to-tree on the point is ignored for now.
            pointStoreView.addToArrayBox(getPointIndex(sibling), aBox);
        } else {
            if (!isInternal(sibling)) {
                throw new IllegalStateException(" incomplete state " + sibling);
            }
            int siblingIdx = translate(sibling);
            if (siblingIdx != Integer.MAX_VALUE) {
                if (rangeSumData[siblingIdx] == 0) {
                    ArrayBox inPlace = new ArrayBox(boundingBoxData, 2 * siblingIdx * dimension, dimension, 0);
                    reconstructArrayBox(inPlace, sibling, pointStoreView);
                    rangeSumData[siblingIdx] = inPlace.getRangeSum();
                }
                aBox.addSlice(boundingBoxData, 2 * siblingIdx * dimension);
                return;
            }
            growArrayBox(aBox, pointStoreView, sibling, nodeStore.getLeftIndex(sibling));
            growArrayBox(aBox, pointStoreView, sibling, nodeStore.getRightIndex(sibling));
        }
    }

    public double probabilityOfCut(int node, float[] point, ArrayBox otherBox) {
        int nodeIdx = translate(node);
        if (nodeIdx != Integer.MAX_VALUE && rangeSumData[nodeIdx] != 0) {
            int base = 2 * nodeIdx * dimension;
            return gapAttribution(boundingBoxData, base, dimension, rangeSumData[nodeIdx], point, null);
        } else if (otherBox != null) {
            return otherBox.probabilityOfCut(point);
        } else {
            ArrayBox box = getArrayBox(node);
            return box.probabilityOfCut(point);
        }
    }

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
     * order: nodeN, node(N-1), ..., node2, node1, and root.
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
        NodeView currentNodeView = new NodeView(this, pointStoreView, root);
        traversePathToLeafAndVisitNodes(point, visitor, currentNodeView, root, 0);
        return visitorFactory.liftResult(this, visitor.getResult());
    }

    protected <R> void traversePathToLeafAndVisitNodes(float[] point, Visitor<R> visitor, NodeView currentNodeView,
            int node, int depthOfNode) {
        if (isLeaf(node)) {
            currentNodeView.setCurrentNode(node, getPointIndex(node), true);
            visitor.acceptLeaf(currentNodeView, depthOfNode);
        } else {
            checkState(isInternal(node), " incomplete state ");
            if (nodeStore.toLeft(point, node)) {
                traversePathToLeafAndVisitNodes(point, visitor, currentNodeView, nodeStore.getLeftIndex(node),
                        depthOfNode + 1);
                currentNodeView.updateToParent(node, nodeStore.getRightIndex(node), !visitor.isConverged());
            } else {
                traversePathToLeafAndVisitNodes(point, visitor, currentNodeView, nodeStore.getRightIndex(node),
                        depthOfNode + 1);
                currentNodeView.updateToParent(node, nodeStore.getLeftIndex(node), !visitor.isConverged());
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
        NodeView currentNodeView = new NodeView(this, pointStoreView, root);
        traverseTreeMulti(point, visitor, currentNodeView, root, 0);
        return visitorFactory.liftResult(this, visitor.getResult());
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

    public int getNumberOfLeaves() {
        return numberOfLeaves;
    }

    public boolean isCenterOfMassEnabled() {
        return centerOfMassEnabled;
    }

    public boolean isStoreSequenceIndexesEnabled() {
        return storeSequenceIndexesEnabled;
    }

    public double getBoundingBoxCacheFraction() {
        return boundingBoxCacheFraction;
    }

    public int getDimension() {
        return dimension;
    }

    /**
     *
     * @return the root of the tree
     */

    public Integer getRoot() {
        return (int) root;
    }

    /**
     * returns the number of samples that needs to be seen before returning
     * meaningful inference
     */
    public int getOutputAfter() {
        return outputAfter;
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

    public long getRandomSeed() {
        return randomSeed;
    }

    public NodeStore getNodeStore() {
        return nodeStore;
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
