/*
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

import static org.streamingalgorithms.randomcutforest.CommonUtils.checkArgument;

import java.util.Arrays;
import java.util.Stack;

import org.streamingalgorithms.randomcutforest.util.ArrayPacking;

/**
 * A fixed-size buffer for storing interior tree nodes. An interior node is
 * defined by its location in the tree (parent and child nodes), its random cut,
 * and its bounding box. The NodeStore class uses arrays to store these field
 * values for a collection of nodes. An index in the store can be used to look
 * up the field values for a particular node.
 *
 * The internal nodes (handled by this store) corresponds to
 * [0..upperRangeLimit]
 *
 * If we think of an array of Node objects as being row-oriented (where each row
 * is a Node), then this class is analogous to a column-oriented database of
 * Nodes.
 *
 */
public class NodeStoreLarge extends NodeStore {

    private final int[] parentIndex;
    private final int[] leftIndex;
    private final int[] rightIndex;
    public final int[] cutDimension;
    private final int[] mass;

    public NodeStoreLarge(int capacity, boolean storeParent) {
        super(capacity, null);
        mass = new int[capacity];
        parentIndex = storeParent ? newSentinelParents() : null;
        leftIndex = new int[capacity];
        rightIndex = new int[capacity];
        cutDimension = new int[capacity];
        Arrays.fill(leftIndex, capacity);
        Arrays.fill(rightIndex, capacity);
    }

    public NodeStoreLarge(int capacity, int root, boolean storeParent, byte[] cutValuesData, int[] packedLeft,
            int[] packedRight, int[] packedCutDim, int size, boolean compressed) {
        super(capacity, ArrayPacking.unpackFloats(cutValuesData, capacity)); // adopts the unpacked cutValue float[]
        mass = new int[capacity];
        parentIndex = storeParent ? newSentinelParents() : null;

        leftIndex = ArrayPacking.unpackInts(packedLeft, capacity, compressed);
        rightIndex = ArrayPacking.unpackInts(packedRight, capacity, compressed);
        cutDimension = ArrayPacking.unpackInts(packedCutDim, capacity, compressed);

        if (compressed) {
            reverseBits(size, leftIndex, rightIndex, capacity); // 0/1 bits -> node numbers, before parent loop
        }
        buildFreeList(root);
        if (parentIndex != null) {
            buildParents(root);
        }
    }

    private int[] newSentinelParents() {
        int[] p = new int[capacity];
        Arrays.fill(p, capacity);
        return p;
    }

    protected void setParentIndex(int node, int parent) {
        parentIndex[node] = parent;
    }

    // shares NodeStoreMedium's int[] reflation semantics
    protected static void reverseBits(int size, int[] leftIndex, int[] rightIndex, int capacity) {
        int nodeCounter = 1;
        for (int i = 0; i < size; i++) {
            leftIndex[i] = (leftIndex[i] != 0) ? nodeCounter++ : capacity;
            rightIndex[i] = (rightIndex[i] != 0) ? nodeCounter++ : capacity;
        }
        for (int i = size; i < leftIndex.length; i++) {
            leftIndex[i] = rightIndex[i] = capacity;
        }
    }

    @Override
    public int addNode(Stack<int[]> pathToRoot, float[] point, long sequenceIndex, int pointIndex, int childIndex,
            int childMassIfLeaf, int cutDimension, float cutValue, BoundingBox box) {
        int index = freeNodeManager.takeIndex();
        this.cutValue[index] = cutValue;
        this.cutDimension[index] = (byte) cutDimension;
        if (leftOf(cutValue, cutDimension, point)) {
            this.leftIndex[index] = (pointIndex + capacity + 1);
            this.rightIndex[index] = childIndex;
        } else {
            this.rightIndex[index] = (pointIndex + capacity + 1);
            this.leftIndex[index] = childIndex;
        }
        this.mass[index] = (((childMassIfLeaf > 0) ? childMassIfLeaf : getMass(childIndex)) + 1) % (capacity + 1);

        int parentIndex = (pathToRoot.size() == 0) ? Null : pathToRoot.lastElement()[0];
        if (this.parentIndex != null) {
            this.parentIndex[index] = parentIndex;
            if (!isLeaf(childIndex)) {
                this.parentIndex[childIndex] = (index);
            }
        }
        if (parentIndex != Null) {
            spliceEdge(parentIndex, childIndex, index);
        }
        return index;
    }

    public int getLeftIndex(int index) {
        return leftIndex[index];
    }

    public int getRightIndex(int index) {
        return rightIndex[index];
    }

    public void setRoot(int index) {
        if (!isLeaf(index) && parentIndex != null) {
            parentIndex[index] = capacity;
        }
    }

    @Override
    protected void decreaseMassOfInternalNode(int node) {
        mass[node] = (mass[node] + capacity) % (capacity + 1);
    }

    @Override
    protected void increaseMassOfInternalNode(int node) {
        mass[node] = (mass[node] + 1) % (capacity + 1);
    }

    @Override
    protected void setMassOfInternalNode(int node, int value) {
        mass[node] = value % (capacity + 1);
    }

    @Override
    protected void addRecord(int node, int left, int right, float cut, int cutD) {
        checkArgument(isInternal(node), "error in record");
        leftIndex[node] = left;
        rightIndex[node] = right;
        cutValue[node] = cut;
        cutDimension[node] = cutD;
    }

    public void deleteInternalNode(int index) {
        leftIndex[index] = capacity;
        rightIndex[index] = capacity;
        if (parentIndex != null) {
            parentIndex[index] = capacity;
        }
        freeNodeManager.releaseIndex(index);
    }

    public int getMass(int index) {
        return mass[index] != 0 ? mass[index] : (capacity + 1);
    }

    @Override
    public void assignInPartialTree(int node, float[] point, int childReference) {
        if (leftOf(node, point)) {
            leftIndex[node] = childReference;
        } else {
            rightIndex[node] = childReference;
        }
    }

    public void spliceEdge(int parent, int node, int newNode) {
        assert (!isLeaf(newNode));
        if (node == leftIndex[parent]) {
            leftIndex[parent] = newNode;
        } else {
            rightIndex[parent] = newNode;
        }
        if (parentIndex != null && isInternal(node)) {
            parentIndex[node] = newNode;
        }
    }

    public void replaceParentBySibling(int grandParent, int parent, int node) {
        int sibling = getSibling(node, parent);
        if (parent == leftIndex[grandParent]) {
            leftIndex[grandParent] = sibling;
        } else {
            rightIndex[grandParent] = sibling;
        }
        if (parentIndex != null && isInternal(sibling)) {
            parentIndex[sibling] = grandParent;
        }
    }

    public int getCutDimension(int index) {
        return cutDimension[index];
    }

    public int[] getCutDimension() {
        return Arrays.copyOf(cutDimension, cutDimension.length);
    }

    public int[] getLeftIndex() {
        return Arrays.copyOf(leftIndex, leftIndex.length);
    }

    public int[] getRightIndex() {
        return Arrays.copyOf(rightIndex, rightIndex.length);
    }

    public int getParentIndex(int index) {
        checkArgument(parentIndex != null, "incorrect call");
        return parentIndex[index];
    }

}
