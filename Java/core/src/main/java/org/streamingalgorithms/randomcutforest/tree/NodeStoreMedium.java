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

public class NodeStoreMedium extends NodeStore {

    private final char[] parentIndex;
    private final int[] leftIndex;
    private final int[] rightIndex;
    public final char[] cutDimension;
    private final char[] mass;

    public NodeStoreMedium(int capacity, boolean storeParent) {
        super(capacity, null);
        mass = new char[capacity];
        parentIndex = storeParent ? newSentinelParents() : null;
        leftIndex = new int[capacity];
        rightIndex = new int[capacity];
        cutDimension = new char[capacity];
        Arrays.fill(leftIndex, capacity);
        Arrays.fill(rightIndex, capacity);
    }

    // size > 0
    public NodeStoreMedium(int capacity, int root, boolean storeParent, byte[] cutValuesData, int[] packedLeft,
            int[] packedRight, int[] packedCutDim, int size, boolean compressed) {
        super(capacity, ArrayPacking.unpackFloats(cutValuesData, capacity)); // adopts the unpacked cutValue float[]
        mass = new char[capacity];
        parentIndex = storeParent ? newSentinelParents() : null;

        leftIndex = ArrayPacking.unpackInts(packedLeft, capacity, compressed);
        rightIndex = ArrayPacking.unpackInts(packedRight, capacity, compressed);
        cutDimension = new char[capacity];
        ArrayPacking.unpackInts(packedCutDim, cutDimension, capacity, compressed);

        if (compressed) {
            reverseBits(size, leftIndex, rightIndex, capacity); // 0/1 bits -> node numbers, before parent loop
        }
        buildFreeList(root);
        if (parentIndex != null) {
            buildParents(root);
        }

    }

    private char[] newSentinelParents() {
        char[] p = new char[capacity];
        Arrays.fill(p, (char) capacity);
        return p;
    }

    /**
     * int[] reflation. Serialized child arrays hold 0/1 (internal/leaf); this
     * assigns breadth-first node numbers to internal slots and the capacity
     * sentinel elsewhere. entries [0,size) reflated, [size,capacity) sentinel.
     */
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

    protected void setParentIndex(int node, int parent) {
        parentIndex[node] = (char) parent;
    }

    @Override
    public int addNode(Stack<int[]> pathToRoot, float[] point, long sequenceIndex, int pointIndex, int childIndex,
            int childMassIfLeaf, int cutDimension, float cutValue, BoundingBox box) {
        int index = freeNodeManager.takeIndex();
        this.cutValue[index] = cutValue;
        this.cutDimension[index] = (char) cutDimension;
        if (leftOf(cutValue, cutDimension, point)) {
            this.leftIndex[index] = (pointIndex + capacity + 1);
            this.rightIndex[index] = childIndex;
        } else {
            this.rightIndex[index] = (pointIndex + capacity + 1);
            this.leftIndex[index] = childIndex;
        }
        this.mass[index] = (char) ((((childMassIfLeaf > 0) ? childMassIfLeaf : getMass(childIndex)) + 1)
                % (capacity + 1));
        int parentIndex = (pathToRoot.size() == 0) ? Null : pathToRoot.lastElement()[0];
        if (this.parentIndex != null) {
            this.parentIndex[index] = (char) parentIndex;
            if (!isLeaf(childIndex)) {
                this.parentIndex[childIndex] = (char) (index);
            }
        }
        if (parentIndex != Null) {
            spliceEdge(parentIndex, childIndex, index);
        }
        return index;
    }

    @Override
    public void assignInPartialTree(int node, float[] point, int childReference) {
        if (leftOf(node, point)) {
            leftIndex[node] = childReference;
        } else {
            rightIndex[node] = childReference;
        }
    }

    @Override
    protected void addRecord(int node, int left, int right, float cut, int cutD) {
        checkArgument(isInternal(node), "error in record");
        leftIndex[node] = left;
        rightIndex[node] = right;
        cutValue[node] = cut;
        cutDimension[node] = (char) cutD;
    }

    public int getLeftIndex(int index) {
        return leftIndex[index];
    }

    public int getRightIndex(int index) {
        return rightIndex[index];
    }

    public int getParentIndex(int index) {
        checkArgument(parentIndex != null, "incorrect call");
        return parentIndex[index];
    }

    public void setRoot(int index) {
        if (!isLeaf(index) && parentIndex != null) {
            parentIndex[index] = (char) capacity;
        }
    }

    @Override
    protected void decreaseMassOfInternalNode(int node) {
        mass[node] = (char) ((mass[node] + capacity) % (capacity + 1)); // this cannot get to 0
    }

    @Override
    protected void increaseMassOfInternalNode(int node) {
        mass[node] = (char) ((mass[node] + 1) % (capacity + 1));
        // mass of root == 0; note capacity = number_of_leaves - 1
    }

    @Override
    protected void setMassOfInternalNode(int node, int value) {
        mass[node] = (char) (value % (capacity + 1));
    }

    public void deleteInternalNode(int index) {
        leftIndex[index] = capacity;
        rightIndex[index] = capacity;
        if (parentIndex != null) {
            parentIndex[index] = (char) capacity;
        }
        freeNodeManager.releaseIndex(index);
    }

    public int getMass(int index) {
        return mass[index] != 0 ? mass[index] : (capacity + 1);
    }

    public void spliceEdge(int parent, int node, int newNode) {
        assert (!isLeaf(newNode));
        if (node == leftIndex[parent]) {
            leftIndex[parent] = newNode;
        } else {
            rightIndex[parent] = newNode;
        }
        if (parentIndex != null && isInternal(node)) {
            parentIndex[node] = (char) newNode;
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
            parentIndex[sibling] = (char) grandParent;
        }
    }

    public int getCutDimension(int index) {
        return cutDimension[index];
    }

    // public int[] getCutDimension() {
    // return toIntArray(cutDimension);
    // }

    // public int[] getLeftIndex() {
    // return Arrays.copyOf(leftIndex, leftIndex.length);
    // }

    // public int[] getRightIndex() {
    // return Arrays.copyOf(rightIndex, rightIndex.length);
    // }

}
