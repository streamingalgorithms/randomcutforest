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

public class NodeStoreSmall extends NodeStore {

    private final byte[] parentIndex;
    private final char[] leftIndex;
    private final char[] rightIndex;
    public final byte[] cutDimension;
    private final byte[] mass;

    public NodeStoreSmall(int capacity, boolean storeParent) {
        super(capacity, null);
        mass = new byte[capacity];
        parentIndex = storeParent ? newSentinelParents() : null;
        leftIndex = new char[capacity];
        rightIndex = new char[capacity];
        cutDimension = new byte[capacity];
        Arrays.fill(leftIndex, (char) capacity);
        Arrays.fill(rightIndex, (char) capacity);
    }

    /**
     * Deserialize constructor: unpack packed left/right/cutDimension DIRECTLY into
     * the retained narrow arrays (no wide int[] intermediate, no conversion copy),
     * reflate the bits to node numbers, then build the parent pointers and free
     * list.
     *
     * @param size number of nodes the bit-arrays encode (state.getSize()); entries
     *             [0,size) reflate to node numbers, [size,capacity) become sentinel
     */
    public NodeStoreSmall(int capacity, int root, boolean storeParent, byte[] cutValuesData, int[] packedLeft,
            int[] packedRight, int[] packedCutDim, int size, boolean compressed) {
        super(capacity, ArrayPacking.unpackFloats(cutValuesData, capacity)); // adopts the unpacked cutValue float[]
        mass = new byte[capacity];
        parentIndex = storeParent ? newSentinelParents() : null;

        leftIndex = new char[capacity];
        rightIndex = new char[capacity];
        cutDimension = new byte[capacity];
        ArrayPacking.unpackInts(packedLeft, leftIndex, capacity, compressed);
        ArrayPacking.unpackInts(packedRight, rightIndex, capacity, compressed);
        ArrayPacking.unpackInts(packedCutDim, cutDimension, capacity, compressed);

        if (compressed) {
            reverseBits(size, leftIndex, rightIndex, capacity); // 0/1 bits -> node numbers, before parent loop
        }
        buildFreeList(root);
        if (parentIndex != null) {
            buildParents(root);
        }
    }

    /**
     * char[] reflation for the Small store. The serialized child arrays store 0/1
     * (internal/leaf); this assigns breadth-first node numbers to the internal
     * slots and writes the capacity sentinel elsewhere. entries [0,size) are
     * reflated, [size,capacity) set to sentinel.
     */
    protected static void reverseBits(int size, char[] leftIndex, char[] rightIndex, int capacity) {
        int nodeCounter = 1;
        for (int i = 0; i < size; i++) {
            leftIndex[i] = (leftIndex[i] != 0) ? (char) nodeCounter++ : (char) capacity;
            rightIndex[i] = (rightIndex[i] != 0) ? (char) nodeCounter++ : (char) capacity;
        }
        for (int i = size; i < leftIndex.length; i++) {
            leftIndex[i] = rightIndex[i] = (char) capacity;
        }
    }

    private byte[] newSentinelParents() {
        byte[] p = new byte[capacity];
        Arrays.fill(p, (byte) capacity);
        return p;
    }

    protected void setParentIndex(int node, int parent) {
        parentIndex[node] = (byte) parent;
    }

    @Override
    public int addNode(Stack<int[]> pathToRoot, float[] point, long sequenceIndex, int pointIndex, int childIndex,
            int childMassIfLeaf, int cutDimension, float cutValue, BoundingBox box) {
        int index = freeNodeManager.takeIndex();
        this.cutValue[index] = cutValue;
        this.cutDimension[index] = (byte) cutDimension;
        if (leftOf(cutValue, cutDimension, point)) {
            this.leftIndex[index] = (char) (pointIndex + capacity + 1);
            this.rightIndex[index] = (char) childIndex;
        } else {
            this.rightIndex[index] = (char) (pointIndex + capacity + 1);
            this.leftIndex[index] = (char) childIndex;
        }
        this.mass[index] = (byte) ((((childMassIfLeaf > 0) ? childMassIfLeaf : getMass(childIndex)) + 1)
                % (capacity + 1));
        int parentIndex = (pathToRoot.size() == 0) ? Null : pathToRoot.lastElement()[0];
        if (this.parentIndex != null) {
            this.parentIndex[index] = (byte) parentIndex;
            if (!isLeaf(childIndex)) {
                this.parentIndex[childIndex] = (byte) (index);
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
            leftIndex[node] = (char) childReference;
        } else {
            rightIndex[node] = (char) childReference;
        }
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
            parentIndex[index] = (byte) capacity;
        }
    }

    @Override
    protected void decreaseMassOfInternalNode(int node) {
        mass[node] = (byte) (((mass[node] & 0xff) + capacity) % (capacity + 1)); // this cannot get to 0
    }

    @Override
    protected void increaseMassOfInternalNode(int node) {
        mass[node] = (byte) (((mass[node] & 0xff) + 1) % (capacity + 1));
        // mass of root == 0; note capacity = number_of_leaves - 1
    }

    @Override
    protected void setMassOfInternalNode(int node, int value) {
        mass[node] = (byte) (value % (capacity + 1));
    }

    @Override
    protected void addRecord(int node, int left, int right, float cut, int cutD) {
        checkArgument(isInternal(node), "error in record");
        leftIndex[node] = (char) left;
        rightIndex[node] = (char) right;
        cutValue[node] = cut;
        cutDimension[node] = (byte) cutD;
    }

    public void deleteInternalNode(int index) {
        leftIndex[index] = (char) capacity;
        rightIndex[index] = (char) capacity;
        if (parentIndex != null) {
            parentIndex[index] = (byte) capacity;
        }
        freeNodeManager.releaseIndex(index);
    }

    public int getMass(int index) {
        return mass[index] != 0 ? (mass[index] & 0xff) : (capacity + 1);
    }

    public void spliceEdge(int parent, int node, int newNode) {
        assert (!isLeaf(newNode));
        if (node == leftIndex[parent]) {
            leftIndex[parent] = (char) newNode;
        } else {
            rightIndex[parent] = (char) newNode;
        }
        if (parentIndex != null && isInternal(node)) {
            parentIndex[node] = (byte) newNode;
        }
    }

    public void replaceParentBySibling(int grandParent, int parent, int node) {
        int sibling = getSibling(node, parent);
        if (parent == leftIndex[grandParent]) {
            leftIndex[grandParent] = (char) sibling;
        } else {
            rightIndex[grandParent] = (char) sibling;
        }
        if (parentIndex != null && isInternal(sibling)) {
            parentIndex[sibling] = (byte) grandParent;
        }
    }

    public int getCutDimension(int index) {
        return cutDimension[index] & 0xff;
    }

    // public int[] getCutDimension() {
    // return toIntArray(cutDimension);
    // }

    // public int[] getLeftIndex() {
    // return toIntArray(leftIndex);
    // }

    // public int[] getRightIndex() {
    // return toIntArray(rightIndex);
    // }

    /**
     * char[] reflation for the Small store. The serialized child arrays store 0/1
     * (internal/leaf); this assigns breadth-first node numbers to the internal
     * slots and writes the capacity sentinel elsewhere. entries [0,size) are
     * reflated, [size,capacity) set to sentinel.
     */

}
