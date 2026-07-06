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

package org.streamingalgorithms.randomcutforest.tree;

import static org.streamingalgorithms.randomcutforest.CommonUtils.checkArgument;

import java.util.BitSet;
import java.util.Stack;
import java.util.concurrent.ArrayBlockingQueue;

import org.streamingalgorithms.randomcutforest.store.IndexIntervalManager;

import lombok.Getter;

public abstract class NodeStore {

    public static int Null = -1;
    public static boolean DEFAULT_STORE_PARENT = false;

    /**
     * Width bands; the single source of truth for store selection (see
     * {@link #widthFor}).
     */
    static final int SMALL = 0, MEDIUM = 1, LARGE = 2;

    @Getter
    protected final int capacity;
    protected final float[] cutValue;
    protected IndexIntervalManager freeNodeManager;

    protected NodeStore(int capacity, float[] cutValues) {
        this.capacity = capacity;
        if (cutValues != null) {
            cutValue = cutValues;
        } else {
            cutValue = new float[capacity];
            freeNodeManager = new IndexIntervalManager(capacity);
        }
    }

    /**
     * Set occupied bits from the reflated child arrays, fill parent pointers, build
     * the free list.
     */
    protected void buildFreeList(int root) {
        BitSet bits = new BitSet(capacity);
        bits.set(root); // we cannot get here if root is not meaningful

        for (int i = 0; i < capacity; i++) {
            if (isInternal(getLeftIndex(i))) {
                bits.set(getLeftIndex(i));
            }
        }
        for (int i = 0; i < capacity; i++) {
            if (isInternal(getRightIndex(i))) {
                bits.set(getRightIndex(i));
            }
        }
        freeNodeManager = new IndexIntervalManager(capacity, capacity, bits);
    }

    // the following should only be run when the parentIndex is set
    protected void buildParents(int root) {
        for (int i = 0; i < capacity; i++) {
            if (isInternal(getLeftIndex(i))) {
                setParentIndex(getLeftIndex(i), i);
            }
            if (isInternal(getRightIndex(i))) {
                setParentIndex(getRightIndex(i), i);
            }
        }
    }

    abstract protected void setParentIndex(int node, int parent);

    /**
     * The single width ladder.
     */
    static int widthFor(int capacity, int dimension) {
        if (capacity < 256 && dimension <= 256) {
            return SMALL;
        } else if (capacity < Character.MAX_VALUE && dimension <= Character.MAX_VALUE) {
            return MEDIUM;
        } else {
            return LARGE;
        }
    }

    public static NodeStore nodeStore(int capacity, int dimension, boolean storeParents) {
        switch (widthFor(capacity, dimension)) {
        case SMALL:
            return new NodeStoreSmall(capacity, storeParents);
        case MEDIUM:
            return new NodeStoreMedium(capacity, storeParents);
        default:
            return new NodeStoreLarge(capacity, storeParents);
        }
    }

    /**
     * Deserialize dispatcher. The concrete store unpacks the packed
     * left/right/cutDim arrays into its own narrow type and reflates the bits
     * internally; this method only chooses the width (via {@link #widthFor}) and
     * forwards. Root has to be nontrivial 0 LE root LT capacity =(number of leaves
     * -1)
     */
    public static NodeStore from(int capacity, int root, int dimension, boolean storeParents, byte[] cutValuesData,
            int[] packedLeft, int[] packedRight, int[] packedCutDim, int size, boolean compressed) {
        checkArgument(size > 0, "use the regular constructor for size 0");
        switch (widthFor(capacity, dimension)) {
        case SMALL:
            return new NodeStoreSmall(capacity, root, storeParents, cutValuesData, packedLeft, packedRight,
                    packedCutDim, size, compressed);
        case MEDIUM:
            return new NodeStoreMedium(capacity, root, storeParents, cutValuesData, packedLeft, packedRight,
                    packedCutDim, size, compressed);
        default:
            return new NodeStoreLarge(capacity, root, storeParents, cutValuesData, packedLeft, packedRight,
                    packedCutDim, size, compressed);
        }
    }

    protected abstract int addNode(Stack<int[]> pathToRoot, float[] point, long sendex, int pointIndex, int childIndex,
            int childMassIfLeaf, int cutDimension, float cutValue, BoundingBox box);

    public boolean isLeaf(int index) {
        return index > capacity;
    }

    public boolean isInternal(int index) {
        // capacity == numberOfLeaves - 1; same boundary the tree calls (index <
        // numberOfLeaves - 1)
        return index < capacity && index >= 0;
    }

    public abstract void assignInPartialTree(int savedParent, float[] point, int childReference);

    public abstract int getLeftIndex(int index);

    public abstract int getRightIndex(int index);

    public abstract int getParentIndex(int index);

    protected abstract void decreaseMassOfInternalNode(int node);

    protected abstract void increaseMassOfInternalNode(int node);

    protected abstract void addRecord(int node, int left, int right, float cut, int cutD);

    public Stack<int[]> getPath(int root, float[] point, boolean verbose) {
        int node = root;
        Stack<int[]> answer = new Stack<>();
        answer.push(new int[] { root, capacity });
        while (isInternal(node)) {
            if (leftOf(node, point)) {
                answer.push(new int[] { getLeftIndex(node), getRightIndex(node) });
                node = getLeftIndex(node);
            } else {
                answer.push(new int[] { getRightIndex(node), getLeftIndex(node) });
                node = getRightIndex(node);
            }
        }
        return answer;
    }

    public abstract void deleteInternalNode(int index);

    public abstract int getMass(int index);

    protected boolean leftOf(float cutValue, int cutDimension, float[] point) {
        return point[cutDimension] <= cutValue;
    }

    public boolean leftOf(int node, float[] point) {
        return leftOf(cutValue[node], getCutDimension(node), point);
    }

    public int getSibling(int node, int parent) {
        int sibling = getLeftIndex(parent);
        if (node == sibling) {
            sibling = getRightIndex(parent);
        }
        return sibling;
    }

    public abstract void spliceEdge(int parent, int node, int newNode);

    public abstract void replaceParentBySibling(int grandParent, int parent, int node);

    public abstract int getCutDimension(int index);

    public double getCutValue(int index) {
        return cutValue[index];
    }

    protected boolean toLeft(float[] point, int currentNodeOffset) {
        return point[getCutDimension(currentNodeOffset)] <= cutValue[currentNodeOffset];
    }

    public float[] getCutValues() {
        return cutValue;
    }

    protected int size() {
        return capacity - freeNodeManager.size();
    }

    protected abstract void setMassOfInternalNode(int node, int value);

    /**
     * The following function reorders the nodes stored in the tree in a breadth
     * first order; Note that a regular binary tree where each internal node has 2
     * chidren, as is the case for AbstractRandomCutTree or any tree produced in a
     * Random Forest ensemble (not restricted to Random Cut Forests), has maxsize -
     * 1 internal nodes for maxSize number of leaves. The leaves are numbered 0 +
     * (maxsize), 1 + (maxSize), ..., etc. in that BFS ordering. The root is node 0.
     *
     * Note that if the binary tree is a complete binary tree, then the numbering
     * would correspond to the well known heuristic where children of node index i
     * are numbered 2*i and 2*i + 1. The trees in AbstractCompactRandomCutTree will
     * not be complete binary trees. But a similar numbering enables us to compress
     * the entire structure of the tree into two bit arrays corresponding to
     * presence of left and right children. The idea can be viewed as similar to
     * Zak's numbering for regular binary trees Lexicographic generation of binary
     * trees, S. Zaks, TCS volume 10, pages 63-82, 1980, that uses depth first
     * numbering. However an extensive literature exists on this topic.
     *
     * The overall relies on the extra advantage that we can use two bit sequences;
     * the left and right child pointers which appears to be simple. While it is
     * feasible to always maintain this order, that would complicate the standard
     * binary search tree pattern and this tranformation is used when the tree is
     * serialized. Note that while there is savings in representing the tree
     * structure into two bit arrays, the bulk of the serialization corresponds to
     * the payload at the nodes (cuts, dimensions for internal nodes and index to
     * pointstore, number of copies for the leaves). The translation to the bits is
     * handled by the NodeStoreMapper. The algorithm here corresponds to just
     * producing the cannoical order.
     *
     * The algorithm renumbers the nodes in BFS ordering.
     */

    public int reorderNodesInBreadthFirstOrder(int[] map, int root, int capacity) {
        if ((root != Null) && (root < capacity)) {
            int currentNode = 0;
            ArrayBlockingQueue<Integer> nodeQueue = new ArrayBlockingQueue<>(capacity);
            nodeQueue.add(root);
            while (!nodeQueue.isEmpty()) {
                int head = nodeQueue.poll();
                int leftChild = getLeftIndex(head); // was leftIndex[head]
                if (leftChild < capacity) {
                    nodeQueue.add(leftChild);
                }
                int rightChild = getRightIndex(head); // was rightIndex[head]
                if (rightChild < capacity) {
                    nodeQueue.add(rightChild);
                }
                map[currentNode] = head;
                currentNode++;
            }
            return currentNode;
        }
        return 0;
    }
}
