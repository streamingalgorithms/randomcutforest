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
import static org.streamingalgorithms.randomcutforest.util.ArrayEncoder.unpackFloats;

import java.util.BitSet;

import org.streamingalgorithms.randomcutforest.store.IndexIntervalManager;
import org.streamingalgorithms.randomcutforest.util.ArrayEncoder;

import lombok.Getter;

/**
 * Single concrete node store. Replaces the abstract NodeStore +
 * Small/Medium/Large + widthFor + the band router. Holds one {@link Column} per
 * internal-node field, each sized independently from its own analytical bound.
 * cutValue stays a real float[] (payload, not narrowable). Leaf mass is NOT
 * here — it lives in the tree's sparse Int2IntOpenHashMap; only internal-node
 * mass is a column.
 */
public class NodeStore {

    public static final int Null = -1;
    public static final boolean DEFAULT_STORE_PARENT = false;

    @Getter
    protected final int capacity; // == numberOfLeaves - 1 (number of internal-node slots)
    protected final int numberOfLeaves; // == capacity + 1 (== the partial-tree placeholder value)
    protected final float[] cutValue;

    protected final Column left; // child ref: internal idx, leaf ref, or placeholder(=capacity)
    protected final Column right;
    protected final Column cutDim; // [0, dimension-1]
    protected final Column mass; // stored internal mass, mod numberOfLeaves; 0 <=> full root
    protected final Column parent; // nullable; internal idx or Null

    protected IndexIntervalManager freeNodeManager;

    // ---- construction ----

    /**
     * @param capacity     numberOfLeaves - 1
     * @param dimension    shingled dimension
     * @param leafRefBound the TRUE max value any left/right slot can hold:
     *                     pointStoreCapacity + numberOfLeaves - 1 (== P + S - 1).
     *                     This is the fix for the router-keyed-off-capacity bug:
     *                     the old widthFor(capacity, dimension) narrowed on S-1,
     *                     but the slots range over leaf refs up to P + S - 1.
     */
    protected NodeStore(int capacity, int dimension, long leafRefBound, boolean storeParents) {
        this.capacity = capacity;
        this.numberOfLeaves = capacity + 1;
        this.cutValue = new float[capacity];
        this.left = Column.of(capacity, leafRefBound, false); // placeholder S-1 is IN range, no reserve
        this.right = Column.of(capacity, leafRefBound, false);
        this.cutDim = Column.of(capacity, dimension - 1, false);
        this.mass = Column.of(capacity, capacity, false); // stored max = S-1; root's S wraps to 0
        this.parent = storeParents ? Column.of(capacity, capacity - 1, true) : null; // sentinel == Null
        this.freeNodeManager = new IndexIntervalManager(capacity);
    }

    public static NodeStore nodeStore(int capacity, int dimension, long leafRefBound, boolean storeParents) {
        return new NodeStore(capacity, dimension, leafRefBound, storeParents);
    }

    public static NodeStore from(int capacity, int root, int dimension, long leafRefBound, boolean storeParents,
            byte[] cutValuesData, int[] packedLeft, int[] packedRight, int[] packedCutDim, int size,
            boolean compressed) {
        checkArgument(size > 0, "use the regular constructor for size 0");
        return new NodeStore(capacity, root, dimension, leafRefBound, storeParents, cutValuesData, packedLeft,
                packedRight, packedCutDim, size, compressed);
    }

    /**
     * Deserialize. left/right arrive as packed 0/1 bit streams (1 == internal
     * child); they are unpacked into the tier-matched primitive and reflated to BFS
     * node numbers by reverseBits, writing the `capacity` placeholder for
     * external/absent children. This preserves the no-wide-intermediate unpack
     * path: unpackInts + reverseBits run on the native char[]/int[], and the Column
     * simply adopts that array afterward.
     *
     * cutDim is a real per-node value array (not bits) -> unpack + adopt, no
     * reflation. mass is NOT serialized -> fresh column, rebuilt post-order by
     * validateAndReconstruct. parent (if kept) -> fresh sentinel column, rebuilt by
     * buildParents.
     */
    protected NodeStore(int capacity, int root, int dimension, long leafRefBound, boolean storeParents,
            byte[] cutValuesData, int[] packedLeft, int[] packedRight, int[] packedCutDim, int size,
            boolean compressed) {
        this.capacity = capacity;
        this.numberOfLeaves = capacity + 1;
        this.cutValue = unpackFloats(cutValuesData, capacity); // adopt, as the old Small/Large did

        // ---- left / right: unpack + reflate in native primitive, then adopt ----
        // tier keyed on leafRefBound (== P + S - 1), the SAME bound construction uses.
        Column l, r;

        switch (Column.tierFor(leafRefBound, false)) {
        case CHAR: {
            char[] la = new char[capacity], ra = new char[capacity];
            ArrayEncoder.unpackInts(packedLeft, la, capacity, compressed);
            ArrayEncoder.unpackInts(packedRight, ra, capacity, compressed);
            if (compressed)
                reverseBits(size, la, ra, capacity);
            l = new Column.CharColumn(la);
            r = new Column.CharColumn(ra);
            break;
        }
        case INT: {
            int[] la = new int[capacity], ra = new int[capacity];
            ArrayEncoder.unpackInts(packedLeft, la, capacity, compressed);
            ArrayEncoder.unpackInts(packedRight, ra, capacity, compressed);
            if (compressed)
                reverseBits(size, la, ra, capacity);
            l = new Column.IntColumn(la);
            r = new Column.IntColumn(ra);
            break;
        }
        default: { // BYTE — only tiny forests reach here; reflate in char[] then narrow-adopt
            char[] la = new char[capacity], ra = new char[capacity];
            ArrayEncoder.unpackInts(packedLeft, la, capacity, compressed);
            ArrayEncoder.unpackInts(packedRight, ra, capacity, compressed);
            if (compressed)
                reverseBits(size, la, ra, capacity);
            l = new Column.ByteColumn(toBytes(la));
            r = new Column.ByteColumn(toBytes(ra));
            break;
        }
        }
        this.left = l;
        this.right = r;

        // ---- cutDim: real values, unpack + adopt, no reflation ----
        switch (Column.tierFor(dimension - 1, false)) {
        case BYTE: {
            byte[] c = new byte[capacity];
            ArrayEncoder.unpackInts(packedCutDim, c, capacity, compressed);
            this.cutDim = new Column.ByteColumn(c);
            break;
        }
        case CHAR: {
            char[] c = new char[capacity];
            ArrayEncoder.unpackInts(packedCutDim, c, capacity, compressed);
            this.cutDim = new Column.CharColumn(c);
            break;
        }
        default: {
            int[] c = new int[capacity];
            ArrayEncoder.unpackInts(packedCutDim, c, capacity, compressed);
            this.cutDim = new Column.IntColumn(c);
            break;
        }
        }

        // ---- mass: not serialized; fresh, rebuilt post-order by the tree ----
        this.mass = Column.of(capacity, capacity, false);

        // ---- parent: not serialized; fresh sentinel column, rebuilt by buildParents
        // ----
        this.parent = storeParents ? Column.of(capacity, capacity - 1, true) : null;

        this.freeNodeManager = null; // set by buildFreeList
        buildFreeList(root);
        if (parent != null)
            buildParents(root);
    }

    // BYTE-tier left/right: reflate in char[] (reverseBits writes `capacity`, which
    // fits
// a byte here precisely because tierFor(leafRefBound)==BYTE implies leafRefBound<=255
// and capacity <= leafRefBound), then narrow.
    private static byte[] toBytes(char[] a) {
        byte[] b = new byte[a.length];
        for (int i = 0; i < a.length; i++)
            b[i] = (byte) a[i];
        return b;
    }

    // ---- free-list / parent reflation (unchanged logic, now column-backed) ----

    protected void buildFreeList(int root) {
        BitSet bits = new BitSet(capacity);
        bits.set(root);
        for (int i = 0; i < capacity; i++) {
            if (isInternal(getLeftIndex(i)))
                bits.set(getLeftIndex(i));
            if (isInternal(getRightIndex(i)))
                bits.set(getRightIndex(i));
        }
        freeNodeManager = new IndexIntervalManager(capacity, capacity, bits);
    }

    protected void buildParents(int root) {
        for (int i = 0; i < capacity; i++) {
            if (isInternal(getLeftIndex(i)))
                setParentIndex(getLeftIndex(i), i);
            if (isInternal(getRightIndex(i)))
                setParentIndex(getRightIndex(i), i);
        }
    }

    // ---- node classification (identical boundaries to the abstract original) ----

    public boolean isLeaf(int index) {
        return index > capacity;
    } // >= numberOfLeaves

    public boolean isInternal(int index) {
        return index < capacity && index >= 0;
    }

    // ---- column accessors ----

    public int getLeftIndex(int i) {
        return left.get(i);
    }

    public int getRightIndex(int i) {
        return right.get(i);
    }

    public int getCutDimension(int i) {
        return cutDim.get(i);
    }

    public double getCutValue(int i) {
        return cutValue[i];
    }

    public float[] getCutValues() {
        return cutValue;
    }

    protected int convertToLeaf(int pointIndex) {
        return pointIndex + numberOfLeaves;
    }

    protected int addLeafPoints(int[] leafArray, int node, int firstEmpty) {
        if (isLeaf(node)) {
            leafArray[firstEmpty++] = translateIndex(node);
            return firstEmpty;
        }
        return addLeafPoints(leafArray, getRightIndex(node), addLeafPoints(leafArray, getLeftIndex(node), firstEmpty));
    }

    int translateIndex(int index) {
        return index - numberOfLeaves;
    }

    public int getParentIndex(int node) {
        return (parent == null) ? Null : (parent.get(node) == parent.sentinel() ? Null : parent.get(node));
    }

    protected void setParentIndex(int node, int p) {
        if (parent != null)
            parent.set(node, (p == Null) ? parent.sentinel() : p);
    }

    // ---- mass: stored mod numberOfLeaves; only the root ever wraps (stored 0 ==
    // full) ----

    public int getMass(int index) {
        int raw = mass.get(index);
        return (raw == 0) ? numberOfLeaves : raw; // 0 is unreachable for non-root live internals (their mass >= 2)
    }

    protected void setMassOfInternalNode(int node, int value) {
        assert value >= 2 && value <= numberOfLeaves : "internal mass out of range " + value; // catches the wrap bug
        mass.set(node, value % numberOfLeaves);
    }

    protected void increaseMassOfInternalNode(int node) {
        setMassOfInternalNode(node, getMass(node) + 1);
    }

    protected void decreaseMassOfInternalNode(int node) {
        setMassOfInternalNode(node, getMass(node) - 1);
    }

    // ---- record mutation ----

    protected void addRecord(int node, int l, int r, float cut, int cutD) {
        left.set(node, l);
        right.set(node, r);
        cutValue[node] = cut;
        cutDim.set(node, cutD);
    }

    public void spliceEdge(int parent, int node, int newNode) {
        if (left.get(parent) == node)
            left.set(parent, newNode);
        else
            right.set(parent, newNode);
    }

    /** The sibling is being promoted to root: it has no parent. */
    public void detachAsRoot(int node) {
        if (parent != null && isInternal(node))
            setParentIndex(node, Null);
    }

    public void replaceParentBySibling(int grandParent, int parentNode, int node) {
        int sibling = getSibling(node, parentNode);
        if (left.get(grandParent) == parentNode)
            left.set(grandParent, sibling);
        else
            right.set(grandParent, sibling);
        if (parent != null && isInternal(sibling))
            setParentIndex(sibling, grandParent);
    }

    public void deleteInternalNode(int index) {
        freeNodeManager.releaseIndex(index);
    }

    /**
     * Partial-tree fill: overwrite the placeholder child (value == capacity ==
     * numberOfLeaves-1) of savedParent with a freshly-sampled leaf ref. The
     * placeholder is what lets serde persist a skeleton and reconstruct leaves by
     * replaying the sampler, instead of storing point payload at every node.
     */
    public void assignInPartialTree(int savedParent, float[] point, int childReference) {
        if (toLeft(point, savedParent)) {
            assert left.get(savedParent) == capacity : "expected placeholder on left of " + savedParent;
            left.set(savedParent, childReference);
        } else {
            assert right.get(savedParent) == capacity : "expected placeholder on right of " + savedParent;
            right.set(savedParent, childReference);
        }
        if (parent != null && isInternal(childReference))
            setParentIndex(childReference, savedParent);
    }

    // ---- traversal / topology (identical logic to the abstract original) ----

    protected boolean leftOf(float cut, int cutDimension, float[] point) {
        return point[cutDimension] <= cut;
    }

    public boolean leftOf(int node, float[] point) {
        return leftOf(cutValue[node], getCutDimension(node), point);
    }

    protected boolean toLeft(float[] point, int off) {
        return point[getCutDimension(off)] <= cutValue[off];
    }

    protected int getSibling(int node, int parentNode) {
        int sibling = getLeftIndex(parentNode);
        if (node == sibling)
            sibling = getRightIndex(parentNode);
        return sibling;
    }

    /**
     * Fills path[0..depth) root→leaf, returns depth. path must hold numberOfLeaves
     * entries.
     */
    protected int getPathInto(int root, float[] point, int[] path) {
        int node = root, depth = 0;
        path[depth++] = node;
        while (isInternal(node)) {
            node = leftOf(node, point) ? getLeftIndex(node) : getRightIndex(node);
            path[depth++] = node;
        }
        return depth;
    }

    public int addNode(int parentIndex, float[] point, long sequenceIndex, int pointIndex, int childIndex,
            int childMassIfLeaf, int cutDimension, float cut) {
        int index = freeNodeManager.takeIndex();
        if (leftOf(cut, cutDimension, point)) {
            addRecord(index, pointIndex + capacity + 1, childIndex, cut, cutDimension);
        } else {
            addRecord(index, childIndex, pointIndex + capacity + 1, cut, cutDimension);
        }
        setMassOfInternalNode(index, ((childMassIfLeaf > 0) ? childMassIfLeaf : getMass(childIndex)) + 1);
        setParentIndex(index, parentIndex);
        if (!isLeaf(childIndex))
            setParentIndex(childIndex, index);
        if (parentIndex != Null)
            spliceEdge(parentIndex, childIndex, index);
        return index;
    }

    public int reorderNodesInBreadthFirstOrder(int[] map, int root, int capacity) {
        if (root == Null || root >= capacity)
            return 0;
        int head = 0, tail = 0;
        map[tail++] = root;
        while (head < tail) {
            int node = map[head++];
            int l = getLeftIndex(node);
            if (l < capacity)
                map[tail++] = l;
            int r = getRightIndex(node);
            if (r < capacity)
                map[tail++] = r;
        }
        return tail;
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

    /**
     * int[] reflation. Serialized child arrays hold 0/1 (internal/leaf); this
     * assigns breadth-first node numbers to internal slots and the capacity
     * sentinel elsewhere. entries [0,size) reflated, [size,capacity) sentinel.
     */
    protected static void reverseBits(int size, char[] leftIndex, char[] rightIndex, int capacity) {
        char nodeCounter = 1;
        for (int i = 0; i < size; i++) {
            leftIndex[i] = (leftIndex[i] != 0) ? nodeCounter++ : (char) capacity;
            rightIndex[i] = (rightIndex[i] != 0) ? nodeCounter++ : (char) capacity;
        }
        for (int i = size; i < leftIndex.length; i++) {
            leftIndex[i] = rightIndex[i] = (char) capacity;
        }
    }
}
