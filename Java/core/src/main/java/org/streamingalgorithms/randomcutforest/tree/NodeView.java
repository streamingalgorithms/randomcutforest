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

import static org.streamingalgorithms.randomcutforest.CommonUtils.checkState;

import java.util.HashMap;

import org.streamingalgorithms.randomcutforest.store.IPointStoreView;

public class NodeView implements INodeView {

    public static double SWITCH_FRACTION = 0.499;

    RandomCutTree tree;
    int currentNodeOffset;
    float[] leafPoint;

    ArrayBox currentBox;
    private final ArrayBox reusableBox;
    private final ArrayBox pathBox;
    protected int[] nodeScratch;

    public NodeView(RandomCutTree tree, IPointStoreView<float[]> pointStoreView, int root) {
        this.currentNodeOffset = root;
        this.tree = tree;
        int dimensions = pointStoreView.getDimensions();
        leafPoint = new float[dimensions];
        reusableBox = new ArrayBox(dimensions);
        pathBox = new ArrayBox(dimensions);
        int numberOfLeaves = (tree != null) ? tree.numberOfLeaves : 256;
        nodeScratch = new int[numberOfLeaves];
    }

    // NodeView
    protected void rearm(RandomCutTree tree, int root) {
        this.tree = tree;
        this.currentNodeOffset = root;
        this.currentBox = null;
        if (nodeScratch == null || nodeScratch.length < tree.numberOfLeaves) {
            nodeScratch = new int[tree.numberOfLeaves];
        }
        // deactivate; pathBox array retained, re-primed by setCurrentNode
    }

    public int getMass() {
        return tree.getMass(currentNodeOffset);
    }

    public IBoundingBoxView getBoundingBox() {
        if (currentBox != null)
            return currentBox;
        tree.fillArrayBox(currentNodeOffset, reusableBox); // fills reusableBox in place, no alloc
        return reusableBox;
    }

    public IBoundingBoxView getSiblingBoundingBox(float[] point) {
        if (toLeft(point)) {
            tree.fillArrayBox(tree.nodeStore.getRightIndex(currentNodeOffset), reusableBox);
        } else {
            tree.fillArrayBox(tree.nodeStore.getLeftIndex(currentNodeOffset), reusableBox);
        }
        return reusableBox;
    }

    public int getCutDimension() {
        return tree.nodeStore.getCutDimension(currentNodeOffset);
    }

    @Override
    public double getCutValue() {
        return tree.nodeStore.getCutValue(currentNodeOffset);
    }

    public float[] getLeafPoint() {
        return leafPoint;
    }

    public HashMap<Long, Integer> getSequenceIndexes() {
        checkState(isLeaf(), "can only be invoked for a leaf");
        if (tree.storeSequenceIndexesEnabled) {
            return tree.getSequenceMap(tree.getPointIndex(currentNodeOffset));
        } else {
            return new HashMap<>();
        }
    }

    // probability of cut which is in 2*d half dimensions
    @Override
    public double probabilityOfSeparation(float[] expandedPoint, float[] components) {
        return tree.probabilityOfCutExpanded(currentNodeOffset, expandedPoint, currentBox, components, reusableBox);
    }

    @Override
    public int getLeafPointIndex() {
        return tree.getPointIndex(currentNodeOffset);
    }

    public boolean isLeaf() {
        return tree.nodeStore.isLeaf(currentNodeOffset);
    }

    protected void setCurrentNode(int newNode, int index, boolean setBox) {
        currentNodeOffset = newNode;
        tree.pointStoreView.getNumericVectorInto(index, leafPoint);
        if (setBox && tree.boundingBoxCacheFraction < SWITCH_FRACTION) {
            // no side effect on tree
            pathBox.fromPoint(leafPoint);
            currentBox = pathBox;
        }
    }

    protected void setCurrentNodeOnly(int newNode) {
        currentNodeOffset = newNode;
    }

    public void updateToParent(int parent, int currentSibling, boolean updateBox) {
        currentNodeOffset = parent;
        if (updateBox && tree.boundingBoxCacheFraction == 0) {
            // bypasses tree to have no side effects in cache
            growBox(currentBox, currentSibling, nodeScratch, tree.pointStoreView);
        } else if (updateBox && tree.boundingBoxCacheFraction < SWITCH_FRACTION) {
            // will change state in tree
            tree.growArrayBox(currentBox, tree.pointStoreView, currentSibling);
        }
    }

    // this function exists for matching the behavior of RCF2.0 and will be replaced
    // this function explicitly uses the encoding of the new nodestore
    protected boolean toLeft(float[] point) {
        return point[tree.nodeStore.getCutDimension(currentNodeOffset)] <= tree.nodeStore
                .getCutValue(currentNodeOffset);
    }

    protected void growBox(ArrayBox box, int sibling, int[] scratch, IPointStoreView<float[]> pointStoreView) {
        int numLeaves = tree.addLeafPoints(scratch, sibling, 0);
        box.rangeSum = pointStoreView.addToSlice(box.values, box.offset, scratch, numLeaves);
    }
}
