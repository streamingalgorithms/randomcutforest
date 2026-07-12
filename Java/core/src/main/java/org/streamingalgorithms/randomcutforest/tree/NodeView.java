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

    // currentBox is a *sentinel reference*, not an owned object:
    // null -> hot cache (>= SWITCH_FRACTION): boxes come from cached slices via
    // reusableBox
    // !null -> cold cache (< SWITCH_FRACTION): points at pathBox, the accumulated
    // path box
    // Demoting it from an allocation to a sentinel is the whole GC hack; every
    // downstream check (getBoundingBox, probabilityOfSeparation, updateToParent)
    // is unchanged because they only ever test null / hand the reference onward.
    ArrayBox currentBox;

    // fill-and-discard scratch for sibling / reconstructed boxes (hot + cold)
    private final ArrayBox reusableBox;
    // reset-and-grow scratch for the accumulated path box (cold cache only).
    // MUST be distinct from reusableBox: getSiblingBoundingBox fills reusableBox
    // while getBoundingBox returns currentBox(==pathBox); a cold-cache visitor can
    // hold both live, so aliasing them would clobber.
    private final ArrayBox pathBox;

    public NodeView(RandomCutTree tree, IPointStoreView<float[]> pointStoreView, int root) {
        this.currentNodeOffset = root;
        this.tree = tree;
        int dimensions = pointStoreView.getDimensions();
        leafPoint = new float[dimensions];
        reusableBox = new ArrayBox(dimensions);
        pathBox = new ArrayBox(dimensions);
    }

    // NodeView
    protected void rearm(RandomCutTree tree, int root) {
        this.tree = tree;
        this.currentNodeOffset = root;
        this.currentBox = null; // deactivate; pathBox array retained, re-primed by setCurrentNode
        // leafPoint, reusableBox, pathBox arrays kept (same dimension across the
        // forest); refilled per node
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

    @Override
    public double probabilityOfSeparation(float[] point) {
        return tree.probabilityOfCut(currentNodeOffset, point, currentBox, null);
    }

    @Override
    public double probabilityOfSeparation(float[] point, float[] components) {
        return tree.probabilityOfCut(currentNodeOffset, point, currentBox, components);
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
            // was: currentBox = new ArrayBox(leafPoint); // per-tree alloc (~42.5KB/op at
            // cache=0)
            // now: reset the owned scratch to the single leaf point and activate the
            // sentinel.
            // fromPoint MUST fully reset (values + rangeSum + any flags) to match the
            // old constructor's state, else tree N carries tree N-1's grown box.
            pathBox.fromPoint(leafPoint);
            currentBox = pathBox;
        }
    }

    protected void setCurrentNodeOnly(int newNode) {
        currentNodeOffset = newNode;
    }

    public void updateToParent(int parent, int currentSibling, boolean updateBox) {
        currentNodeOffset = parent;
        if (updateBox && tree.boundingBoxCacheFraction < SWITCH_FRACTION) {
            tree.growArrayBox(currentBox, tree.pointStoreView, parent, currentSibling); // in-place, currentBox==pathBox
        }
    }

    // this function exists for matching the behavior of RCF2.0 and will be replaced
    // this function explicitly uses the encoding of the new nodestore
    protected boolean toLeft(float[] point) {
        return point[tree.nodeStore.getCutDimension(currentNodeOffset)] <= tree.nodeStore
                .getCutValue(currentNodeOffset);
    }
}
