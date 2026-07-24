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
import static org.streamingalgorithms.randomcutforest.CommonUtils.checkState;

import java.util.Arrays;
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
    private float[] expanded;
    private float[] gapBuf; // armed at rearm, visitor-owned
    private final double[] fuseOut = new double[2];
    private int fusedNode = -1;
    private double fusedS, fusedRange;
    private boolean needsGap;

    public NodeView(int dimensions, int sampleSize) {
        leafPoint = new float[dimensions];
        reusableBox = new ArrayBox(dimensions);
        pathBox = new ArrayBox(dimensions);
        nodeScratch = new int[sampleSize];
        expanded = new float[2 * dimensions];
        gapBuf = new float[2 * dimensions];
    }

    public NodeView(RandomCutTree tree, IPointStoreView<float[]> pointStoreView, int root, float[] point) {
        this(pointStoreView.getDimensions(), (tree != null) ? tree.numberOfLeaves : 256);
        this.currentNodeOffset = root;
        this.tree = tree;
        set(point);
    }

    public void set(float[] point) {
        checkArgument(point != null && point.length == expanded.length / 2, "incorrect set in nodeview");
        VectorSupport.expandInto(point, 0, expanded, 0, expanded.length / 2);
    }

    public float[] expanded() {
        return expanded;
    }

    // NodeView
    protected void rearm(RandomCutTree tree, int root, boolean needsGap) {
        this.tree = tree;
        this.currentNodeOffset = root;
        this.currentBox = null;
        if (nodeScratch == null || nodeScratch.length < tree.numberOfLeaves) {
            nodeScratch = new int[tree.numberOfLeaves];
        }
        fusedNode = -1;
        fusedRange = 0;
        fusedS = 0;
        Arrays.fill(fuseOut, 0);
        Arrays.fill(gapBuf, 0);
        this.needsGap = needsGap;
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

    public IBoundingBoxView getSiblingBoundingBox() {
        return getSiblingBoundingBox(leafPoint);
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
    public double probabilityAndSeparation(float[] expandedInputPoint, float[] components) {
        return tree.probabilityOfCutExpanded(currentNodeOffset, expandedInputPoint, currentBox, components, reusableBox,
                null);
    }

    @Override
    public double probabilityAndSeparation(ArrayBox box, float[] components) {
        return tree.probabilityOfCutExpanded(currentNodeOffset, expanded, box, components, reusableBox, null);
    }

    @Override
    public double probabilityAndSeparation(float[] components) {
        if (fusedNode == currentNodeOffset && fusedNode != -1) {
            if (components != null) {
                checkArgument(components.length == gapBuf.length && needsGap, "incorrect call");
                System.arraycopy(gapBuf, 0, components, 0, gapBuf.length);
            }
            return (fusedS == 0.0) ? 0.0 : (fusedRange == 0.0 ? 1.0 : fusedS / (fusedS + fusedRange));
        }
        return probabilityAndSeparation(expanded, components);
    }

    @Override
    public float[] separation(double[] ranges) {
        if (fusedNode == currentNodeOffset && fusedNode != -1) {
            ranges[0] = fusedS;
            ranges[1] = fusedRange;
            return gapBuf;
        }
        tree.probabilityOfCutExpanded(currentNodeOffset, expanded, currentBox, gapBuf, reusableBox, ranges);
        return gapBuf;
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
            int numLeaves = tree.addLeafPoints(nodeScratch, currentSibling, 0);
            fusedS = tree.pointStoreView.addToSliceWithGap(currentBox.values, currentBox.offset, nodeScratch, 0,
                    numLeaves, expanded, 0, (needsGap) ? gapBuf : null, 0, fuseOut);
            fusedRange = fuseOut[0];
            currentBox.rangeSum = fusedRange;
            fusedNode = parent;

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

}
