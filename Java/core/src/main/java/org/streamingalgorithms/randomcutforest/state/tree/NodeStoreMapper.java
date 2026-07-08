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

package org.streamingalgorithms.randomcutforest.state.tree;

import static org.streamingalgorithms.randomcutforest.tree.NodeStore.Null;
import static org.streamingalgorithms.randomcutforest.tree.NodeStore.nodeStore;
import static org.streamingalgorithms.randomcutforest.util.ArrayEncoder.bitEncode;
import static org.streamingalgorithms.randomcutforest.util.ArrayEncoder.moveAndPack;

import org.streamingalgorithms.randomcutforest.config.Precision;
import org.streamingalgorithms.randomcutforest.state.IContextualStateMapper;
import org.streamingalgorithms.randomcutforest.state.Version;
import org.streamingalgorithms.randomcutforest.state.store.NodeStoreState;
import org.streamingalgorithms.randomcutforest.tree.NodeStore;

import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
public class NodeStoreMapper implements IContextualStateMapper<NodeStore, NodeStoreState, CompactRandomCutTreeContext> {

    private int root;
    private boolean storeParent;

    @Override
    public NodeStore toModel(NodeStoreState state, CompactRandomCutTreeContext context, long seed) {
        int capacity = state.getCapacity();

        if (root == Null || root >= capacity) {
            return nodeStore(capacity, context.getDimension(), storeParent);
        }

        // PASS-THROUGH: every encoded array is forwarded packed; the concrete node
        // store unpacks each into its own retained (narrow) type and reflates the bits.
        // no unpacking, no reverseBits -- it only routes state -> store.
        // note size cannot be 0 if the root is nontrivial

        return NodeStore.from(capacity, 0, context.getDimension(), storeParent, state.getCutValueData(),
                state.getLeftIndex(), state.getRightIndex(), state.getCutDimension(), state.getSize(),
                state.isCompressed());
    }

    @Override
    public NodeStoreState toState(NodeStore model) {
        NodeStoreState state = new NodeStoreState();
        int capacity = model.getCapacity();
        state.setVersion(Version.V3_0);
        state.setCapacity(capacity);
        state.setCompressed(true);
        state.setPartialTreeStateEnabled(true);
        state.setPrecision(Precision.FLOAT_32.name());

        float[] cutValues = model.getCutValues();

        int[] map = new int[capacity];
        int size = model.reorderNodesInBreadthFirstOrder(map, root, capacity);
        // note that the point of this is that reorder is not inplace -- the
        // original nodestore is unaffected
        state.setSize(size); // (A) BFS count -- the value kept when check==false
        boolean check = root != Null && root < capacity;
        state.setCanonicalAndNotALeaf(check);
        if (check) {
            boolean compress = state.isCompressed();
            state.setLeftIndex(bitEncode(size, compress, m -> (model.getLeftIndex(map[m]) < capacity) ? 1 : 0));
            state.setRightIndex(bitEncode(size, compress, m -> (model.getRightIndex(map[m]) < capacity) ? 1 : 0));
            state.setCutDimension(moveAndPack(map, size, compress, model::getCutDimension));
            state.setCutValueData(moveAndPack(map, cutValues, size));
        }
        return state;
    }

}
