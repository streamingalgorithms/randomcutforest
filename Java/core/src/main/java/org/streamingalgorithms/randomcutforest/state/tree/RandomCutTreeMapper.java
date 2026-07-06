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

package org.streamingalgorithms.randomcutforest.state.tree;

import static org.streamingalgorithms.randomcutforest.tree.NodeStore.Null;

import org.streamingalgorithms.randomcutforest.state.IContextualStateMapper;
import org.streamingalgorithms.randomcutforest.state.Version;
import org.streamingalgorithms.randomcutforest.tree.NodeStore;
import org.streamingalgorithms.randomcutforest.tree.RandomCutTree;

import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
public class RandomCutTreeMapper
        implements IContextualStateMapper<RandomCutTree, CompactRandomCutTreeState, CompactRandomCutTreeContext> {

    @Override
    public RandomCutTree toModel(CompactRandomCutTreeState state, CompactRandomCutTreeContext context, long seed) {

        int dimension = (state.getDimensions() != 0) ? state.getDimensions() : context.getPointStore().getDimensions();
        context.setDimension(dimension);
        NodeStoreMapper nodeStoreMapper = new NodeStoreMapper();
        nodeStoreMapper.setRoot(state.getRoot());
        NodeStore nodeStore = nodeStoreMapper.toModel(state.getNodeStoreState(), context);

        // boundingBoxcache is set
        int newRoot = nodeStore.isLeaf(state.getRoot()) ? nodeStore.getCapacity() : state.getRoot();
        RandomCutTree tree = new RandomCutTree.Builder().dimension(dimension)
                .storeSequenceIndexesEnabled(state.isStoreSequenceIndexesEnabled()).capacity(state.getMaxSize())
                .setRoot(newRoot).randomSeed(state.getSeed()).pointStoreView(context.getPointStore())
                .nodeStore(nodeStore).centerOfMassEnabled(state.isCenterOfMassEnabled())
                .boundingBoxCacheFraction(state.getBoundingBoxCacheFraction()).outputAfter(state.getOutputAfter())
                .build();
        return tree;
    }

    @Override
    public CompactRandomCutTreeState toState(RandomCutTree model) {
        CompactRandomCutTreeState state = new CompactRandomCutTreeState();
        state.setVersion(Version.V3_0);
        int root = model.getRoot();
        NodeStoreMapper nodeStoreMapper = new NodeStoreMapper();
        nodeStoreMapper.setRoot(root);
        state.setNodeStoreState(nodeStoreMapper.toState(model.getNodeStore()));
        // the compression of nodeStore would change the root
        if ((root != Null) && (root < model.getNumberOfLeaves() - 1)) {
            root = 0; // reordering is forced
        }
        state.setRoot(root);
        state.setMaxSize(model.getNumberOfLeaves());
        state.setPartialTreeState(true);
        state.setStoreSequenceIndexesEnabled(model.isStoreSequenceIndexesEnabled());
        state.setCenterOfMassEnabled(model.isCenterOfMassEnabled());
        state.setBoundingBoxCacheFraction(model.getBoundingBoxCacheFraction());
        state.setOutputAfter(model.getOutputAfter());
        state.setSeed(model.getRandomSeed());
        state.setDimensions(model.getDimension());

        return state;
    }
}
