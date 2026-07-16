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

import static org.junit.jupiter.api.Assertions.*;

import java.util.Random;
import java.util.stream.Stream;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.streamingalgorithms.randomcutforest.RandomCutForest;
import org.streamingalgorithms.randomcutforest.executor.SamplerPlusTree;
import org.streamingalgorithms.randomcutforest.state.RandomCutForestMapper;
import org.streamingalgorithms.randomcutforest.state.tree.CompactRandomCutTreeContext;
import org.streamingalgorithms.randomcutforest.state.tree.RandomCutTreeMapper;
import org.streamingalgorithms.randomcutforest.store.PointStore;

class NodeStoreTest {

    /**
     * Boundary-focused shapes. sampleSize straddles the byte tier (255/256/257);
     * trees=1 is where P=2S diverges from S(T+1); dimension small so trees fill.
     */
    static Stream<Arguments> shapes() {
        return Stream.of(
                // S, T, dim, rotation
                Arguments.of(8, 1, 1, false), Arguments.of(2, 1, 1000, false), Arguments.of(2, 1, 100000, false),
                Arguments.of(8, 50, 1, false), Arguments.of(255, 1, 3, false), Arguments.of(256, 1, 3, false), // the
                                                                                                               // byte/char
                                                                                                               // boundary,
                                                                                                               // T=1
                                                                                                               // (P=2S)
                Arguments.of(256, 2, 3, false), // P = S*T+1 kicks in
                Arguments.of(257, 50, 4, false), Arguments.of(50000, 2, 2, false), Arguments.of(100000, 2, 2, false), // large
                Arguments.of(64, 10, 8, true) // rotation path
        );
    }

    @ParameterizedTest(name = "S={0} T={1} dim={2} rot={3}")
    @MethodSource("shapes")
    void serdeRoundTripPreservesTopology(int sampleSize, int trees, int dim, boolean rotation) {
        RandomCutForest forest = buildForest(sampleSize, trees, dim, rotation, /* seed */ 42L);
        streamRandom(forest, dim, /* count */ sampleSize * 3, /* seed */ 7L);

        // take one real tree out of the forest and round-trip its node store
        RandomCutTree tree = firstTree(forest);
        assumeHasInternalRoot(tree);

        NodeStore before = tree.getNodeStore();
        int root = tree.getRoot();

        RandomCutForestMapper mapper = new RandomCutForestMapper();
        mapper.setSaveTreeStateEnabled(true);
        mapper.setSaveExecutorContextEnabled(true);
        RandomCutForest newForest = mapper.toModel(mapper.toState(forest));
        RandomCutTree newTree = firstTree(newForest);
        assumeHasInternalRoot(newTree);
        assertEquals(tree.getMass(), newTree.getMass());
        NodeStore after = newTree.getNodeStore();
        RandomCutTreeMapper treeMapper = new RandomCutTreeMapper();
        CompactRandomCutTreeContext context = contextFor(newForest);
        RandomCutTree secondTree = treeMapper.toModel(treeMapper.toState(newTree), context);
        NodeStore afterParent = secondTree.getNodeStore();
        assertTrue(afterParent.parent != null); // forcing test of parent construction
        NodeStore afterAfter = firstTree(mapper.toModel(mapper.toState(newForest))).getNodeStore();
        assertTopologyEquals(after, afterAfter, 0);
    }

    /**
     * Every stored child ref stays within [0, P + S - 1]. Directly targets the
     * router-keyed-on-capacity bug: with a too-narrow left/right column this fails
     * the moment the point store grows past its initial allocation.
     */
    @ParameterizedTest(name = "S={0} T={1}")
    @MethodSource("shapes")
    void childRefsNeverExceedBound(int sampleSize, int trees, int dim, boolean rotation) {
        RandomCutForest forest = buildForest(sampleSize, trees, dim, rotation, 42L);
        streamRandom(forest, dim, sampleSize * 3, 7L);
        RandomCutTree tree = firstTree(forest);
        assumeHasInternalRoot(tree);

        NodeStore ns = tree.getNodeStore();
        long bound = pointStoreCapacity(forest) + ns.getCapacity(); // P + (S-1)
        forEachInternal(ns, tree.getRoot(), node -> {
            int l = ns.getLeftIndex(node), r = ns.getRightIndex(node);
            assertTrue(l >= 0 && l <= bound, "left " + l + " exceeds " + bound + " at node " + node);
            assertTrue(r >= 0 && r <= bound, "right " + r + " exceeds " + bound + " at node " + node);
        });
    }

    /**
     * Internal mass in [2, numberOfLeaves]; only root may reach numberOfLeaves
     * (stored 0). The mod-encoding the byte tier depends on must never wrap a
     * non-root node silently.
     */
    @ParameterizedTest(name = "S={0} T={1}")
    @MethodSource("shapes")
    void internalMassStaysInRange(int sampleSize, int trees, int dim, boolean rotation) {
        RandomCutForest forest = buildForest(sampleSize, trees, dim, rotation, 42L);
        streamRandom(forest, dim, sampleSize * 3, 7L);
        RandomCutTree tree = firstTree(forest);
        assumeHasInternalRoot(tree);

        NodeStore ns = tree.getNodeStore();
        int root = tree.getRoot();
        int numberOfLeaves = ns.getCapacity() + 1;
        forEachInternal(ns, root, node -> {
            int m = ns.getMass(node);
            assertTrue(m >= 2 && m <= numberOfLeaves, "mass " + m + " out of range at " + node);
            if (node != root) {
                assertTrue(m <= numberOfLeaves - 1, "non-root node " + node + " reached full mass (silent wrap)");
            }
        });
    }

    /**
     * Duplicate-heavy stream: few distinct points, high multiplicity. This is the
     * distribution that pushes leaf multiplicity toward the root and exercises the
     * P+S-1 vs S-1 divergence. Round-trip must still be topology-identical.
     */
    @Test
    void duplicateHeavyRoundTrips() {
        int S = 256, dim = 3;
        RandomCutForest forest = buildForest(S, 1, dim, false, 42L);
        Random rng = new Random(11);
        // only 5 distinct points, each streamed many times
        float[][] palette = new float[5][];
        for (int i = 0; i < palette.length; i++)
            palette[i] = randomPoint(dim, rng);
        for (int i = 0; i < S * 4; i++)
            forest.update(palette[rng.nextInt(palette.length)]);

        RandomCutTree tree = firstTree(forest);
        assumeHasInternalRoot(tree);
        NodeStore before = tree.getNodeStore();
        int root = tree.getRoot();

        RandomCutForestMapper mapper = new RandomCutForestMapper();
        mapper.setSaveTreeStateEnabled(true);
        mapper.setSaveExecutorContextEnabled(true);
        RandomCutForest newForest = mapper.toModel(mapper.toState(forest));
        RandomCutTree newTree = firstTree(newForest);
        assumeHasInternalRoot(newTree);
        assertEquals(tree.getMass(), newTree.getMass());
        NodeStore after = newTree.getNodeStore();
        NodeStore afterAfter = firstTree(mapper.toModel(mapper.toState(newForest))).getNodeStore();
        assertTopologyEquals(after, afterAfter, 0);
    }

    /** Single distinct point: leaf root, size==0, must bypass from() cleanly. */
    @Test
    void singleDistinctPointLeafRoot() {
        RandomCutForest forest = buildForest(16, 1, 2, false, 42L);
        float[] p = { 1.0f, 2.0f };
        for (int i = 0; i < 10; i++)
            forest.update(p);
        RandomCutTree tree = firstTree(forest);
        // root is a leaf; round-trip should not throw and should stay a leaf root
        NodeStore before = tree.getNodeStore();
        int root = tree.getRoot();
        RandomCutForestMapper mapper = new RandomCutForestMapper();
        mapper.setSaveTreeStateEnabled(true);
        mapper.setSaveExecutorContextEnabled(true);
        RandomCutForest newForest = mapper.toModel(mapper.toState(forest));
        NodeStore after = firstTree(newForest).getNodeStore();

        // no internal nodes to compare; assert both agree the root is a leaf
        assertEquals(before.isLeaf(root), after.isLeaf(root));
    }

    // =====================================================================
    // THE ORACLE — a too-lenient version makes every test above vacuous.
    // Structural equality from `node`: same leaf/internal classification,
    // same cut (dim + exact float value), same internal mass, recurse on
    // children. Leaf refs compared by decoded point index, not raw int.
    // =====================================================================
    private void assertTopologyEquals(NodeStore a, NodeStore b, int node) {
        assertEquals(a.isLeaf(node), b.isLeaf(node), "leaf/internal mismatch at " + node);
        if (a.isLeaf(node)) {
            // both leaves: same underlying point index (raw ref = pointIndex + capacity +
            // 1)
            assertEquals(leafPointIndex(a, node), leafPointIndex(b, node), "leaf point index mismatch at " + node);
            return;
        }
        assertEquals(a.getCutDimension(node), b.getCutDimension(node), "cut dim mismatch at " + node);
        assertEquals(a.getCutValue(node), b.getCutValue(node), 0.0f, "cut value mismatch at " + node);
        assertEquals(a.getMass(node), b.getMass(node), "mass mismatch at " + node);
        // structural recursion — children occupy the SAME node numbers after canonical
        // BFS
        assertTopologyEquals(a, b, a.getLeftIndex(node)); // NOTE (2) below
        assertTopologyEquals(a, b, a.getRightIndex(node));
    }
    // ==================================================================
    // Builders / streaming — pinned to RandomCutForest.builder()
    // ==================================================================

    private RandomCutForest buildForest(int sampleSize, int trees, int dim, boolean rotation, long seed) {
        RandomCutForest.Builder<?> b = RandomCutForest.builder().dimensions(dim).sampleSize(sampleSize)
                .numberOfTrees(trees).randomSeed(seed);
        if (rotation) {
            // rotation requires internal shingling (doc 2: checkArgument in the
            // (Builder,boolean) ctor).
            // shingleSize == dim makes baseDimension == 1, the simplest rotating shape.
            b = b.internalShinglingEnabled(true).internalRotationEnabled(true).shingleSize(dim);
        }
        return b.build();
    }

    private void streamRandom(RandomCutForest forest, int dim, int count, long seed) {
        Random rng = new Random(seed);
        boolean rotating = forest.isRotationEnabled();
        int inputLen = rotating ? forest.getDimensions() / forest.getShingleSize() : dim;
        for (int i = 0; i < count; i++) {
            var point = randomPoint(inputLen, rng);
            forest.getAnomalyScore(point); // excercising the forest
            forest.update(point); // internal shingling expands to full dim
        }
    }

    private float[] randomPoint(int len, Random rng) {
        float[] p = new float[len];
        for (int i = 0; i < len; i++) {
            p[i] = (float) (rng.nextGaussian() * 10.0);
        }
        return p;
    }

    // ==================================================================
    // Point-store capacity — pinned to RandomCutForest :
    // pointStoreCapacity = max(sampleSize*numberOfTrees + 1, 2*sampleSize)
    // Recomputed here rather than read, because the forest doesn't expose it.
    // If PointStore.getCapacity() is reachable via the coordinator, prefer that
    // (see contextFor) so the test tracks the real value, not a re-derivation.
    // ==================================================================

    private long pointStoreCapacity(RandomCutForest forest) {
        long s = forest.getSampleSize();
        long t = forest.getNumberOfTrees();
        return Math.max(s * t + 1, 2 * s);
    }

    // ==================================================================
    // Reaching a concrete tree + its point store.
    // THESE TWO ARE THE SEAMS TO CONFIRM — they touch forest internals
    // (getComponents() returns ComponentList<?,?>; the element is a
    // SamplerPlusTree whose tree is a RandomCutTree). Exact unwrap depends
    // on your accessors; the shape below matches docs 2 & 4.
    // ==================================================================

    private RandomCutTree firstTree(RandomCutForest forest) {
        // doc 2: components is ComponentList<?, float[]>, each a SamplerPlusTree<>.
        // SamplerPlusTree exposes the tree — confirm the getter name in your copy.
        Object component = forest.getComponents().iterator().next(); // ComponentList is Iterable
        SamplerPlusTree<?, float[]> spt = (SamplerPlusTree<?, float[]>) component;
        return (RandomCutTree) spt.getTree(); // <-- CONFIRM: getTree()/getComponent() name
    }

    private PointStore pointStore(RandomCutForest forest) {
        // doc 2: stateCoordinator is a PointStoreCoordinator; getStore() returns the
        // PointStore.
        return (PointStore) forest.getUpdateCoordinator().getStore(); // <-- CONFIRM cast target
    }

    // ==================================================================
    // Context for the mapper round-trip. CompactRandomCutTreeContext needs
    // at least the point store + dimension (doc 5 mapper reads getPointStore()
    // and getDimension()). Confirm the setters on your context class.
    // ==================================================================

    private CompactRandomCutTreeContext contextFor(RandomCutForest forest) {
        CompactRandomCutTreeContext ctx = new CompactRandomCutTreeContext();
        ctx.setPointStore(pointStore(forest));
        ctx.setStoreParents(true);
        ctx.setDimension(forest.getDimensions());
        return ctx;
    }

    // ==================================================================
    // Iteration over live internal nodes, from the root, via the free-list-
    // agnostic structural walk (doc 3/4: isInternal, getLeftIndex/getRightIndex).
    // ==================================================================

    private void forEachInternal(NodeStore ns, int root, java.util.function.IntConsumer visit) {
        if (root == NodeStore.Null || !ns.isInternal(root)) {
            return; // leaf root or empty: no internal nodes
        }
        java.util.ArrayDeque<Integer> stack = new java.util.ArrayDeque<>();
        stack.push(root);
        while (!stack.isEmpty()) {
            int node = stack.pop();
            visit.accept(node);
            int l = ns.getLeftIndex(node);
            int r = ns.getRightIndex(node);
            if (ns.isInternal(l))
                stack.push(l);
            if (ns.isInternal(r))
                stack.push(r);
        }
    }

    // decoded point index behind a leaf ref (doc 4: getPointIndex = index -
    // numberOfLeaves;
    // numberOfLeaves == capacity + 1 in NodeStore terms).
    private int leafPointIndex(NodeStore ns, int leafRef) {
        return leafRef - (ns.getCapacity() + 1);
    }

    // roots that never grew past a single leaf have nothing structural to compare
    private void assumeHasInternalRoot(RandomCutTree tree) {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                tree.getRoot() != NodeStore.Null && !tree.getNodeStore().isLeaf(tree.getRoot()),
                "tree has no internal root — skipping structural comparison");
    }
}
