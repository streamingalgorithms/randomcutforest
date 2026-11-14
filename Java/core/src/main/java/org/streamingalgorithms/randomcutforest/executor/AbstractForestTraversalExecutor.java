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
 *
 * The file is substantially modified from its prior version which had the
 * following notice.
 *
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

package org.streamingalgorithms.randomcutforest.executor;

import java.util.ArrayList;
import java.util.List;
import java.util.function.BinaryOperator;
import java.util.function.Function;
import java.util.stream.Collector;

import org.streamingalgorithms.randomcutforest.ComponentList;
import org.streamingalgorithms.randomcutforest.IComponentModel;
import org.streamingalgorithms.randomcutforest.IMultiVisitorFactory;
import org.streamingalgorithms.randomcutforest.IRFVisitor;
import org.streamingalgorithms.randomcutforest.IVisitorFactory;
import org.streamingalgorithms.randomcutforest.returntypes.ConvergingAccumulator;
import org.streamingalgorithms.randomcutforest.tree.ITree;
import org.streamingalgorithms.randomcutforest.tree.NodeView;
import org.streamingalgorithms.randomcutforest.tree.RandomCutTree;

public abstract class AbstractForestTraversalExecutor {

    protected final ComponentList<?, ?> components;

    protected AbstractForestTraversalExecutor(ComponentList<?, ?> components) {
        this.components = components;
    }

    /**
     * Visit each of the trees in the forest and combine the individual results into
     * an aggregate result. A visitor is constructed for each tree using the visitor
     * factory, and then submitted to a tree. The results from all the trees are
     * combined using the accumulator and then transformed using the finisher before
     * being returned.
     *
     * @param point          The point that defines the traversal path.
     * @param visitorFactory A factory method which is invoked for each tree to
     *                       construct a visitor.
     * @param accumulator    A function that combines the results from individual
     *                       trees into an aggregate result.
     * @param finisher       A function called on the aggregate result in order to
     *                       produce the final result.
     * @param <R>            The visitor result type. This is the type that will be
     *                       returned after traversing each individual tree.
     * @param <S>            The final type, after any final normalization at the
     *                       forest level.
     * @return The aggregated and finalized result after sending a visitor through
     *         each tree in the forest.
     */
    public abstract <R, S> S traverseForest(float[] point, IVisitorFactory<R> visitorFactory,
            BinaryOperator<R> accumulator, Function<R, S> finisher);

    /**
     * The same visitor for a batch of points, against a forest that is not updated
     * meanwhile: result i is what the single-point method gives for points[i], up
     * to the order of the floating-point sums. The order of the work is turned
     * around: each tree is walked for every point before the next tree is touched,
     * with one visitor and one node view, so a tree's nodes stay in the processor's
     * cache and a tree that does not keep every box borrows a full box cache for
     * its turn ({@link RandomCutTree#lendCache}), building each box once for the
     * whole batch. This implementation walks the trees on the calling thread; the
     * parallel executor gives each thread a share of the trees.
     *
     * A factory that is not reusable is served point by point. A factory that folds
     * gets one visitor per point of the batch, kept between batches when the
     * factory allows, and each adds the trees in as they are read; otherwise one
     * visitor serves and each tree's detached result is combined by the
     * accumulator. The accumulator must not be null: it also joins the shares of
     * the threads.
     */
    public <R, S> List<S> traverseForest(float[][] points, IVisitorFactory<R> visitorFactory,
            BinaryOperator<R> accumulator, Function<R, S> finisher) {
        List<S> out = new ArrayList<>(points.length);
        if (points.length == 0) {
            return out;
        }
        if (!visitorFactory.isReusable() || components.isEmpty()) {
            for (float[] point : points) {
                out.add(traverseForest(point, visitorFactory, accumulator, finisher));
            }
            return out;
        }
        if (batchState == null) {
            batchState = new BatchState();
        }
        Object[] acc = new Object[points.length];
        IRFVisitor<R>[] folders = batchSlice(0, components.size(), points, visitorFactory, accumulator, acc,
                batchState);
        if (folders != null) {
            for (int i = 0; i < points.length; i++) {
                acc[i] = visitorFactory.liftResult(null, folders[i].getFoldResult());
            }
        }
        return finish(acc, finisher, out);
    }

    /**
     * Folds the forest for one point and copies the fold into out, building no
     * result: see {@link IRFVisitor#copyFold}. Returns false, with out undefined,
     * when the factory does not fold or its visitor has no array to copy; the
     * caller then takes the result the usual way.
     */
    public <R> boolean foldInto(float[] point, IVisitorFactory<R> visitorFactory, double[] out) {
        return false;
    }

    /** What one thread keeps between its batches. */
    protected static final class BatchState {
        NodeView view;
        final RandomCutTree.BoxLoan loan = new RandomCutTree.BoxLoan();
        // for a factory that folds and may be kept: a visitor per point of the batch
        IVisitorFactory<?> foldKey;
        IRFVisitor<?>[] fold = new IRFVisitor<?>[0];

        @SuppressWarnings("unchecked")
        <R> IRFVisitor<R>[] folders(IVisitorFactory<R> vf, float[][] points) {
            boolean keep = vf.isReusableAcrossQueries();
            IRFVisitor<?>[] v = (keep && foldKey == vf) ? fold : new IRFVisitor<?>[0];
            if (v.length < points.length) {
                v = java.util.Arrays.copyOf(v, points.length);
            }
            for (int i = 0; i < points.length; i++) {
                if (v[i] == null) {
                    v[i] = vf.newReusableVisitor(points[i]);
                }
                if (keep) {
                    v[i].resetAcrossQueries(points[i]);
                }
            }
            if (keep) {
                foldKey = vf;
                fold = v;
            }
            return (IRFVisitor<R>[]) v;
        }
    }

    private BatchState batchState;

    @SuppressWarnings("unchecked")
    protected static <R, S> List<S> finish(Object[] acc, Function<R, S> finisher, List<S> out) {
        for (Object a : acc) {
            out.add(finisher.apply((R) a));
        }
        return out;
    }

    /**
     * Trees [from, to), each walked for every point. For a factory that does not
     * fold, acc[i] takes the trees' results for points[i], combined in the order of
     * the trees, and null is returned. For one that folds, the visitors are
     * returned, one per point, each holding its fold of these trees: the caller
     * joins the folds of its slices and reads each result once.
     */
    @SuppressWarnings("unchecked")
    protected <R> IRFVisitor<R>[] batchSlice(int from, int to, float[][] points, IVisitorFactory<R> visitorFactory,
            BinaryOperator<R> accumulator, Object[] acc, BatchState state) {
        // a factory that folds gets a visitor per point, which adds each tree in as it
        // is read; otherwise one visitor serves, and each tree's result is detached
        boolean folding = visitorFactory.isFoldable();
        IRFVisitor<R>[] folders = folding ? state.folders(visitorFactory, points) : null;
        IRFVisitor<R> visitor = folding ? null : visitorFactory.newReusableVisitor(points[0]);
        NodeView tower = state.view;
        if (tower == null) {
            tower = new NodeView(points[0].length, 256); // the scratch grows to each tree at rearm
        }
        for (int t = from; t < to; t++) {
            IComponentModel<?, ?> component = components.get(t);
            RandomCutTree lent = null;
            if (component instanceof SamplerPlusTree) {
                ITree<?, ?> tree = ((SamplerPlusTree<?, ?>) component).getTree();
                if (tree instanceof RandomCutTree && ((RandomCutTree) tree).lendCache(state.loan)) {
                    lent = (RandomCutTree) tree;
                }
            }
            try {
                for (int i = 0; i < points.length; i++) {
                    tower.set(points[i]);
                    if (folding) {
                        tower = component.reusableTraverse(points[i], folders[i], tower);
                        folders[i].foldOut();
                        continue;
                    }
                    tower = component.reusableTraverse(points[i], visitor, tower);
                    R r = visitorFactory.liftResult(null, visitor.getResult()); // detached per tree
                    acc[i] = (acc[i] == null) ? r : accumulator.apply((R) acc[i], r);
                }
            } finally {
                if (lent != null) {
                    lent.returnCache(state.loan);
                }
            }
        }
        state.view = tower;
        return folders;
    }

    /**
     * Visit each of the trees in the forest and combine the individual results into
     * an aggregate result. A visitor is constructed for each tree using the visitor
     * factory, and then submitted to each tree. The results from individual trees
     * are collected using the {@link java.util.stream.Collector} and returned.
     * Trees are visited in parallel using
     * {@link java.util.Collection#parallelStream()}.
     *
     * @param point          The point that defines the traversal path.
     * @param visitorFactory A factory method which is invoked for each tree to
     *                       construct a visitor.
     * @param collector      A collector used to aggregate individual tree results
     *                       into a final result.
     * @param <R>            The visitor result type. This is the type that will be
     *                       returned after traversing each individual tree.
     * @param <S>            The final type, after any final normalization at the
     *                       forest level.
     * @return The aggregated and finalized result after sending a visitor through
     *         each tree in the forest.
     */
    public abstract <R, S> S traverseForest(float[] point, IVisitorFactory<R> visitorFactory,
            Collector<R, ?, S> collector);

    /**
     * Visit each of the trees in the forest sequentially and combine the individual
     * results into an aggregate result. A visitor is constructed for each tree
     * using the visitor factory, and then submitted to each tree. The results from
     * all the trees are combined using the {@link ConvergingAccumulator}, and the
     * method stops visiting trees after convergence is reached. The result is
     * transformed using the finisher before being returned.
     *
     * @param point          The point that defines the traversal path.
     * @param visitorFactory A factory method which is invoked for each tree to
     *                       construct a visitor.
     * @param accumulator    An accumulator that combines the results from
     *                       individual trees into an aggregate result and checks to
     *                       see if the result can be returned without further
     *                       processing.
     * @param finisher       A function called on the aggregate result in order to
     *                       produce the final result.
     * @param <R>            The visitor result type. This is the type that will be
     *                       returned after traversing each individual tree.
     * @param <S>            The final type, after any final normalization at the
     *                       forest level.
     * @return The aggregated and finalized result after sending a visitor through
     *         each tree in the forest.
     */
    public abstract <R, S> S traverseForest(float[] point, IVisitorFactory<R> visitorFactory,
            ConvergingAccumulator<R> accumulator, Function<R, S> finisher);

    /**
     * Visit each of the trees in the forest and combine the individual results into
     * an aggregate result. A multi-visitor is constructed for each tree using the
     * visitor factory, and then submitted to a tree. The results from all the trees
     * are combined using the accumulator and then transformed using the finisher
     * before being returned.
     *
     * @param point          The point that defines the traversal path.
     * @param visitorFactory A factory method which is invoked for each tree to
     *                       construct a multi-visitor.
     * @param accumulator    A function that combines the results from individual
     *                       trees into an aggregate result.
     * @param finisher       A function called on the aggregate result in order to
     *                       produce the final result.
     * @param <R>            The visitor result type. This is the type that will be
     *                       returned after traversing each individual tree.
     * @param <S>            The final type, after any final normalization at the
     *                       forest level.
     * @return The aggregated and finalized result after sending a visitor through
     *         each tree in the forest.
     */
    public abstract <R, S> S traverseForestMulti(float[] point, IMultiVisitorFactory<R> visitorFactory,
            BinaryOperator<R> accumulator, Function<R, S> finisher);

    /**
     * Visit each of the trees in the forest and combine the individual results into
     * an aggregate result. A multi-visitor is constructed for each tree using the
     * visitor factory, and then submitted to a tree. The results from individual
     * trees are collected using the {@link java.util.stream.Collector} and
     * returned. Trees are visited in parallel using
     * {@link java.util.Collection#parallelStream()}.
     *
     * @param point          The point that defines the traversal path.
     * @param visitorFactory A factory method which is invoked for each tree to
     *                       construct a visitor.
     * @param collector      A collector used to aggregate individual tree results
     *                       into a final result.
     * @param <R>            The visitor result type. This is the type that will be
     *                       returned after traversing each individual tree.
     * @param <S>            The final type, after any final normalization at the
     *                       forest level.
     * @return The aggregated and finalized result after sending a visitor through
     *         each tree in the forest.
     */
    public abstract <R, S> S traverseForestMulti(float[] point, IMultiVisitorFactory<R> visitorFactory,
            Collector<R, ?, S> collector);

    public abstract void setMultiRead(boolean multiRead);
}
