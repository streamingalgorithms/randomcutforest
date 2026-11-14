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
import java.util.concurrent.Callable;
import java.util.concurrent.ForkJoinPool;
import java.util.function.BiConsumer;
import java.util.function.BinaryOperator;
import java.util.function.Function;
import java.util.stream.Collector;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import org.streamingalgorithms.randomcutforest.ComponentList;
import org.streamingalgorithms.randomcutforest.IMultiVisitorFactory;
import org.streamingalgorithms.randomcutforest.IRFVisitor;
import org.streamingalgorithms.randomcutforest.IVisitorFactory;
import org.streamingalgorithms.randomcutforest.config.Config;
import org.streamingalgorithms.randomcutforest.returntypes.ConvergingAccumulator;
import org.streamingalgorithms.randomcutforest.tree.NodeView;

/**
 * An implementation of forest traversal methods that uses a private thread pool
 * to visit trees in parallel.
 *
 * <p>
 * A reusable visitor factory is served by slices: the trees are dealt into at
 * most threadPoolSize contiguous slices, and each slice walks its trees with
 * one visitor and one node view, as the sequential executor walks the whole
 * forest. A query then allocates per thread and not per tree. Unless multi-read
 * is set, a slice keeps its visitor across queries when the factory allows it
 * (isReusableAcrossQueries), under the rule of the sequential executor; with
 * multi-read, queries may overlap, so each query gets its own. A factory that
 * is not reusable takes the stream over the trees, as before.
 */
public class ParallelForestTraversalExecutor extends AbstractForestTraversalExecutor {

    ForkJoinPool forkJoinPool;
    private final int threadPoolSize;
    private boolean multiRead;
    private Slice[] slices;

    /** Trees [from, to), with what one thread keeps between its queries. */
    private static final class Slice {
        final int from, to;
        NodeView view;
        final IVisitorFactory<?>[] slotKey = new IVisitorFactory<?>[4];
        final IRFVisitor<?>[] slotVal = new IRFVisitor<?>[4];

        Slice(int from, int to) {
            this.from = from;
            this.to = to;
        }

        @SuppressWarnings("unchecked")
        <R> IRFVisitor<R> slotFor(IVisitorFactory<R> vf, float[] point, boolean keep) {
            if (!keep || !vf.isReusableAcrossQueries()) {
                return vf.newReusableVisitor(point);
            }
            for (int i = 0; i < slotKey.length; i++) {
                if (slotKey[i] == vf) {
                    IRFVisitor<R> v = (IRFVisitor<R>) slotVal[i];
                    v.resetAcrossQueries(point);
                    return v;
                }
                if (slotKey[i] == null) {
                    slotKey[i] = vf;
                    IRFVisitor<R> v = vf.newReusableVisitor(point);
                    v.resetAcrossQueries(point);
                    slotVal[i] = v;
                    return v;
                }
            }
            return vf.newReusableVisitor(point);
        }

        NodeView viewFor(float[] point, boolean keep) {
            NodeView v = keep ? view : null;
            if (v == null) {
                v = new NodeView(point.length, 256); // the scratch grows to each tree at rearm
                if (keep) {
                    view = v;
                }
            }
            v.set(point);
            return v;
        }
    }

    private BatchState[] batchStates;

    /**
     * The batch of the base class with the trees shared out: each thread walks its
     * slice of trees for every point, with its own visitor, view and box loan, and
     * the slices' results are combined per point in the order of the trees. A
     * thread then has a whole batch of work per hand-over, not one tree of one
     * query. In multi-read the batch is served point by point.
     */
    @Override
    public <R, S> List<S> traverseForest(float[][] points, IVisitorFactory<R> visitorFactory,
            BinaryOperator<R> accumulator, Function<R, S> finisher) {
        if (points.length == 0 || !visitorFactory.isReusable() || components.isEmpty() || multiRead) {
            List<S> out = new ArrayList<>(points.length);
            for (float[] point : points) {
                out.add(traverseForest(point, visitorFactory, accumulator, finisher));
            }
            return out;
        }
        Slice[] all = slices();
        synchronized (this) {
            if (batchStates == null || batchStates.length != all.length) {
                batchStates = new BatchState[all.length];
                for (int i = 0; i < all.length; i++) {
                    batchStates[i] = new BatchState();
                }
            }
        }
        BatchState[] states = batchStates;
        Object[][] part = new Object[all.length][points.length];
        @SuppressWarnings("unchecked")
        IRFVisitor<R>[][] folders = new IRFVisitor[all.length][];
        if (all.length == 1) {
            folders[0] = batchSlice(all[0].from, all[0].to, points, visitorFactory, accumulator, part[0], states[0]);
        } else {
            submitAndJoin(() -> {
                IntStream.range(0, all.length).parallel().forEach(i -> folders[i] = batchSlice(all[i].from, all[i].to,
                        points, visitorFactory, accumulator, part[i], states[i]));
                return null;
            });
        }
        Object[] acc = part[0];
        for (int i = 0; i < points.length; i++) {
            if (folders[0] != null) {
                acc[i] = joinFolds(visitorFactory, accumulator, folders, i);
                continue;
            }
            for (int s = 1; s < all.length; s++) {
                @SuppressWarnings("unchecked")
                R joined = accumulator.apply((R) acc[i], (R) part[s][i]);
                acc[i] = joined;
            }
        }
        return finish(acc, finisher, new ArrayList<>(points.length));
    }

    /**
     * The folds of the slices for point i as one result: added in place into the
     * first slice's visitor where the visitor can absorb, so that one result is
     * built; otherwise each slice's result, joined by the accumulator.
     */
    private <R> R joinFolds(IVisitorFactory<R> vf, BinaryOperator<R> accumulator, IRFVisitor<R>[][] folders, int i) {
        IRFVisitor<R> first = folders[0][i];
        R acc = null;
        for (int s = 1; s < folders.length; s++) {
            if (acc == null && first.absorb(folders[s][i])) {
                continue;
            }
            if (acc == null) { // cannot absorb: the first slice's result, then the rest
                acc = vf.liftResult(null, first.getFoldResult());
            }
            acc = accumulator.apply(acc, vf.liftResult(null, folders[s][i].getFoldResult()));
        }
        return (acc != null) ? acc : vf.liftResult(null, first.getFoldResult());
    }

    private synchronized Slice[] slices() {
        int n = components.size();
        if (slices == null || slices[slices.length - 1].to != n) {
            int count = Math.max(1, Math.min(threadPoolSize, n));
            Slice[] fresh = new Slice[count];
            for (int i = 0; i < count; i++) {
                fresh[i] = new Slice((int) ((long) i * n / count), (int) ((long) (i + 1) * n / count));
            }
            slices = fresh;
        }
        return slices;
    }

    @Override
    public <R> boolean foldInto(float[] point, IVisitorFactory<R> vf, double[] out) {
        if (!vf.isReusable() || !vf.isFoldable() || components.isEmpty()) {
            return false;
        }
        Slice[] all = slices();
        if (all.length == 1) {
            return foldSlice(all[0], point, vf).copyFold(out);
        }
        List<IRFVisitor<R>> folds = submitAndJoin(() -> IntStream.range(0, all.length).parallel()
                .mapToObj(i -> foldSlice(all[i], point, vf)).collect(Collectors.toList()));
        IRFVisitor<R> first = folds.get(0);
        for (int s = 1; s < folds.size(); s++) {
            if (!first.absorb(folds.get(s))) {
                return false;
            }
        }
        return first.copyFold(out);
    }

    /** One slice folded, its visitor returned with the fold still in it. */
    private <R> IRFVisitor<R> foldSlice(Slice slice, float[] point, IVisitorFactory<R> vf) {
        boolean keep = !multiRead;
        IRFVisitor<R> slot = slice.slotFor(vf, point, keep);
        NodeView tower = slice.viewFor(point, keep);
        for (int t = slice.from; t < slice.to; t++) {
            tower = components.get(t).reusableTraverse(point, slot, tower);
            slot.foldOut();
        }
        if (keep) {
            slice.view = tower;
        }
        return slot;
    }

    /** One slice folded, or reduced tree by tree when the factory does not fold. */
    private <R> R reduceSlice(Slice slice, float[] point, IVisitorFactory<R> vf, BinaryOperator<R> accumulator) {
        boolean keep = !multiRead;
        IRFVisitor<R> slot = slice.slotFor(vf, point, keep);
        NodeView tower = slice.viewFor(point, keep);
        R acc = null;
        boolean fold = vf.isFoldable();
        for (int t = slice.from; t < slice.to; t++) {
            tower = components.get(t).reusableTraverse(point, slot, tower);
            if (fold) {
                slot.foldOut();
            } else {
                R r = vf.liftResult(null, slot.getResult()); // detached per tree
                acc = (acc == null) ? r : accumulator.apply(acc, r);
            }
        }
        if (keep) {
            slice.view = tower;
        }
        return fold ? vf.liftResult(null, slot.getFoldResult()) : acc;
    }

    private <R> List<R> collectSlice(Slice slice, float[] point, IVisitorFactory<R> vf) {
        boolean keep = !multiRead;
        IRFVisitor<R> slot = slice.slotFor(vf, point, keep);
        NodeView tower = slice.viewFor(point, keep);
        List<R> out = new ArrayList<>(slice.to - slice.from);
        for (int t = slice.from; t < slice.to; t++) {
            tower = components.get(t).reusableTraverse(point, slot, tower);
            out.add(vf.liftResult(null, slot.getResult())); // detached per tree
        }
        if (keep) {
            slice.view = tower;
        }
        return out;
    }

    public ParallelForestTraversalExecutor(ComponentList<?, ?> treeExecutors, int threadPoolSize) {
        super(treeExecutors);
        this.threadPoolSize = threadPoolSize;
        forkJoinPool = new ForkJoinPool(threadPoolSize);
    }

    @Override
    public <R, S> S traverseForest(float[] point, IVisitorFactory<R> visitorFactory, BinaryOperator<R> accumulator,
            Function<R, S> finisher) {

        if (visitorFactory.isReusable() && !components.isEmpty()) {
            Slice[] all = slices();
            if (accumulator == null) { // foldable, by the forest's check: nothing combines slices, so one fold
                return finisher.apply(reduceSlice(new Slice(0, components.size()), point, visitorFactory, null));
            }
            if (visitorFactory.isFoldable() && all.length > 1) { // join the folds, read one result
                List<IRFVisitor<R>> folds = submitAndJoin(() -> IntStream.range(0, all.length).parallel()
                        .mapToObj(i -> foldSlice(all[i], point, visitorFactory)).collect(Collectors.toList()));
                @SuppressWarnings("unchecked")
                IRFVisitor<R>[][] folders = new IRFVisitor[all.length][];
                for (int s = 0; s < all.length; s++) {
                    @SuppressWarnings("unchecked")
                    IRFVisitor<R>[] one = new IRFVisitor[] { folds.get(s) };
                    folders[s] = one;
                }
                return finisher.apply(joinFolds(visitorFactory, accumulator, folders, 0));
            }
            List<R> parts = (all.length == 1)
                    ? java.util.Collections.singletonList(reduceSlice(all[0], point, visitorFactory, accumulator))
                    : submitAndJoin(() -> IntStream.range(0, all.length).parallel()
                            .mapToObj(i -> reduceSlice(all[i], point, visitorFactory, accumulator))
                            .collect(Collectors.toList()));
            R acc = parts.get(0);
            for (int i = 1; i < parts.size(); i++) {
                acc = accumulator.apply(acc, parts.get(i));
            }
            return finisher.apply(acc);
        }
        return submitAndJoin(() -> components.parallelStream().map(c -> c.traverse(point, visitorFactory))
                .reduce(accumulator).map(finisher))
                .orElseThrow(() -> new IllegalStateException("accumulator returned an empty result"));
    }

    @Override
    public <R, S> S traverseForest(float[] point, IVisitorFactory<R> visitorFactory, Collector<R, ?, S> collector) {

        if (visitorFactory.isReusable() && !components.isEmpty()) {
            return collectSlices(point, visitorFactory, collector);
        }
        return submitAndJoin(
                () -> components.parallelStream().map(c -> c.traverse(point, visitorFactory)).collect(collector));
    }

    private <R, A, S> S collectSlices(float[] point, IVisitorFactory<R> visitorFactory, Collector<R, A, S> collector) {
        Slice[] all = slices();
        List<List<R>> parts = submitAndJoin(() -> IntStream.range(0, all.length).parallel()
                .mapToObj(i -> collectSlice(all[i], point, visitorFactory)).collect(Collectors.toList()));
        A container = collector.supplier().get();
        BiConsumer<A, R> add = collector.accumulator();
        for (List<R> part : parts) { // in the order of the trees
            for (R r : part) {
                add.accept(container, r);
            }
        }
        return collector.finisher().apply(container);
    }

    @Override
    public <R, S> S traverseForest(float[] point, IVisitorFactory<R> visitorFactory,
            ConvergingAccumulator<R> accumulator, Function<R, S> finisher) {

        for (int i = 0; i < components.size(); i += threadPoolSize) {
            final int start = i;
            final int end = Math.min(start + threadPoolSize, components.size());

            List<R> results = submitAndJoin(() -> components.subList(start, end).parallelStream()
                    .map(c -> c.traverse(point, visitorFactory)).collect(Collectors.toList()));
            results.forEach(accumulator::accept);

            if (accumulator.isConverged()) {
                break;
            }
        }

        return finisher.apply(accumulator.getAccumulatedValue());
    }

    @Override
    public <R, S> S traverseForestMulti(float[] point, IMultiVisitorFactory<R> visitorFactory,
            BinaryOperator<R> accumulator, Function<R, S> finisher) {

        return submitAndJoin(() -> components.parallelStream().map(c -> c.traverseMulti(point, visitorFactory))
                .reduce(accumulator).map(finisher))
                .orElseThrow(() -> new IllegalStateException("accumulator returned an empty result"));
    }

    @Override
    public <R, S> S traverseForestMulti(float[] point, IMultiVisitorFactory<R> visitorFactory,
            Collector<R, ?, S> collector) {

        return submitAndJoin(
                () -> components.parallelStream().map(c -> c.traverseMulti(point, visitorFactory)).collect(collector));
    }

    @Override
    public void setMultiRead(boolean multiRead) {
        this.multiRead = multiRead;
        components.forEach(x -> x.setConfig(Config.MULTI_READ, multiRead));
    }

    <T> T submitAndJoin(Callable<T> callable) {
        if (forkJoinPool == null) {
            forkJoinPool = new ForkJoinPool(threadPoolSize);
        }
        return forkJoinPool.submit(callable).join();
    }
}
