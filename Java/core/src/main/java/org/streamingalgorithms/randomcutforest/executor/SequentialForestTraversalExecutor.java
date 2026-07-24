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

import static org.streamingalgorithms.randomcutforest.CommonUtils.checkArgument;

import java.util.function.BiConsumer;
import java.util.function.BinaryOperator;
import java.util.function.Function;
import java.util.stream.Collector;

import org.streamingalgorithms.randomcutforest.*;
import org.streamingalgorithms.randomcutforest.returntypes.ConvergingAccumulator;
import org.streamingalgorithms.randomcutforest.tree.NodeView;

public class SequentialForestTraversalExecutor extends AbstractForestTraversalExecutor {

    private final NodeView resusableView;
    private boolean multiRead;
    private final int dimensions; // ctor already takes it; just retain it
    private final IVisitorFactory<?>[] slotKey = new IVisitorFactory<?>[4];
    private final IRFVisitor<?>[] slotVal = new IRFVisitor<?>[4];

    public SequentialForestTraversalExecutor(ComponentList<?, ?> components, int dimensions, int sampleSize) {
        super(components);
        this.dimensions = dimensions;
        resusableView = new NodeView(dimensions, sampleSize);
    }

    private <R, S> S foldForest(float[] point, IVisitorFactory<R> vf, ConvergingAccumulator<R> conv,
            Function<R, S> finisher) {
        /// slotFor(vf, point) can be used someday. TODAY!!
        IRFVisitor<R> slot = slotFor(vf, point); // vf.newReusableVisitor(point);
        NodeView viewTower = resusableView;
        resusableView.set(point);
        if (conv == null) { // EXACT-foldable spine
            for (ITraversable c : components) {
                viewTower = c.reusableTraverse(point, slot, viewTower);
                slot.foldOut();
            }
        } else { // APPROX spine
            for (ITraversable c : components) {
                viewTower = c.reusableTraverse(point, slot, viewTower);
                conv.acceptValue(slot.convergingValue());
                slot.foldOut();
                if (conv.isConverged())
                    break;
            }
        }
        return finisher.apply(vf.liftResult(null, slot.getFoldResult()));
    }

    @Override
    public <R, S> S traverseForest(float[] point, IVisitorFactory<R> visitorFactory, BinaryOperator<R> accumulator,
            Function<R, S> finisher) {
        if (!visitorFactory.isReusable()) {
            R result = components.stream().map(c -> c.traverse(point, visitorFactory)).reduce(accumulator)
                    .orElseThrow(() -> new IllegalStateException("empty"));
            return finisher.apply(result);
        }
        if (visitorFactory.isFoldable()) {
            return foldForest(point, visitorFactory, null, finisher); // unified fold spine
        }

        IRFVisitor<R> slot = slotFor(visitorFactory, point);
        // IRFVisitor<R> slot = visitorFactory.newReusableVisitor(point);
        NodeView viewTower = resusableView;
        resusableView.set(point);
        R acc = null;
        for (ITraversable c : components) {
            viewTower = c.reusableTraverse(point, slot, viewTower);
            R r = visitorFactory.liftResult(null, slot.getResult()); // detached per-tree result
            acc = (acc == null) ? r : accumulator.apply(acc, r);
        }
        return finisher.apply(acc);
    }

    @Override
    public <R, S> S traverseForest(float[] point, IVisitorFactory<R> visitorFactory, Collector<R, ?, S> collector) {
        if (visitorFactory.isReusable()) {
            return collectReusable(point, visitorFactory, collector);
        }
        return components.stream().map(c -> c.traverse(point, visitorFactory)).collect(collector);
    }

    private <R, A, S> S collectReusable(float[] point, IVisitorFactory<R> vf, Collector<R, A, S> collector) {
        IRFVisitor<R> slot = slotFor(vf, point); // vf.newReusableVisitor(point);
        A container = collector.supplier().get();
        BiConsumer<A, R> acc = collector.accumulator();
        NodeView viewTower = resusableView;
        resusableView.set(point);
        for (ITraversable c : components) {
            viewTower = c.reusableTraverse(point, slot, viewTower); // threads the view; no foldOut
            acc.accept(container, vf.liftResult(null, slot.getResult())); // getResult() MUST be detached per tree
        }
        return collector.finisher().apply(container);
    }

    @Override
    public <R, S> S traverseForest(float[] point, IVisitorFactory<R> visitorFactory,
            ConvergingAccumulator<R> accumulator, Function<R, S> finisher) {
        if (visitorFactory.isReusable() && visitorFactory.isFoldable() && accumulator.isPrimitive()) {
            return foldForest(point, visitorFactory, accumulator, finisher);
        }
        // fallback unchanged
        for (IComponentModel<?, ?> component : components) {
            accumulator.accept(component.traverse(point, visitorFactory));
            if (accumulator.isConverged()) {
                break;
            }
        }
        return finisher.apply(accumulator.getAccumulatedValue());
    }

    @Override
    public <R, S> S traverseForestMulti(float[] point, IMultiVisitorFactory<R> visitorFactory,
            BinaryOperator<R> accumulator, Function<R, S> finisher) {
        // No reusable multi-fold: impute forks (multiple live copies) and its forest
        // reduction is a population, so it cannot fold. Legacy stream, unchanged.
        R unnormalizedResult = components.stream().map(c -> c.traverseMulti(point, visitorFactory)).reduce(accumulator)
                .orElseThrow(() -> new IllegalStateException("accumulator returned an empty result"));
        return finisher.apply(unnormalizedResult);
    }

    @Override
    public <R, S> S traverseForestMulti(float[] point, IMultiVisitorFactory<R> visitorFactory,
            Collector<R, ?, S> collector) {
        // The collector dive: impute (getConditionalField ->
        // ConditionalTreeSample.collector)
        // lands here. Reuse the ROOT visitor + one NodeView across trees; forks still
        // allocate via newPartialCopy() (a later, independent cut). Gated on
        // isReusable(), so the legacy stream is bit-unchanged for current callers.
        if (visitorFactory.isReusable()) {
            return collectReusableMulti(point, visitorFactory, collector);
        }
        return components.stream().map(c -> c.traverseMulti(point, visitorFactory)).collect(collector);
    }

    private <R, A, S> S collectReusableMulti(float[] point, IMultiVisitorFactory<R> vf, Collector<R, A, S> collector) {
        IRFMultiVisitor<R> slot = vf.newReusableMultiVisitor(point);
        A container = collector.supplier().get();
        BiConsumer<A, R> acc = collector.accumulator();
        NodeView viewTower = null;
        for (ITraversable c : components) {
            viewTower = c.reusableTraverseMulti(point, slot, viewTower);
            acc.accept(container, slot.getResult()); // detached per-tree sample; no lift, no sink
        }
        return collector.finisher().apply(container);
    }

    @SuppressWarnings("unchecked")
    private <R> IRFVisitor<R> slotFor(IVisitorFactory<R> vf, float[] point) {
        if (multiRead || !vf.isReusableAcrossQueries()) {
            return vf.newReusableVisitor(point);
        }
        checkArgument(point.length == dimensions, "unexpected point length");

        for (int i = 0; i < slotKey.length; i++) {
            if (slotKey[i] == vf) {
                IRFVisitor<R> v = (IRFVisitor<R>) slotVal[i];
                v.resetAcrossQueries(point);
                return v;
            }
            if (slotKey[i] == null) {
                slotKey[i] = vf;
                IRFVisitor<R> v = vf.newReusableVisitor(point);
                v.resetAcrossQueries(point); // redundant on a fresh visitor — deliberately
                slotVal[i] = v;
                return v;
            }
        }
        return vf.newReusableVisitor(point);
    }
}
