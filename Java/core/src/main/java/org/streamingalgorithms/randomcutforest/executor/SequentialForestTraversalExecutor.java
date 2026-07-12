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

package org.streamingalgorithms.randomcutforest.executor;

import java.util.function.BinaryOperator;
import java.util.function.Function;
import java.util.stream.Collector;

import org.streamingalgorithms.randomcutforest.*;
import org.streamingalgorithms.randomcutforest.returntypes.ConvergingAccumulator;
import org.streamingalgorithms.randomcutforest.tree.NodeView;

/**
 * Traverse the trees in a forest sequentially.
 */
public class SequentialForestTraversalExecutor extends AbstractForestTraversalExecutor {

    public SequentialForestTraversalExecutor(ComponentList<?, ?> components) {
        super(components);
    }

    @Override
    public <R, S> S traverseForest(float[] point, IVisitorFactory<R> visitorFactory, BinaryOperator<R> accumulator,
            Function<R, S> finisher) {
        if (!visitorFactory.isReusable()) {
            R result = components.stream().map(c -> c.traverse(point, visitorFactory)).reduce(accumulator)
                    .orElseThrow(() -> new IllegalStateException("empty"));
            return finisher.apply(result);
        }
        // SequentialForestTraversalExecutor, reuse path
        IRFVisitor<R> slot = visitorFactory.newReusableVisitor(point);
        NodeView viewTower = null; // a viewing structure and not the view
                                   // trees arm the view
        if (visitorFactory.isFoldable()) {
            for (ITraversable c : components) {
                viewTower = c.reusableFoldableTraverse(point, slot, viewTower); // arms + walks; leaves per-tree buffer
                                                                                // full
                slot.foldOut(); // scale + add into query accumulator in the visitor itself
                                // visitor keeps the score
            }
            return finisher.apply(visitorFactory.liftResult(/* tree? */ null, slot.getFoldResult()));
            // getResult() returns the ONE DiVector(foldedAttribution)
            // foldresult is the accumulator
        } else {
            R acc = null;
            for (ITraversable c : components) {
                R r = visitorFactory.liftResult(null, c.reusableTraverse(point, slot));
                acc = (acc == null) ? r : accumulator.apply(acc, r);
            }
            return finisher.apply(acc);
        }
    }

    @Override
    public <R, S> S traverseForest(float[] point, IVisitorFactory<R> visitorFactory, Collector<R, ?, S> collector) {

        return components.stream().map(c -> c.traverse(point, visitorFactory)).collect(collector);
    }

    @Override
    public <R, S> S traverseForest(float[] point, IVisitorFactory<R> visitorFactory,
            ConvergingAccumulator<R> accumulator, Function<R, S> finisher) {

        // fast path: reusable + foldable + primitive-converging → no per-tree alloc, no
        // boxing
        if (visitorFactory.isReusable() && visitorFactory.isFoldable() && accumulator.isPrimitive()) {
            IRFVisitor<R> slot = visitorFactory.newReusableVisitor(point);
            NodeView viewTower = null;
            for (ITraversable c : components) {
                viewTower = c.reusableFoldableTraverse(point, slot, viewTower);
                accumulator.acceptValue(slot.convergingValue()); // probe FIRST (before foldOut may scale buffer)
                slot.foldOut(); // then fold into the visitor's sum
                if (accumulator.isConverged()) {
                    break;
                }
            }
            return finisher.apply(visitorFactory.liftResult(null, slot.getFoldResult()));
        }

        // legacy fallback: non-reusable converging factories (allocates + boxes,
        // unchanged)
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

        R unnormalizedResult = components.stream().map(c -> c.traverseMulti(point, visitorFactory)).reduce(accumulator)
                .orElseThrow(() -> new IllegalStateException("accumulator returned an empty result"));

        return finisher.apply(unnormalizedResult);
    }

    @Override
    public <R, S> S traverseForestMulti(float[] point, IMultiVisitorFactory<R> visitorFactory,
            Collector<R, ?, S> collector) {

        return components.stream().map(c -> c.traverseMulti(point, visitorFactory)).collect(collector);
    }
}
