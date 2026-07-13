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
 * This file has changed significantly from its original form which carried
 * this notice.
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

package org.streamingalgorithms.randomcutforest.imputation;

import static java.lang.Math.min;
import static org.streamingalgorithms.randomcutforest.CommonUtils.checkArgument;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.streamingalgorithms.randomcutforest.returntypes.ConditionalTreeSample;
import org.streamingalgorithms.randomcutforest.returntypes.SampleSummary;
import org.streamingalgorithms.randomcutforest.store.IPointStoreView;
import org.streamingalgorithms.randomcutforest.summarization.Summarizer;
import org.streamingalgorithms.randomcutforest.util.Weighted;

public class ConditionalSampleSummarizer {

    /**
     * this limits the number of valueswe would see per dimension; note that it may
     * be hard to interpret a larger list
     */
    public static int MAX_NUMBER_OF_TYPICAL_PER_DIMENSION = 2;

    /**
     * the maximum size of the typical points array, irrespective of the number of
     * missing dimensions
     */
    public static int MAX_NUMBER_OF_TYPICAL_ELEMENTS = 5;

    /**
     * the array of missing dimension indices
     */
    protected int[] missingDimensions;

    /**
     * the query point, where we are inferring the missing values indicated by
     * missingDimensions[0], missingDimensions[1], ... etc.
     */
    protected float[] queryPoint;

    /**
     * a control parameter; =0 corresponds to (near) random samples and =1
     * correponds to more central (low anomaly score) samples
     */
    protected double centrality;

    /**
     * a boolean that determines if the summarization should use the missing
     * dimensions or the full dimensions.
     */
    protected boolean project = false;

    protected int numberOfReps = 1;

    protected double shrinkage = 0;

    protected int shingleSize = 1;

    protected final IPointStoreView<float[]> store;
    protected final boolean[] isMissing;

    public ConditionalSampleSummarizer(int[] missingDimensions, float[] queryPoint, double centrality, boolean project,
            int numberOfReps, double shrinkage, int shingleSize, IPointStoreView<float[]> store) {
        this.missingDimensions = Arrays.copyOf(missingDimensions, missingDimensions.length);
        this.queryPoint = Arrays.copyOf(queryPoint, queryPoint.length);
        this.centrality = centrality;
        this.project = project;
        this.numberOfReps = numberOfReps;
        this.shrinkage = shrinkage;
        this.shingleSize = shingleSize;
        this.store = store;
        this.isMissing = new boolean[queryPoint.length];
        for (int d : this.missingDimensions) {
            this.isMissing[d] = true;
        }
    }

    private float coord(ConditionalTreeSample e, int i) {
        if (store == null) {
            return e.leafPoint[i]; // test/legacy path: sample carries its own coords
        }
        return isMissing[i] ? store.valueAt(e.pointStoreIndex, i) : queryPoint[i];
    }

    // the coordinate indices the summary is computed over (identity / last-block /
    // projected-to-missing), computed ONCE per summarize call.
    private int[] outputIndices() {
        int dimension = queryPoint.length;
        if (project) {
            return missingDimensions; // summary over the missing coords only
        }
        if (shingleSize == 1) {
            int[] all = new int[dimension];
            for (int i = 0; i < dimension; i++) {
                all[i] = i;
            }
            return all;
        }
        int block = dimension / shingleSize; // last shingle block
        int[] tail = new int[block];
        for (int j = 0; j < block; j++) {
            tail[j] = dimension - block + j;
        }
        return tail;
    }

    // one sample -> its coordinate row over outIdx, read query-or-store (owned
    // array).
    private float[] rowOf(ConditionalTreeSample e, int[] outIdx) {
        float[] r = new float[outIdx.length];
        for (int j = 0; j < outIdx.length; j++) {
            r[j] = coord(e, outIdx[j]);
        }
        return r;
    }

    public SampleSummary summarize(List<ConditionalTreeSample> alist) {
        checkArgument(alist.size() > 0, "incorrect call to summarize");
        return summarize(alist, true);
    }

    public SampleSummary summarize(List<ConditionalTreeSample> alist, boolean addTypical) {
        double totalWeight = alist.size();
        List<ConditionalTreeSample> newList = ConditionalTreeSample.dedup(alist);
        newList.sort((o1, o2) -> Double.compare(o1.distance, o2.distance));

        final int[] outIdx = outputIndices();

        if (!addTypical) {
            final List<ConditionalTreeSample> samples = newList;
            final int[] out = outIdx;
            return new SampleSummary(new SampleSummary.CoordReader() {
                public int count() {
                    return samples.size();
                }

                public int dimension() {
                    return out.length;
                }

                public float value(int k, int j) {
                    return coord(samples.get(k), out[j]); // query-or-store, zero copy
                }

                public float weight(int k) {
                    return (float) samples.get(k).weight;
                }
            }, SampleSummary.DEFAULT_PERCENTILE);
        }

        // ---- centrality filtering: distance/weight only, coordinates untouched ----
        int num = 0;
        if (centrality > 0) {
            double threshold = centrality * newList.get(0).distance + 1e-6;
            double currentWeight = 0;
            int alwaysInclude = 0;
            double remainderWeight = totalWeight;
            while (newList.get(alwaysInclude).distance == 0) {
                remainderWeight -= newList.get(alwaysInclude).weight;
                ++alwaysInclude;
                if (alwaysInclude == newList.size()) {
                    break;
                }
            }
            for (int j = 1; j < newList.size(); j++) {
                if ((currentWeight < remainderWeight / 3
                        && currentWeight + newList.get(j).weight >= remainderWeight / 3)
                        || (currentWeight < remainderWeight / 2
                                && currentWeight + newList.get(j).weight >= remainderWeight / 2)) {
                    threshold = centrality * newList.get(j).distance;
                }
                currentWeight += newList.get(j).weight;
            }
            threshold += (1 - centrality) * newList.get(newList.size() - 1).distance;
            while (num < newList.size() && newList.get(num).distance <= threshold) {
                ++num;
            }
        } else {
            num = newList.size();
        }

        // typical points still wrap in Weighted for Summarizer.summarize (clustering).
        // That Weighted stays until the deferred columnar PR; the rows are now built
        // from the store (owned arrays), so nothing reads leafPoint here either.
        ArrayList<Weighted<float[]>> typicalPoints = new ArrayList<>();
        for (int j = 0; j < num; j++) {
            ConditionalTreeSample e = newList.get(j);
            typicalPoints.add(new Weighted<>(rowOf(e, outIdx), (float) e.weight));
        }
        int maxAllowed = min(queryPoint.length * MAX_NUMBER_OF_TYPICAL_PER_DIMENSION, MAX_NUMBER_OF_TYPICAL_ELEMENTS);
        maxAllowed = min(maxAllowed, num);

        SampleSummary projectedSummary = Summarizer.summarize(typicalPoints, maxAllowed, num, false,
                Summarizer::L2distance, 72, false, numberOfReps, shrinkage);

        return new SampleSummary(typicalPoints, projectedSummary);
    }

}
