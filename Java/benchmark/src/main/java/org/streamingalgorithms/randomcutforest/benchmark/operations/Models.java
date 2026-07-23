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

package org.streamingalgorithms.randomcutforest.benchmark.operations;

import java.util.EnumMap;

import org.streamingalgorithms.randomcutforest.RandomCutForest;
import org.streamingalgorithms.randomcutforest.parkservices.RCFCaster;
import org.streamingalgorithms.randomcutforest.parkservices.ThresholdedRandomCutForest;
import org.streamingalgorithms.randomcutforest.parkservices.state.RCFCasterMapper;
import org.streamingalgorithms.randomcutforest.parkservices.state.RCFCasterState;
import org.streamingalgorithms.randomcutforest.parkservices.state.ThresholdedRandomCutForestMapper;
import org.streamingalgorithms.randomcutforest.parkservices.state.ThresholdedRandomCutForestState;
import org.streamingalgorithms.randomcutforest.state.RandomCutForestMapper;
import org.streamingalgorithms.randomcutforest.state.RandomCutForestState;

/**
 * The three streaming models under measurement, built and saturated identically
 * so the Process and Serialization suites can't drift in setup.
 * 
 * <pre>
 *   RCF    -- score+update (RCF has no process(); one tuple = getAnomalyScore then update)
 *   TRCF   -- process(point, ts).getRCFScore()
 *   CASTER -- process(point, ts).getRCFScore()  (same call as TRCF; forecast is internal)
 * </pre>
 * 
 * The RCF member is the baseline that isolates the TRCF thresholder / Caster
 * forecast overhead ABOVE raw RCF.
 *
 */
public final class Models {

    public enum Kind {
        RCF, TRCF, CASTER
    }

    public enum TreeMode {
        SAVE, REBUILD
    }

    private static final int NUMBER_OF_TREES = 50;
    private static final int SAMPLE_SIZE = 256;
    private static final int FORECAST_HORIZON = 3;

    // Mappers are effectively stateless; share one configured instance each
    // (single-threaded suites).

    private static final RandomCutForestMapper RCF_SAVE = configuredRcfMapper(true);
    private static final RandomCutForestMapper RCF_REBUILD = configuredRcfMapper(false);
    private static final ThresholdedRandomCutForestMapper TRCF_MAPPER = new ThresholdedRandomCutForestMapper();
    private static final RCFCasterMapper CASTER_MAPPER = new RCFCasterMapper();

    private Models() {
    }

    public static final class Prepared {
        public final Kind kind;
        public final Object model; // trained + saturated
        public final double[][] stream; // SCORED-row hold-out to drive the measured pass
        public final long clock0; // next timestamp after saturation
        public final int dims;

        Prepared(Kind kind, Object model, double[][] stream, long clock0, int dims) {
            this.kind = kind;
            this.model = model;
            this.stream = stream;
            this.clock0 = clock0;
            this.dims = dims;
        }
    }

    public static EnumMap<Kind, Prepared> prepareAll(Datasets.Id dataset, double cacheFraction, long seed,
            int saturation, Kind... kinds) {
        Kind[] selected = (kinds.length == 0) ? Kind.values() : kinds;
        EnumMap<Kind, Prepared> prepared = new EnumMap<>(Kind.class);
        for (Kind kind : selected) {
            prepared.put(kind, prepare(kind, dataset, cacheFraction, seed, saturation));
        }
        return prepared;
    }

    private static void saturate(Kind kind, Object model, double[][] data, int count) {
        for (int i = 0; i < count; i++) {
            switch (kind) {
            case RCF:
                ((RandomCutForest) model).update(data[i]);
                break;
            case TRCF:
                ((ThresholdedRandomCutForest) model).process(data[i], i);
                break;
            case CASTER:
                ((RCFCaster) model).process(data[i], i);
                break;
            default:
                throw new IllegalStateException();
            }
        }
    }

    public static Prepared prepare(Kind kind, Datasets.Id dataset, double cacheFraction, long seed) {
        return prepare(kind, dataset, cacheFraction, seed, Datasets.INITIAL);
    }

    public static Prepared prepare(Kind kind, Datasets.Id dataset, double cacheFraction, long seed, int saturation) {
        final int dimensions = dataset.effectiveDims();
        final int shingleSize = (dataset == Datasets.Id.D1) ? 1 : 10;
        final double[][] data = Datasets.rawData(dataset);
        final Object model;

        switch (kind) {
        case RCF:
            model = RandomCutForest.builder().numberOfTrees(NUMBER_OF_TREES).dimensions(dimensions)
                    .shingleSize(shingleSize).boundingBoxCacheFraction(cacheFraction).randomSeed(seed)
                    .internalShinglingEnabled(true).sampleSize(SAMPLE_SIZE).parallelExecutionEnabled(false).build();
            break;
        case TRCF:
            model = ThresholdedRandomCutForest.builder().numberOfTrees(NUMBER_OF_TREES).dimensions(dimensions)
                    .shingleSize(shingleSize).boundingBoxCacheFraction(cacheFraction).randomSeed(seed)
                    .internalShinglingEnabled(true).sampleSize(SAMPLE_SIZE).parallelExecutionEnabled(false).build();
            break;
        case CASTER:
            if (shingleSize == 1) {
                throw new IllegalArgumentException("Caster needs shingleSize>1");
            }
            model = RCFCaster.builder().numberOfTrees(NUMBER_OF_TREES).dimensions(dimensions).shingleSize(shingleSize)
                    .boundingBoxCacheFraction(cacheFraction).randomSeed(seed).internalShinglingEnabled(true)
                    .sampleSize(SAMPLE_SIZE).parallelExecutionEnabled(false).forecastHorizon(FORECAST_HORIZON).build();
            break;
        default:
            throw new IllegalStateException("unknown model " + kind);
        }

        // Single feed: saturate() is the sole source of truth for how many points the
        // model consumes. (Each case previously ALSO pre-fed INITIAL points, which
        // double-processed the [0,saturation) prefix and, for TRCF/Caster, rewound the
        // timestamp axis before the measured pass.)
        saturate(kind, model, data, saturation);

        double[][] stream = new double[Datasets.SCORED][];
        for (int i = 0; i < Datasets.SCORED; i++) {
            stream[i] = data[saturation + i];
        }
        return new Prepared(kind, model, stream, saturation, dimensions); // clock0 = saturation
    }

    /**
     * One streaming tuple -> one comparable scalar. RCF ignores the timestamp;
     * TRCF/Caster use it.
     */
    public static double processOne(Kind kind, Object model, double[] point, long ts) {
        switch (kind) {
        case RCF: {
            RandomCutForest f = (RandomCutForest) model;
            double s = f.getAnomalyAttribution(point).getHighLowSum();
            // f.getAnomalyScore(point);
            f.update(point);
            return s;
        }
        case TRCF:
            return ((ThresholdedRandomCutForest) model).process(point, ts).getRCFScore();
        case CASTER:
            return ((RCFCaster) model).process(point, ts).getRCFScore();
        default:
            throw new IllegalStateException("unknown model " + kind);
        }
    }

    // ---- serialization bridges (used only by the Serialization suite) ----

    public static Object toState(Kind kind, Object model, TreeMode mode) {
        if (mode == TreeMode.REBUILD && kind != Kind.RCF) {
            throw new IllegalArgumentException("REBUILD only supported for RCF");
        }
        RandomCutForestMapper rcf = (mode == TreeMode.REBUILD) ? RCF_REBUILD : RCF_SAVE;
        switch (kind) {
        case RCF:
            return rcf.toState((RandomCutForest) model);
        case TRCF:
            return TRCF_MAPPER.toState((ThresholdedRandomCutForest) model);
        case CASTER:
            return CASTER_MAPPER.toState((RCFCaster) model);
        default:
            throw new IllegalStateException("unknown model " + kind);
        }
    }

    public static Object toModel(Kind kind, Object state, TreeMode mode) {
        if (mode == TreeMode.REBUILD && kind != Kind.RCF) {
            throw new IllegalArgumentException("REBUILD only supported for RCF");
        }
        RandomCutForestMapper rcf = (mode == TreeMode.REBUILD) ? RCF_REBUILD : RCF_SAVE;
        switch (kind) {
        case RCF:
            return rcf.toModel((RandomCutForestState) state);
        case TRCF:
            return TRCF_MAPPER.toModel((ThresholdedRandomCutForestState) state);
        case CASTER:
            return CASTER_MAPPER.toModel((RCFCasterState) state);
        default:
            throw new IllegalStateException("unknown model " + kind);
        }
    }

    public static Class<?> stateClass(Kind kind) {
        switch (kind) {
        case RCF:
            return RandomCutForestState.class;
        case TRCF:
            return ThresholdedRandomCutForestState.class;
        case CASTER:
            return RCFCasterState.class;
        default:
            throw new IllegalStateException("unknown model " + kind);
        }
    }

    private static RandomCutForestMapper configuredRcfMapper(boolean saveTrees) {
        RandomCutForestMapper m = new RandomCutForestMapper();
        m.setSaveExecutorContextEnabled(true);
        m.setSaveTreeStateEnabled(saveTrees);
        return m;
    }

}
