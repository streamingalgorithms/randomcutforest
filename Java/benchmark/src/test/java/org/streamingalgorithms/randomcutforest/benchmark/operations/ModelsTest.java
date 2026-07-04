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

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.EnumMap;
import java.util.Map;
import java.util.concurrent.TimeUnit;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import org.streamingalgorithms.randomcutforest.parkservices.state.RCFCasterState;
import org.streamingalgorithms.randomcutforest.parkservices.state.ThresholdedRandomCutForestState;
import org.streamingalgorithms.randomcutforest.state.RandomCutForestState;

/**
 * Models is the three-way dispatch (RCF / TRCF / Caster) shared by the Process
 * and Serialization suites. Dispatch correctness, the shingle-1 guard as a real
 * contract, and a mapper-only smoke -- NOT behavioral codec fidelity (that is
 * SerializationFidelityTest's job).
 *
 * <p>
 * All three models are built ONCE in {@link BeforeAll} at {@link #SAT} points
 * (not benchmark equilibrium) and shared read-only across the finiteness /
 * mapper / metadata assertions -- 3 builds, not 9. The class timeout is a
 * backstop: a genuine wedge (the Caster scare) fails loud in seconds instead of
 * hanging the reactor for minutes.
 */

@Timeout(value = 120, unit = TimeUnit.SECONDS, threadMode = Timeout.ThreadMode.SEPARATE_THREAD)
public class ModelsTest {

    private static final int SAT = 1000;
    private static final Map<Models.Kind, Models.Prepared> MODELS = new EnumMap<>(Models.Kind.class);

    @BeforeAll
    static void setUp() {
        MODELS.putAll(Models.prepareAll(Datasets.Id.D2, 0.2, 99L, SAT));
    }

    @Test
    public void casterRejectsShingleOne() {
        // D1 is shingle-1; the Caster has no temporal axis to forecast into. Assert the
        // guard
        // throws rather than building a broken (or hanging) model. Fast: throws
        // immediately.
        assertThrows(IllegalArgumentException.class,
                () -> Models.prepare(Models.Kind.CASTER, Datasets.Id.D1, 0.2, 99L, SAT));
    }

    @ParameterizedTest
    @EnumSource(Models.Kind.class)
    public void processOneIsFiniteOnD2(Models.Kind kind) {
        Models.Prepared p = MODELS.get(kind);
        assertNotNull(p.model, kind + ": model");
        long ts = p.clock0;
        double sum = 0;
        for (int i = 0; i < 200; i++) {
            double s = Models.processOne(kind, p.model, p.stream[i], ts++);
            assertTrue(Double.isFinite(s), kind + ": non-finite process score at " + i);
            sum += s;
        }
        assertTrue(sum != 0.0, kind + ": degenerate (all-zero) process sum");
    }

    @Test
    public void stateClassMapping() {
        assertEquals(RandomCutForestState.class, Models.stateClass(Models.Kind.RCF));
        assertEquals(ThresholdedRandomCutForestState.class, Models.stateClass(Models.Kind.TRCF));
        assertEquals(RCFCasterState.class, Models.stateClass(Models.Kind.CASTER));
    }

    @ParameterizedTest
    @EnumSource(Models.Kind.class)
    public void mapperRoundTripSmokeOnD2(Models.Kind kind) {
        smoke(kind, Models.TreeMode.SAVE);
    }

    @Test
    public void rebuildMapperRoundTripSmokeOnD2_rcfOnly() {
        // REBUILD is RCF-only and lossy-of-structure: trees are reconstructed from
        // sampler points, so the forest is NOT point-identical to the original.
        // Fidelity (score equality) therefore can't cover it -- this is the only
        // place REBUILD's "rebuilt forest still scores sanely" is asserted.
        smoke(Models.Kind.RCF, Models.TreeMode.REBUILD);
    }

    private static void smoke(Models.Kind kind, Models.TreeMode mode) {
        Models.Prepared p = MODELS.get(kind);
        Object state = Models.toState(kind, p.model, mode);
        assertNotNull(state, kind + "/" + mode + ": toState null");
        Object rebuilt = Models.toModel(kind, state, mode);
        assertNotNull(rebuilt, kind + "/" + mode + ": toModel null");

        long ts = p.clock0;
        double sum = 0;
        for (int i = 0; i < 50; i++) {
            double s = Models.processOne(kind, rebuilt, p.stream[i], ts++);
            assertTrue(Double.isFinite(s), kind + "/" + mode + ": rebuilt non-finite at " + i);
            sum += s;
        }
        assertTrue(sum != 0.0, kind + "/" + mode + ": degenerate (all-zero) rebuilt scores");
    }

    @ParameterizedTest
    @EnumSource(Models.Kind.class)
    public void preparedCarriesConsistentMetadata(Models.Kind kind) {
        Models.Prepared p = MODELS.get(kind);
        assertEquals(kind, p.kind, "kind round-trips into Prepared");
        assertEquals(100, p.dims, "effective dims");
        assertEquals(Datasets.SCORED, p.stream.length, "stream length");
        assertEquals(SAT, p.clock0, "clock starts after saturation");
    }
}
