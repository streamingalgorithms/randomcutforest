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

package org.streamingalgorithms.randomcutforest.benchmark;

import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;
import static org.streamingalgorithms.randomcutforest.benchmark.operations.Models.TreeMode.SAVE;

import java.util.EnumMap;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.TimeUnit;
import java.util.stream.Stream;

import org.apache.fory.logging.LoggerFactory;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Timeout;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.streamingalgorithms.randomcutforest.benchmark.operations.Codec;
import org.streamingalgorithms.randomcutforest.benchmark.operations.Datasets;
import org.streamingalgorithms.randomcutforest.benchmark.operations.Models;

/**
 * Codec fidelity: a model round-tripped through a codec must BEHAVE identically
 * to the original. This is the correctness oracle deliberately kept OUT of the
 * benchmark's measured path (SerializationBenchmark / SerializationColdMain
 * time the round-trip; they do not check it) -- here running both models to
 * compare them is the point, not a perturbation.
 *
 * <p>
 * The check is BEHAVIORAL, not structural: we do not assert state.equals(),
 * because array-vs-object reshaping can make structurally-different states
 * score identically (or structurally-identical states diverge). Instead we feed
 * the original and the round-tripped model the same held-out stream and assert
 * their getRCFScore() agree. A codec that "completes" a round-trip but rebuilds
 * a corrupted model -- e.g. a mangled DeviationState that skews the very
 * statistics TRCF uses to decide what is anomalous -- fails HERE, which is
 * exactly where a silently-wrong model must be caught rather than shipped.
 *
 * <p>
 * Policy: this library does NOT expand serialization support beyond upstream.
 * Known-unfaithful (codec, model) pairs are listed in {@link #KNOWN_UNFAITHFUL}
 * so they are reported but do not block the build; ANY OTHER unfaithful pair is
 * a hard failure. Removing a pair from the allowlist turns it back into a gate.
 *
 * <p>
 * Runs on D2 only (shingleSize>1) so the Caster is valid; RCF/TRCF/Caster x all
 * codecs except the no-wire controls (STATE/NONE round-trip trivially).
 */
public class SerializationFidelityTest {

    private static final Datasets.Id DATASET = Datasets.Id.D2;
    private static final long SEED = 99L;
    private static final int SAT = 1000;
    private static final double CACHE = 0.2; // fidelity is cache-independent; one representative value
    private static final int CHECK_TUPLES = 20;
    private static final double TOL = 1e-6;

    /**
     * (codec, model) pairs known not to round-trip faithfully upstream. Reported,
     * not fatal -- the state classes are not strict-serializer-clean (see the
     * parkservices POJO-contract test), and fixing that is out of scope for this
     * library. A NEW unfaithful pair not listed here fails the build.
     */
    private static final Set<String> KNOWN_UNFAITHFUL = Set.of(
    // e.g. "JACKSON3/TRCF", "JACKSON2/TRCF", "JACKSON3/CASTER" -- populate from the
    // first run; leave empty to see the full current truth as failures first.
    );

    private static final Map<Models.Kind, Object> STATE = new EnumMap<>(Models.Kind.class);
    private static final Map<Models.Kind, double[]> REF = new EnumMap<>(Models.Kind.class);
    private static final Map<Models.Kind, float[][]> STREAM = new EnumMap<>(Models.Kind.class);
    private static final Map<Models.Kind, Long> CLOCK0 = new EnumMap<>(Models.Kind.class);

    @BeforeAll
    static void setUp() {
        LoggerFactory.disableLogging();
        for (Models.Kind kind : Models.Kind.values()) {
            if (kind == Models.Kind.CASTER && DATASET == Datasets.Id.D1) {
                continue; // Caster needs shingleSize>1
            }
            Models.Prepared original = Models.prepare(kind, DATASET, CACHE, SEED, SAT); // once
            Object state = Models.toState(kind, original.model, SAVE); // pristine
            int limit = Math.min(CHECK_TUPLES, original.stream.length);

            // Score a FRESH model rebuilt from the pristine state, so neither `state`
            // nor original.model is mutated by baseline generation.
            Object refModel = Models.toModel(kind, state, SAVE);
            double[] ref = new double[limit];
            long ts = original.clock0;
            for (int i = 0; i < limit; i++) {
                ref[i] = Models.processOne(kind, refModel, original.stream[i], ts++);
            }

            STATE.put(kind, state);
            REF.put(kind, ref);
            STREAM.put(kind, original.stream);
            CLOCK0.put(kind, original.clock0);
        }
    }

    static Stream<Arguments> cases() {
        Stream.Builder<Arguments> b = Stream.builder();
        for (Models.Kind model : Models.Kind.values()) {
            if (model == Models.Kind.CASTER && DATASET == Datasets.Id.D1) {
                continue;
            }
            for (Codec.Id id : Codec.Id.values()) {
                b.add(Arguments.of(id, model));
            }
        }
        return b.build();
    }

    @ParameterizedTest(name = "{0} / {1}")
    @MethodSource("cases")
    @Timeout(value = 30, unit = TimeUnit.SECONDS, threadMode = Timeout.ThreadMode.SEPARATE_THREAD)
    public void roundTripPreservesScores(Codec.Id codecId, Models.Kind model) {
        String pair = codecId + "/" + model;
        Codec codec = Codec.of(codecId);

        Object state = STATE.get(model);
        double[] ref = REF.get(model);
        float[][] stream = STREAM.get(model);

        Object rebuilt;
        try {
            if (codec.isControl()) {
                rebuilt = Models.toModel(model, state, SAVE); // STATE/NONE: mapper-only round-trip
            } else {
                byte[] wire = codec.encode(state);
                Object decoded = codec.decode(wire, Models.stateClass(model));
                rebuilt = Models.toModel(model, decoded, SAVE);
            }
        } catch (RuntimeException e) {
            reportOrFail(pair, "round-trip threw " + e.getClass().getSimpleName() + ": " + e.getMessage());
            return;
        }

        long ts = CLOCK0.get(model);
        int mismatches = 0;
        double firstRef = Double.NaN, firstRebuilt = Double.NaN;
        for (int i = 0; i < ref.length; i++) {
            double sRebuilt = Models.processOne(model, rebuilt, stream[i], ts++);
            assertTrue(Double.isFinite(ref[i]), pair + ": baseline non-finite at " + i);
            if (Double.compare(ref[i], sRebuilt) != 0 && Math.abs(ref[i] - sRebuilt) > TOL) {
                if (mismatches == 0) {
                    firstRef = ref[i];
                    firstRebuilt = sRebuilt;
                }
                mismatches++;
            }
        }
        if (mismatches > 0) {
            reportOrFail(pair, String.format("%d/%d scores diverged (first: ref=%.8g rebuilt=%.8g)", mismatches,
                    ref.length, firstRef, firstRebuilt));
        }
        System.out.println(pair + " completed");
    }

    /**
     * Known-unfaithful pairs are reported to stderr; anything else fails the build.
     */
    private static void reportOrFail(String pair, String detail) {
        if (KNOWN_UNFAITHFUL.contains(pair)) {
            System.err.printf("KNOWN-UNFAITHFUL %-16s : %s%n", pair, detail);
        } else {
            fail(pair + " unfaithful: " + detail);
        }
    }
}
