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
 * This file is substantially modified from a file of the same name which
 * had the following notice.
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

package org.streamingalgorithms.randomcutforest.state.store;

import static org.junit.jupiter.api.Assertions.*;

import java.util.Random;
import java.util.stream.Stream;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.streamingalgorithms.randomcutforest.config.Precision;
import org.streamingalgorithms.randomcutforest.store.PointStore;

/**
 * Serde oracle for the unified {@link PointStore}. The previous test compared
 * raw backing arrays ({@code getStore()}), which is not an invariant:
 * compaction, store trimming, and a legitimate BYTE/CHAR/INT tier change all
 * rewrite that array while preserving every point. Here the invariant is
 * semantic — index {@code i} must decode to the same vector and carry the same
 * reference count after a round trip — plus idempotence across a second round
 * trip.
 *
 * Shapes straddle the tier ladder that {@code Column.tierFor} walks, including
 * the rotation x2 value-bound factor the old {@code build()} predicate omitted.
 * Tier is a function of capacity, not fill level, so we probe INT cheaply with
 * a large capacity but a small number of adds.
 */
public class PointStoreMapperTest {

    private PointStoreMapper mapper;

    @BeforeEach
    public void setUp() {
        mapper = new PointStoreMapper();
    }

    /**
     * dimensions, shingleSize, capacity, rotation.
     * <ul>
     * <li>200 -> BYTE (bound 199, top 200)
     * <li>256 -> CHAR (bound 255, top 256 — the byte/char edge)
     * <li>1000 -> CHAR
     * <li>70000 -> INT (bound 69999) — probed with few adds, not a full fill
     * <li>rotation: bound = 2*cap*shingle-1, exercises the x2 factor + rotated
     * decode
     * </ul>
     */
    static Stream<Arguments> shapes() {
        return Stream.of(Arguments.of(2, 1, 200, false), Arguments.of(2, 1, 256, false),
                Arguments.of(3, 1, 1000, false), Arguments.of(1, 1, 70000, false), Arguments.of(4, 4, 300, true));
    }

    @ParameterizedTest(name = "dim={0} shingle={1} cap={2} rot={3}")
    @MethodSource("shapes")
    void roundTripPreservesEveryLivePoint(int dimensions, int shingleSize, int capacity, boolean rotation) {
        PointStore store = newStore(dimensions, shingleSize, capacity, rotation);
        int feasible = fill(store, dimensions, shingleSize, rotation, Math.min(capacity, 800), 7L);
        assertTrue(feasible > 0, "no points landed — test would be vacuous");
        assertEquals(feasible, store.size());

        PointStore after = mapper.toModel(mapper.toState(store));
        PointStore afterAfter = mapper.toModel(mapper.toState(after));

        assertEquals(store.getCapacity(), after.getCapacity());
        assertEquals(store.getDimensions(), after.getDimensions());
        assertEquals(store.size(), after.size());

        assertPointStoreEquals(store, after); // round trip preserves
        assertPointStoreEquals(after, afterAfter); // and is idempotent
    }

    /**
     * Multiplicity past the byte tier forces the refCount overflow map, which
     * serializes through the duplicate refs. Round trip must preserve the exact
     * count at the index.
     */
    @Test
    void refCountOverflowRoundTrips() {
        PointStore store = new PointStore.Builder<>().dimensions(2).capacity(8).build();
        int idx = store.add(new float[] { 1.1f, -22.2f }, 1, false);
        for (int i = 0; i < 300; i++) {
            store.incrementRefCount(idx); // past 255 -> into refCountMap
        }
        int expected = store.getRefCount(idx);
        assertTrue(expected > 255, "did not exercise the overflow map");

        PointStore after = mapper.toModel(mapper.toState(store));
        assertEquals(expected, after.getRefCount(idx));
        assertArrayEquals(store.getNumericVector(idx), after.getNumericVector(idx), 0.0f);
    }

    /**
     * The mapper's own state validation — carried over from the original test,
     * which was the one part worth keeping.
     */
    @Test
    void mapperValidatesState() {
        PointStore store = new PointStore.Builder<>().dimensions(2).capacity(8).build();
        store.add(new float[] { 1.1f, -22.2f }, 1, false);
        store.add(new float[] { 3.3f, -4.4f }, 2, false);
        store.add(new float[] { 10.1f, 100.1f }, 3, false);

        PointStoreState state = mapper.toState(store);
        state.setDuplicateRefs(null);
        assertDoesNotThrow(() -> mapper.toModel(state));
        state.setDuplicateRefs(new int[1]);
        assertThrows(IllegalArgumentException.class, () -> mapper.toModel(state));
        state.setDuplicateRefs(new int[2]);
        assertDoesNotThrow(() -> mapper.toModel(state));
        state.setPrecision(Precision.FLOAT_64.name());
        assertThrows(IllegalArgumentException.class, () -> mapper.toModel(state));
    }

    // =====================================================================
    // THE ORACLE. Same live indices, same decoded vector per index, same
    // per-index reference count (including overflow). Decodes via
    // getNumericVector so it goes through the real location -> address ->
    // (rotated) unpack path rather than peeking at raw storage.
    // =====================================================================
    private void assertPointStoreEquals(PointStore a, PointStore b) {
        assertEquals(a.size(), b.size(), "live count mismatch");

        int[] ra = a.getRefCount();
        int[] rb = b.getRefCount();
        assertEquals(ra.length, rb.length, "index-capacity mismatch");
        assertArrayEquals(ra, rb, "reference-count vector mismatch");

        for (int i = 0; i < ra.length; i++) {
            if (ra[i] > 0) {
                assertArrayEquals(a.getNumericVector(i), b.getNumericVector(i), 0.0f,
                        "point at index " + i + " differs after round trip");
            }
        }
    }

    // =====================================================================
    // Builders / streaming.
    // =====================================================================

    private PointStore newStore(int dimensions, int shingleSize, int capacity, boolean rotation) {
        PointStore.Builder<?> b = PointStore.builder().dimensions(dimensions).shingleSize(shingleSize)
                .capacity(capacity).initialSize(Math.min(capacity, 64)); // small initial -> exercises growth
        if (rotation) {
            // rotation requires internal shingling
            b = b.internalShinglingEnabled(true).internalRotationEnabled(true);
        }
        return b.build();
    }

    /**
     * Adds up to {@code count} points, returning how many landed feasibly. With
     * internal shingling the first (shingleSize - 1) updates warm the shingle and
     * return INFEASIBLE; without it every add takes an index. Every add consumes an
     * index, so callers keep count <= capacity.
     */
    private int fill(PointStore store, int dimensions, int shingleSize, boolean rotation, int count, long seed) {
        Random rng = new Random(seed);
        int inputLen = rotation ? dimensions / shingleSize : dimensions;
        int landed = 0;
        for (int i = 0; i < count; i++) {
            int idx = store.add(randomPoint(inputLen, rng), i + 1, false);
            if (idx >= 0) {
                landed++;
            }
        }
        return landed;
    }

    private float[] randomPoint(int len, Random rng) {
        float[] p = new float[len];
        for (int i = 0; i < len; i++) {
            p[i] = (float) (rng.nextGaussian() * 10.0);
        }
        return p;
    }
}