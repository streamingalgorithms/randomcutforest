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

package org.streamingalgorithms.randomcutforest.util;

import static org.junit.jupiter.api.Assertions.*;
import static org.streamingalgorithms.randomcutforest.util.ArrayEncoder.unpackFloats;

import java.util.Arrays;
import java.util.Random;
import java.util.stream.Stream;

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

class ArrayEncoderTest {

    /** A named test scenario: the raw values, and whether to compress. */
    record Case(String name, int[] values, boolean compress) {
        @Override
        public String toString() {
            return name + "[n=" + values.length + ",c=" + compress + "]";
        }
    }

    static int[] bits(int n, double p1, long seed) {
        Random r = new Random(seed);
        int[] a = new int[n];
        for (int i = 0; i < n; i++)
            a[i] = r.nextDouble() < p1 ? 1 : 0;
        return a;
    }

    static int[] rand(int n, int lo, int hiExcl, long seed) {
        Random r = new Random(seed);
        int[] a = new int[n];
        for (int i = 0; i < n; i++)
            a[i] = lo + r.nextInt(hiExcl - lo);
        return a;
    }

    static int[] constant(int n, int v) {
        int[] a = new int[n];
        Arrays.fill(a, v);
        return a;
    }

    static Stream<Case> cases() {
        return Stream.of(
                // base==2 : the NodeStore 0/1 mask case
                new Case("bits_0.5_100", bits(100, 0.5, 1), true), new Case("bits_0.1_1000", bits(1000, 0.1, 2), true),
                new Case("bits_0.9_1000", bits(1000, 0.9, 3), true), new Case("bits_all0_500", bits(500, 0.0, 4), true), // collapses
                                                                                                                         // to
                                                                                                                         // base==1
                new Case("bits_all1_500", bits(500, 1.0, 5), true), // base==1
                new Case("bits_exact30", bits(30, 0.5, 6), true), // one code word
                new Case("bits_31", bits(31, 0.5, 7), true), new Case("bits_60", bits(60, 0.5, 8), true),
                new Case("bits_nocomp", bits(100, 0.5, 9), false),
                // two-consecutive, min != 0 -> base==2 with offset
                new Case("twoconsec_5_6", rand(200, 5, 7, 10), true),
                // packNum==1 large range
                new Case("bigrange_50pct", rand(5000, 0, 300_000, 11), true),
                new Case("bigrange_nocomp", rand(5000, 0, 300_000, 12), false),
                // packNum>1 general
                new Case("midrange_base64", rand(2000, 0, 64, 13), true),
                new Case("midrange_base256", rand(2000, 100, 356, 14), true),
                // base==1 constant
                new Case("const_12345", constant(1000, 12345), true),
                // size<3 copy path
                new Case("size2", rand(2, 0, 300_000, 15), true), new Case("size1", rand(1, 0, 300_000, 16), true),
                new Case("size0", new int[0], true));
    }

    static Stream<Case> bitCases() {
        return Stream.of(
                // base==2 : the NodeStore 0/1 mask case
                new Case("bits_0.5_100", bits(100, 0.5, 1), true), new Case("bits_0.1_1000", bits(1000, 0.1, 2), true),
                new Case("bits_0.9_1000", bits(1000, 0.9, 3), true), new Case("bits_all0_500", bits(500, 0.0, 4), true), // collapses
                                                                                                                         // to
                                                                                                                         // base==1
                new Case("bits_all1_500", bits(500, 1.0, 5), true), // base==1
                new Case("bits_exact30", bits(30, 0.5, 6), true), // one code word
                new Case("bits_31", bits(31, 0.5, 7), true), new Case("bits_60", bits(60, 0.5, 8), true),
                new Case("bits_nocomp", bits(100, 0.5, 9), false));
    }

    /**
     * encodeCore must byte-match the REAL ArrayPacking.moveAndPack (identity map).
     */
    @ParameterizedTest(name = "encode {0}")
    @MethodSource("cases")
    void encodeMatchesLibrary(Case c) {
        int size = c.values().length;
        int[] map = new int[size];
        for (int i = 0; i < size; i++)
            map[i] = i;
        int[] fromLibrary = ArrayPacking.moveAndPack(map, size, c.compress(), m -> c.values()[m]);
        int[] fromCore = ArrayEncoder.encodeCore(size, c.compress(), k -> c.values()[k]);
        assertArrayEquals(fromLibrary, fromCore, c.name() + ": encodeCore != ArrayPacking.moveAndPack");
    }

    /** Full round trip: encode then decode must return the original values. */
    @ParameterizedTest(name = "roundtrip {0}")
    @MethodSource("cases")
    void roundTrip(Case c) {
        int size = c.values().length;
        int[] packed = ArrayEncoder.encodeCore(size, c.compress(), k -> c.values()[k]);
        final int[] out = new int[size];
        if (!c.compress() || packed.length < 3) {
            System.arraycopy(packed, 0, out, 0, Math.min(size, packed.length));
        } else {
            int[] cur = { 0 };
            ArrayEncoder.decodeCore(packed, size, true, v -> out[cur[0]++] = v);
        }
        assertArrayEquals(c.values(), out, c.name() + ": round trip lost data");
    }

    @ParameterizedTest(name = "bitDirect roundtrip {0}")
    @MethodSource("bitCases") // only 0/1 arrays
    void bitDirectRoundTrip(Case c) {
        int size = c.values().length;
        int[] packed = ArrayEncoder.bitEncode(size, c.compress(), k -> c.values()[k]);
        final int[] out = new int[size];
        if (!c.compress() || packed.length < 3) {
            System.arraycopy(packed, 0, out, 0, Math.min(size, packed.length));
        } else {
            int[] cur = { 0 };
            ArrayEncoder.bitDecode(packed, size, v -> out[cur[0]++] = v);
        }
        assertArrayEquals(c.values(), out, c.name() + ": bit round trip lost data");
    }

    public void testPackFloatsWithLength() {
        int inputLength = 100;
        int packLength = 76;
        Random rng = new Random();
        float[] inputArray = new float[inputLength];
        for (int i = 0; i < inputLength; i++) {
            inputArray[i] = rng.nextFloat();
        }
        byte[] bytes = ArrayEncoder.pack(inputArray, packLength);
        assertThrows(IllegalArgumentException.class, () -> ArrayEncoder.pack(inputArray, inputLength + 10));
        float[] outputArray = unpackFloats(bytes);

        assertEquals(packLength, outputArray.length);
        assertArrayEquals(Arrays.copyOf(inputArray, packLength), outputArray);
        assertDoesNotThrow(() -> ArrayEncoder.pack(new float[0], 0));
        assertThrows(IllegalArgumentException.class, () -> ArrayEncoder.pack(new float[10], -1));
    }
}