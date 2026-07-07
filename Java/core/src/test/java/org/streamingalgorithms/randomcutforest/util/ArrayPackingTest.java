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
 * The file has substantially modified from the original which had this notice.
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

package org.streamingalgorithms.randomcutforest.util;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.streamingalgorithms.randomcutforest.util.ArrayPacking.pack;
import static org.streamingalgorithms.randomcutforest.util.ArrayPacking.unpackDoubles;
import static org.streamingalgorithms.randomcutforest.util.ArrayPacking.unpackFloats;
import static org.streamingalgorithms.randomcutforest.util.ArrayPacking.unpackInts;
import static org.streamingalgorithms.randomcutforest.util.ArrayPacking.unpackShorts;

import java.util.Arrays;
import java.util.Random;
import java.util.function.IntUnaryOperator;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.junit.jupiter.params.provider.ValueSource;

public class ArrayPackingTest {
    private Random rng;

    @BeforeEach
    public void setUp() {
        rng = new Random();
        ArrayPacking arrayPacking = new ArrayPacking();
    }

    @Test
    public void testLogMax() {
        long[] bases = new long[] { 2, 101, 3_456_789 };
        Arrays.stream(bases).forEach(base -> {
            int log = ArrayPacking.logMax(base);
            assertTrue(Math.pow(base, log + 1) >= Integer.MAX_VALUE);
            assertTrue(Math.pow(base, log) < Integer.MAX_VALUE);
        });
    }

    @Test
    public void testLogMaxInvalid() {
        assertThrows(IllegalArgumentException.class, () -> ArrayPacking.logMax(1));
        assertThrows(IllegalArgumentException.class, () -> ArrayPacking.logMax(0));
        assertThrows(IllegalArgumentException.class, () -> ArrayPacking.logMax(-123467890));
    }

    @ParameterizedTest
    @ValueSource(ints = { 0, 1, 2, 3, 11, 100 })
    public void testIntsPackRoundTrip(int inputLength) {
        int[] inputArray = rng.ints().limit(inputLength).toArray();
        assertArrayEquals(inputArray, ArrayPacking.unpackInts(ArrayPacking.pack(inputArray, false), false));
        assertArrayEquals(inputArray, ArrayPacking.unpackInts(ArrayPacking.pack(inputArray, true), true));
    }

    @ParameterizedTest
    @ValueSource(ints = { 0, 1, 2, 3, 17, 100 })
    public void testShortsPackRoundTrip(int inputLength) {
        short[] inputArray = new short[inputLength];
        for (int i = 0; i < inputLength; i++) {
            inputArray[i] = (short) (rng.nextInt() % 100);
        }
        assertArrayEquals(inputArray, ArrayPacking.unpackShorts(ArrayPacking.pack(inputArray, false), false));
        assertArrayEquals(inputArray, ArrayPacking.unpackShorts(ArrayPacking.pack(inputArray, true), true));
    }

    @ParameterizedTest
    @ValueSource(ints = { 0, 1, 2, 3, 11, 100 })
    public void testIdenticalInts(int inputLength) {
        int[] inputArray = new int[inputLength];
        Arrays.fill(inputArray, rng.nextInt());
        assertArrayEquals(inputArray, ArrayPacking.unpackInts(ArrayPacking.pack(inputArray, false), false));
        int[] result = ArrayPacking.pack(inputArray, true);
        assertTrue(result.length == 3 || inputLength < 3 && result.length == inputLength);
        assertArrayEquals(inputArray, ArrayPacking.unpackInts(result, true));
    }

    @ParameterizedTest
    @ValueSource(ints = { 0, 1, 2, 3, 17, 100 })
    public void testIdenticalShorts(int inputLength) {
        short item = (short) (rng.nextInt() % 100);
        short[] inputArray = new short[inputLength];
        for (int i = 0; i < inputLength; i++) {
            inputArray[i] = item;
        }
        assertArrayEquals(inputArray, ArrayPacking.unpackShorts(ArrayPacking.pack(inputArray, false), false));
        int[] result = ArrayPacking.pack(inputArray, true);
        assertTrue(result.length == 3 || inputLength < 3 && result.length == inputLength);
        assertArrayEquals(inputArray, ArrayPacking.unpackShorts(result, true));
    }

    @Test
    public void testUnpackIntsWithLengthGiven() {
        int inputLength = 100;
        int[] inputArray = rng.ints().limit(inputLength).toArray();

        assertThrows(IllegalArgumentException.class, () -> pack(inputArray, inputLength + 1, false));
        assertThrows(IllegalArgumentException.class, () -> pack(inputArray, inputLength + 1, true));
        assertThrows(IllegalArgumentException.class, () -> pack(inputArray, -1, false));
        assertThrows(IllegalArgumentException.class, () -> pack(inputArray, -1, true));
        assertDoesNotThrow(() -> pack(inputArray, 0, true));
        assertDoesNotThrow(() -> pack(inputArray, 0, false));

        int[] uncompressed = ArrayPacking.pack(inputArray, false);
        int[] compressed = ArrayPacking.pack(inputArray, true);

        int[] result = ArrayPacking.unpackInts(uncompressed, 50, false);
        assertThrows(IllegalArgumentException.class, () -> unpackInts(compressed, -1, true));
        assertEquals(50, result.length);
        assertArrayEquals(Arrays.copyOf(inputArray, 50), result);

        result = ArrayPacking.unpackInts(compressed, 50, true);
        assertEquals(50, result.length);
        assertArrayEquals(Arrays.copyOf(inputArray, 50), result);

        result = ArrayPacking.unpackInts(uncompressed, 200, false);
        assertEquals(200, result.length);
        assertArrayEquals(inputArray, Arrays.copyOf(result, 100));
        for (int i = 100; i < 200; i++) {
            assertEquals(0, result[i]);
        }

        result = ArrayPacking.unpackInts(compressed, 200, true);
        assertEquals(200, result.length);
        assertArrayEquals(inputArray, Arrays.copyOf(result, 100));
        for (int i = 100; i < 200; i++) {
            assertEquals(0, result[i]);
        }
    }

    @Test
    public void testUnpackShortsWithLengthGiven() {
        int inputLength = 100;
        short[] inputArray = new short[50];
        Arrays.fill(inputArray, (short) 2);
        short[] test = new short[2];
        short[] test2 = new short[3];
        int[] uncompressed = ArrayPacking.pack(inputArray, false);
        int[] compressed = ArrayPacking.pack(inputArray, true);

        assertArrayEquals(test, unpackShorts(new int[2], true));
        assertArrayEquals(test, unpackShorts(new int[2], false));
        assertArrayEquals(test2, unpackShorts(new int[3], false));

        assertThrows(IllegalArgumentException.class, () -> unpackShorts(uncompressed, -1, false));

        short[] result = ArrayPacking.unpackShorts(uncompressed, 50, false);
        assertEquals(50, result.length);
        assertArrayEquals(Arrays.copyOf(inputArray, 50), result);

        result = ArrayPacking.unpackShorts(compressed, 100, true);
        assertEquals(100, result.length);
        for (int y = 0; y < 50; y++) {
            assertTrue(result[y] == 2);
        }
        for (int y = 50; y < 100; y++) {
            assertTrue(result[y] == 0);
        }
    }

    @ParameterizedTest
    @ValueSource(ints = { 0, 1, 2, 3, 17, 100 })
    public void testPackDoublesRoundTrip(int inputLength) {
        double[] inputArray = rng.doubles().limit(inputLength).toArray();
        assertArrayEquals(inputArray, ArrayPacking.unpackDoubles(ArrayPacking.pack(inputArray)));
    }

    @ParameterizedTest
    @ValueSource(ints = { 0, 1, 2, 3, 5, 100 })
    public void testPackFloatsRoundTrip(int inputLength) {
        float[] inputArray = new float[inputLength];
        for (int i = 0; i < inputLength; i++) {
            inputArray[i] = rng.nextFloat();
        }
        assertArrayEquals(inputArray, unpackFloats(ArrayPacking.pack(inputArray)));
    }

    @ParameterizedTest
    @ValueSource(booleans = { true, false })
    public void testPackShortsWithLength(boolean compress) {
        int inputLength = 100;
        int packLength = 76;
        short[] inputArray = new short[inputLength];
        for (int i = 0; i < inputLength; i++) {
            inputArray[i] = (short) (rng.nextInt() % 100);
        }
        assertThrows(IllegalArgumentException.class, () -> pack(inputArray, inputLength + 10, compress));
        assertThrows(IllegalArgumentException.class, () -> pack(inputArray, -10, compress));
        int[] array = ArrayPacking.pack(inputArray, packLength, compress);
        short[] outputArray = ArrayPacking.unpackShorts(array, compress);

        assertEquals(packLength, outputArray.length);
        assertArrayEquals(Arrays.copyOf(inputArray, packLength), outputArray);
    }

    @Test
    public void testPackDoublesWithLength() {
        int inputLength = 100;
        int packLength = 76;
        double[] inputArray = rng.doubles().limit(inputLength).toArray();
        byte[] bytes = ArrayPacking.pack(inputArray, packLength);
        double[] outputArray = ArrayPacking.unpackDoubles(bytes);

        assertEquals(packLength, outputArray.length);
        assertArrayEquals(Arrays.copyOf(inputArray, packLength), outputArray);
        assertDoesNotThrow(() -> pack(new double[0], 0));
        assertThrows(IllegalArgumentException.class, () -> pack(new double[10], 11));
        assertThrows(IllegalArgumentException.class, () -> pack(new double[10], -1));
    }

    @Test
    public void testPackFloatsWithLength() {
        int inputLength = 100;
        int packLength = 76;
        float[] inputArray = new float[inputLength];
        for (int i = 0; i < inputLength; i++) {
            inputArray[i] = rng.nextFloat();
        }
        byte[] bytes = ArrayPacking.pack(inputArray, packLength);
        assertThrows(IllegalArgumentException.class, () -> pack(inputArray, inputLength + 10));
        float[] outputArray = unpackFloats(bytes);

        assertEquals(packLength, outputArray.length);
        assertArrayEquals(Arrays.copyOf(inputArray, packLength), outputArray);
        assertDoesNotThrow(() -> pack(new float[0], 0));
        assertThrows(IllegalArgumentException.class, () -> pack(new float[10], -1));
    }

    @Test
    public void testUnpackDoublesWithLength() {
        int inputLength = 100;
        double[] inputArray = rng.doubles().limit(inputLength).toArray();
        byte[] bytes = ArrayPacking.pack(inputArray);

        int unpackLength1 = 25;
        double[] outputArray1 = ArrayPacking.unpackDoubles(bytes, unpackLength1);
        assertEquals(unpackLength1, outputArray1.length);
        assertArrayEquals(Arrays.copyOf(inputArray, unpackLength1), outputArray1);

        int unpackLength2 = 123;
        assertThrows(IllegalArgumentException.class, () -> pack(inputArray, unpackLength2));
        double[] outputArray2 = ArrayPacking.unpackDoubles(bytes, unpackLength2);
        assertEquals(unpackLength2, outputArray2.length);
        assertArrayEquals(inputArray, Arrays.copyOf(outputArray2, inputLength));
        for (int i = inputLength; i < unpackLength2; i++) {
            assertEquals(0.0, outputArray2[i]);
        }
    }

    @Test
    public void testUnpackFloatWithLength() {
        int inputLength = 100;
        float[] inputArray = new float[inputLength];
        for (int i = 0; i < inputLength; i++) {
            inputArray[i] = rng.nextFloat();
        }
        byte[] bytes = ArrayPacking.pack(inputArray);

        int unpackLength1 = 25;
        float[] outputArray1 = unpackFloats(bytes, unpackLength1);
        assertEquals(unpackLength1, outputArray1.length);
        assertArrayEquals(Arrays.copyOf(inputArray, unpackLength1), outputArray1);

        int unpackLength2 = 123;
        float[] outputArray2 = unpackFloats(bytes, unpackLength2);
        assertEquals(unpackLength2, outputArray2.length);
        assertArrayEquals(inputArray, Arrays.copyOf(outputArray2, inputLength));
        for (int i = inputLength; i < unpackLength2; i++) {
            assertEquals(0.0, outputArray2[i]);
        }
    }

    @Test
    public void testConfig() {
        byte[] array = new byte[1];
        assertThrows(IllegalArgumentException.class, () -> unpackFloats(array, 1));
        assertThrows(IllegalArgumentException.class, () -> unpackDoubles(array, 1));

        byte[] newArray = new byte[Double.BYTES];
        assertDoesNotThrow(() -> unpackDoubles(newArray, 1));
        assertDoesNotThrow(() -> unpackFloats(newArray, 1));
        assertThrows(IllegalArgumentException.class, () -> unpackFloats(newArray, -1));
        assertThrows(IllegalArgumentException.class, () -> unpackDoubles(newArray, -1));
    }

    /**
     * The invariant: moveAndPack == pack(gather(map, provider), compress), byte for
     * byte.
     */
    private static void assertEquivalent(int[] map, int size, boolean compress, IntUnaryOperator provider) {
        int[] gathered = new int[size];
        for (int i = 0; i < size; i++) {
            gathered[i] = provider.applyAsInt(map[i]);
        }
        int[] expected = ArrayPacking.pack(gathered, size, compress);
        int[] actual = ArrayPacking.moveAndPack(map, size, compress, provider);
        assertArrayEquals(expected, actual, () -> "size=" + size + " compress=" + compress + "\n exp="
                + Arrays.toString(expected) + "\n act=" + Arrays.toString(actual));
    }

    // ---- branch: !compress and size < 3 both take the copyOf path ----

    @ParameterizedTest
    @ValueSource(ints = { 0, 1, 2, 3, 5, 17, 256, 1000 })
    void identityProvider_matchesPack_bothModes(int size) {
        int[] map = shuffledIdentity(size, 42);
        assertEquivalent(map, size, true, m -> m); // compressed
        assertEquivalent(map, size, false, m -> m); // stored (copyOf path)
    }

    // ---- branch: base == 1 (all values equal) — the all-ones child-bit case ----

    @ParameterizedTest
    @ValueSource(ints = { 3, 4, 255, 256, 999 })
    void constantProvider_hitsBaseOnePath(int size) {
        int[] map = shuffledIdentity(size, 7);
        assertEquivalent(map, size, true, m -> 1); // every element 1 -> {1,1,size}
        assertEquivalent(map, size, true, m -> 0);
        assertEquivalent(map, size, true, m -> -13); // negative constant, base still 1
    }

    // ---- branch: real mapper-shaped providers (child bit, cut dimension) ----

    @ParameterizedTest
    @MethodSource("mapperShapedCases")
    void mapperShapedProviders_matchPack(int size, int capacity, IntUnaryOperator provider) {
        int[] map = shuffledIdentity(Math.max(size, capacity), 99);
        assertEquivalent(map, size, true, provider);
    }

    static Stream<Arguments> mapperShapedCases() {
        Random r = new Random(1234);
        // childBit: 1 if node's (fake) child index < capacity else 0 — mixed 0/1 ->
        // base 2
        int[] leftIdx = IntStream.range(0, 4096).map(i -> r.nextInt(8192)).toArray();
        int cap = 4096;
        IntUnaryOperator childBit = m -> (leftIdx[m] < cap) ? 1 : 0;
        // cutDimension: small non-negative range -> narrow base
        IntUnaryOperator cutDim = m -> m % 100;
        return Stream.of(Arguments.of(300, cap, childBit), Arguments.of(1000, cap, childBit),
                Arguments.of(300, cap, cutDim), Arguments.of(4096, cap, cutDim));
    }

    // ---- size < map.length: the real call passes new int[capacity] but packs only
    // `size` ----

    @Test
    void packsOnlyFirstSize_ignoresTail() {
        int capacity = 512;
        int size = 300;
        int[] map = shuffledIdentity(capacity, 5); // length capacity, BFS count 300
        assertEquivalent(map, size, true, m -> (m * 31) & 0xFF);
    }

    // ---- randomized differential fuzz across value ranges (drives varied
    // base/packNum) ----

    @Test
    void randomizedDifferential() {
        Random r = new Random(20240607L);
        for (int trial = 0; trial < 2000; trial++) {
            int size = r.nextInt(600);
            int[] map = shuffledIdentity(size, r.nextLong());
            int span = 1 << (1 + r.nextInt(20)); // 2 .. ~1M -> exercises many packNum values
            int lo = r.nextInt(2001) - 1000;
            IntUnaryOperator provider = m -> lo + (Math.abs(m * 2654435761L % span) == 0 ? 0 : (int) (m % span));
            boolean compress = r.nextBoolean();
            assertEquivalent(map, size, compress, provider);
        }
    }

    /**
     * map of the given length whose entries are a shuffle of [0,length): a valid
     * node-id permutation.
     */
    private static int[] shuffledIdentity(int length, long seed) {
        int[] a = IntStream.range(0, length).toArray();
        Random r = new Random(seed);
        for (int i = length - 1; i > 0; i--) {
            int j = r.nextInt(i + 1);
            int t = a[i];
            a[i] = a[j];
            a[j] = t;
        }
        return a;
    }
}
