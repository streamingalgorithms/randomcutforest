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

package org.streamingalgorithms.randomcutforest.state.tree;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

import java.util.Random;
import java.util.stream.IntStream;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.streamingalgorithms.randomcutforest.util.ArrayPacking;

class NodeStoreMapperTest {

    private static void assertEquivalent(int[] map, float[] source, int size) {
        float[] gathered = new float[size];
        for (int i = 0; i < size; i++) {
            gathered[i] = source[map[i]];
        }
        byte[] expected = ArrayPacking.pack(gathered); // reference marshal
        byte[] actual = NodeStoreMapper.moveAndPack(map, source, size);
        assertArrayEquals(expected, actual);
    }

    @ParameterizedTest
    @ValueSource(ints = { 0, 1, 2, 3, 255, 256, 1000 })
    void matchesPackFloat(int size) {
        Random r = new Random(size + 1);
        float[] source = randomFloats(Math.max(size, 1), r);
        int[] map = shuffledIdentity(source.length, size * 7L);
        assertEquivalent(map, source, size);
    }

    @Test
    void preservesSpecialBitPatterns() {
        // raw-bit fidelity: NaN, +/-Inf, -0.0 must survive the gather+marshal unchanged
        float[] source = { Float.NaN, Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY, -0.0f, 0.0f, Float.MIN_VALUE,
                Float.MAX_VALUE, 1.5f };
        int[] map = { 7, 3, 0, 4, 1, 6, 2, 5 }; // arbitrary reorder
        assertEquivalent(map, source, source.length);
    }

    @Test
    void packsOnlyFirstSize() {
        float[] source = randomFloats(512, new Random(9));
        int[] map = shuffledIdentity(512, 123);
        assertEquivalent(map, source, 300); // map length 512, marshal only 300
    }

    private static float[] randomFloats(int n, Random r) {
        float[] a = new float[n];
        for (int i = 0; i < n; i++) {
            a[i] = Float.intBitsToFloat(r.nextInt());
        }
        return a;
    }

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
