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

package org.streamingalgorithms.randomcutforest.util;

import static java.lang.Math.min;
import static org.streamingalgorithms.randomcutforest.CommonUtils.checkArgument;
import static org.streamingalgorithms.randomcutforest.CommonUtils.checkNotNull;

import java.util.Arrays;
import java.util.function.IntUnaryOperator;

public class ArrayPacking {

    /**
     * For a given base value, return the smallest int value {@code p} so that
     * {@code base^(p + 1) >= Integer.MAX_VALUE}. If
     * {@code base >= Integer.MAX_VALUE}, return 1.
     * 
     * @param base Compute the approximate log of {@code Integer.MAX_VALUE} in this
     *             base.
     * @return the largest int value {@code p} so that
     *         {@code base^p >= Integer.MAX_VALUE} or 1 if
     *         {@code base >= Integer.MAX_VALUE}.
     */
    public static int logMax(long base) {
        checkArgument(base > 1, "Absolute value of base must be greater than 1");

        int pack = 0;
        long num = base;
        while (num < Integer.MAX_VALUE) {
            num = num * base;
            ++pack;
        }
        return Math.max(pack, 1); // pack can be 0 for max - min being more than Integer.MaxValue
    }

    /**
     * Pack an array of ints. If {@code compress} is true, then this method will
     * apply arithmetic compression to the inputs, otherwise it returns a copy of
     * the input.
     *
     * @param inputArray An array of ints to pack.
     * @param compress   A flag indicating whether to apply arithmetic compression.
     * @return an array of packed ints.
     */
    public static int[] pack(int[] inputArray, boolean compress) {
        return pack(inputArray, inputArray.length, compress);
    }

    /**
     * Pack an array of ints. If {@code compress} is true, then this method will
     * apply arithmetic compression to the inputs, otherwise it returns a copy of
     * the input.
     *
     * @param inputArray An array of ints to pack.
     * @param length     The length of the output array. Only the first
     *                   {@code length} values in {@code inputArray} will be packed.
     * @param compress   A flag indicating whether to apply arithmetic compression.
     * @return an array of packed ints.
     */
    public static int[] pack(int[] inputArray, int length, boolean compress) {
        checkNotNull(inputArray, "inputArray must not be null");
        checkArgument(0 <= length && length <= inputArray.length,
                "length must be between 0 and inputArray.length (inclusive)");

        if (!compress || length < 3) {
            return Arrays.copyOf(inputArray, length);
        }

        int min = inputArray[0];
        int max = inputArray[0];
        for (int i = 1; i < length; i++) {
            min = min(min, inputArray[i]);
            max = Math.max(max, inputArray[i]);
        }
        long base = (long) max - min + 1;
        if (base == 1) {
            return new int[] { min, max, length };
        } else {
            int packNum = logMax(base);

            int[] output = new int[3 + (int) Math.ceil(1.0 * length / packNum)];
            output[0] = min;
            output[1] = max;
            output[2] = length;
            int len = 0;
            int used = 0;
            while (len < length) {
                long code = 0;
                int reach = min(len + packNum - 1, length - 1);
                for (int i = reach; i >= len; i--) {
                    code = base * code + (inputArray[i] - min);
                }
                output[3 + used++] = (int) code;
                len += packNum;
            }
            // uncomment for debug; should be always true
            // checkArgument(used + 3 == output.length, "incorrect state");
            return output;
        }
    }

    /**
     * Unpack an array previously created by {@link #pack(int[], int, boolean)}.
     * 
     * @param packedArray An array previously created by
     *                    {@link #pack(int[], int, boolean)}.
     * @param decompress  A flag indicating whether the packed array was created
     *                    with arithmetic compression enabled.
     * @return the array of unpacked ints.
     */
    public static int[] unpackInts(int[] packedArray, boolean decompress) {
        checkNotNull(packedArray, " array unpacking invoked on null arrays");

        if (!decompress) {
            return Arrays.copyOf(packedArray, packedArray.length);
        }

        return (packedArray.length < 3) ? unpackInts(packedArray, packedArray.length, decompress)
                : unpackInts(packedArray, packedArray[2], decompress);
    }

    /**
     * Unpack an array previously created by {@link #pack(int[], int, boolean)}.
     * 
     * @param packedArray An array previously created by
     *                    {@link #pack(int[], int, boolean)}.
     * @param length      The desired length of the output array. If this number is
     *                    different from the length of the array that was originally
     *                    packed, then the result will be truncated or padded with
     *                    zeros as needed.
     * @param decompress  A flag indicating whether the packed array was created
     *                    with arithmetic compression enabled.
     * @return the array of unpacked ints.
     */
    public static int[] unpackInts(int[] packedArray, int length, boolean decompress) {
        checkNotNull(packedArray, " array unpacking invoked on null arrays");
        checkArgument(length >= 0, "incorrect length parameter");

        if (packedArray.length < 3 || !decompress) {
            return Arrays.copyOf(packedArray, length);
        }
        int min = packedArray[0];
        int max = packedArray[1];
        int[] output = new int[length];
        if (min == max) {
            if (packedArray[2] >= length) {
                Arrays.fill(output, min);
            } else {
                for (int i = 0; i < packedArray[2]; i++) {
                    output[i] = min;
                }
            }
        } else {
            long base = ((long) max - min + 1);
            int packNum = logMax(base);
            int count = 0;
            for (int i = 3; i < packedArray.length; i++) {
                long code = packedArray[i];
                for (int j = 0; j < packNum && count < min(packedArray[2], length); j++) {
                    output[count++] = (int) (min + code % base);
                    code = (int) (code / base);
                }
            }
        }
        return output;
    }

    private static short[] copyToShort(int[] array, int length) {
        short[] ret = new short[length];
        for (int i = 0; i < Math.min(length, array.length); i++) {
            ret[i] = (short) array[i];
        }
        return ret;
    }

    /**
     * Gather-and-pack without the intermediate array. Equivalent to pack(g,
     * compress) where g[i] = valueOfNode.applyAsInt(map[i]) for i in [0,size).
     *
     * valueOfNode is called up to twice per element (min/max pass, then encode
     * pass), so it MUST be a pure, side-effect-free read of immutable state.
     */
    public static int[] moveAndPack(int[] map, int size, boolean compress, IntUnaryOperator valueOfNode) {
        checkNotNull(map, "map must not be null");
        checkNotNull(valueOfNode, "valueOfNode must not be null");
        checkArgument(0 <= size && size <= map.length, "size must be between 0 and map.length");

        if (!compress || size < 3) {
            int[] out = new int[size];
            for (int i = 0; i < size; i++) {
                out[i] = valueOfNode.applyAsInt(map[i]);
            }
            return out; // matches Arrays.copyOf(reordered, size)
        }

        int min = valueOfNode.applyAsInt(map[0]);
        int max = min;
        for (int i = 1; i < size; i++) {
            int v = valueOfNode.applyAsInt(map[i]);
            if (v < min)
                min = v;
            if (v > max)
                max = v;
        }
        long base = (long) max - min + 1;
        if (base == 1) {
            return new int[] { min, max, size };
        }
        int packNum = logMax(base);
        int[] output = new int[3 + (int) Math.ceil(1.0 * size / packNum)];
        output[0] = min;
        output[1] = max;
        output[2] = size;
        int len = 0, used = 0;
        while (len < size) {
            long code = 0;
            int reach = Math.min(len + packNum - 1, size - 1);
            for (int i = reach; i >= len; i--) {
                code = base * code + (valueOfNode.applyAsInt(map[i]) - min);
            }
            output[3 + used++] = (int) code;
            len += packNum;
        }
        return output;
    }
}
