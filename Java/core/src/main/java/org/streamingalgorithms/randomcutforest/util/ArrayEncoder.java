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

import static org.streamingalgorithms.randomcutforest.CommonUtils.checkArgument;
import static org.streamingalgorithms.randomcutforest.CommonUtils.checkNotNull;

import java.nio.ByteBuffer;
import java.util.function.IntUnaryOperator;

import it.unimi.dsi.fastutil.ints.Int2IntOpenHashMap;

/**
 * Encode/decode cores for the RCF arithmetic array codec.
 *
 * Fast paths, all PROVABLY byte-identical to the general arithmetic path: -
 * base == 1 (min==max) : every value is min. - base == 2^k (power of two) :
 * {@literal %base == &(base-1), /base == >>>k, } code stays non-negative (lt 30
 * bits, bit31 clear). base==2 is the common case (0/1 masks) and additionally
 * gets a direct bit codec (bitEncode/bitDecode) for callers that KNOW their
 * input is 0/1 and want to skip the min/max scan entirely. - packNum == 1
 * (large base) : one value per word, no modular arithmetic.
 *
 * The power-of-2 fast path is DECODE-SIDE ONLY and keyed on the base read from
 * the header, so it applies to already-serialized data with no format change
 * and no wasted bits: the encoder never rounds the base. base==2^k is detected,
 * not forced.
 *
 */

public final class ArrayEncoder {

    private ArrayEncoder() {
    }

    @FunctionalInterface
    public interface IntSink {
        void accept(int value);
    }

    @FunctionalInterface
    public interface IntIndexConsumer {
        void accept(int index, int value);
    }

    /**
     * {@literal Bits per code word for base==2. logMax(2)==30; named so the non-negativity}
     * {@literal proof (bit31 clear => &1==%2, >>>1==/2) is visible at the call site.}
     */

    public static final int BIT_PACK_NUM = 30;

    // logMax: smallest p such that base^(p+1) >= Integer.MAX_VALUE, min 1.
    static int logMax(long base) {
        if (base <= 1)
            throw new IllegalArgumentException("base must be > 1");
        int pack = 0;
        long num = base;
        while (num < Integer.MAX_VALUE) {
            num = num * base;
            ++pack;
        }
        return Math.max(pack, 1);
    }

    public static int unpackedLength(int[] packed, boolean decompress) {
        if (!decompress || packed.length < 3) {
            return packed.length; // uncompressed / too-short: full array
        }
        return packed[2]; // compressed header: element count
    }

    // ==================================================================
    // ENCODE
    // ==================================================================

    /**
     * Encode `size` values (source.applyAsInt(k) for k in [0,size)) to a packed
     * int[]. Byte-identical to the historical
     * ArrayPacking.pack(dense,size,compress). source is called twice (min/max, then
     * encode) and MUST be a pure read.
     */
    public static int[] encodeCore(int size, boolean compress, IntUnaryOperator source) {
        if (!compress || size < 3) {
            int[] out = new int[size];
            for (int k = 0; k < size; k++)
                out[k] = source.applyAsInt(k);
            return out;
        }
        int min = source.applyAsInt(0), max = min;
        for (int k = 1; k < size; k++) {
            int v = source.applyAsInt(k);
            if (v < min)
                min = v;
            if (v > max)
                max = v;
        }
        long base = (long) max - min + 1;
        if (base == 1) { // all equal
            return new int[] { min, max, size };
        }
        int packNum = logMax(base);
        int[] out = new int[3 + (int) Math.ceil(1.0 * size / packNum)];
        out[0] = min;
        out[1] = max;
        out[2] = size;

        if (packNum == 1) { // large base: one value per word
            final int mn = min;
            for (int k = 0; k < size; k++)
                out[3 + k] = source.applyAsInt(k) - mn;
            return out;
        }
        if ((base & (base - 1)) == 0) { // base is a power of two: shift encode
            int shift = Long.numberOfTrailingZeros(base);
            final int mn = min;
            int used = 0, len = 0;
            while (len < size) {
                int code = 0;
                int reach = Math.min(len + packNum - 1, size - 1);
                for (int i = reach; i >= len; i--)
                    code = (code << shift) | (source.applyAsInt(i) - mn);
                out[3 + used++] = code;
                len += packNum;
            }
            return out;
        }
        // general arithmetic encode
        int len = 0, used = 0;
        while (len < size) {
            long code = 0;
            int reach = Math.min(len + packNum - 1, size - 1);
            for (int i = reach; i >= len; i--)
                code = base * code + (source.applyAsInt(i) - min);
            out[3 + used++] = (int) code;
            len += packNum;
        }
        return out;
    }

    public static int[] pack(int[] inputArray, int length, boolean compress) {
        checkNotNull(inputArray, "inputArray must not be null");
        checkArgument(0 <= length && length <= inputArray.length,
                "length must be between 0 and inputArray.length (inclusive)");
        return ArrayEncoder.encodeCore(length, compress, k -> inputArray[k]);
    }

    public static int[] pack(byte[] inputArray, int length, boolean compress) {
        checkNotNull(inputArray, "inputArray must not be null");
        checkArgument(0 <= length && length <= inputArray.length,
                "length must be between 0 and inputArray.length (inclusive)");
        return ArrayEncoder.encodeCore(length, compress, k -> inputArray[k] & 0xFF);
    }

    /**
     * Gather-and-encode over a reorder map. valueOf is applied to map[i] — the
     * lambda receives the REORDERED index, matching the historical moveAndPack
     * convention. Do NOT index map[] inside the lambda; the indirection is here.
     */
    public static int[] moveAndPack(int[] map, int size, boolean compress, IntUnaryOperator valueOf) {
        checkNotNull(map, "map must not be null");
        checkNotNull(valueOf, "valueOf must not be null");
        checkArgument(0 <= size && size <= map.length, "size must be between 0 and map.length");
        return encodeCore(size, compress, k -> valueOf.applyAsInt(map[k]));
    }

    /**
     * Gather-and-encode over an implicit occupancy map: emit valueOf(i) for every i
     * in [0,len) with refCountBytes[i] != 0, ascending. size = #occupied.
     * Byte-identical to encodeCore over the dense live sequence.
     */
    public static int[] moveAndPackLive(byte[] refCountBytes, int len, boolean compress, IntUnaryOperator valueOf) {
        checkNotNull(refCountBytes, "refCountBytes must not be null");
        checkNotNull(valueOf, "valueOf must not be null");
        // Precompute the live-index list once so encodeCore's source(k) is O(1) and
        // called exactly twice per element (min/max, encode) — no repeated scans.
        int size = 0;
        for (int i = 0; i < len; i++)
            if (refCountBytes[i] != 0)
                size++;
        int[] live = new int[size];
        int k = 0;
        for (int i = 0; i < len; i++)
            if (refCountBytes[i] != 0)
                live[k++] = i;
        return encodeCore(size, compress, idx -> valueOf.applyAsInt(live[idx]));
    }

    // ---- base==2 direct bit codec (rule (a): hot 0/1 path, no min/max scan) ----

    /**
     * Encode a 0/1 array directly (source over [0,size)), skipping the min/max
     * scan. Byte-identical to encodeCore on a genuine two-value {0,1} array. Emits
     * header {0,1,size} + ceil(size/30) code words (high index -> high bit). For an
     * all-0 or all-1 array this still emits the {0,1,size} 2-value header (NOT
     * base==1); decode handles both, but see NOTE: pair bitEncode with bitDecode
     * consistently.
     */
    public static int[] bitEncode(int size, boolean compress, IntUnaryOperator source) {
        if (!compress || size < 3) {
            int[] out = new int[size];
            for (int k = 0; k < size; k++) {
                int v = source.applyAsInt(k);
                assert v == 0 || v == 1 : "bitEncode requires 0/1 input";
                out[k] = v;
            }
            return out;
        }
        int[] out = new int[3 + (size + BIT_PACK_NUM - 1) / BIT_PACK_NUM];
        out[0] = 0;
        out[1] = 1;
        out[2] = size;
        int used = 0, k = 0;
        while (k < size) {
            int code = 0;
            int reach = Math.min(k + BIT_PACK_NUM - 1, size - 1);
            for (int i = reach; i >= k; i--) {
                int v = source.applyAsInt(i);
                assert v == 0 || v == 1 : "bitEncode requires 0/1 input";
                code = (code << 1) | v;
            }
            out[3 + used++] = code;
            k += BIT_PACK_NUM;
        }
        return out;
    }

    /**
     * Gather bit-encode over a reorder map (moveAndPack convention:
     * valueOf(map[i])).
     */
    public static int[] bitEncode(int[] map, int size, boolean compress, IntUnaryOperator valueOf) {
        checkNotNull(map, "map must not be null");
        return bitEncode(size, compress, k -> valueOf.applyAsInt(map[k]));
    }

    /**
     * Gather-and-marshal floats to bytes over a reorder map. Byte-identical to
     * packing source[map[i]] with a big-endian ByteBuffer (same order as the float
     * pack path), without an intermediate float[].
     */
    public static byte[] moveAndPack(int[] map, float[] source, int size) {
        checkNotNull(map, "map must not be null");
        checkNotNull(source, "source must not be null");
        checkArgument(0 <= size && size <= map.length, "size must be between 0 and map.length");
        byte[] out = new byte[size * Float.BYTES];
        ByteBuffer buf = ByteBuffer.wrap(out);
        for (int i = 0; i < size; i++)
            buf.putFloat(source[map[i]]);
        return out;
    }

    // ==================================================================
    // DECODE
    // ==================================================================

    /**
     * Emits (index, value) for index in [0, lim), lim = min(stored count, limit).
     * Handles every shape a packed array can take: {@literal uncompressed/size<3}
     * {@literal  copy, base==1, base==2^k (shift), packNum==1, general.}
     * Byte-identical to the historical ArrayPacking.unpackInts. Every typed unpack
     * is a thin sink over this.
     */
    public static void decodeCore(int[] packed, int limit, boolean decompress, IntIndexConsumer sink) {
        if (packed.length < 3 || !decompress) { // raw copy — no header
            int n = Math.min(limit, packed.length);
            for (int i = 0; i < n; i++)
                sink.accept(i, packed[i]);
            return;
        }
        int min = packed[0], max = packed[1];
        int lim = Math.min(packed[2], limit);
        if (min == max) { // base==1
            for (int i = 0; i < lim; i++)
                sink.accept(i, min);
            return;
        }
        long base = (long) max - min + 1;
        int packNum = logMax(base);
        int count = 0;

        if (packNum == 1) { // large base: value = min + word
            for (int i = 3; i < packed.length && count < lim; i++, count++) {
                sink.accept(count, min + packed[i]);
            }
            return;
        }
        if ((base & (base - 1)) == 0) { // power of two: shift/mask decode
            int shift = Long.numberOfTrailingZeros(base);
            int mask = (int) (base - 1);
            for (int i = 3; i < packed.length && count < lim; i++) {
                int code = packed[i];
                for (int j = 0; j < packNum && count < lim; j++) {
                    sink.accept(count, min + (code & mask));
                    code >>>= shift;
                    count++;
                }
            }
            return;
        }
        for (int i = 3; i < packed.length && count < lim; i++) { // general arithmetic
            long code = packed[i];
            for (int j = 0; j < packNum && count < lim; j++) {
                sink.accept(count, (int) (min + code % base));
                code = code / base; // no (int) cast: code>=0, shrinks, in range
                count++;
            }
        }
    }

    /** Plain (unindexed) decode for callers that only need values in order. */
    public static void decodeCore(int[] packed, int limit, boolean decompress, IntSink sink) {
        decodeCore(packed, limit, decompress, (i, v) -> sink.accept(v));
    }

    /**
     * Direct bit decode for 0/1 (base==2) arrays.
     */
    public static void bitDecode(int[] packed, int limit, IntSink sink) {
        if (packed.length < 3) { // raw copy (size<3)
            int n = Math.min(limit, packed.length);
            for (int i = 0; i < n; i++)
                sink.accept(packed[i]);
            return;
        }
        int min = packed[0], max = packed[1];
        int lim = Math.min(packed[2], limit);
        if (min == max) { // base==1 (all-same)
            for (int i = 0; i < lim; i++)
                sink.accept(min);
            return;
        }
        int count = 0; // base==2 bits
        for (int i = 3; i < packed.length && count < lim; i++) {
            int code = packed[i];
            for (int j = 0; j < BIT_PACK_NUM && count < lim; j++) {
                sink.accept(min + (code & 1));
                code >>>= 1;
                count++;
            }
        }
    }

    public static int[] unpackInts(int[] packedArray, int length, boolean decompress) {
        checkNotNull(packedArray, " array unpacking invoked on null arrays");
        checkArgument(length >= 0, "incorrect length parameter");
        int[] out = new int[length];
        ArrayEncoder.decodeCore(packedArray, length, decompress, (i, v) -> out[i] = v);
        return out;
    }

    public static void unpackInts(int[] packedArray, int[] out, int length, boolean decompress) {
        checkNotNull(packedArray, " array unpacking invoked on null arrays");
        checkNotNull(out, "output cannot be null");
        checkArgument(length >= 0, "incorrect length parameter");
        checkArgument(out.length >= length, "output buffer too small");
        ArrayEncoder.decodeCore(packedArray, length, decompress, (i, v) -> out[i] = v);
    }

    public static void unpackInts(int[] packedArray, char[] out, int length, boolean decompress) {
        checkNotNull(packedArray, " array unpacking invoked on null arrays");
        checkNotNull(out, "output cannot be null");
        checkArgument(length >= 0, "incorrect length parameter");
        checkArgument(out.length >= length, "output buffer too small");
        ArrayEncoder.decodeCore(packedArray, length, decompress, (i, v) -> out[i] = (char) v);
    }

    public static void unpackInts(int[] packedArray, byte[] out, int length, boolean decompress) {
        checkNotNull(packedArray, " array unpacking invoked on null arrays");
        checkNotNull(out, "output cannot be null");
        checkArgument(length >= 0, "incorrect length parameter");
        checkArgument(out.length >= length, "output buffer too small");
        ArrayEncoder.decodeCore(packedArray, length, decompress, (i, v) -> out[i] = (byte) v);
    }

    public static Int2IntOpenHashMap unpackRefCounts(int[] packedArray, byte[] out, int length, boolean decompress) {
        checkNotNull(packedArray, " array unpacking invoked on null arrays");
        checkNotNull(out, "output cannot be null");
        checkArgument(length >= 0, "incorrect length parameter");
        checkArgument(out.length >= length, "output buffer too small");
        Int2IntOpenHashMap[] overflow = { null }; // holder: lambda can't reassign a captured local
        ArrayEncoder.decodeCore(packedArray, length, decompress, (i, v) -> {
            out[i] = (byte) Math.min(v, 255);
            if (v > 255) {
                if (overflow[0] == null) {
                    overflow[0] = new Int2IntOpenHashMap();
                    overflow[0].defaultReturnValue(0);
                }
                overflow[0].put(i, v - 255);
            }
        });
        return overflow[0];
    }

    // ==================================================================
    // PointStore and generic overflow maps
    // ==================================================================

    public static void mergeDuplicateRefs(int[] packedDup, boolean decompress, byte[] refCountBytes,
            Int2IntOpenHashMap overflow) {
        checkNotNull(packedDup, "duplicate array must not be null");
        // decodeCore emits decoded values in order; pair them (index, excess) via a
        // toggling sink. The dense count == packedDup[2] (or packedDup.length on the
        // copy path), always even for well-formed duplicates.
        int[] savedIdx = { 0 };
        boolean[] haveIdx = { false };
        ArrayEncoder.decodeCore(packedDup, Integer.MAX_VALUE, decompress, (i, v) -> {
            if (!haveIdx[0]) {
                savedIdx[0] = v;
                haveIdx[0] = true;
            } else {
                applyDuplicate(refCountBytes, overflow, savedIdx[0], v);
                haveIdx[0] = false;
            }
        });
        checkArgument(!haveIdx[0], " corrupt duplicates"); // odd count => a dangling index
    }

    private static void applyDuplicate(byte[] refCountBytes, Int2IntOpenHashMap overflow, int idx, int excess) {
        int trueCount = (refCountBytes[idx] & 0xff) + overflow.get(idx) + excess;
        refCountBytes[idx] = (byte) Math.min(trueCount, 255);
        if (trueCount > 255) {
            overflow.put(idx, trueCount - 255);
        }
        // trueCount can never fall <= 255 here (excess >= 1), so no stale-entry removal
        // needed
    }

    // ==================================================================
    // Floats -- just to have encoding in one place
    // ==================================================================

    public static byte[] pack(float[] array) {
        return pack(array, array.length);
    }

    public static byte[] pack(float[] array, int length) {
        checkArgument(0 <= length, "incorrect length parameter");
        checkArgument(length <= array.length, "length must be between 0 and inputArray.length (inclusive)");

        byte[] out = new byte[length * Float.BYTES];
        ByteBuffer.wrap(out).asFloatBuffer().put(array, 0, length); // bulk marshal, replaces putFloat() loop
        return out;
    }

    /**
     * Unpack an array of bytes as an array of floats.
     *
     * @param bytes An array of bytes.
     * @return an array of floats obtained by marshalling consecutive bytes in the
     *         input array into floats.
     */
    public static float[] unpackFloats(byte[] bytes) {
        checkNotNull(bytes, "bytes must not be null");
        return unpackFloats(bytes, bytes.length / Float.BYTES);
    }

    /**
     * Unpack an array of bytes as an array of floats.
     *
     * @param bytes  An array of bytes.
     * @param length The desired length of the resulting float array. The input will
     *               be truncated or padded with zeros as needed.
     * @return an array of doubles obtained by marshalling consecutive bytes in the
     *         input array into floats.
     */
    public static float[] unpackFloats(byte[] bytes, int length) {
        checkNotNull(bytes, "bytes must not be null");
        checkArgument(length >= 0, "length must be greater than or equal to 0");
        checkArgument(bytes.length % Float.BYTES == 0, "bytes.length must be divisible by Float.BYTES");
        float[] result = new float[length];
        int m = Math.min(length, bytes.length / Float.BYTES);
        ByteBuffer.wrap(bytes).asFloatBuffer().get(result, 0, m); // bulk marshal, replaces the getFloat() loop
        return result;
    }

}
