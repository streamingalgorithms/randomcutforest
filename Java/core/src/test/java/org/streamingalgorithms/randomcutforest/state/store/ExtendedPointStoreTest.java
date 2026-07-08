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

package org.streamingalgorithms.randomcutforest.state.store;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

import java.util.Arrays;
import java.util.Random;
import java.util.function.IntUnaryOperator;

import org.junit.jupiter.api.Test;
import org.streamingalgorithms.randomcutforest.RandomCutForest;
import org.streamingalgorithms.randomcutforest.store.PointStore;
import org.streamingalgorithms.randomcutforest.util.ArrayPacking;

/**
 * Standalone, dependency-free test proving the streaming ("sequential")
 * location scatter reconstructs the identical locationList[] as the original
 * indexed loop, across the axes that actually broke things:
 *
 * - occupancy: sparse (many holes) and dense (few holes) - compression: on and
 * off - location range: small (<= 65535, Small-store regime) and large (>
 * 65535, the regime the 0xFFFF mask corrupted) - packNum: forced to 1 (large
 * range) and > 1 (small range, tight base)
 *
 * It re-implements the pack / unpack / moveAndPackLive / unpackIntsSequential
 * logic verbatim from ArrayPacking so the algorithms are tested in isolation
 * from the PointStore machinery. If this passes, the two scatters are
 * equivalent and the streaming version is safe to keep. If it fails, the
 * failing case + first index is printed.
 *
 * INTENT: this is a checked-in-quality algorithm test. The four copied helpers
 * (pack, unpackInts, moveAndPackLive, unpackIntsSequential) MUST stay
 * byte-for-byte identical to ArrayPacking; if ArrayPacking changes, mirror it
 * here or replace these with direct calls once this runs on the RCF classpath.
 */
public class ExtendedPointStoreTest {

    // ------------------------------------------------------------------
    // Verbatim copies of the ArrayPacking algorithms under test.
    // ------------------------------------------------------------------

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

    /** ArrayPacking.pack(int[], int, boolean) */
    static int[] pack(int[] inputArray, int length, boolean compress) {
        if (length < 0 || length > inputArray.length)
            throw new IllegalArgumentException("length");
        if (!compress || length < 3) {
            return Arrays.copyOf(inputArray, length);
        }
        int min = inputArray[0], max = inputArray[0];
        for (int i = 1; i < length; i++) {
            min = Math.min(min, inputArray[i]);
            max = Math.max(max, inputArray[i]);
        }
        long base = (long) max - min + 1;
        if (base == 1) {
            return new int[] { min, max, length };
        }
        int packNum = logMax(base);
        int[] output = new int[3 + (int) Math.ceil(1.0 * length / packNum)];
        output[0] = min;
        output[1] = max;
        output[2] = length;
        int len = 0, used = 0;
        while (len < length) {
            long code = 0;
            int reach = Math.min(len + packNum - 1, length - 1);
            for (int i = reach; i >= len; i--) {
                code = base * code + (inputArray[i] - min);
            }
            output[3 + used++] = (int) code;
            len += packNum;
        }
        return output;
    }

    /** ArrayPacking.unpackInts(int[], int, boolean) -> new int[length] */
    static int[] unpackInts(int[] packedArray, int length, boolean decompress) {
        if (length < 0)
            throw new IllegalArgumentException("length");
        if (packedArray.length < 3 || !decompress) {
            return Arrays.copyOf(packedArray, length);
        }
        int min = packedArray[0], max = packedArray[1];
        int[] output = new int[length];
        if (min == max) {
            if (packedArray[2] >= length) {
                Arrays.fill(output, min);
            } else {
                for (int i = 0; i < packedArray[2]; i++)
                    output[i] = min;
            }
        } else {
            long base = ((long) max - min + 1);
            int packNum = logMax(base);
            int count = 0;
            for (int i = 3; i < packedArray.length; i++) {
                long code = packedArray[i];
                for (int j = 0; j < packNum && count < Math.min(packedArray[2], length); j++) {
                    output[count++] = (int) (min + code % base);
                    code = (int) (code / base);
                }
            }
        }
        return output;
    }

    /** ArrayPacking.unpackInts(int[], boolean) -> new int[stored length] */
    static int[] unpackInts(int[] packedArray, boolean decompress) {
        if (!decompress)
            return Arrays.copyOf(packedArray, packedArray.length);
        return (packedArray.length < 3) ? unpackInts(packedArray, packedArray.length, decompress)
                : unpackInts(packedArray, packedArray[2], decompress);
    }

    /** ArrayPacking.moveAndPackLive(byte[], int, boolean, IntUnaryOperator) */
    static int[] moveAndPackLive(byte[] refCountBytes, int len, boolean compress, IntUnaryOperator valueOf) {
        int size = 0;
        for (int i = 0; i < len; i++)
            if (refCountBytes[i] != 0)
                size++;
        if (!compress || size < 3) {
            int[] out = new int[size];
            int k = 0;
            for (int i = 0; i < len; i++)
                if (refCountBytes[i] != 0)
                    out[k++] = valueOf.applyAsInt(i);
            return out;
        }
        int min = Integer.MAX_VALUE, max = Integer.MIN_VALUE;
        for (int i = 0; i < len; i++)
            if (refCountBytes[i] != 0) {
                int v = valueOf.applyAsInt(i);
                if (v < min)
                    min = v;
                if (v > max)
                    max = v;
            }
        long base = (long) max - min + 1;
        if (base == 1)
            return new int[] { min, max, size };
        int packNum = logMax(base);
        int[] output = new int[3 + (int) Math.ceil(1.0 * size / packNum)];
        output[0] = min;
        output[1] = max;
        output[2] = size;
        int used = 0, emitted = 0;
        int i = 0;
        while (emitted < size) {
            long code = 0;
            int[] window = new int[Math.min(packNum, size - emitted)];
            int w = 0;
            while (w < window.length) {
                if (refCountBytes[i] != 0)
                    window[w++] = valueOf.applyAsInt(i);
                i++;
            }
            for (int r = window.length - 1; r >= 0; r--)
                code = base * code + (window[r] - min);
            output[3 + used++] = (int) code;
            emitted += window.length;
        }
        return output;
    }

    @FunctionalInterface
    interface IntSink {
        void accept(int value);
    }

    /** ArrayPacking.unpackIntsSequential(int[], int, boolean, IntSink) */
    static void unpackIntsSequential(int[] packed, int length, boolean decompress, IntSink sink) {
        if (length < 0)
            throw new IllegalArgumentException("length");
        if (packed.length < 3 || !decompress) {
            int n = Math.min(length, packed.length);
            for (int i = 0; i < n; i++)
                sink.accept(packed[i]);
            return;
        }
        int min = packed[0], max = packed[1];
        if (min == max) {
            int n = Math.min(packed[2], length);
            for (int i = 0; i < n; i++)
                sink.accept(min);
            return;
        }
        long base = ((long) max - min + 1);
        int packNum = logMax(base);
        int count = 0;
        for (int i = 3; i < packed.length && count < Math.min(packed[2], length); i++) {
            long code = packed[i];
            for (int j = 0; j < packNum && count < Math.min(packed[2], length); j++) {
                sink.accept((int) (min + code % base));
                code = (code / base);
                count++;
            }
        }
    }

    // ------------------------------------------------------------------
    // The two reconstruction paths under comparison.
    // Both take: the packed location array (as produced by getPackedLocation),
    // the occupancy byte[] refCountBytes, and indexCapacity.
    // Both return the full-length locationList[] with INFEASIBLE (-1) in holes.
    // ------------------------------------------------------------------

    static final int INFEASIBLE = -1;

    /**
     * ORIGINAL: decode whole dense array, index by nextLocation. Allocates
     * int[size].
     */
    static int[] scatterIndexed(int[] packedLoc, byte[] refCountBytes, int indexCapacity, boolean compressed) {
        int[] locationList = new int[indexCapacity];
        Arrays.fill(locationList, INFEASIBLE);
        int[] tempList = unpackInts(packedLoc, compressed);
        int nextLocation = 0;
        for (int i = 0; i < indexCapacity; i++) {
            if (refCountBytes[i] != 0) {
                locationList[i] = tempList[nextLocation++];
            }
        }
        return locationList;
    }

    /**
     * SEQUENTIAL: stream decode straight into occupied slots. No int[size]
     * materialized.
     */
    static int[] scatterSequential(int[] packedLoc, byte[] refCountBytes, int indexCapacity, boolean compressed) {
        int[] locationList = new int[indexCapacity];
        Arrays.fill(locationList, INFEASIBLE);
        final int[] cursor = { 0 };
        final int[] placed = { 0 };
        unpackIntsSequential(packedLoc, indexCapacity, compressed, v -> {
            int i = cursor[0];
            while (i < indexCapacity && refCountBytes[i] == 0)
                i++; // bounded
            if (i >= indexCapacity)
                throw new IllegalStateException("emitted more locations than occupied slots");
            locationList[i] = v;
            cursor[0] = i + 1;
            placed[0]++;
        });
        // count invariant: exactly one location per occupied slot
        int occupied = 0;
        for (int i = 0; i < indexCapacity; i++)
            if (refCountBytes[i] != 0)
                occupied++;
        if (placed[0] != occupied) {
            throw new IllegalStateException("placed " + placed[0] + " != occupied " + occupied);
        }
        return locationList;
    }

    // ------------------------------------------------------------------
    // Test scaffolding.
    // ------------------------------------------------------------------

    static int failures = 0;
    static int cases = 0;

    /**
     * Build one random store scenario and assert both scatters agree, AND that both
     * round-trip the original locations (getPackedLocation -> scatter == original).
     *
     * @param indexCapacity full index space
     * @param occupancyPct  fraction of indices that are live
     * @param maxLocation   locations drawn from [0, maxLocation); >65535 exercises
     *                      the Large regime
     * @param compressed    pack/unpack compression flag
     */
    static void runCase(String name, int indexCapacity, double occupancyPct, int maxLocation, boolean compressed,
            long seed) {
        cases++;
        Random rnd = new Random(seed);

        // occupancy: refCountBytes[i] != 0 for live indices. Use counts 1..2 (like
        // maxRef=2 in the field).
        byte[] refCountBytes = new byte[indexCapacity];
        // the "true" location stored at each live index (this is what
        // getLocationList()[i] would return)
        int[] trueLocation = new int[indexCapacity];
        Arrays.fill(trueLocation, INFEASIBLE);

        for (int i = 0; i < indexCapacity; i++) {
            if (rnd.nextDouble() < occupancyPct) {
                refCountBytes[i] = (byte) (1 + rnd.nextInt(2)); // 1 or 2, nonzero
                trueLocation[i] = rnd.nextInt(maxLocation); // real location value
            }
            // else refCountBytes[i]=0, trueLocation[i]=INFEASIBLE
        }

        // WRITE side: getPackedLocation == moveAndPackLive gating on refCountBytes,
        // reading trueLocation.
        // (This mirrors PointStoreLarge.getPackedLocation with the CORRECT lambda i ->
        // locationList[i],
        // i.e. NO 0xFFFF mask.)
        int[] packedLoc = moveAndPackLive(refCountBytes, indexCapacity, compressed, i -> trueLocation[i]);

        // Reference: what getLocationList() would look like after a correct round trip
        // is exactly
        // trueLocation for live indices, INFEASIBLE elsewhere. Build the "expected"
        // reconstruction.
        int[] expected = new int[indexCapacity];
        Arrays.fill(expected, INFEASIBLE);
        for (int i = 0; i < indexCapacity; i++)
            if (refCountBytes[i] != 0)
                expected[i] = trueLocation[i];

        // Two reconstructions.
        int[] viaIndexed, viaSequential;
        try {
            viaIndexed = scatterIndexed(packedLoc, refCountBytes, indexCapacity, compressed);
        } catch (RuntimeException e) {
            fail(name, "indexed scatter threw: " + e.getMessage());
            return;
        }
        try {
            viaSequential = scatterSequential(packedLoc, refCountBytes, indexCapacity, compressed);
        } catch (RuntimeException e) {
            fail(name, "sequential scatter threw: " + e.getMessage());
            return;
        }

        // Assertion 1: indexed reconstructs the expected locations.
        int d1 = firstDiff(expected, viaIndexed);
        if (d1 >= 0) {
            fail(name, "INDEXED != expected @" + d1 + " expected=" + expected[d1] + " got=" + viaIndexed[d1]
                    + " refByte=" + (refCountBytes[d1] & 0xff));
            return;
        }
        // Assertion 2: sequential reconstructs the expected locations.
        int d2 = firstDiff(expected, viaSequential);
        if (d2 >= 0) {
            fail(name, "SEQUENTIAL != expected @" + d2 + " expected=" + expected[d2] + " got=" + viaSequential[d2]
                    + " refByte=" + (refCountBytes[d2] & 0xff));
            return;
        }
        // Assertion 3: indexed == sequential (they must agree with each other too).
        int d3 = firstDiff(viaIndexed, viaSequential);
        if (d3 >= 0) {
            fail(name,
                    "INDEXED != SEQUENTIAL @" + d3 + " indexed=" + viaIndexed[d3] + " sequential=" + viaSequential[d3]);
            return;
        }

        // report the shape so we know the case actually exercised what we think
        int live = 0, maxLoc = 0;
        for (int i = 0; i < indexCapacity; i++)
            if (refCountBytes[i] != 0) {
                live++;
                maxLoc = Math.max(maxLoc, trueLocation[i]);
            }
        System.out.printf("PASS  %-34s cap=%-7d live=%-7d maxLoc=%-7d packLen=%-7d compressed=%s%n", name,
                indexCapacity, live, maxLoc, packedLoc.length, compressed);
    }

    static int firstDiff(int[] a, int[] b) {
        if (a.length != b.length)
            return Math.min(a.length, b.length); // length mismatch reported at min
        for (int i = 0; i < a.length; i++)
            if (a[i] != b[i])
                return i;
        return -1;
    }

    static void fail(String name, String msg) {
        failures++;
        System.out.println("FAIL  " + name + " : " + msg);
    }

    public static void main(String[] args) {
        System.out.println("Scatter equivalence: original indexed loop vs streaming sequential scatter\n");

        // --- The regime that actually broke: Large store, locations > 65535, ~50%
        // occupancy ---
        runCase("large_50pct_bigloc_comp", 400_000, 0.50, 300_000, true, 1);
        runCase("large_50pct_bigloc_nocomp", 400_000, 0.50, 300_000, false, 2);

        // --- Small-store regime: locations <= 65535 ---
        runCase("small_50pct_smallloc_comp", 60_000, 0.50, 65_535, true, 3);
        runCase("small_50pct_smallloc_nocomp", 60_000, 0.50, 65_535, false, 4);

        // --- Sparse occupancy (few live, many holes) ---
        runCase("sparse_5pct_bigloc_comp", 400_000, 0.05, 300_000, true, 5);
        runCase("sparse_5pct_bigloc_nocomp", 400_000, 0.05, 300_000, false, 6);

        // --- Dense occupancy (almost all live) ---
        runCase("dense_95pct_bigloc_comp", 400_000, 0.95, 300_000, true, 7);
        runCase("dense_99pct_bigloc_nocomp", 400_000, 0.99, 300_000, false, 8);

        // --- Fully dense (every index live) : stresses moveAndPackLive window at the
        // array end ---
        runCase("full_100pct_bigloc_comp", 50_000, 1.00, 300_000, true, 9);
        runCase("full_100pct_smallloc_comp", 50_000, 1.00, 65_535, true, 10);

        // --- Tight location range -> forces packNum > 1 (the reversed-grouping path)
        // ---
        runCase("tightrange_50pct_comp", 100_000, 0.50, 64, true, 11); // base ~64 -> packNum > 1
        runCase("tightrange_dense_comp", 100_000, 0.90, 256, true, 12); // base ~256 -> packNum > 1

        // --- Tiny stores : size < 3 early-return paths in pack/moveAndPackLive ---
        runCase("tiny_size2_comp", 10, 0.20, 300_000, true, 13);
        runCase("tiny_size1_comp", 10, 0.10, 300_000, true, 14);
        runCase("tiny_size0_comp", 5, 0.00, 300_000, true, 15);

        // --- All-same-location : min==max degenerate pack path ---
        runCaseConstant("const_loc_50pct_comp", 100_000, 0.50, 12345, true, 16);
        runCaseConstant("const_loc_50pct_nocomp", 100_000, 0.50, 12345, false, 17);

        System.out.println();
        System.out.printf("%d cases, %d failures%n", cases, failures);
        if (failures == 0) {
            System.out.println("RESULT: sequential scatter is equivalent to the indexed loop across all cases.");
        } else {
            System.out.println("RESULT: DIVERGENCE FOUND — see FAIL lines above.");
        }
        System.exit(failures == 0 ? 0 : 1);
    }

    /**
     * Variant where every live index holds the SAME location value (min==max pack
     * path).
     */
    static void runCaseConstant(String name, int indexCapacity, double occupancyPct, int constLoc, boolean compressed,
            long seed) {
        cases++;
        Random rnd = new Random(seed);
        byte[] refCountBytes = new byte[indexCapacity];
        int[] trueLocation = new int[indexCapacity];
        Arrays.fill(trueLocation, INFEASIBLE);
        for (int i = 0; i < indexCapacity; i++) {
            if (rnd.nextDouble() < occupancyPct) {
                refCountBytes[i] = (byte) (1 + rnd.nextInt(2));
                trueLocation[i] = constLoc;
            }
        }
        int[] packedLoc = moveAndPackLive(refCountBytes, indexCapacity, compressed, i -> trueLocation[i]);
        int[] expected = new int[indexCapacity];
        Arrays.fill(expected, INFEASIBLE);
        for (int i = 0; i < indexCapacity; i++)
            if (refCountBytes[i] != 0)
                expected[i] = trueLocation[i];

        int[] viaIndexed, viaSequential;
        try {
            viaIndexed = scatterIndexed(packedLoc, refCountBytes, indexCapacity, compressed);
        } catch (RuntimeException e) {
            fail(name, "indexed threw: " + e.getMessage());
            return;
        }
        try {
            viaSequential = scatterSequential(packedLoc, refCountBytes, indexCapacity, compressed);
        } catch (RuntimeException e) {
            fail(name, "sequential threw: " + e.getMessage());
            return;
        }

        int d1 = firstDiff(expected, viaIndexed);
        if (d1 >= 0) {
            fail(name, "INDEXED != expected @" + d1);
            return;
        }
        int d2 = firstDiff(expected, viaSequential);
        if (d2 >= 0) {
            fail(name, "SEQUENTIAL != expected @" + d2);
            return;
        }
        int d3 = firstDiff(viaIndexed, viaSequential);
        if (d3 >= 0) {
            fail(name, "INDEXED != SEQUENTIAL @" + d3);
            return;
        }

        int live = 0;
        for (int i = 0; i < indexCapacity; i++)
            if (refCountBytes[i] != 0)
                live++;
        System.out.printf("PASS  %-34s cap=%-7d live=%-7d maxLoc=%-7d packLen=%-7d compressed=%s%n", name,
                indexCapacity, live, constLoc, packedLoc.length, compressed);
    }

    @Test
    void stressTest() {
        int shingleSize = 1;
        int baseDimension = 1;
        int dimensions = shingleSize * baseDimension;
        RandomCutForest forest = RandomCutForest.builder().dimensions(dimensions).numberOfTrees(5)
                .shingleSize(shingleSize).internalShinglingEnabled(true).sampleSize(1000).randomSeed(0)
                .boundingBoxCacheFraction(1.0).build();
        float[] input = new float[baseDimension];
        Random v = new Random(7);
        for (int j = 0; j < 100000; j++) {
            for (int i = 0; i < baseDimension; i++) {
                input[i] = v.nextInt(200);
            }
            forest.update(input);
        }
        PointStore store = (PointStore) forest.getUpdateCoordinator().getStore();
        int[] refCounts = store.getRefCount();
        int[] tempList = store.getLocationList();
        int[] locationList = new int[store.getIndexCapacity()];
        int size = 0;
        int treecount = 0;
        for (int i = 0; i < refCounts.length; i++) {
            if (refCounts[i] > 0) {
                locationList[size++] = tempList[i];
            }
            if (refCounts[i] > forest.getNumberOfTrees()) {
                treecount++;
            }
        }
        int over255 = 0, maxRef = 0;
        for (int i = 0; i < refCounts.length; i++) {
            if (refCounts[i] > 255)
                over255++;
            if (refCounts[i] > maxRef)
                maxRef = refCounts[i];
        }
        System.out.println("over 255 " + over255 + " maxRef " + maxRef);
        System.out.println("dedup size " + size);
        System.out.println("more than trees " + treecount);
        System.out.println("index capacity " + store.getIndexCapacity());
        int[] old = ArrayPacking.pack(locationList, size, true);
        int[] array = store.getPackedLocation(true);
        assertArrayEquals(old, array);

    }
}
