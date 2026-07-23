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
 * This file has been substantially modified from the version in original repo
 * which had the following notice.
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

package org.streamingalgorithms.randomcutforest.store;

import static java.lang.Math.max;
import static java.lang.Math.min;
import static org.streamingalgorithms.randomcutforest.CommonUtils.checkArgument;
import static org.streamingalgorithms.randomcutforest.CommonUtils.checkNotNull;
import static org.streamingalgorithms.randomcutforest.CommonUtils.checkState;
import static org.streamingalgorithms.randomcutforest.CommonUtils.toFloatArray;
import static org.streamingalgorithms.randomcutforest.summarization.Summarizer.iterativeClustering;
import static org.streamingalgorithms.randomcutforest.util.ArrayEncoder.moveAndPackLive;

import java.util.*;
import java.util.function.BiFunction;

import org.streamingalgorithms.randomcutforest.summarization.ICluster;
import org.streamingalgorithms.randomcutforest.summarization.MultiCenter;
import org.streamingalgorithms.randomcutforest.tree.Column;
import org.streamingalgorithms.randomcutforest.tree.VectorSupport;
import org.streamingalgorithms.randomcutforest.util.ArrayEncoder;
import org.streamingalgorithms.randomcutforest.util.ArrayUtils;
import org.streamingalgorithms.randomcutforest.util.Weighted;

import it.unimi.dsi.fastutil.ints.Int2IntOpenHashMap;
import lombok.Getter;

/**
 * Single concrete point store. Collapses the former width-specialized
 * {@code PointStoreSmall} (char[] locationList) and {@code PointStoreLarge}
 * (int[] locationList) into one class whose per-index location slot is a single
 * {@link Column}. Tier (BYTE / CHAR / INT) is chosen once by
 * {@link Column#tierFor(long, boolean)} from the true semantic bound of a
 * location value, and the same ladder is used on construction and on
 * deserialize so the tier can never diverge between build and reload.
 *
 * <p>
 * A location slot goes empty -> filled -> freed, so the sentinel IS reserved
 * (reserveSentinel = true): a live value can never collide with the empty
 * marker regardless of tier.
 */
public class PointStore implements IPointStore<Integer, float[]> {

    public static int INFEASIBLE_POINTSTORE_INDEX = -1;

    public static int INFEASIBLE_LOCN = (int) -1;
    /**
     * an index manager to manage free locations. Deliberately NOT columnized: its
     * arrays are interval-count-sized (tiny) and it sits on the hot insert path;
     * columnizing would add virtual dispatch for no memory win.
     */
    protected IndexIntervalManager indexManager;
    /**
     * generic store class
     */
    protected float[] store;
    /**
     * generic internal shingle, note that input is doubles
     */
    protected float[] internalShingle;
    /**
     * enable rotation of shingles; use a cyclic buffer instead of sliding window
     */
    @Getter
    boolean rotationEnabled;
    /**
     * last seen timestamp for internal shingling
     */
    @Getter
    protected long nextSequenceIndex;

    /**
     * refCount[i] counts of the number of times this point is being used across
     * different trees (a point can be used multiple times by the same tree -- for
     * example when duplicate points keep showing up
     */
    protected byte[] refCount;

    private final Int2IntOpenHashMap refCountMap;

    /**
     * per-index location slot. Stored value is {@code location / baseDimension};
     * {@link #getLocation(int)} multiplies back. Empty slots hold
     * {@code location.sentinel()}. Length (index capacity) grows via
     * {@link Column#extend(int)} independently of the value bound.
     */
    protected final Column location;

    /**
     * first location where new data can be safely copied;
     */
    @Getter
    int startOfFreeSegment;
    /**
     * overall dimension of the point (after shingling)
     */
    int dimensions;
    /**
     * shingle size, if known. Setting shingle size = 1 rules out overlapping
     */
    @Getter
    int shingleSize;
    /**
     * number of original dimensions which are shingled to produce and overall point
     * dimensions = shingleSize * baseDimensions. However there is a possibility
     * that even though the data is shingled, we may not choose to use the
     * overlapping (say for out of order updates).
     */
    int baseDimension;

    /**
     * maximum capacity
     */
    @Getter
    int capacity;
    /**
     * current capacity of store (number of shingled points)
     */
    @Getter
    int currentStoreCapacity;

    /**
     * enabling internal shingling
     */
    @Getter
    boolean internalShinglingEnabled;

    // ------------------------------------------------------------------
    // location column: bound, tier-from-legacy factory, and slot helpers
    // ------------------------------------------------------------------

    /**
     * Semantic upper bound of a stored location value (a quotient
     * {@code location / baseDimension}). The store array can grow to
     * {@code maxCapacity * dimensions} where {@code maxCapacity} is
     * {@code (rotationEnabled ? 2 : 1) * capacity} (see {@link #resizeStore()}), so
     * the largest quotient is {@code (rotationEnabled ? 2 : 1) * capacity *
     * shingleSize - 1}. Shared by constructor AND mapper so Column.tierFor picks
     * the same tier on build and reload. Long math: (rot?2:1)*cap*shingle can
     * exceed int.
     */
    public static long locationBound(int capacity, int shingleSize, boolean rotationEnabled) {
        return (long) (rotationEnabled ? 2 : 1) * capacity * shingleSize - 1;
    }

    private long locationMaxValue() {
        return locationBound(capacity, shingleSize, rotationEnabled);
    }

    /**
     * A slot is live iff its stored value is not the sentinel. This replaces the
     * old {@code compact()} liveness test, which relied on a numeric-range
     * coincidence that happened to catch Large's {@code -baseDimension} via
     * {@code >= 0} and Small's {@code baseDimension*65535} via the upper bound —
     * two different halves of one check. Now explicit and tier-independent.
     */
    boolean isFeasible(int index) {
        return location.get(index) != location.sentinel();
    }

    void setInfeasiblePointstoreLocationIndex(int index) {
        location.set(index, location.sentinel());
    }

    void extendLocationList(int newCapacity) {
        checkArgument(newCapacity > 0, "cannot be 0 or negative");
        location.extend(newCapacity); // fills the new tail with sentinel
    }

    void setLocation(int index, int loc) {
        int quotient = loc / baseDimension;
        location.set(index, quotient);
        assert baseDimension * quotient == loc : "location not aligned to baseDimension: " + loc;
    }

    int getLocation(int index) {
        checkArgument(index >= 0 && index < locationListLength(), " index not supported by store");
        return baseDimension * location.get(index);
    }

    int locationListLength() {
        return location.length();
    }

    /**
     * Decrement the reference count for the given index.
     *
     * @param index The index value.
     * @throws IllegalArgumentException if the index value is not valid.
     * @throws IllegalArgumentException if the current reference count for this
     *                                  index is non positive.
     */
    @Override
    public int decrementRefCount(int index) {
        checkArgument(index >= 0 && index < locationListLength(), " index not supported by store");
        checkArgument((refCount[index] & 0xff) > 0, " cannot decrement index");
        int value = refCountMap.remove(index); // 0 if absent (drv), else the stored overflow (>=1)
        if (value == 0) { // was absent
            if ((refCount[index] & 0xff) == 1) {
                indexManager.releaseIndex(index);
                refCount[index] = (byte) 0;
                setInfeasiblePointstoreLocationIndex(index);
                return 0;
            } else {
                int newVal = ((refCount[index] & 0xff) - 1);
                refCount[index] = (byte) newVal;
                return newVal;
            }
        } else { // overflow entry existed
            if (value > 1) {
                refCountMap.put(index, value - 1);
            }
            return value - 1 + (refCount[index] & 0xff);
        }
    }

    /**
     * takes an index from the index manager and rezises if necessary also adjusts
     * refCount size to have increment/decrement be seamless
     *
     * @return an index from the index manager
     */
    int takeIndex() {
        if (indexManager.isEmpty()) {
            if (indexManager.getCapacity() < capacity) {
                int oldCapacity = indexManager.getCapacity();
                int newCapacity = min(capacity, 1 + (int) Math.floor(1.1 * oldCapacity));
                indexManager.extendCapacity(newCapacity);
                refCount = Arrays.copyOf(refCount, newCapacity);
                extendLocationList(newCapacity);
            } else {
                throw new IllegalStateException(" index manager in point store is full ");
            }
        }
        return indexManager.takeIndex();
    }

    protected int getAmountToWrite(float[] tempPoint) {
        if (checkShingleAlignment(startOfFreeSegment, tempPoint)) {
            if (!rotationEnabled
                    || startOfFreeSegment % dimensions == (nextSequenceIndex - 1) * baseDimension % dimensions) {
                return baseDimension;
            }
        } else if (!rotationEnabled) {
            return dimensions;

        }
        // the following adds the padding for what exists;
        // then the padding for the new part; all mod (dimensions)
        // note that the expression is baseDimension when the condition
        // startOfFreeSegment % dimensions == (nextSequenceIndex-1)*baseDimension %
        // dimension is met
        return dimensions + (dimensions - startOfFreeSegment % dimensions
                + (int) ((nextSequenceIndex) * baseDimension) % dimensions) % dimensions;
    }

    /**
     * Add a point to the point store and return the index of the stored point.
     *
     * @param point       The point being added to the store.
     * @param sequenceNum sequence number of the point
     * @return the index value of the stored point.
     * @throws IllegalArgumentException if the length of the point does not match
     *                                  the point store's dimensions.
     * @throws IllegalStateException    if the point store is full.
     */
    public int add(double[] point, long sequenceNum) {
        return add(toFloatArray(point), sequenceNum, false);
    }

    public Integer add(float[] point, long sequenceNum, boolean updateShingleOnly) {
        checkArgument(internalShinglingEnabled || point.length == dimensions,
                "point.length must be equal to dimensions");
        checkArgument(!internalShinglingEnabled || point.length == baseDimension,
                "point.length must be equal to dimensions");

        float[] tempPoint = point;
        nextSequenceIndex++;
        if (internalShinglingEnabled) {
            // rotation is supported via the output and input is unchanged
            tempPoint = constructShingleInPlace(internalShingle, point, false);
            if (nextSequenceIndex < shingleSize || updateShingleOnly) {
                return INFEASIBLE_POINTSTORE_INDEX;
            }
        }
        int nextIndex;

        int amountToWrite = getAmountToWrite(tempPoint);

        if (startOfFreeSegment > currentStoreCapacity * dimensions - amountToWrite) {
            // try compaction and then resizing
            compact();
            // the compaction can change the array contents
            amountToWrite = getAmountToWrite(tempPoint);
            if (startOfFreeSegment > currentStoreCapacity * dimensions - amountToWrite) {
                resizeStore();
                checkState(startOfFreeSegment + amountToWrite <= currentStoreCapacity * dimensions, "out of space");
            }
        }

        nextIndex = takeIndex();

        setLocation(nextIndex, startOfFreeSegment - dimensions + amountToWrite);
        if (amountToWrite <= dimensions) {
            copyPoint(tempPoint, dimensions - amountToWrite, startOfFreeSegment, amountToWrite);
        } else {
            copyPoint(tempPoint, 0, startOfFreeSegment + amountToWrite - dimensions, dimensions);
        }
        startOfFreeSegment += amountToWrite;

        refCount[nextIndex] = 1;
        return nextIndex;
    }

    /**
     * Increment the reference count for the given index. This operation assumes
     * that there is currently a point stored at the given index and will throw an
     * exception if that's not the case.
     *
     * @param index The index value.
     * @throws IllegalArgumentException if the index value is not valid.
     * @throws IllegalArgumentException if the current reference count for this
     *                                  index is non positive.
     */
    public int incrementRefCount(int index) {
        checkArgument(index >= 0 && index < locationListLength(), " index not supported by store");
        checkArgument((refCount[index] & 0xff) > 0, " not in use");
        int value = refCountMap.remove(index); // 0 if absent
        if (value == 0) { // was absent
            if ((refCount[index] & 0xff) == 255) {
                refCountMap.put(index, 1);
                return 256;
            } else {
                int newVal = ((refCount[index] & 0xff) + 1);
                refCount[index] = (byte) newVal;
                return newVal;
            }
        } else { // overflow existed
            refCountMap.put(index, value + 1);
            return (refCount[index] & 0xff) + value + 1;
        }
    }

    @Override
    public int getDimensions() {
        return dimensions;
    }

    /**
     * capacity of the indices
     */
    public int getIndexCapacity() {
        return indexManager.getCapacity();
    }

    /**
     * used for mappers
     *
     * @return the store that stores the values
     */
    public float[] getStore() {
        return store;
    }

    /**
     * used for mapper
     *
     * @return the array of counts referring to different points
     */

    public int[] getPackedRefCount(boolean compress) {
        return ArrayEncoder.pack(refCount, refCount.length, compress);
    }

    public int[] getRefCount() {
        int[] newarray = new int[refCount.length];
        for (int i = 0; i < refCount.length; i++) {
            newarray[i] = (refCount[i] & 0xff) + refCountMap.get(i); // primitive get, 0 if absent
        }
        return newarray;
    }

    /**
     * used to obtain the most recent shingle seen so far in case of internal
     * shingling
     *
     * @return for internal shingling, returns the last seen shingle
     */
    public float[] getInternalShingle() {
        checkState(internalShinglingEnabled, "internal shingling is not enabled");
        return copyShingle();
    }

    void alignBoundaries(int initial, int freshStart) {
        int locn = freshStart;
        for (int i = 0; i < initial; i++) {
            store[locn] = 0;
            ++locn;
        }

    }

    // Now that both widths share one Column, packing gathers straight from it.
    public int[] getPackedLocation(boolean compress) {
        return moveAndPackLive(refCount, location.length(), compress, i -> location.get(i)); // raw already
    }

    /**
     * The following function eliminates redundant information that builds up in the
     * point store and shrinks the point store
     */
    public void compact() {
        int live = 0;
        int len = locationListLength();
        for (int i = 0; i < len; i++) {
            if (isFeasible(i)) {
                live++;
            }
        }
        long[] ref = new long[live];
        int n = 0;
        for (int i = 0; i < len; i++) {
            if (isFeasible(i)) {
                int locn = getLocation(i);
                ref[n++] = ((long) locn << 32) | (i & 0xFFFFFFFFL);
            }
        }
        Arrays.sort(ref);

        int freshStart = 0;
        int jStatic = 0;
        int jDynamic = 0;
        int jEnd = live;
        while (jStatic < jEnd) {
            int blockStart = (int) (ref[jStatic] >>> 32);
            int blockEnd = blockStart + dimensions;
            int initial = 0;
            if (rotationEnabled) {
                initial = (dimensions - freshStart + blockStart) % dimensions;
            }
            int k = jStatic + 1;
            jDynamic = jStatic + 1;
            while (k < jEnd) {
                int newElem = (int) (ref[k] >>> 32);
                if (blockEnd >= newElem) {
                    k += 1;
                    jDynamic += 1;
                    blockEnd = max(blockEnd, newElem + dimensions);
                } else {
                    k = jEnd;
                }
            }

            alignBoundaries(initial, freshStart);
            freshStart += initial;

            int start = freshStart;
            for (int i = blockStart; i < blockEnd; i++) {
                assert (!rotationEnabled || freshStart % dimensions == i % dimensions);

                if (jStatic < jEnd) {
                    int locn = (int) (ref[jStatic] >>> 32);
                    if (i == locn) {
                        int newIdx = (int) ref[jStatic];
                        setLocation(newIdx, freshStart);
                        jStatic += 1;
                    }
                }
                freshStart += 1;
            }
            copyTo(start, blockStart, blockEnd - blockStart);

            if (jStatic != jDynamic) {
                throw new IllegalStateException("There is discepancy in indices");
            }
        }
        startOfFreeSegment = freshStart;
    }

    /**
     * returns the number of copies of a point
     *
     * @param i index of a point
     * @return number of copies of the point managed by the store
     */
    public int getRefCount(int i) {
        return (refCount[i] & 0xff) + refCountMap.get(i); // 0 if absent
    }

    @Override
    public boolean isInternalRotationEnabled() {
        return rotationEnabled;
    }

    /**
     * @return the number of indices stored
     */
    public int size() {
        int count = 0;
        int len = location.length();
        for (int i = 0; i < len; i++) {
            if (location.get(i) != location.sentinel()) {
                ++count;
            }
        }
        return count;
    }

    public int[] getLocationList() {
        int len = location.length();
        int[] answer = new int[len];
        for (int i = 0; i < len; i++) {
            answer[i] = location.get(i);
        }
        return answer;
    }

    public int[] getPackedDuplicates(boolean compress) {
        int n = refCountMap.size();
        if (n == 0) {
            return ArrayEncoder.pack(new int[0], 0, compress);
        }
        // collect keys, sort ascending -> matches the old peel loop's index order,
        // so the packed bytes are byte-identical to the pre-refactor duplicateRefs.
        int[] keys = refCountMap.keySet().toIntArray(); // fastutil primitive extract
        Arrays.sort(keys);
        int[] dup = new int[2 * n];
        int k = 0;
        for (int idx : keys) {
            dup[k++] = idx;
            dup[k++] = refCountMap.get(idx); // primitive get, excess for this index
        }
        return ArrayEncoder.pack(dup, 2 * n, compress);
    }

    /**
     * transforms a point to a shingled point if internal shingling is turned on
     *
     * @param point new input values
     * @return shingled point
     */
    @Override
    public float[] transformToShingledPoint(float[] point) {
        checkNotNull(point, "point must not be null");
        if (internalShinglingEnabled && point.length == baseDimension) {
            return constructShingleInPlace(copyShingle(), point, rotationEnabled);
        }
        return ArrayUtils.cleanCopy(point);
    }

    private float[] copyShingle() {
        if (!rotationEnabled) {
            return Arrays.copyOf(internalShingle, dimensions);
        } else {
            float[] answer = new float[dimensions];
            int offset = (int) (nextSequenceIndex * baseDimension);
            for (int i = 0; i < dimensions; i++) {
                answer[(offset + i) % dimensions] = internalShingle[i];
            }
            return answer;
        }
    }

    /**
     * the following function is used to update the shingle in place; it can be used
     * to produce new copies as well
     *
     * @param target the array containing the shingled point
     * @param point  the new values
     * @return the array which now contains the updated shingle
     */
    protected float[] constructShingleInPlace(float[] target, float[] point, boolean rotationEnabled) {
        if (!rotationEnabled) {
            for (int i = 0; i < dimensions - baseDimension; i++) {
                target[i] = target[i + baseDimension];
            }
            for (int i = 0; i < baseDimension; i++) {
                target[dimensions - baseDimension + i] = (point[i] == 0.0) ? 0.0f : point[i];
            }
        } else {
            int offset = ((int) (nextSequenceIndex * baseDimension) % dimensions);
            for (int i = 0; i < baseDimension; i++) {
                target[offset + i] = (point[i] == 0.0) ? 0.0f : point[i];
            }
        }
        return target;
    }

    /**
     * Zero-alloc shingle transform. Fills caller-owned {@code output} (length ==
     * dimensions) with the shingled point for {@code input}, without allocating on
     * the linear path and without advancing internalShingle.
     *
     * Non-destructive (a peek, not a commit): unlike add(), it does not roll the
     * internal shingle forward, so it must be called for the point being scored
     * BEFORE that point is committed via add() — same ordering as the old
     * array-returning version.
     */
    @Override
    public void transformToShingledPoint(float[] input, float[] output) {
        checkNotNull(input, "input must not be null");
        checkArgument(output != null && output.length == dimensions,
                () -> "output must be preallocated to dimensions, currently " + output.length + ", need " + dimensions);

        if (internalShinglingEnabled && input.length == baseDimension) {
            if (!rotationEnabled) {
                // shift old shingle left by baseDimension into output (src != dst, no
                // overlap)...
                System.arraycopy(internalShingle, baseDimension, output, 0, dimensions - baseDimension);
                // ...then append the normalized new base block at the tail
                final int tail = dimensions - baseDimension;
                for (int i = 0; i < baseDimension; i++) {
                    float v = input[i];
                    output[tail + i] = (v == 0.0f) ? 0.0f : v; // canonicalize -0.0 -> +0.0
                }
            } else {
                // rotated: allowed to allocate a scratch, then land it in output
                float[] rotated = constructShingleInPlace(copyShingle(), input, true);
                System.arraycopy(rotated, 0, output, 0, dimensions);
            }
            return;
        }

        // not internally shingling (input already full-dimension) or external shingle:
        // normalized copy into output, still no allocation
        checkArgument(input.length == dimensions, "non-shingled input must equal dimensions");
        for (int i = 0; i < dimensions; i++) {
            float v = input[i];
            output[i] = (v == 0.0f) ? 0.0f : v;
        }
    }

    /**
     * for extrapolation and imputation, in presence of internal shingling we need
     * to update the list of missing values from the space of the input dimensions
     * to the shingled dimensions
     *
     * @param indexList list of missing values in the input point
     * @return list of missing values in the shingled point
     */
    @Override
    public int[] transformIndices(int[] indexList) {
        checkArgument(internalShinglingEnabled, " only allowed for internal shingling");
        checkArgument(indexList.length <= baseDimension, " incorrect length");
        int[] results = Arrays.copyOf(indexList, indexList.length);
        if (!rotationEnabled) {
            for (int i = 0; i < indexList.length; i++) {
                checkArgument(results[i] < baseDimension, "incorrect index");
                results[i] += dimensions - baseDimension;
            }
        } else {
            int offset = ((int) (nextSequenceIndex * baseDimension) % dimensions);
            for (int i = 0; i < indexList.length; i++) {
                checkArgument(results[i] < baseDimension, "incorrect index");
                results[i] = (results[i] + offset) % dimensions;
            }
        }
        return results;
    }

    /**
     * a builder
     */

    public static class Builder<T extends Builder<T>> {

        // We use Optional types for optional primitive fields when it doesn't make
        // sense to use a constant default.

        protected int dimensions;
        protected int shingleSize = 1;
        protected int baseDimension;
        protected boolean internalRotationEnabled = false;
        protected boolean internalShinglingEnabled = false;
        protected int capacity;
        protected Optional<Integer> initialPointStoreSize = Optional.empty();
        protected int currentStoreCapacity = 0;
        protected int indexCapacity = 0;
        protected float[] store = null;
        protected double[] knownShingle = null;
        protected Column location = null;
        protected int[] locationList = null;
        protected byte[] refCountBytes = null;
        protected Int2IntOpenHashMap overflow = null;
        protected long nextTimeStamp = 0;
        protected int startOfFreeSegment = 0;
        protected int[] refCount = null;

        // dimension of the points being stored
        public T dimensions(int dimensions) {
            this.dimensions = dimensions;
            return (T) this;
        }

        // maximum number of points in the store
        public T capacity(int capacity) {
            this.capacity = capacity;
            return (T) this;
        }

        // initial size of the pointstore, dynamicResizing must be on
        // and value cannot exceed capacity
        public T initialSize(int initialPointStoreSize) {
            this.initialPointStoreSize = Optional.of(initialPointStoreSize);
            return (T) this;
        }

        // shingleSize for opportunistic compression
        public T shingleSize(int shingleSize) {
            this.shingleSize = shingleSize;
            return (T) this;
        }

        // is internal shingling enabled
        public T internalShinglingEnabled(boolean internalShinglingEnabled) {
            this.internalShinglingEnabled = internalShinglingEnabled;
            return (T) this;
        }

        // are shingles rotated
        public T internalRotationEnabled(boolean internalRotationEnabled) {
            this.internalRotationEnabled = internalRotationEnabled;
            return (T) this;
        }

        @Deprecated
        public T directLocationEnabled(boolean value) {
            return (T) this;
        }

        @Deprecated
        public T dynamicResizingEnabled(boolean value) {
            return (T) this;
        }

        // the size of the array storing the specific points
        // this is used for serialization
        public T currentStoreCapacity(int currentStoreCapacity) {
            this.currentStoreCapacity = currentStoreCapacity;
            return (T) this;
        }

        // the size of the pointset being tracked
        // this is used for serialization
        public T indexCapacity(int indexCapacity) {
            this.indexCapacity = indexCapacity;
            return (T) this;
        }

        // last known shingle, if internalshingle is on
        // this shingle is not rotated
        // this is used for serialization
        public T knownShingle(double[] knownShingle) {
            this.knownShingle = knownShingle;
            return (T) this;
        }

        // count of the points being tracked
        // used for serialization
        public T refCountBytes(byte[] refCountBytes) {
            this.refCountBytes = refCountBytes;
            return (T) this;
        }

        public T refCount(int[] refCount) {
            this.refCount = refCount;
            return (T) this;
        }

        public T overflow(Int2IntOpenHashMap overflow) {
            this.overflow = overflow;
            return (T) this;
        }

        // location of the points being tracked, if not directmapped
        // used for serialization
        public T location(Column location) {
            this.location = location;
            return (T) this;
        }

        public T locationList(int[] locationList) {
            this.locationList = locationList;
            return (T) this;
        }

        public T store(float[] store) {
            this.store = store;
            return (T) this;
        }

        // location of where points can be written
        // used for serialization
        public T startOfFreeSegment(int startOfFreeSegment) {
            this.startOfFreeSegment = startOfFreeSegment;
            return (T) this;
        }

        // the next timeStamp to accept
        // used for serialization
        public T nextTimeStamp(long nextTimeStamp) {
            this.nextTimeStamp = nextTimeStamp;
            return (T) this;
        }

        // No more Small/Large routing: one concrete PointStore, one location Column,
        // tier picked internally from the true value bound.
        public PointStore build() {
            return new PointStore(this);
        }
    }

    public PointStore(PointStore.Builder builder) {
        checkArgument(builder.dimensions > 0, "dimensions must be greater than 0");
        checkArgument(builder.capacity > 0, "capacity must be greater than 0");
        checkArgument(builder.shingleSize == 1 || builder.dimensions == builder.shingleSize
                || builder.dimensions % builder.shingleSize == 0, "incorrect use of shingle size");
        /**
         * the following checks are due to mappers (kept for future)
         */
        if (builder.refCountBytes != null || builder.location != null || builder.knownShingle != null
                || builder.refCount != null || builder.locationList != null) {
            checkArgument(builder.refCountBytes != null || builder.refCount != null, "reference count must be present");
            checkArgument(builder.location != null || builder.locationList != null, "location column must be present");
            int length = (builder.refCountBytes != null) ? builder.refCountBytes.length
                    : (builder.refCount != null) ? builder.refCount.length : 0;
            checkArgument(length == builder.indexCapacity, "incorrect reference count length");
            int locLength = (builder.location != null) ? builder.location.length()
                    : (builder.locationList != null) ? builder.locationList.length : 0;
            checkArgument(locLength == builder.indexCapacity, " incorrect length of locations");
            checkArgument(
                    builder.knownShingle == null
                            || builder.internalShinglingEnabled && builder.knownShingle.length == builder.dimensions,
                    "incorrect shingling information");
        }

        this.shingleSize = builder.shingleSize;
        this.dimensions = builder.dimensions;
        this.internalShinglingEnabled = builder.internalShinglingEnabled;
        this.rotationEnabled = builder.internalRotationEnabled;
        this.baseDimension = this.dimensions / this.shingleSize;
        this.capacity = builder.capacity;

        if (builder.refCountBytes == null && builder.refCount == null) {
            int size = (int) builder.initialPointStoreSize.orElse(builder.capacity);
            currentStoreCapacity = size;
            this.indexManager = new IndexIntervalManager(size);
            startOfFreeSegment = 0;
            refCount = new byte[size];
            if (internalShinglingEnabled) {
                nextSequenceIndex = 0;
                internalShingle = new float[dimensions];
            }
            store = new float[currentStoreCapacity * dimensions];
            this.refCountMap = new Int2IntOpenHashMap();
            this.refCountMap.defaultReturnValue(0);
            // fresh column: length == index capacity, all slots sentinel
            this.location = Column.of(currentStoreCapacity, locationMaxValue(), true);
        } else {
            if (builder.overflow == null) {
                this.refCountMap = new Int2IntOpenHashMap();
                this.refCountMap.defaultReturnValue(0);
            } else {
                this.refCountMap = builder.overflow;
                this.refCountMap.defaultReturnValue(0);
            }
            if (builder.refCount == null) {
                this.refCount = builder.refCountBytes;
            } else {
                this.refCount = new byte[builder.refCount.length];
                for (int i = 0; i < builder.refCount.length; i++) {
                    if (builder.refCount[i] > 255) {
                        this.refCount[i] = (byte) 255;
                        this.refCountMap.put(i, builder.refCount[i] - 255);
                    } else {
                        this.refCount[i] = (byte) (builder.refCount[i]);
                    }
                }
            }
            this.startOfFreeSegment = builder.startOfFreeSegment;
            this.nextSequenceIndex = builder.nextTimeStamp;
            this.currentStoreCapacity = builder.currentStoreCapacity;
            if (internalShinglingEnabled) {
                this.internalShingle = (builder.knownShingle != null)
                        ? Arrays.copyOf(toFloatArray(builder.knownShingle), dimensions)
                        : new float[dimensions];
            }
            indexManager = new IndexIntervalManager(this.refCount, builder.indexCapacity);
            store = (builder.store == null) ? new float[currentStoreCapacity * dimensions] : builder.store;
            this.location = (builder.location != null) ? builder.location
                    : columnFrom(builder.locationList, locationMaxValue(), this.refCount);

        }
    }

    /**
     * Build the location column from a legacy int[] locationList on deserialize.
     * <p>
     * Legacy empties are encoded as {@code -1} (old Large) or {@code 65535} (old
     * Small, char sentinel widened to int).
     */
    protected static Column columnFrom(int[] legacyRaw, long maxValue, byte[] refCount) {
        Column col = Column.of(legacyRaw.length, maxValue, true);
        for (int i = 0; i < legacyRaw.length; i++) {
            if ((refCount[i] & 0xff) != 0) {
                // real quotient; < sentinel by construction, so col.set fits.
                col.set(i, legacyRaw[i]);
            }
            // dead slot: Column.of already filled it with sentinel; leaving it
            // there is exactly the normalization we want (legacy -1 and 65535
            // both become the new tier's sentinel).
        }
        return col;
    }

    void resizeStore() {
        int maxCapacity = (rotationEnabled) ? 2 * capacity : capacity;
        int newCapacity = (int) Math.floor(min(1.1 * currentStoreCapacity, maxCapacity));
        if (newCapacity > currentStoreCapacity) {
            float[] newStore = new float[newCapacity * dimensions];
            System.arraycopy(store, 0, newStore, 0, currentStoreCapacity * dimensions);
            currentStoreCapacity = newCapacity;
            store = newStore;
        }
    }

    boolean checkShingleAlignment(int location, float[] point) {
        boolean test = (location - dimensions + baseDimension >= 0);
        for (int i = 0; i < dimensions - baseDimension && test; i++) {
            test = (((float) point[i]) == store[location - dimensions + baseDimension + i]);
        }
        return test;
    }

    void copyPoint(float[] point, int src, int location, int length) {
        for (int i = 0; i < length; i++) {
            store[location + i] = point[src + i];
        }
    }

    protected void checkFeasible(int index) {
        checkArgument(isFeasible(index), " invalid point");
    }

    /**
     * Get a copy of the point at the given index.
     *
     * @param index An index value corresponding to a storage location in this point
     *              store.
     * @return a copy of the point stored at the given index.
     * @throws IllegalArgumentException if the index value is not valid.
     * @throws IllegalArgumentException if the current reference count for this
     *                                  index is nonpositive.
     */
    @Override
    public float[] getNumericVector(int index) {
        float[] answer = new float[dimensions];
        getNumericVectorInto(index, answer);
        return answer;
    }

    @Override
    public void getNumericVectorInto(int index, float[] answer) {
        getNumericVectorInto(index, answer, 0);
    }

    void getNumericVectorInto(int index, float[] answer, int offset) {
        int address = getLocation(index);
        checkFeasible(index);
        checkArgument(answer != null && answer.length == dimensions, "incorrect array for 0-alloc");

        if (!rotationEnabled) {
            System.arraycopy(store, address, answer, offset, dimensions);
        } else {
            // clearly every cell
            for (int i = 0; i < dimensions; i++) {
                answer[(address + i) % dimensions + offset] = store[address + i];
            }
        }
    }

    @Override
    public void setAsSlice(int index, float[] values, int offset) {
        int address = getLocation(index);
        checkFeasible(index);
        if (!rotationEnabled) {
            VectorSupport.expandInto(store, address, values, offset, dimensions);
        } else {
            VectorSupport.expandInto(getNumericVector(index), 0, values, offset, dimensions);
        }
    }

    double addToLinearSlice(float[] values, int offset, int[] array, int start, int finish) {
        final int dim = dimensions;
        for (int k = start; k < finish; k++)
            array[k] = getLocation(array[k]);

        return VectorSupport.updateBoundsAll(values, offset, dim, store, array, start, finish);
    }

    double addToRotatedSlice(float[] values, int offset, int[] array, int start, int finish) {
        double val = 0;
        for (int i = start; i < finish; i++) { // allocates
            val = VectorSupport.addPointInPlace(values, offset, dimensions, getNumericVector(array[i]), 0);
        }
        return val;
    }

    public double addToSlice(float[] values, int offset, int[] array, int start, int finish) {
        if (!rotationEnabled) {
            return addToLinearSlice(values, offset, array, start, finish);
        } else {
            return addToRotatedSlice(values, offset, array, start, finish);
        }
    }

    @Override
    // return the rangesum
    public double addToSlice(int index, float[] boxLocation, int offset) {
        int address = getLocation(index);
        checkFeasible(index);
        if (!rotationEnabled) {
            return VectorSupport.addPointInPlace(boxLocation, offset, dimensions, store, address);
        } else {
            return VectorSupport.addPointInPlace(boxLocation, offset, dimensions, getNumericVector(index), 0);
        }
    }

    public double addToSliceWithGap(float[] values, int offset, int[] array, int start, int finish, float[] exp,
            int expOff, float[] gapOut, int gapOff, double[] out) {
        checkState(!rotationEnabled, "fused path is linear-only");
        for (int k = start; k < finish; k++)
            array[k] = getLocation(array[k]);
        return VectorSupport.updateBoundsAndGap(values, offset, dimensions, store, array, start, finish, exp, expOff,
                gapOut, gapOff, out);
    }

    @Override
    public boolean isEqual(int index, float[] candidate) {
        int address = getLocation(index);
        checkFeasible(index);
        checkArgument(candidate != null && candidate.length == dimensions, "incorrect array for 0-alloc");

        if (!rotationEnabled) {
            return Arrays.equals(store, address, address + dimensions, candidate, 0, dimensions);
        } else {
            // clearly every cell
            for (int i = 0; i < dimensions; i++) {
                if (candidate[(address + i) % dimensions] != store[address + i]) {
                    return false;
                }
            }
            return true;
        }
    }

    @Override
    public float valueAt(int index, int coord) {
        checkArgument(coord < dimensions, "incorrect coord in valueAt");
        int address = getLocation(index);
        checkFeasible(index);

        if (!rotationEnabled) {
            return store[address + coord];
        } else {
            int currentColumn = address % dimensions;
            int offset = (coord - currentColumn + dimensions) % dimensions;
            int targetIndex = address + offset;
            return store[targetIndex];
        }
    }

    @Override
    public boolean leftOf(int index, float cutValue, int dim) {
        return valueAt(index, dim) <= cutValue;
    }

    public String toString(int index) {
        return Arrays.toString(getNumericVector(index));
    }

    void copyTo(int dest, int source, int length) {
        if (dest < source) {
            for (int i = 0; i < length; i++) {
                store[dest + i] = store[source + i];
            }
        }
    }

    public static Builder builder() {
        return new Builder();
    }

    /**
     * a function that exposes an L1 clustering of the points stored in pointstore
     *
     * @param maxAllowed              the maximum number of clusters one is
     *                                interested in
     * @param shrinkage               a parameter used in CURE algorithm that can
     *                                produce a combination of behaviors (=1
     *                                corresponds to centroid clustering, =0
     *                                resembles robust Minimum Spanning Tree)
     * @param numberOfRepresentatives another parameter used to control the
     *                                plausible (potentially non-spherical) shapes
     *                                of the clusters
     * @param separationRatio         a parameter that controls how aggressively we
     *                                go below maxAllowed -- this is often set to a
     *                                DEFAULT_SEPARATION_RATIO_FOR_MERGE
     * @param previous                a (possibly null) list of previous clusters
     *                                which can be used to seed the current clusters
     *                                to ensure some smoothness
     * @return a list of clusters
     */

    public List<ICluster<float[]>> summarize(int maxAllowed, double shrinkage, int numberOfRepresentatives,
            double separationRatio, BiFunction<float[], float[], Double> distance, List<ICluster<float[]>> previous) {
        int[] counts = getRefCount();
        ArrayList<Weighted<Integer>> refs = new ArrayList<>();
        for (int i = 0; i < counts.length; i++) {
            if (counts[i] != 0) {
                refs.add(new Weighted<>(i, (float) counts[i]));
            }
        }
        BiFunction<float[], Float, ICluster<float[]>> clusterInitializer = (a, b) -> MultiCenter.initialize(a, b,
                shrinkage, numberOfRepresentatives);
        return iterativeClustering(maxAllowed, 4 * maxAllowed, 1, refs, this::getNumericVector, distance,
                clusterInitializer, 42, false, true, separationRatio, previous);
    }

}
