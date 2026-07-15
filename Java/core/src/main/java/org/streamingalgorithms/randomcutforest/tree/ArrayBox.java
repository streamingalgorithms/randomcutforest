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

package org.streamingalgorithms.randomcutforest.tree;

import static java.lang.Math.max;
import static org.streamingalgorithms.randomcutforest.CommonUtils.checkArgument;

import java.util.Arrays;

import lombok.Getter;

/**
 * Single precision bounding box backed by ONE float[] viewed at an offset.
 *
 * The box owns the slice values[offset .. offset + 2*dimensions), encoded as:
 * values[offset + i] = maxValue(i) for i in [0, dimensions) values[offset + i +
 * dimensions] = -minValue(i) for i in [0, dimensions)
 *
 * `values` may be longer than 2*dimensions (a pooled / shared buffer); `offset`
 * says where this box lives inside it. Negating the min half makes the
 * per-coordinate gap of a query point a single branch-free expression: gap[i] =
 * max(0f, newValues[i] - values[offset + i]) with newValues[i] = point[i],
 * newValues[i+dimensions] = -point[i].
 */
public class ArrayBox implements IBoundingBoxView {

    /** Backing store. This box uses [offset, offset + 2*dimensions). */
    public final float[] values;

    /** Start of this box's slice inside `values`. */
    public final int offset;

    /** Real dimension count; the slice length is 2 * dimensions. */
    @Getter
    private final int dimensions;

    /** Sum of side lengths, accumulated in double. */
    @Getter
    protected double rangeSum;

    // ---- constructors ------------------------------------------------------

    /**
     * General reference constructor: view `values` at `offset` as a box of the
     * given dimension count and precomputed range sum. The array is stored by
     * reference (no copy) — this is the entry point for pooled/shared buffers.
     */
    public ArrayBox(final float[] values, int offset, int dimensions, double sum) {
        checkArgument(offset >= 0 && offset + 2 * dimensions <= values.length, "slice out of bounds");
        this.values = values;
        this.offset = offset;
        this.dimensions = dimensions;
        this.rangeSum = sum;
    }

    /** Convenience: whole array is the box, offset 0, dimensions = length / 2. */
    public ArrayBox(final float[] encoded, double sum) {
        this(encoded, 0, encoded.length / 2, sum);
    }

    /**
     * Degenerate box around a single point (min == max == point). Kept for the
     * tests/callers that build a point box. Allocates its own 2*dimensions array at
     * offset 0 and is genuinely mutable.
     */
    public ArrayBox(float[] point) {
        this.dimensions = point.length;
        this.offset = 0;
        this.values = new float[2 * dimensions];
        for (int i = 0; i < dimensions; ++i) {
            values[i] = point[i];
            values[i + dimensions] = -point[i]; // what about 0?
        }
        this.rangeSum = 0.0;
    }

    public void fromPoint(float[] point, int pointOffset) {
        checkArgument(point.length >= pointOffset + dimensions, "incorrect point");
        System.arraycopy(point, pointOffset, values, offset, dimensions);
        for (int i = 0; i < dimensions; ++i) {
            values[i + offset + dimensions] = -point[i + pointOffset];
        }
        this.rangeSum = 0.0;
    }

    public void fromPoint(float[] point) {
        this.fromPoint(point, 0);
    }

    /** Smallest box containing two points. Owns a fresh array at offset 0. */
    public ArrayBox(final float[] first, final float[] second) {
        checkArgument(first.length == second.length, " incorrect lengths in box");
        this.dimensions = first.length;
        this.offset = 0;
        this.values = new float[2 * dimensions];
        double sum = 0.0;
        for (int i = 0; i < dimensions; ++i) {
            float mx = max(first[i], second[i]);
            float mn = Math.min(first[i], second[i]);
            values[i] = mx;
            values[i + dimensions] = -mn;
            sum += (values[i] + values[i + dimensions]); // (mx - mn), promoted to double
        }
        this.rangeSum = sum;
    }

    // ---- raw slice movement (no allocation) --------------------------------

    /**
     * Copies this box's encoded 2*dimensions floats out to `dest` starting at
     * `destOffset`. Pure System.arraycopy; nothing is allocated.
     */
    public void to(float[] dest, int destOffset) {
        System.arraycopy(values, offset, dest, destOffset, 2 * dimensions);
    }

    public ArrayBox copyFromSlice(float[] source, int srcOffset, double sum) {
        System.arraycopy(source, srcOffset, values, offset, 2 * dimensions);
        rangeSum = sum;
        return this;
    }

    public ArrayBox copyFrom(ArrayBox box) {
        return copyFromSlice(box.values, box.offset, box.rangeSum);
    }

    // ---- duplication / merging --------------------------------------------

    public ArrayBox copy() {
        float[] c = Arrays.copyOfRange(values, offset, offset + 2 * dimensions);
        return new ArrayBox(c, 0, dimensions, rangeSum);
    }

    // all of its user are for tests and legacy
    public ArrayBox getMergedBox(IBoundingBoxView otherBox) {
        float[] merged = new float[2 * dimensions];
        double sum = 0.0;
        for (int i = 0; i < dimensions; ++i) {
            merged[i] = max(values[offset + i], (float) otherBox.getMaxValue(i));
            merged[i + dimensions] = max(values[offset + i + dimensions], (float) (-otherBox.getMinValue(i)));
            sum += (merged[i] + merged[i + dimensions]);
        }
        return new ArrayBox(merged, 0, dimensions, sum);
    }

    // tests and legacy
    public ArrayBox getMergedBox(float[] point) {
        checkArgument(point.length == dimensions, "incorrect length");
        return copy().addPoint(point);
    }

    // ---- in-place mutation -------------------------------------------------

    public ArrayBox addPoint(float[] point) {
        return addPoint(point, 0);
    }

    public ArrayBox addPoint(float[] point, int pointOffset) {
        rangeSum = VectorSupport.addPointInPlace(values, offset, dimensions, point, pointOffset);
        return this;
    }

    public static double addPointInPlace(float[] values, int offset, int dimensions, float[] point, int pointOffset) {
        return VectorSupport.addPointInPlace(values, offset, dimensions, point, pointOffset);
    }

    public ArrayBox addBox(ArrayBox otherBox) {
        rangeSum = VectorSupport.addSlice(values, offset, dimensions, otherBox.values, otherBox.offset);
        return this;
    }

    public ArrayBox addSlice(float[] otherValues, int otherOffset) {
        rangeSum = VectorSupport.addSlice(values, offset, dimensions, otherValues, otherOffset);
        return this;
    }

    // ---- accessors ---------------------------------------------------------

    public float getMaxValue(final int dimension) {
        return values[offset + dimension];
    }

    public float getMinValue(final int dimension) {
        return -values[offset + dimension + dimensions];
    }

    public double getRange(final int dimension) {
        return values[offset + dimension] + values[offset + dimension + dimensions]; // max - min
    }

    public float[] getMaxValues() {
        return Arrays.copyOfRange(values, offset, offset + dimensions);
    }

    public float[] getMinValues() {
        float[] min = new float[dimensions];
        for (int i = 0; i < dimensions; ++i) {
            min[i] = -values[offset + i + dimensions];
        }
        return min;
    }

    // ---- containment / equality -------------------------------------------

    public boolean contains(float[] point) {
        checkArgument(point.length == dimensions, " incorrect lengths");
        for (int i = 0; i < dimensions; ++i) {
            if (-values[offset + i + dimensions] > point[i] || values[offset + i] < point[i]) {
                return false;
            }
        }
        return true;
    }

    public boolean contains(ArrayBox otherBox) {
        checkArgument(otherBox.dimensions == dimensions, " incorrect lengths");
        // contains <=> this.max >= other.max AND this.min <= other.min,
        // which under the encoding is just values >= other.values across the slice.
        for (int i = 0; i < 2 * dimensions; ++i) {
            if (values[offset + i] < otherBox.values[otherBox.offset + i]) {
                return false;
            }
        }
        return true;
    }

    @Override
    public String toString() {
        return String.format("BoundingBox(min=%s, max=%s)", Arrays.toString(getMinValues()),
                Arrays.toString(getMaxValues()));
    }

    @Override
    public boolean equals(Object other) {
        if (!(other instanceof ArrayBox)) {
            return false;
        }
        ArrayBox otherBox = (ArrayBox) other;
        return dimensions == otherBox.dimensions && Arrays.equals(values, offset, offset + 2 * dimensions,
                otherBox.values, otherBox.offset, otherBox.offset + 2 * dimensions);
    }

    @Override
    public int hashCode() {
        int result = 1;
        for (int i = 0; i < 2 * dimensions; ++i) {
            result = 31 * result + Float.floatToIntBits(values[offset + i]);
        }
        return result;
    }

    public BoundingBox getBoundingBox() {
        return getBoundingBox(values, offset, dimensions, rangeSum);
    }

    protected static BoundingBox getBoundingBox(float[] values, int offset, int dimensions, double rangeSum) {
        float[] minvalues = new float[dimensions];
        for (int i = 0; i < dimensions; i++) {
            minvalues[i] = -values[offset + dimensions + i];
        }
        float[] maxValues = Arrays.copyOfRange(values, offset, offset + dimensions);
        return new BoundingBox(minvalues, maxValues, rangeSum);
    }

    public ArrayBox(BoundingBox box) {
        dimensions = box.minValues.length;
        values = new float[2 * dimensions];
        offset = 0;
        System.arraycopy(box.maxValues, 0, values, 0, dimensions);
        for (int i = 0; i < dimensions; i++) {
            values[i + dimensions] = -box.getMinValue(i);
        }
        rangeSum = box.rangeSum;
    }

    public ArrayBox(int dimensions) {
        checkArgument(dimensions > 0, "incorrect dimensions");
        values = new float[2 * dimensions];
        rangeSum = 0;
        offset = 0;
        this.dimensions = dimensions;
    }

    //
    public double probabilityOfCut(float[] expandedPoint, float[] components) {
        return VectorSupport.gapAttribution(values, offset, dimensions, rangeSum, expandedPoint, 0, components);
    }

}
