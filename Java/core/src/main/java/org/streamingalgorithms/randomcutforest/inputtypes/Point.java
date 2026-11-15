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
 * The file is substantially modified from its prior version which had the
 * following notice.
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

package org.streamingalgorithms.randomcutforest.inputtypes;

import java.util.Arrays;

/**
 * a basic class that defines a proto-point
 */
public class Point {

    // current values
    protected double[] currentInput;

    // input timestamp
    protected long inputTimestamp;

    public Point(double[] input, long inputTimestamp) {
        this.currentInput = copyIfNotnull(input);
        this.inputTimestamp = inputTimestamp;
    }

    public double[] getCurrentInput() {
        return copyIfNotnull(currentInput);
    }

    public long getInputTimestamp() {
        return inputTimestamp;
    }

    protected double[] copyIfNotnull(double[] array) {
        return array == null ? null : Arrays.copyOf(array, array.length);
    }

    /**
     * Overwrites the input and its timestamp, in place when the lengths agree; for
     * a point (or descriptor) that is reused.
     *
     * @param input          the new input
     * @param inputTimestamp its timestamp
     */
    protected void resetInput(double[] input, long inputTimestamp) {
        this.currentInput = copyInto(currentInput, input);
        this.inputTimestamp = inputTimestamp;
    }

    /**
     * A copy of source: written into target when the lengths agree, a new array
     * otherwise, and null when source is null.
     *
     * @param target the array to reuse, or null
     * @param source the values
     * @return the copy
     */
    protected static double[] copyInto(double[] target, double[] source) {
        if (source == null) {
            return null;
        }
        if (target == null || target.length != source.length) {
            return Arrays.copyOf(source, source.length);
        }
        System.arraycopy(source, 0, target, 0, source.length);
        return target;
    }
}
