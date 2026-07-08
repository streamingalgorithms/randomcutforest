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

import java.util.Arrays;

/**
 * Fixed-width, unsigned-ish integer column. Reads widen to int; writes narrow
 * with a round-trip assert. The top raw value of each tier is exposed as
 * {@link #sentinel()}; stores that need an empty/Null marker size with
 * reserveSentinel=true so a live value never collides with it. Stores that
 * track freedom elsewhere (NodeStore left/right via the free list) ignore
 * sentinel() and size on maxValue alone.
 *
 * All the {@literal & 0xFF / & 0xFFFF} narrowing lives here instead of
 * sprinkled across Small/Medium/Large.
 */
public interface Column {

    enum Tier {
        BYTE, CHAR, INT
    }

    static Tier tierFor(long maxValue, boolean reserveSentinel) {
        long top = reserveSentinel ? maxValue + 1 : maxValue;
        if (top <= 0xFFL)
            return Tier.BYTE;
        if (top <= 0xFFFFL)
            return Tier.CHAR;
        return Tier.INT;
    }

    int get(int i);

    void set(int i, int v);

    int length();

    int sentinel();

    void extend(int newLen);

    /**
     * Narrowest column over [0, maxValue]; +1 headroom reserved for sentinel if
     * asked.
     */
    static Column of(int length, long maxValue, boolean reserveSentinel) {
        switch (tierFor(maxValue, reserveSentinel)) {
        case BYTE:
            return new ByteColumn(length);
        case CHAR:
            return new CharColumn(length);
        default:
            return new IntColumn(length);
        }
    }

    final class ByteColumn implements Column {
        private byte[] a;

        ByteColumn(int n) {
            a = new byte[n];
            Arrays.fill(a, (byte) 0xFF);
        }

        ByteColumn(byte[] r) {
            a = r;
        }

        public int get(int i) {
            return a[i] & 0xFF;
        }

        public void set(int i, int v) {
            assert (v & 0xFF) == v : "byte overflow " + v;
            a[i] = (byte) v;
        }

        public int length() {
            return a.length;
        }

        public int sentinel() {
            return 0xFF;
        }

        public void extend(int n) {
            int o = a.length;
            a = Arrays.copyOf(a, n);
            Arrays.fill(a, o, n, (byte) 0xFF);
        }
    }

    final class CharColumn implements Column {
        private char[] a;

        CharColumn(int n) {
            a = new char[n];
            Arrays.fill(a, (char) 0xFFFF);
        }

        CharColumn(char[] r) {
            a = r;
        }

        public int get(int i) {
            return a[i];
        } // char widens unsigned; no mask

        public void set(int i, int v) {
            assert (v & 0xFFFF) == v : "char overflow " + v;
            a[i] = (char) v;
        }

        public int length() {
            return a.length;
        }

        public int sentinel() {
            return 0xFFFF;
        }

        public void extend(int n) {
            int o = a.length;
            a = Arrays.copyOf(a, n);
            Arrays.fill(a, o, n, (char) 0xFFFF);
        }
    }

    final class IntColumn implements Column {
        private int[] a;

        IntColumn(int n) {
            a = new int[n];
            Arrays.fill(a, -1);
        }

        IntColumn(int[] r) {
            a = r;
        }

        public int get(int i) {
            return a[i];
        }

        public void set(int i, int v) {
            a[i] = v;
        }

        public int length() {
            return a.length;
        }

        public int sentinel() {
            return -1;
        } // matches legacy INFEASIBLE_LOCN

        public void extend(int n) {
            int o = a.length;
            a = Arrays.copyOf(a, n);
            Arrays.fill(a, o, n, -1);
        }
    }
}
