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

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.streamingalgorithms.randomcutforest.tree.Column.Tier.*;

import java.util.stream.Stream;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.streamingalgorithms.randomcutforest.tree.Column.Tier;

class ColumnTest {

    // ---- tier ladder lands exactly on the byte/char boundaries ----
    @Test
    void tierBoundariesNoSentinel() {
        assertEquals(Tier.BYTE, Column.tierFor(0x00, false));
        assertEquals(Tier.BYTE, Column.tierFor(0xFF, false)); // 255 fits a byte
        assertEquals(Tier.CHAR, Column.tierFor(0x100, false)); // 256 spills to char
        assertEquals(Tier.CHAR, Column.tierFor(0xFFFF, false)); // 65535 fits a char
        assertEquals(Tier.INT, Column.tierFor(0x10000, false));// 65536 spills to int
    }

    @Test
    void tierBoundariesWithSentinelShiftDown() {
        // reserving the top value for a sentinel pushes each boundary down by one
        assertEquals(Tier.CHAR, Column.tierFor(0xFF, true)); // needs 256 distinct -> char
        assertEquals(Tier.INT, Column.tierFor(0xFFFF, true)); // needs 65536 distinct -> int
    }

    // ---- get/set round-trips every representable value, all three tiers ----
    static Stream<Arguments> tierAndValue() {
        // (factory for a length-4 column, tier, a value guaranteed representable in
        // that tier)
        return Stream.of(Arguments.of("byte", (java.util.function.IntFunction<Column>) Column.ByteColumn::new, 0),
                Arguments.of("byte", (java.util.function.IntFunction<Column>) Column.ByteColumn::new, 1),
                Arguments.of("byte", (java.util.function.IntFunction<Column>) Column.ByteColumn::new, 0xFE), // FF =
                                                                                                             // sentinel
                Arguments.of("char", (java.util.function.IntFunction<Column>) Column.CharColumn::new, 0),
                Arguments.of("char", (java.util.function.IntFunction<Column>) Column.CharColumn::new, 0x1234),
                Arguments.of("char", (java.util.function.IntFunction<Column>) Column.CharColumn::new, 0xFFFE),
                Arguments.of("int", (java.util.function.IntFunction<Column>) Column.IntColumn::new, 0),
                Arguments.of("int", (java.util.function.IntFunction<Column>) Column.IntColumn::new, 123456),
                Arguments.of("int", (java.util.function.IntFunction<Column>) Column.IntColumn::new,
                        Integer.MAX_VALUE - 1));
    }

    @ParameterizedTest(name = "{0} round-trips {2}")
    @MethodSource("tierAndValue")
    void getSetRoundTrips(String tier, java.util.function.IntFunction<Column> factory, int value) {
        Column col = factory.apply(4);
        col.set(2, value);
        assertEquals(value, col.get(2), tier + " lost value " + value);
        // untouched slots are still sentinel
        assertEquals(col.sentinel(), col.get(0));
    }

    // ---- overflow is LOUD, not a silent truncation (requires -ea) ----
    static Stream<Arguments> overflowingWrites() {
        return Stream.of(Arguments.of((java.util.function.IntFunction<Column>) Column.ByteColumn::new, 0x100), // 256 >
                                                                                                               // byte
                Arguments.of((java.util.function.IntFunction<Column>) Column.CharColumn::new, 0x10000) // 65536 > char
        );
    }

    @ParameterizedTest
    @MethodSource("overflowingWrites")
    void overflowTripsAssert(java.util.function.IntFunction<Column> factory, int tooBig) {
        Column col = factory.apply(4);
        // NOTE: only meaningful with -ea (set in parent surefire argLine).
        assertThrows(AssertionError.class, () -> col.set(0, tooBig),
                "overflow " + tooBig + " must throw, not truncate");
    }

    // ---- fresh column is all-sentinel ----
    @ParameterizedTest(name = "{0} fresh = sentinel")
    @MethodSource("freshColumns")
    void freshIsAllSentinel(String tier, Column col) {
        for (int i = 0; i < col.length(); i++) {
            assertEquals(col.sentinel(), col.get(i), tier + " slot " + i + " not sentinel");
        }
    }

    static Stream<Arguments> freshColumns() {
        return Stream.of(Arguments.of("byte", new Column.ByteColumn(5)), Arguments.of("char", new Column.CharColumn(5)),
                Arguments.of("int", new Column.IntColumn(5)));
    }

    // ---- extend preserves data and sentinel-fills the tail (the unpack path
    // depends on this) ----
    @ParameterizedTest(name = "{0} extend")
    @MethodSource("freshColumns")
    void extendPreservesAndSentinelFills(String tier, Column col) {
        int old = col.length();
        col.set(0, 1);
        col.set(1, 2);
        col.extend(old + 3);
        assertEquals(old + 3, col.length());
        assertEquals(1, col.get(0), tier);
        assertEquals(2, col.get(1), tier);
        for (int i = old; i < old + 3; i++) {
            assertEquals(col.sentinel(), col.get(i), tier + " tail slot " + i);
        }
    }

    // ---- adopt-constructor sees the backing array unsigned (the reflation path
    // relies on this) ----
    @Test
    void charAdoptWidensUnsigned() {
        char[] backing = { 0, 1, 0xFFFE, 0x8000 };
        Column col = new Column.CharColumn(backing);
        assertEquals(0x8000, col.get(3), "char must widen unsigned, not sign-extend");
    }

    @Test
    void byteAdoptWidensUnsigned() {
        byte[] backing = { 0, 1, (byte) 0xFE, (byte) 0x80 };
        Column col = new Column.ByteColumn(backing);
        assertEquals(0x80, col.get(3), "byte must mask 0xFF, not sign-extend");
    }
}