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

package org.streamingalgorithms.randomcutforest.parkservices.calibration;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Random;

import org.junit.jupiter.api.Test;
import org.streamingalgorithms.randomcutforest.parkservices.config.Calibration;
import org.streamingalgorithms.randomcutforest.returntypes.RangeVector;
import org.streamingalgorithms.randomcutforest.statistics.Deviation;

public class ErrorHandlerTest {

    @Test
    public void errorHandlerConstructorTest() {
        ErrorHandler.Builder builder = new ErrorHandler.Builder();
        // builder().compact(true).dimensions(dimensions).randomSeed(seed + 1)
        // .numberOfTrees(numberOfTrees).shingleSize(shingleSize).sampleSize(sampleSize)
        // .internalShinglingEnabled(true).precision(precision).anomalyRate(0.01).forestMode(ForestMode.STANDARD)
        // .transformMethod(transformMethod).outputAfter(outputAfter).forecastHorizon(forecastHorizon)
        // .calibration(calibration).errorHorizon(errorHorizon).initialAcceptFraction(0.125);
        assertThrows(IllegalArgumentException.class, () -> new ErrorHandler(builder));
        builder.errorHorizon(1).forecastHorizon(2);
        assertThrows(IllegalArgumentException.class, () -> new ErrorHandler(builder));
        builder.errorHorizon(2).forecastHorizon(2);
        assertThrows(IllegalArgumentException.class, () -> new ErrorHandler(builder));
        builder.dimensions(1);
        assertDoesNotThrow(() -> new ErrorHandler(builder));
        builder.errorHorizon(10000);
        assertThrows(IllegalArgumentException.class, () -> new ErrorHandler(builder));
    }

    @Test
    public void testCalibrate() {
        ErrorHandler e = ErrorHandler.builder().errorHorizon(2).forecastHorizon(2).dimensions(2).build();
        assertThrows(IllegalArgumentException.class,
                () -> e.calibrate(new double[2], Calibration.SIMPLE, new RangeVector(5)));
        RangeVector r = new RangeVector(4);
        e.sequenceIndex = 5;
        e.lastDataDeviations = new float[] { 1.0f, 1.3f };
        float v = new Random().nextFloat();
        r.shift(0, v);
        assertThrows(IllegalArgumentException.class,
                () -> e.calibrate(new double[1], Calibration.SIMPLE, new RangeVector(r)));
        e.calibrate(new double[2], Calibration.SIMPLE, new RangeVector(r));
        assertEquals(r.values[0], v);
        e.calibrate(new double[2], Calibration.NONE, r);
        assertEquals(r.values[0], v);
        assertEquals(r.upper[0], v);
        assertEquals(r.values[1], 0);
        e.lastDataDeviations = new float[] { v + 1.0f, 1.3f };
        e.calibrate(new double[2], Calibration.MINIMAL, r);
        assertEquals(r.values[0], v);
        assertEquals(r.values[1], 0);
    }

    @Test
    public void testSerializedConstructor() {
        assertThrows(IllegalArgumentException.class,
                () -> new ErrorHandler(0, 0, 0, 0, 0, null, null, null, null, null, null));
        assertThrows(IllegalArgumentException.class,
                () -> new ErrorHandler(0, 1, 0, 0, 0, null, null, null, null, null, null));
        assertThrows(IllegalArgumentException.class,
                () -> new ErrorHandler(1, 1, 0, 0, 0, null, null, null, null, null, null));
        assertThrows(IllegalArgumentException.class,
                () -> new ErrorHandler(1, 1, -1, 0, 1, null, null, null, null, null, null));
        assertThrows(IllegalArgumentException.class,
                () -> new ErrorHandler(1, 1, 0, 0, 1, null, null, null, null, null, null));
        assertThrows(IllegalArgumentException.class,
                () -> new ErrorHandler(1, 1, 0, 0.1, 1, null, null, null, new Deviation[2], null, null));
        assertThrows(IllegalArgumentException.class,
                () -> new ErrorHandler(1, 1, 0, 0.1, 1, null, null, new double[1], new Deviation[3], null, null));
        assertThrows(IllegalArgumentException.class, () -> new ErrorHandler(1, 1, 0, 0.1, 1, null, new float[0],
                new double[2], new Deviation[3], null, null));
        Deviation[] deviations = new Deviation[3];
        deviations[0] = deviations[1] = deviations[2] = new Deviation(0);
        assertThrows(IllegalArgumentException.class, () -> new ErrorHandler(1, 1, 0, 0.1, 1, new float[2], new float[1],
                new double[2], deviations, null, null));
        assertDoesNotThrow(() -> new ErrorHandler(1, 1, 0, 0.1, 1, new float[0], new float[1], new double[2],
                deviations, null, null));
    }

    @Test
    public void limitsKeepTheIntervalAroundItsPoint() {
        ErrorHandler e = ErrorHandler.builder().dimensions(1).forecastHorizon(1).errorHorizon(10).useRCF(false)
                .lowerLimit(new float[] { -5 }).upperLimit(new float[] { 8 }).build();
        // RCF's range above the upper limit, and a positive lower offset
        e.errorDistribution.values[0] = 2;
        e.errorDistribution.upper[0] = 5;
        e.errorDistribution.lower[0] = 1;
        RangeVector above = new RangeVector(new float[] { 9 }, new float[] { 10 }, new float[] { 8.5f });
        e.calibrate(new double[1], Calibration.SIMPLE, above);
        assertEquals(8, above.values[0]);
        assertTrue(above.lower[0] <= above.values[0] && above.values[0] <= above.upper[0]);
        // RCF's range below the lower limit, and a negative upper offset
        e.errorDistribution.values[0] = -2;
        e.errorDistribution.upper[0] = -1;
        e.errorDistribution.lower[0] = -5;
        RangeVector below = new RangeVector(new float[] { -7 }, new float[] { -6.5f }, new float[] { -8 });
        e.calibrate(new double[1], Calibration.SIMPLE, below);
        assertEquals(-5, below.values[0]);
        assertTrue(below.lower[0] <= below.values[0] && below.values[0] <= below.upper[0]);
    }

    @Test
    public void restoreRepairsAnIntervalThatMissesItsPoint() {
        Deviation[] deviations = new Deviation[] { new Deviation(0.1), new Deviation(0.1), new Deviation(0.1) };
        // value 8, upper 8, lower 8.0012: as published with an upper limit of 8
        float[] past = new float[] { 8, 8, 8.0012f };
        ErrorHandler e = assertDoesNotThrow(
                () -> new ErrorHandler(1, 1, 1, 0.1, 1, past, new float[1], new double[2], deviations, null, null));
        assertEquals(8, e.getPastForecasts()[0].lower[0]);
        assertEquals(8, e.getPastForecasts()[0].upper[0]);
    }
}