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

package org.streamingalgorithms.randomcutforest.parkservices;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.streamingalgorithms.randomcutforest.CommonUtils.toFloatArray;
import static org.streamingalgorithms.randomcutforest.config.TransformMethod.NORMALIZE;
import static org.streamingalgorithms.randomcutforest.parkservices.PredictorCorrector.DEFAULT_SAMPLING_SUPPORT;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import org.junit.jupiter.api.Test;
import org.streamingalgorithms.randomcutforest.RandomCutForest;
import org.streamingalgorithms.randomcutforest.config.ForestMode;
import org.streamingalgorithms.randomcutforest.config.TransformMethod;
import org.streamingalgorithms.randomcutforest.parkservices.config.ScoringStrategy;
import org.streamingalgorithms.randomcutforest.parkservices.returntypes.RCFComputeDescriptor;
import org.streamingalgorithms.randomcutforest.parkservices.state.predictorcorrector.PredictorCorrectorMapper;
import org.streamingalgorithms.randomcutforest.parkservices.threshold.BasicThresholder;
import org.streamingalgorithms.randomcutforest.returntypes.DiVector;
import org.streamingalgorithms.randomcutforest.statistics.Deviation;

public class PredictorCorrectorTest {

    @Test
    void AttributorTest() {
        int sampleSize = 256;
        int baseDimensions = 10;
        int shingleSize = 10;
        int dimensions = baseDimensions * shingleSize;
        ThresholdedRandomCutForest forest = ThresholdedRandomCutForest.builder().dimensions(dimensions).randomSeed(0L)
                .forestMode(ForestMode.STANDARD).shingleSize(shingleSize).anomalyRate(0.01).transformMethod(NORMALIZE)
                .build();
        DiVector test = new DiVector(baseDimensions * shingleSize);
        assert (forest.predictorCorrector.getExpectedPoint(test, 0, baseDimensions, null, null) == null);
        assertThrows(IllegalArgumentException.class, () -> forest.predictorCorrector.setNumberOfAttributors(-1));
        forest.predictorCorrector.setNumberOfAttributors(baseDimensions);
        assertThrows(NullPointerException.class,
                () -> forest.predictorCorrector.getExpectedPoint(test, 0, baseDimensions, null, null));
        double[] array = new double[20];
        Arrays.fill(array, 1.0);
        DiVector testTwo = new DiVector(array, array);
        assertThrows(NullPointerException.class,
                () -> forest.predictorCorrector.getExpectedPoint(test, 0, baseDimensions, null, null));
    }

    @Test
    void configTest() {
        int sampleSize = 256;
        int baseDimensions = 2;
        int shingleSize = 10;
        int dimensions = baseDimensions * shingleSize;
        double[] testOne = new double[] { new Random().nextDouble(), new Random().nextDouble() };
        double[] testTwo = new double[] { new Random().nextDouble(), new Random().nextDouble() };
        double[] testThree = new double[] { new Random().nextDouble(), new Random().nextDouble() };
        double[] testFour = new double[] { new Random().nextDouble(), new Random().nextDouble() };
        ThresholdedRandomCutForest forest = ThresholdedRandomCutForest.builder().dimensions(dimensions).randomSeed(0L)
                .forestMode(ForestMode.STANDARD).shingleSize(shingleSize).anomalyRate(0.01)
                .scoringStrategy(ScoringStrategy.DISTANCE).transformMethod(NORMALIZE).randomSeed(1110).autoAdjust(true)
                .ignoreNearExpectedFromAbove(testOne).ignoreNearExpectedFromBelow(testTwo)
                .ignoreNearExpectedFromAboveByRatio(testThree).ignoreNearExpectedFromBelowByRatio(testFour).build();
        PredictorCorrector predictorCorrector = forest.getPredictorCorrector();
        assertEquals(predictorCorrector.getSamplingSupport(), DEFAULT_SAMPLING_SUPPORT);
        assertThrows(IllegalArgumentException.class, () -> predictorCorrector.setSamplingSupport(-1.0));
        assertThrows(IllegalArgumentException.class, () -> predictorCorrector.setSamplingSupport(2.0));
        assertDoesNotThrow(() -> predictorCorrector.setSamplingSupport(1.5 * DEFAULT_SAMPLING_SUPPORT));
        double[] test = new double[1];
        assertThrows(IllegalArgumentException.class, () -> predictorCorrector.setIgnoreNearExpected(test));
        assertDoesNotThrow(() -> predictorCorrector.setIgnoreNearExpected(null));
        assertArrayEquals(predictorCorrector.ignoreNearExpectedFromAbove, testOne, 1e-10);
        assertDoesNotThrow(() -> predictorCorrector.setIgnoreNearExpectedFromAbove(null));
        assertArrayEquals(predictorCorrector.ignoreNearExpectedFromAbove, testOne, 1e-10);
        assertArrayEquals(predictorCorrector.ignoreNearExpectedFromBelow, testTwo, 1e-10);
        assertDoesNotThrow(() -> predictorCorrector.setIgnoreNearExpectedFromBelow(null));
        assertArrayEquals(predictorCorrector.ignoreNearExpectedFromBelow, testTwo, 1e-10);
        assertArrayEquals(predictorCorrector.ignoreNearExpectedFromAboveByRatio, testThree, 1e-10);
        assertDoesNotThrow(() -> predictorCorrector.setIgnoreNearExpectedFromAboveByRatio(null));
        assertArrayEquals(predictorCorrector.ignoreNearExpectedFromAboveByRatio, testThree, 1e-10);
        assertArrayEquals(predictorCorrector.ignoreNearExpectedFromBelowByRatio, testFour, 1e-10);
        assertDoesNotThrow(() -> predictorCorrector.setIgnoreNearExpectedFromBelowByRatio(null));
        assertArrayEquals(predictorCorrector.ignoreNearExpectedFromBelowByRatio, testFour, 1e-10);
        assertNotNull(predictorCorrector.getDeviations());
        assertEquals(predictorCorrector.lastStrategy, ScoringStrategy.DISTANCE);
        assertThrows(IllegalArgumentException.class,
                () -> predictorCorrector.getCachedAttribution(1, null, new DiVector[2], null));

        PredictorCorrectorMapper mapper = new PredictorCorrectorMapper();
        PredictorCorrector copy = mapper.toModel(mapper.toState(predictorCorrector));
        assertEquals(1.5 * DEFAULT_SAMPLING_SUPPORT, copy.getSamplingSupport(), 1e-12);
        assertArrayEquals(copy.ignoreNearExpectedFromAbove, testOne, 1e-10);
        assertArrayEquals(copy.ignoreNearExpectedFromBelow, testTwo, 1e-10);
        assertArrayEquals(copy.ignoreNearExpectedFromAboveByRatio, testThree, 1e-10);
        assertArrayEquals(copy.ignoreNearExpectedFromBelowByRatio, testFour, 1e-10);
        assertNotNull(copy.getDeviations());
        assertEquals(copy.lastStrategy, ScoringStrategy.DISTANCE);
        copy.deviationsActual = new Deviation[1]; // changing the state
        IllegalArgumentException exception = assertThrows(IllegalArgumentException.class, () -> copy.getDeviations());
        assertEquals("incorrect state", exception.getMessage());
        copy.deviationsExpected = new Deviation[1];
        exception = assertThrows(IllegalArgumentException.class, () -> copy.getDeviations());
        assertEquals("length should be base dimension", exception.getMessage());

        double[] another = new double[4 * baseDimensions];
        assertDoesNotThrow(() -> predictorCorrector.setIgnoreNearExpected(another));
        assertArrayEquals(predictorCorrector.ignoreNearExpectedFromAbove, new double[2]);
        assertArrayEquals(predictorCorrector.ignoreNearExpectedFromBelow, new double[2]);
        assertArrayEquals(predictorCorrector.ignoreNearExpectedFromAboveByRatio, new double[2]);
        assertArrayEquals(predictorCorrector.ignoreNearExpectedFromBelowByRatio, new double[2]);
        another[0] = -1;
        assertThrows(IllegalArgumentException.class, () -> predictorCorrector.setIgnoreNearExpected(another));
        forest.setIgnoreNearExpectedFromAbove(testOne);
        forest.setIgnoreNearExpectedFromBelow(testTwo);
        forest.setIgnoreNearExpectedFromAboveByRatio(testThree);
        forest.setIgnoreNearExpectedFromBelowByRatio(testFour);
        assertArrayEquals(predictorCorrector.ignoreNearExpectedFromAbove, testOne, 1e-10);
        assertArrayEquals(predictorCorrector.ignoreNearExpectedFromBelow, testTwo, 1e-10);
        assertArrayEquals(predictorCorrector.ignoreNearExpectedFromAboveByRatio, testThree, 1e-10);
        assertArrayEquals(predictorCorrector.ignoreNearExpectedFromBelowByRatio, testFour, 1e-10);

        Random testRandom = new Random(1110L);
        assertEquals(predictorCorrector.getRandomSeed(), 1110L);
        double nextDouble = predictorCorrector.nextDouble();
        assertEquals(predictorCorrector.getRandomSeed(), testRandom.nextLong());
        assertEquals(nextDouble, testRandom.nextDouble(), 1e-10);

    }

    @Test
    public void mapperTest() {
        assertThrows(IllegalArgumentException.class, () -> new PredictorCorrector(new BasicThresholder[0], null, 1, 0));
        assertThrows(NullPointerException.class, () -> new PredictorCorrector(new BasicThresholder[1], null, 1, 0));
        assertThrows(IllegalArgumentException.class,
                () -> new PredictorCorrector(new BasicThresholder[] { new BasicThresholder(0) }, new Deviation[1], 1,
                        0));
    }

    @Test
    public void expectedValueTest() {
        ThresholdedRandomCutForest forest = ThresholdedRandomCutForest.builder().dimensions(20).randomSeed(0L)
                .forestMode(ForestMode.STANDARD).shingleSize(1).anomalyRate(0.01)
                .scoringStrategy(ScoringStrategy.DISTANCE).transformMethod(NORMALIZE).build();
        PredictorCorrector predictorCorrector = forest.getPredictorCorrector();
        double[] vector = new double[20];
        Arrays.fill(vector, 1.0);
        DiVector diVec = new DiVector(vector, vector);
        assertNull(predictorCorrector.getExpectedPoint(diVec, 0, 20, null, null));
        assertTrue(predictorCorrector.trigger(diVec, 1, 20, null, null, 1.0));
        assertTrue(predictorCorrector.trigger(diVec, 21, 20, null, null, 1.0));
        assertTrue(predictorCorrector.trigger(diVec, 21, 20, diVec, null, 1.0));
        assertEquals(1, predictorCorrector.centeredTransformPass(new AnomalyDescriptor(null, 0), toFloatArray(vector)));
        Arrays.fill(vector, 0);
        assertEquals(0, predictorCorrector.centeredTransformPass(new AnomalyDescriptor(null, 0), toFloatArray(vector)));
    }

    @Test
    public void runLengthTest() {
        ThresholdedRandomCutForest forest = ThresholdedRandomCutForest.builder().dimensions(4).randomSeed(0L)
                .forestMode(ForestMode.STANDARD).shingleSize(4).anomalyRate(0.01).autoAdjust(false)
                .scoringStrategy(ScoringStrategy.MULTI_MODE).transformMethod(NORMALIZE).build();
        for (int i = 0; i < 100; i++) {
            forest.process(new double[] { 10 }, 0);
        }
        for (int i = 0; i < 100; i++) {
            forest.process(new double[] { 20 }, 0);
        }
        double[] scores = forest.getPredictorCorrector().getLastScore();
        forest.predictorCorrector.setLastScore(null);
        assertArrayEquals(forest.predictorCorrector.getLastScore(), scores, 1e-6);
    }

    @Test
    public void expectedPointImputesOnlySelectedCoordinates() {
        List<int[]> asked = new ArrayList<>();
        RandomCutForest forest = new RandomCutForest(RandomCutForest.builder().dimensions(6).randomSeed(1)) {
            @Override
            public float[] imputeMissingValues(float[] point, int numberOfMissingValues, int[] missingIndexes) {
                asked.add(Arrays.copyOf(missingIndexes, missingIndexes.length));
                return point;
            }
        };
        PredictorCorrector predictorCorrector = new PredictorCorrector(0.01, 0.005, false, 3, 0L);
        // two blocks of three; the block at position 3 is blamed. With a block total
        // above 1, contributors are counted against 0.1 but selected against 10% of the
        // total, so only the first is selected; nothing else may be imputed.
        DiVector large = new DiVector(6);
        large.high[3] = 5;
        large.high[4] = 0.5;
        large.high[5] = 0.3;
        predictorCorrector.getExpectedPoint(large, 3, 3, new float[6], forest);
        assertArrayEquals(new int[] { 3 }, asked.get(0));
        // with a total below 1 both thresholds agree
        DiVector small = new DiVector(6);
        small.high[3] = 0.5;
        small.high[4] = 0.2;
        small.high[5] = 0.05;
        predictorCorrector.getExpectedPoint(small, 3, 3, new float[6], forest);
        assertArrayEquals(new int[] { 3, 4 }, asked.get(1));
    }

    @Test
    public void expectedPointIsScoredWhenTheLastAnomalyIsAShingleAgo() {
        int baseDimension = 2;
        int shingleSize = 4;
        int dimensions = baseDimension * shingleSize;
        float[] point = new float[dimensions];
        Arrays.fill(point, 1.0f);
        point[dimensions - baseDimension] = 9.0f; // the newest block holds the anomaly
        float[] expected = Arrays.copyOf(point, dimensions);
        expected[dimensions - baseDimension] = 1.0f;
        List<float[]> scored = new ArrayList<>();
        RandomCutForest forest = new RandomCutForest(RandomCutForest.builder().dimensions(dimensions).randomSeed(1)) {
            @Override
            public double getAnomalyScore(float[] p) {
                scored.add(Arrays.copyOf(p, p.length));
                return Arrays.equals(p, point) ? 5.0 : 0.5;
            }

            @Override
            public DiVector getAnomalyAttribution(float[] p) {
                DiVector attribution = new DiVector(dimensions);
                attribution.high[dimensions - baseDimension] = 5.0;
                return attribution;
            }

            @Override
            public float[] imputeMissingValues(float[] p, int numberOfMissingValues, int[] missingIndexes) {
                return Arrays.copyOf(expected, expected.length);
            }
        };
        PredictorCorrector predictorCorrector = new PredictorCorrector(0.01, 0.005, false, baseDimension, 0L);
        AnomalyDescriptor result = new AnomalyDescriptor(new double[baseDimension], 100L);
        result.setScoringStrategy(ScoringStrategy.EXPECTED_INVERSE_DEPTH);
        result.setRCFPoint(point);
        result.setShingleSize(shingleSize);
        result.setDimension(dimensions);
        result.setInternalTimeStamp(100L);
        result.setTransformMethod(TransformMethod.NONE);
        result.setScale(new double[] { 1.0, 1.0 });
        result.setShift(new double[baseDimension]);
        result.setDeviations(new double[2 * baseDimension]);
        // the last anomaly is far more than a shingle ago
        predictorCorrector.detect(result, new RCFComputeDescriptor(null, 0L), forest);

        assertTrue(result.getAnomalyGrade() > 0);
        // first the point itself, then the expected point (not the point again)
        assertEquals(2, scored.size());
        assertArrayEquals(point, scored.get(0));
        assertArrayEquals(expected, scored.get(1));
    }
}
