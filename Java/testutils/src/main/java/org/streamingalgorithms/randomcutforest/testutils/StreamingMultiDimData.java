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

package org.streamingalgorithms.randomcutforest.testutils;

import static java.lang.Math.PI;

import java.util.Random;

/**
 * The streaming form of
 * {@link ShingledMultiDimDataWithKeys#getMultiDimData(int, int, double, double, long, int, double, boolean)}.
 * Emits BASE-dimension points one at a time into a buffer it owns, and keeps no
 * history.
 *
 * <p>
 * Produces the identical sequence to the batch generator for the same seed and
 * parameters: same {@code prg}/{@code noiseprg} construction, same per-point
 * draw order. A throughput number from this producer is comparable to one taken
 * from {@code getMultiDimData(...).data}.
 *
 * <h2>Why this exists</h2>
 * <p>
 * The batch generator materializes {@code double[num][baseDimension]} up front.
 * At 100k points that is a live array the GC must trace, and -- more to the
 * point for a throughput harness -- every forest reads the same rows, so N
 * threads share one working set and the measurement quietly becomes a study of
 * how well those rows cache. Give each forest its own producer with its own
 * seed and the threads share nothing.
 *
 * <h2>Buffer contract</h2>
 * <p>
 * <b>{@link #next()} returns a buffer this object owns and overwrites on the
 * next call.</b> Do not retain it. This is deliberate: it makes the producer
 * allocation-free, so a {@code ThreadMXBean} allocation count over the
 * measurement loop attributes bytes to RCF rather than to the data source. Both
 * {@code update} and {@code getAnomalyScore} copy what they need, so this is
 * safe; if you ever stash the reference, you will see every stored point mutate
 * underneath you.
 *
 * <h2>Not thread safe</h2>
 * <p>
 * One instance per thread. That is the whole design.
 */
public final class StreamingMultiDimData {

    private final int baseDimension;
    private final int period;
    private final double noise;
    private final double anomalyFactor;

    private final double[] phase;
    private final double[] amp;
    private final double[] slope;
    private final double[] shift;

    private final Random noiseprg;

    private final double[] dbuf;
    private final float[] fbuf;

    private int index;
    private boolean lastWasAnomaly;

    /**
     * Matches getMultiDimData(num, period, amplitude, noise, seed, baseDimension).
     */
    public StreamingMultiDimData(int period, double amplitude, double noise, long seed, int baseDimension) {
        this(period, amplitude, noise, seed, baseDimension, 5.0, false);
    }

    public StreamingMultiDimData(int period, double amplitude, double noise, long seed, int baseDimension,
            double anomalyFactor, boolean useSlope) {
        this.baseDimension = baseDimension;
        this.period = period;
        this.noise = noise;
        this.anomalyFactor = anomalyFactor;

        this.phase = new double[baseDimension];
        this.amp = new double[baseDimension];
        this.slope = new double[baseDimension];
        this.shift = new double[baseDimension];
        this.dbuf = new double[baseDimension];
        this.fbuf = new float[baseDimension];

        // identical draw order to the batch generator, so the streams agree
        Random prg = new Random(seed);
        this.noiseprg = new Random(prg.nextLong());
        for (int i = 0; i < baseDimension; i++) {
            phase[i] = prg.nextInt(period);
            if (useSlope) {
                shift[i] = (4 * prg.nextDouble() - 1) * amplitude;
            }
            amp[i] = (1 + 0.2 * prg.nextDouble()) * amplitude;
            if (useSlope) {
                slope[i] = (0.25 - prg.nextDouble() * 0.5) * amplitude / period;
            }
        }
    }

    /**
     * Next point, as double[]. Reused buffer -- see the class javadoc.
     */
    public double[] next() {
        final int i = index++;
        final boolean flag = (noiseprg.nextDouble() < 0.01);
        boolean used = false;
        for (int j = 0; j < baseDimension; j++) {
            double v = amp[j] * Math.cos(2 * PI * (i + phase[j]) / period) + slope[j] * i + shift[j];
            // ensures that the noise does not cancel the anomaly or change its magnitude
            if (flag && noiseprg.nextDouble() < 0.3) {
                double factor = anomalyFactor * (1 + noiseprg.nextDouble());
                v += noiseprg.nextDouble() < 0.5 ? factor * noise : -factor * noise;
                used = true;
            } else {
                v += noise * (2 * noiseprg.nextDouble() - 1);
            }
            dbuf[j] = v;
        }
        lastWasAnomaly = used;
        return dbuf;
    }

    /**
     * Next point, as float[]. Reused buffer -- see the class javadoc.
     */
    public float[] nextFloat() {
        next();
        for (int j = 0; j < baseDimension; j++) {
            fbuf[j] = (float) dbuf[j];
        }
        return fbuf;
    }

    /** True if the point just returned carried an injected anomaly. */
    public boolean lastWasAnomaly() {
        return lastWasAnomaly;
    }

    /**
     * Count of points emitted so far; equals the row index in the batch generator.
     */
    public int index() {
        return index;
    }

    public int baseDimension() {
        return baseDimension;
    }
}
