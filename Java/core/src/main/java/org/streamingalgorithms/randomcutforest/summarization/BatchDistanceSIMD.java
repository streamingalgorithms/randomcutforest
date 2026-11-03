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

package org.streamingalgorithms.randomcutforest.summarization;

import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorMask;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorShape;
import jdk.incubator.vector.VectorSpecies;

/**
 * The Vector API side of BatchDistance.nearest: one lane per point. Reached
 * only when BatchDistance.SIMD is true, so the class never loads without the
 * incubator module. No vector value leaves this class; the signatures are
 * arrays and ints.
 *
 * <p>
 * A lane performs exactly the scalar steps of BatchDistance.pair: the float
 * difference (FloatVector.sub is IEEE float subtraction), widened to double
 * (exact), squared or made absolute (exact), summed in double into the same
 * four partial sums combined as (s0 + s1) + (s2 + s3) and then the tail, and
 * the square root (correctly rounded). The nearest representative is kept with
 * a strict less-than, so the first of equals wins, as in the scalar loop.
 */
final class BatchDistanceSIMD {

    private static final VectorSpecies<Double> DS = DoubleVector.SPECIES_PREFERRED;

    /** Floats with as many lanes as DS, so one conversion widens them all. */
    private static final VectorSpecies<Float> FS = VectorSpecies.of(float.class,
            VectorShape.forBitSize(DS.vectorBitSize() / 2));

    private static final int LANES = DS.length();

    private BatchDistanceSIMD() {
    }

    /**
     * As BatchDistance.nearest, for whole vectors of points; returns the first
     * point not done.
     */
    static int nearest(boolean l2, int d, float[] points, int n, int from, int to, float[] reps, int r0, int r1,
            double[] first, double[] best, double[] arg, int off) {
        int p = from;
        for (; p + LANES <= to; p += LANES) {
            DoubleVector b = distance(l2, d, points, n, p, reps, r0 * d);
            b.intoArray(first, off + p - from);
            DoubleVector a = DoubleVector.zero(DS);
            for (int m = r0 + 1; m < r1; m++) {
                DoubleVector t = distance(l2, d, points, n, p, reps, m * d);
                VectorMask<Double> closer = t.lt(b);
                b = b.blend(t, closer);
                a = a.blend((double) (m - r0), closer);
            }
            b.intoArray(best, off + p - from);
            a.intoArray(arg, off + p - from);
        }
        return p;
    }

    private static DoubleVector distance(boolean l2, int d, float[] points, int n, int p, float[] reps, int rb) {
        DoubleVector s0 = DoubleVector.zero(DS);
        DoubleVector s1 = s0, s2 = s0, s3 = s0;
        int j = 0;
        final int bound = d & ~3;
        for (; j < bound; j += 4) {
            s0 = s0.add(term(l2, reps[rb + j], points, j * n + p));
            s1 = s1.add(term(l2, reps[rb + j + 1], points, (j + 1) * n + p));
            s2 = s2.add(term(l2, reps[rb + j + 2], points, (j + 2) * n + p));
            s3 = s3.add(term(l2, reps[rb + j + 3], points, (j + 3) * n + p));
        }
        DoubleVector s = s0.add(s1).add(s2.add(s3));
        for (; j < d; j++) {
            s = s.add(term(l2, reps[rb + j], points, j * n + p));
        }
        return l2 ? s.sqrt() : s;
    }

    /** (double) (r - x) squared, or its absolute value, for LANES consecutive x. */
    private static DoubleVector term(boolean l2, float r, float[] points, int at) {
        FloatVector x = FloatVector.broadcast(FS, r).sub(FloatVector.fromArray(FS, points, at));
        DoubleVector y = (DoubleVector) x.convertShape(VectorOperators.F2D, DS, 0);
        return l2 ? y.mul(y) : y.abs();
    }
}
