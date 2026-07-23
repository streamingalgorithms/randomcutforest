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
import static org.junit.jupiter.api.Assumptions.assumeTrue;
import static org.streamingalgorithms.randomcutforest.summarization.Summarizer.*;

import java.util.Random;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

/**
 * {@link VectorSupportSIMD} vs {@link VectorSupportLegacy} agree to 1e-6
 * relative.
 *
 * <p>
 * Deliberately calls {@link VectorSupportSIMD} directly rather than going
 * through {@link VectorSupport}. The facade's whole job is to send small inputs
 * to Legacy -- which is precisely what must NOT happen here, or both arms would
 * be Legacy and every assertion below would pass while comparing a thing to
 * itself. Asking the facade to stand down (via
 * {@code -Drcf.vector.minLength=0}) cannot be done reliably from a test class:
 * the property is read in VectorSupport's {@code <clinit>}, surefire reuses one
 * JVM across test classes, and any earlier test that touches a bounding box
 * wins the race. So: skip the facade, test the kernels. That is what this file
 * is for anyway.
 *
 * <p>
 * Requires {@code --add-modules jdk.incubator.vector} on the surefire argLine.
 *
 * <p>
 * The dimension list mixes even, odd, prime and just-off-a-power-of-two sizes
 * so that on any host every path gets hit: the 4-chunk block loop, the
 * single-chunk remainder loop, the 4-way-unrolled scalar tail, and the
 * 1-at-a-time tail remainder. Which dim exercises which depends on LANES (4 on
 * NEON, 8 on AVX2, 16 on AVX-512), so the list is deliberately dense rather
 * than clever.
 *
 * <p>
 * 1e-6 is loose on purpose. A sums in float within a block of 4*LANES before
 * promoting to double; B accumulates in double throughout. They differ by a few
 * ulp (~1e-7 relative) by construction, and A's own answer moves with lane
 * count from host to host. Anything tighter would be testing the host, not the
 * code. Arrays written elementwise should be bit-identical, but are held to the
 * same 1e-6 here to keep this file simple.
 */
public class VectorSupportParityTest {

    private static final double TOL = 1e-6;

    @BeforeAll
    static void requireModule() {
        // Under -Dvector.runtime.addmodules=java.base the incubator module is absent
        // and VectorSupportSIMD cannot load at all. Skip rather than fail: proving the
        // fallback works is that CI job's business, and it does it via the other ~1000
        // tests, not this one.
        //
        // Safe because constant-pool entries resolve on first EXECUTION: an aborted
        // assumption means no test method body runs, so VectorSupportSIMD is never
        // resolved. (Loading this class is fine either way -- every signature below is
        // primitives and primitive arrays, so the verifier needs no assignability
        // check and never loads a vector class to typecheck it.)
        assumeTrue(VectorSupport.isVectorized(),
                "incubator module not resolved, nothing to compare: " + VectorSupport.describe());
    }

    // ---- kernels -----------------------------------------------------------
    // The dim lists are spelled out per test rather than shared: @ValueSource needs
    // a compile-time constant, and the alternative (@MethodSource) buys nothing
    // here.

    @ParameterizedTest
    @ValueSource(ints = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 13, 15, 16, 17, 23, 24, 30, 31, 32, 33, 47, 63, 64, 65, 100,
            127, 128, 129, 255, 256, 257 })
    void gapAttributionProbOnly(int dim) {
        Fixture f = new Fixture(dim);
        double a = VectorSupportSIMD.gapAttribution(f.values, 0, dim, f.rangeSum, f.newValues, 0, null);
        double b = VectorSupportLegacy.gapAttribution(f.values, 0, dim, f.rangeSum, f.newValues, 0, null);
        close("gapAttribution(prob)", dim, a, b);
    }

    @ParameterizedTest
    @ValueSource(ints = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 13, 15, 16, 17, 23, 24, 30, 31, 32, 33, 47, 63, 64, 65, 100,
            127, 128, 129, 255, 256, 257 })
    void gapAttributionWithFill(int dim) {
        Fixture f = new Fixture(dim);
        float[] cA = new float[f.n], cB = new float[f.n];
        double a = VectorSupportSIMD.gapAttribution(f.values, 0, dim, f.rangeSum, f.newValues, 0, cA);
        double b = VectorSupportLegacy.gapAttribution(f.values, 0, dim, f.rangeSum, f.newValues, 0, cB);
        close("gapAttribution(fill)", dim, a, b);
        close("gapAttribution.contrib", dim, cA, cB);
    }

    /** The pooled-buffer case: contrib LONGER than n must still be filled. */
    @ParameterizedTest
    @ValueSource(ints = { 3, 8, 17, 30, 64, 129 })
    void gapAttributionPooledContrib(int dim) {
        Fixture f = new Fixture(dim);
        float[] exact = new float[f.n];
        float[] pooled = new float[f.n + 37]; // oversized, as a pool hands out
        VectorSupportSIMD.gapAttribution(f.values, 0, dim, f.rangeSum, f.newValues, 0, exact);
        VectorSupportSIMD.gapAttribution(f.values, 0, dim, f.rangeSum, f.newValues, 0, pooled);
        for (int i = 0; i < f.n; i++) {
            assertEquals(exact[i], pooled[i], 0f, "pooled contrib differs at " + i + " (dim=" + dim + ")");
        }
    }

    @ParameterizedTest
    @ValueSource(ints = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 13, 15, 16, 17, 23, 24, 30, 31, 32, 33, 47, 63, 64, 65, 100,
            127, 128, 129, 255, 256, 257 })
    void addSlice(int dim) {
        Fixture f = new Fixture(dim);
        float[] vA = f.values.clone(), vB = f.values.clone();
        double a = VectorSupportSIMD.addSlice(vA, 0, dim, f.other, 0);
        double b = VectorSupportLegacy.addSlice(vB, 0, dim, f.other, 0);
        close("addSlice", dim, a, b);
        close("addSlice.values", dim, vA, vB);
    }

    @ParameterizedTest
    @ValueSource(ints = { 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 30, 31, 33, 63, 64, 65, 127, 128, 129, 256, 257 })
    void contains(int dim) {
        Fixture f = new Fixture(dim);
        // other is strictly inside values -> true, the full-scan case
        assertEquals(VectorSupportLegacy.contains(f.values, 0, dim, f.other, 0),
                VectorSupportSIMD.contains(f.values, 0, dim, f.other, 0), "contains(true) dim=" + dim);
        // poke one coordinate out of the box -> false, at every position in turn
        for (int k = 0; k < f.n; k++) {
            float[] o = f.other.clone();
            o[k] = f.values[k] + 1f;
            assertEquals(VectorSupportLegacy.contains(f.values, 0, dim, o, 0),
                    VectorSupportSIMD.contains(f.values, 0, dim, o, 0), "contains(false@" + k + ") dim=" + dim);
        }
    }

    @ParameterizedTest
    @ValueSource(ints = { 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 23, 24, 30, 31, 33, 63, 64, 65, 127, 128, 129, 256, 257 })
    void multiply(int dim) {
        Fixture f = new Fixture(dim);
        double[] dA = f.dir.clone(), dB = f.dir.clone();
        VectorSupportSIMD.multiply(dA, 0.97);
        VectorSupportLegacy.multiply(dB, 0.97);
        close("multiply", dim, dA, dB);
    }

    @ParameterizedTest
    @ValueSource(ints = { 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 23, 24, 30, 31, 33, 63, 64, 65, 127, 128, 129, 256, 257 })
    void updateRecurrence(int dim) {
        Fixture f = new Fixture(dim);
        double[] dA = f.dir.clone(), dB = f.dir.clone();
        VectorSupport.updateRecurrence(dA, f.comp, 0.25, 0.97);
        VectorSupportLegacy.updateRecurrence(dB, f.comp, 0.25, 0.97);
        close("updateRecurrence", dim, dA, dB);
    }

    @ParameterizedTest
    @ValueSource(ints = { 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 23, 24, 30, 31, 33, 63, 64, 65, 127, 128, 129, 256, 257 })
    void axpyInto(int dim) {
        Fixture f = new Fixture(dim);
        double[] dA = f.dst.clone(), dB = f.dst.clone();
        VectorSupportSIMD.axpyInto(dA, f.src, 1e-3);
        VectorSupportLegacy.axpyInto(dB, f.src, 1e-3);
        close("axpyInto", dim, dA, dB);
    }

    @ParameterizedTest
    @ValueSource(ints = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 13, 15, 16, 17, 23, 24, 30, 31, 32, 33, 47, 63, 64, 65, 100,
            127, 128, 129, 255, 256, 257 })
    void signedGapInto(int dim) {
        Fixture f = new Fixture(dim);
        float[] gA = new float[f.n], gB = new float[f.n];
        double aHi = VectorSupportSIMD.signedGapInto(f.expanded, 0, +1f, f.point, 0, gA, 0, dim);
        double aLo = VectorSupportSIMD.signedGapInto(f.expanded, dim, -1f, f.point, 0, gA, dim, dim);
        double bHi = VectorSupportLegacy.signedGapInto(f.expanded, 0, +1f, f.point, 0, gB, 0, dim);
        double bLo = VectorSupportLegacy.signedGapInto(f.expanded, dim, -1f, f.point, 0, gB, dim, dim);
        close("signedGapInto(hi)", dim, aHi, bHi);
        close("signedGapInto(lo)", dim, aLo, bLo);
        close("signedGapInto.dst", dim, gA, gB);
    }

    @ParameterizedTest
    @ValueSource(ints = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 13, 15, 16, 17, 23, 24, 30, 31, 32, 33, 47, 63, 64, 65, 100,
            127, 128, 129, 255, 256, 257 })
    void addPointInPlace(int dim) {
        Fixture f = new Fixture(dim);
        float[] vA = f.values.clone(), vB = f.values.clone();
        double a = VectorSupportSIMD.addPointInPlace(vA, 0, dim, f.point, 0);
        double b = VectorSupportLegacy.addPointInPlace(vB, 0, dim, f.point, 0);
        close("addPointInPlace", dim, a, b);
        close("addPointInPlace.values", dim, vA, vB);
    }

    @ParameterizedTest
    @ValueSource(ints = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 13, 15, 16, 17, 23, 24, 30, 31, 32, 33, 47, 63, 64, 65, 100,
            127, 128, 129, 255, 256, 257 })
    void gapInto(int dim) {
        Fixture f = new Fixture(dim);
        float[] gA = new float[f.n], gB = new float[f.n];
        double a = VectorSupportSIMD.gapInto(f.newValues, 0, f.values, 0, gA, 0, f.n);
        double b = VectorSupportLegacy.gapInto(f.newValues, 0, f.values, 0, gB, 0, f.n);
        close("gapInto", dim, a, b);
        close("gapInto.dst", dim, gA, gB);
    }

    @ParameterizedTest
    @ValueSource(ints = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 13, 15, 16, 17, 23, 24, 30, 31, 32, 33, 47, 63, 64, 65, 100,
            127, 128, 129, 255, 256, 257 })
    void probAndDistInto(int dim) {
        Fixture f = new Fixture(dim);
        float[] gA = f.gap.clone(), gB = f.gap.clone();
        float[] dA = new float[f.n], dB = new float[f.n];
        VectorSupportSIMD.probAndDistInto(gA, dA, f.values, 0, dim, f.invSum);
        VectorSupportLegacy.probAndDistInto(gB, dB, f.values, 0, dim, f.invSum);
        close("probAndDistInto.gap", dim, gA, gB);
        close("probAndDistInto.dist", dim, dA, dB);
    }

    @ParameterizedTest
    @ValueSource(ints = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 13, 15, 16, 17, 23, 24, 30, 31, 32, 33, 47, 63, 64, 65, 100,
            127, 128, 129, 255, 256, 257 })
    void probOnlyInto(int dim) {
        Fixture f = new Fixture(dim);
        float[] gA = f.gap.clone(), gB = f.gap.clone();
        float[] dA = new float[f.n], dB = new float[f.n];
        VectorSupportSIMD.probOnlyInto(gA, dA, f.n, f.invSum);
        VectorSupportLegacy.probOnlyInto(gB, dB, f.n, f.invSum);
        close("probOnlyInto.gap", dim, gA, gB);
        close("probOnlyInto.dist", dim, dA, dB);
    }

    @ParameterizedTest
    @ValueSource(ints = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 13, 15, 16, 17, 23, 24, 30, 31, 32, 33, 47, 63, 64, 65, 100,
            127, 128, 129, 255, 256, 257 })
    void distances(int dim) {
        Fixture f = new Fixture(dim);
        close("L1", dim, VectorSupportSIMD.L1distance(f.va, f.vb), VectorSupportLegacy.L1distance(f.va, f.vb));
        close("L2", dim, VectorSupportSIMD.L2distance(f.va, f.vb), VectorSupportLegacy.L2distance(f.va, f.vb));
        close("LInf", dim, VectorSupportSIMD.LInfinitydistance(f.va, f.vb),
                VectorSupportLegacy.LInfinitydistance(f.va, f.vb));
    }

    /** Degenerate inputs: all-zero gaps, identical boxes, S == 0 branches. */
    @ParameterizedTest
    @ValueSource(ints = { 1, 3, 8, 17, 30, 64, 129 })
    void degenerate(int dim) {
        final int n = 2 * dim;
        float[] box = new float[n];
        java.util.Arrays.fill(box, 2f);
        float[] same = box.clone();
        float[] cA = new float[n], cB = new float[n];

        // newValues == values -> every gap 0 -> S == 0 -> probability 0, contrib zeroed
        close("degenerate.prob", dim, VectorSupportSIMD.gapAttribution(box, 0, dim, 4.0 * dim, same, 0, cA),
                VectorSupportLegacy.gapAttribution(box, 0, dim, 4.0 * dim, same, 0, cB));
        close("degenerate.contrib", dim, cA, cB);

        // rangeSum == 0 with a real gap -> probability 1.0
        float[] bigger = new float[n];
        java.util.Arrays.fill(bigger, 3f);
        close("degenerate.rangeSum0", dim, VectorSupportSIMD.gapAttribution(box, 0, dim, 0.0, bigger, 0, null),
                VectorSupportLegacy.gapAttribution(box, 0, dim, 0.0, bigger, 0, null));

        // identical vectors -> all distances 0
        close("degenerate.L2", dim, VectorSupportSIMD.L2distance(same, box), VectorSupportLegacy.L2distance(same, box));
    }

    // ---- fixture + comparison ---------------------------------------------

    /**
     * Seeded per dimension, so a failure is reproducible from the test name alone.
     */
    private static final class Fixture {
        final int n;
        final float[] values, other, newValues, point, expanded, gap, va, vb, comp;
        final double[] dir, src, dst;
        final double rangeSum, invSum;

        Fixture(int dim) {
            final Random r = new Random(9176L * dim + 13);
            n = 2 * dim;
            values = new float[n];
            other = new float[n];
            newValues = new float[n];
            expanded = new float[n];
            gap = new float[n];
            comp = new float[n];
            point = new float[dim];
            va = new float[dim];
            vb = new float[dim];
            dir = new double[n];
            src = new double[n];
            dst = new double[n];

            for (int i = 0; i < dim; i++) {
                values[i] = 1f + Math.abs((float) r.nextGaussian()); // max_i
                values[i + dim] = 1f + Math.abs((float) r.nextGaussian()); // -min_i
                other[i] = values[i] * 0.5f;
                other[i + dim] = values[i + dim] * 0.5f;
                // straddles the box: a mix of zero and non-zero gaps, which is the point
                newValues[i] = values[i] + (float) r.nextGaussian();
                newValues[i + dim] = values[i + dim] + (float) r.nextGaussian();
                point[i] = (float) (2.0 * r.nextGaussian());
                va[i] = (float) r.nextGaussian();
                vb[i] = (float) r.nextGaussian();
            }
            for (int i = 0; i < n; i++) {
                gap[i] = Math.abs((float) r.nextGaussian());
                comp[i] = (float) r.nextGaussian();
                dir[i] = r.nextGaussian();
                src[i] = r.nextGaussian();
                dst[i] = r.nextGaussian();
            }
            VectorSupport.expandInto(point, 0, expanded, 0, dim);

            double s = 0;
            for (float v : values) {
                s += v;
            }
            rangeSum = s;
            invSum = 1.0 / (s + 1e-9);
        }
    }

    private static void close(String what, int dim, double a, double b) {
        double scale = Math.max(1.0, Math.max(Math.abs(a), Math.abs(b)));
        assertTrue(Math.abs(a - b) <= TOL * scale,
                what + " dim=" + dim + ": A=" + a + " B=" + b + " (err " + Math.abs(a - b) + ")");
    }

    private static void close(String what, int dim, float[] a, float[] b) {
        assertEquals(a.length, b.length, what + " length");
        for (int i = 0; i < a.length; i++) {
            close(what + "[" + i + "]", dim, a[i], b[i]);
        }
    }

    private static void close(String what, int dim, double[] a, double[] b) {
        assertEquals(a.length, b.length, what + " length");
        for (int i = 0; i < a.length; i++) {
            close(what + "[" + i + "]", dim, a[i], b[i]);
        }
    }

    private float[] randomPoint(int len, Random rng) {
        float[] p = new float[len];
        for (int i = 0; i < len; i++) {
            p[i] = (float) (rng.nextGaussian() * 10.0);
        }
        return p;
    }

    @ParameterizedTest
    @ValueSource(ints = { 1, 5, 50 })
    void testLP(int dim) {
        float[] point1 = randomPoint(dim, new Random());
        float[] point2 = randomPoint(dim, new Random());
        double dist = L1distance(point1, point2);
        assertEquals(dist, VectorSupport.L1distance(point1, point2), dist * 1e-6f);
        assertEquals(L2distance(point1, point2), VectorSupport.L2distance(point1, point2), dist * 1e-6f);
        assertEquals(LInfinitydistance(point1, point2), VectorSupport.LInfinitydistance(point1, point2), 1e-6f);
    }

    @ParameterizedTest
    @ValueSource(ints = { 2, 10, 50 })
    void gapBox(int dim) {
        // has to be even
        float[] box1 = randomPoint(dim, new Random());
        float[] box2 = randomPoint(dim, new Random());
        float[] dst1 = new float[dim];
        float[] dst2 = new float[dim];
        double t1 = VectorSupportSIMD.gapInto(box1, 0, box2, 0, dst1, 0, dim / 2);
        double t2 = VectorSupportLegacy.gapInto(box1, 0, box2, 0, dst2, 0, dim / 2);
        double t3 = VectorSupport.gapInto(box1, 0, box2, 0, dst2, 0, dim / 2);
        assertEquals(t1, t2, t1 * 1e-6);
        assertTrue(t1 == t3 || t2 == t3);
        assertArrayEquals(dst1, dst2, 1e-6f);
        assertTrue(VectorSupportSIMD.contains(box1, 0, dim / 2, box1, 0));
        assertTrue(VectorSupportLegacy.contains(box1, 0, dim / 2, box1, 0));
        assertTrue(VectorSupport.contains(box1, 0, dim / 2, box1, 0));
    }

    @Test
    public void fusedEqualsUnfused() {
        for (int dim : new int[] { 1, 3, 4, 7, 8, 16, 30, 100 }) {
            Random r = new Random(42);
            int nPoints = 5, cap = 64;
            float[] store = new float[cap * dim];
            for (int i = 0; i < store.length; i++)
                store[i] = (float) r.nextGaussian();

            float[] pt = new float[dim];
            for (int i = 0; i < dim; i++)
                pt[i] = (float) r.nextGaussian();
            float[] exp = new float[2 * dim];
            VectorSupport.expandInto(pt, 0, exp, 0, dim);

            int[] offs = new int[nPoints];
            for (int i = 0; i < nPoints; i++)
                offs[i] = r.nextInt(cap) * dim;

            // seed two identical boxes from a leaf point
            float[] a = new float[2 * dim], b = new float[2 * dim];
            for (int i = 0; i < dim; i++) {
                a[i] = b[i] = pt[i];
                a[dim + i] = b[dim + i] = -pt[i];
            }

            // reference: union, then gap, as two separate passes
            double refR = VectorSupport.updateBoundsAll(a, 0, dim, store, offs.clone(), 0, nPoints);
            float[] refGap = new float[2 * dim];
            double refP = VectorSupport.gapAttribution(a, 0, dim, refR, exp, 0, refGap);

            // fused
            double[] out = new double[1];
            float[] fusedGap = new float[2 * dim];
            double fusedS = VectorSupport.updateBoundsAndGap(b, 0, dim, store, offs.clone(), 0, nPoints, exp, 0,
                    fusedGap, 0, out);
            double fusedR = out[0];

            assertArrayEquals(a, b, 0f); // boxes bit-identical: max is exact
            assertEquals(refR, fusedR, 1e-5 * Math.max(1.0, Math.abs(refR))); // float ULP, not double

            double fusedP = (fusedS == 0) ? 0 : (fusedR == 0 ? 1 : fusedS / (fusedS + fusedR));
            assertEquals(refP, fusedP, 1e-9);
            // refGap is scaled by 1/(R+S); fusedGap is raw
            for (int i = 0; i < 2 * dim; i++)
                assertEquals(refGap[i], fusedGap[i] / (float) (refR + fusedS), 1e-6f);
        }
    }
}
