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

/**
 * The only entry point. Chooses between {@link VectorSupportSIMD} (Vector API)
 * and {@link VectorSupportLegacy} (plain Java) and forwards.
 *
 * <h2>Why this class exists</h2>
 * <p>
 * The Vector API lives in an incubator module, which is not in the default root
 * set. Compiling against it is not enough: without
 * {@code --add-modules jdk.incubator.vector} on the <em>runtime</em> command
 * line the module never resolves, and the first code to touch
 * {@code FloatVector} dies with {@code NoClassDefFoundError}. There is no way
 * to bake the flag into a jar -- JEP 261 gave executable jars
 * {@code Add-Exports} and {@code Add-Opens} and deliberately not
 * {@code Add-Modules} -- so any consumer of this library from Maven Central who
 * does not know to pass it gets a crash from ten frames inside a scoring call
 * stack, nowhere near anything they wrote. They would be right to file that as
 * a bug.
 *
 * <p>
 * So the flag has to be optional, and that means the decision has to be made by
 * a class that has <b>no compile-time reference to {@code jdk.incubator.vector}
 * at all</b> -- otherwise it has already crashed by the time it could decide.
 * That is this class. Note there is not one incubator import below, and every
 * signature is primitives and primitive arrays: the verifier needs no
 * assignability checks, so it never loads the vector classes to typecheck this
 * one.
 *
 * <h2>How the graceful path works</h2>
 * <p>
 * Constant-pool entries resolve on <em>first execution</em>, not at compile or
 * class-load. {@link #SIMD} is {@code static final}, so when it is false C2
 * folds {@code SIMD && n >= MIN_LEN} to a constant false and dead-code
 * eliminates the entire {@code VectorSupportSIMD} branch. The branch never
 * executes, so its constant-pool entry never resolves, so the missing module
 * never surfaces. Not caught -- never reached.
 *
 * <p>
 * When the module IS present, {@code SIMD} folds to true and what remains is a
 * single {@code n >= MIN_LEN} compare against a {@code static final} int, on a
 * ~12-bytecode method that inlines into its caller. {@code n} is fixed for the
 * life of a forest, so the branch is perfectly predicted and C2 prunes the
 * untaken side behind an uncommon trap. The dispatch is free; confirm with
 * ScoringBenchmark rather than taking that on faith.
 *
 * <h2>Invariant</h2>
 * <p>
 * <b>No vector value ever crosses this boundary.</b> Every {@code FloatVector}
 * is created and consumed inside one {@code VectorSupportSIMD} method, which is
 * the condition under which C2's box elimination works -- C2 special-cases
 * vector instances to avoid boxing rather than relying on general escape
 * analysis, and letting one escape (into a field, or across a boundary like
 * this one) costs real heap allocation in the hot loop. If a method here ever
 * takes or returns a {@code Vector}, both that invariant and the verifier one
 * above break at once.
 *
 * <h2>Knobs</h2>
 * <ul>
 * <li>{@code -Drcf.vector.enabled=false} -- force the scalar path even where
 * the module is available. Also the only way to test the fallback without
 * uninstalling the JDK module.</li>
 * <li>{@code -Drcf.vector.minLength=N} -- scalar below n=N. Default
 * {@code LANES}, below which the vector loop cannot execute a single iteration.
 * Raise it once the A/B crossover has been measured on a target arch; set 0 to
 * force the vector path at every size (the benchmark does this).</li>
 * </ul>
 */
public final class VectorSupport {

    /**
     * True iff the incubator module resolved AND its class initialized cleanly AND
     * the user did not opt out. static final, hence constant-folded.
     */
    private static final boolean SIMD = probe();

    /**
     * Minimum n worth vectorizing. Integer.MAX_VALUE when SIMD is false, but note
     * the guard tests SIMD first: {@code n >= Integer.MAX_VALUE} would NOT fold,
     * since C2 cannot bound n, whereas {@code SIMD &&} short-circuits at compile
     * time and removes the branch entirely.
     */
    private static final int MIN_LEN = SIMD ? minLength() : Integer.MAX_VALUE;

    private VectorSupport() {
    }

    private static boolean probe() {
        if (!Boolean.parseBoolean(System.getProperty("rcf.vector.enabled", "true"))) {
            return false;
        }
        try {
            // Both of these must happen INSIDE the try. The forName gives a clean
            // failure when the module is unresolved; the lanes() call forces
            // VectorSupportSIMD's <clinit>, which is where FloatVector.SPECIES_PREFERRED
            // is actually touched, so a failure there is caught here rather than at
            // some arbitrary later call site.
            Class.forName("jdk.incubator.vector.FloatVector", false, VectorSupport.class.getClassLoader());
            return VectorSupportSIMD.lanes() > 0;
        } catch (Throwable t) {
            // ClassNotFoundException / NoClassDefFoundError: no --add-modules.
            // ExceptionInInitializerError: the class loaded but <clinit> blew up.
            // Either way: run scalar, do not crash the caller.
            return false;
        }
    }

    private static int minLength() {
        return Integer.getInteger("rcf.vector.minLength", VectorSupportSIMD.lanes());
    }

    /**
     * Constant-folded. When SIMD is false this is a compile-time false and the
     * caller's SIMD branch is eliminated outright.
     */
    private static boolean simd(int n) {
        return SIMD && n >= MIN_LEN;
    }

    /**
     * True if calls large enough to qualify will actually take the Vector API path.
     */
    public static boolean isVectorized() {
        return SIMD;
    }

    /** Diagnostics, and the benchmark's arm labelling. */
    public static String describe() {
        return "VectorSupport[simd=" + SIMD + ", minLength=" + MIN_LEN + (SIMD ? ", " + VectorSupportSIMD.shape() : "")
                + "]";
    }

    // ---- gap / cut probability + optional attribution fill ----------------

    /**
     * Single source of probability computation. Returns S / (S + rangeSum) and, if
     * {@code contrib} is non-null and at least {@code 2 * dimensions} long, fills
     * it with the per-half-dimension attribution
     * {@code max(0f, newValues[i] - values[offset+i]) / (rangeSum + S)}, so the
     * entries sum to the returned probability. Per-coordinate gaps stay float; the
     * reduction into S is done in double. Passing null takes the same path and just
     * returns the probability.
     */
    public static double gapAttribution(float[] values, int offset, int dimensions, double rangeSum, float[] newValues,
            int nvOffset, float[] contrib) {
        return simd(2 * dimensions)
                ? VectorSupportSIMD.gapAttribution(values, offset, dimensions, rangeSum, newValues, nvOffset, contrib)
                : VectorSupportLegacy.gapAttribution(values, offset, dimensions, rangeSum, newValues, nvOffset,
                        contrib);
    }

    // ---- box U= box (returns the new rangeSum) -----------------------------

    public static double addSlice(float[] values, int offset, int dimensions, float[] other, int otherOffset) {
        return simd(2 * dimensions) ? VectorSupportSIMD.addSlice(values, offset, dimensions, other, otherOffset)
                : VectorSupportLegacy.addSlice(values, offset, dimensions, other, otherOffset);
    }

    public static boolean contains(float[] values, int offset, int dimensions, float[] other, int otherOffset) {
        return simd(2 * dimensions) ? VectorSupportSIMD.contains(values, offset, dimensions, other, otherOffset)
                : VectorSupportLegacy.contains(values, offset, dimensions, other, otherOffset);
    }

    // ---- expanded query [p, -p] -------------------------------------------

    /**
     * In-place refill (no allocation) for reused visitors. Never had a Vector API
     * form, so it lives here rather than in either implementation.
     */
    public static void expandInto(float[] point, int pointOffset, float[] dest, int destOffset, int dimensions) {
        final int d = dimensions;
        System.arraycopy(point, pointOffset, dest, destOffset, d);
        for (int i = 0; i < d; i++) {
            dest[i + d + destOffset] = -point[i + pointOffset];
        }
    }

    // ---- AttributionVisitor accumulator kernels ---------------------------

    /**
     * Elementwise, no reduction -- C2's SuperWord already emits vmulpd for the
     * scalar form. Benchmark A vs B here specifically: if the Vector API adds
     * nothing, delete it from {@link VectorSupportSIMD} and route this
     * unconditionally to Legacy.
     */
    public static void multiply(double[] dir, double decay) {
        if (simd(dir.length)) {
            VectorSupportSIMD.multiply(dir, decay);
        } else {
            VectorSupportLegacy.multiply(dir, decay);
        }
    }

    /**
     * Fused into one pass: fma(comp, a, decay*dir). Scalar unconditionally -- the
     * widening float->double load of {@code comp} is exactly the F2D conversion
     * that does not reliably intrinsify, and the loop is memory-bound anyway.
     */
    public static void updateRecurrence(double[] dir, float[] comp, double a, double decay) {
        VectorSupportLegacy.updateRecurrence(dir, comp, a, decay);
    }

    /** Elementwise fma. Same autovectorize caveat as {@link #multiply}. */
    public static void axpyInto(double[] dst, double[] src, double factor) {
        if (simd(dst.length)) {
            VectorSupportSIMD.axpyInto(dst, src, factor);
        } else {
            VectorSupportLegacy.axpyInto(dst, src, factor);
        }
    }

    /**
     * PER-TREE sum reduction. Scalar unconditionally: C2 emits a strictly ordered
     * FP reduction, so there is nothing to vectorize without reassociating.
     */
    public static double sum(double[] a) {
        return VectorSupportLegacy.sum(a);
    }

    /**
     * dst[dstOff+i] = max(0, exp[expOff+i] - sign*pt[ptOff+i]); returns the sum
     * over the half. exp = canonical [p,-p] (read-only); pt = RAW point, length
     * dim. Two calls do the leaf: high: (exp,0,+1f,pt,0,dst,0,dim); low:
     * (exp,dim,-1f,pt,0,dst,dim,dim). No leaf box, no copy of exp.
     */
    public static double signedGapInto(float[] exp, int expOff, float sign, float[] pt, int ptOff, float[] dst,
            int dstOff, int dim) {
        return simd(dim) ? VectorSupportSIMD.signedGapInto(exp, expOff, sign, pt, ptOff, dst, dstOff, dim)
                : VectorSupportLegacy.signedGapInto(exp, expOff, sign, pt, ptOff, dst, dstOff, dim);
    }

    public static double addPointInPlace(float[] values, int offset, int dim, float[] point, int pOff) {
        return simd(dim) ? VectorSupportSIMD.addPointInPlace(values, offset, dim, point, pOff)
                : VectorSupportLegacy.addPointInPlace(values, offset, dim, point, pOff);
    }

    /** shadow path: raw box-vs-box gap dst = max(0, nv - v); returns sum. */
    public static double gapInto(float[] nv, int nvOff, float[] v, int vOff, float[] dst, int dstOff, int n) {
        return simd(n) ? VectorSupportSIMD.gapInto(nv, nvOff, v, vOff, dst, dstOff, n)
                : VectorSupportLegacy.gapInto(nv, nvOff, v, vOff, dst, dstOff, n);
    }

    /**
     * gap -> prob in place; dist = prob*(gap + oldRange), oldRange read inline from
     * the SMALL box: oldRange[i] = box[boxOff+i] + box[boxOff+i+dim] (= max_i -
     * min_i), identical across both halves.
     */
    public static void probAndDistInto(float[] gap, float[] dist, float[] box, int boxOff, int dim, double invSumNew) {
        if (simd(dim)) {
            VectorSupportSIMD.probAndDistInto(gap, dist, box, boxOff, dim, invSumNew);
        } else {
            VectorSupportLegacy.probAndDistInto(gap, dist, box, boxOff, dim, invSumNew);
        }
    }

    /**
     * leaf seed: prob = gap*inv (in place), dist = prob*gap. oldRange == 0
     * (degenerate box).
     */
    public static void probOnlyInto(float[] gap, float[] dist, int len, double invSumNew) {
        if (simd(len)) {
            VectorSupportSIMD.probOnlyInto(gap, dist, len, invSumNew);
        } else {
            VectorSupportLegacy.probOnlyInto(gap, dist, len, invSumNew);
        }
    }

    // ---- distances ---------------------------------------------------------

    public static double L1distance(float[] a, float[] b) {
        return simd(a.length) ? VectorSupportSIMD.L1distance(a, b) : VectorSupportLegacy.L1distance(a, b);
    }

    public static double L2distance(float[] a, float[] b) {
        return simd(a.length) ? VectorSupportSIMD.L2distance(a, b) : VectorSupportLegacy.L2distance(a, b);
    }

    public static double LInfinitydistance(float[] a, float[] b) {
        return simd(a.length) ? VectorSupportSIMD.LInfinitydistance(a, b) : VectorSupportLegacy.LInfinitydistance(a, b);
    }
}
