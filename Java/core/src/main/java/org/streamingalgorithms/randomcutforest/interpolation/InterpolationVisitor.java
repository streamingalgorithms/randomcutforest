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

package org.streamingalgorithms.randomcutforest.interpolation;

import java.util.Arrays;

import org.streamingalgorithms.randomcutforest.DefaultScoreFunctions;
import org.streamingalgorithms.randomcutforest.IRFVisitor;
import org.streamingalgorithms.randomcutforest.IVisitorFactory;
import org.streamingalgorithms.randomcutforest.RFVisitor;
import org.streamingalgorithms.randomcutforest.Visitor;
import org.streamingalgorithms.randomcutforest.returntypes.DiVector;
import org.streamingalgorithms.randomcutforest.returntypes.InterpolationMeasure;
import org.streamingalgorithms.randomcutforest.tree.*;

/**
 * Flat 2*dim interpolation walk (follows AttributionVisitor). Three
 * double[2*dim] accumulators, [0,dim)=high / [dim,2*dim)=low;
 * InterpolationMeasure materialized only at the getResult / getFoldResult
 * boundary via DiVector(double[]).
 *
 * <p>
 * Gap is computed ONCE per node via a single gapInto over the SMALL box: point
 * mode -&gt; newValues = expandedPoint [p,-p]; shadow mode -&gt; newValues =
 * node box slice. sumOfNewRange = small.rangeSum + S (the rangeSum field
 * already holds Σ oldRange). oldRange for the distance term is read inline from
 * the small box in probAndDistInto.
 *
 * <p>
 * <b>What the accumulators mean.</b> The walk induces a first-passage
 * distribution over (face j, level v) with weight S(v)·π_j(v) summing to
 * exactly 1, where S is the survival product over ancestors and π_j the
 * probability that the separating cut lands in half-dimension j. Every
 * accumulator is an expectation against that one distribution, differing only
 * in the integrand:
 * <ul>
 * <li>{@code probMass} — the distribution itself (integrand 1)</li>
 * <li>{@code measure} — node mass, so its high-low sum is DISP(x, S) +
 * pointMass exactly, per Lemma 1 of Guha et al. (2016). The walk integrates
 * over the cut analytically instead of sampling an insertion, so it is
 * Rao-Blackwellized.</li>
 * <li>{@code distances} — the length scale at separation</li>
 * <li>{@code savedHeight} — the score function, i.e. expected inverse
 * height</li>
 * </ul>
 *
 * <p>
 * <b>savedScore is redundant, savedHeight is not.</b> The scalar and
 * directional recurrences share a scalar fieldVal, so savedScore is identically
 * measure.getHighLowSum() and no override can make them differ. Expected
 * inverse height needs a genuinely separate recurrence with a depth-aware
 * weight, which is what savedHeight is. Both now leave through the result
 * rather than being computed and discarded.
 *
 * <p>
 * <b>Duplicate policy.</b> Like AttributionVisitor and unlike ScoreVisitor,
 * this visitor does not converge at a duplicate leaf: it sets pointEqualsLeaf
 * and keeps climbing with the shadow box. So savedHeight agrees with
 * ScoreVisitor exactly for queries that are not present in the tree, and
 * deliberately differs for those that are. A regression test asserting equality
 * must exclude duplicates. This visitor also has no ignoreLeafMassThreshold;
 * add it here if the height is to match ScoreVisitor under that setting too.
 */
public class InterpolationVisitor extends RFVisitor<InterpolationMeasure> {

    private final double pointMass;
    private final boolean centerOfMass;

    protected final int dim;
    protected final int len; // 2 * dim

    protected final DefaultScoreFunctions.ScoreFn scoreSeenFn;
    protected final DefaultScoreFunctions.ScoreFn scoreUnseenFn;
    protected final DefaultScoreFunctions.DampFn dampFn;
    protected final DefaultScoreFunctions.Normalizer normalizer;

    // flat accumulators: [0,dim)=high, [dim,2*dim)=low
    private final double[] measure;
    private final double[] distances;
    private final double[] probMass;
    private int sampleSize;

    // per-query folds — NOT cleared in reset()
    private final double[] foldedMeasure;
    private final double[] foldedDistances;
    private final double[] foldedProbMass;
    private int foldedSampleSize;
    private int foldedTreeCount;

    private double savedScore;
    private double foldedScore; // NOT cleared in reset()

    private double savedHeight;
    private double foldedHeight; // NOT cleared in reset()

    private boolean pointEqualsLeaf;
    private double savedMass;

    // float scratch, fully overwritten per node — never cleared
    private final float[] gap; // expandedPoint gap -> prob, in place
    private final float[] distComp; // prob * (gap + oldRange)

    private double sumOfNewRange;

    InterpolationVisitor(int dimension, double pointMass, boolean centerOfMass) {
        this(dimension, pointMass, centerOfMass, DefaultScoreFunctions.DEFAULT_SCORE_SEEN,
                DefaultScoreFunctions.DEFAULT_SCORE_UNSEEN, DefaultScoreFunctions.DEFAULT_DAMP,
                DefaultScoreFunctions.DEFAULT_NORMALIZER);
    }

    InterpolationVisitor(int dimension, double pointMass, boolean centerOfMass,
            DefaultScoreFunctions.ScoreFn scoreSeenFn, DefaultScoreFunctions.ScoreFn scoreUnseenFn,
            DefaultScoreFunctions.DampFn dampFn, DefaultScoreFunctions.Normalizer normalizer) {
        this.pointMass = pointMass;
        this.centerOfMass = centerOfMass;
        this.dim = dimension;
        this.len = 2 * dimension;
        this.scoreSeenFn = scoreSeenFn;
        this.scoreUnseenFn = scoreUnseenFn;
        this.dampFn = dampFn;
        this.normalizer = normalizer;
        this.measure = new double[len];
        this.distances = new double[len];
        this.probMass = new double[len];
        this.foldedMeasure = new double[len];
        this.foldedDistances = new double[len];
        this.foldedProbMass = new double[len];
        this.gap = new float[len];
        this.distComp = new float[len];
        setDefaults();
    }

    public InterpolationVisitor(float[] pointToScore, int treeMass, double pointMass, boolean centerOfMass) {
        this(pointToScore.length, pointMass, centerOfMass);
        this.treeMass = treeMass;
    }

    private void setDefaults() {
        savedScore = 0.0;
        savedHeight = 0.0;
        pointEqualsLeaf = false;
        pointInsideBox = false;
        shadowBoxActive = false;
        sampleSize = 0;
        Arrays.fill(measure, 0.0);
        Arrays.fill(distances, 0.0);
        Arrays.fill(probMass, 0.0);
        // folded*, foldedScore, foldedHeight deliberately NOT cleared
    }

    @Override
    protected void reset() {
        setDefaults();
    }

    /**
     * One gapInto over the small box. Sets sumOfNewRange; returns S.
     */
    private double computeGap(ArrayBox small, float[] nv, int nvOff) {
        double S = VectorSupport.gapInto(nv, nvOff, small.values, small.offset, gap, 0, len);
        sumOfNewRange = small.getRangeSum() + S; // rangeSum field == Σ oldRange
        return S;
    }

    /**
     * gap -&gt; prob + distComp (reads small box for oldRange), then the three
     * recurrences. decay = 0 seeds the leaf (comp*a, no prior); 1 - probOfCut on
     * interior nodes.
     */
    private void recur(double fieldVal, double influenceVal, double decay) {
        VectorSupport.updateRecurrence(measure, gap, fieldVal, decay);
        VectorSupport.updateRecurrence(probMass, gap, influenceVal, decay);
        VectorSupport.updateRecurrence(distances, distComp, influenceVal, decay);
    }

    @Override
    public void accept(INodeView node, int depthOfNode) {
        if (pointInsideBox) {
            return;
        }

        ArrayBox small;
        double S;

        if (pointEqualsLeaf) {
            small = growShadow((ArrayBox) node.getSiblingBoundingBox());
            ArrayBox large = (ArrayBox) node.getBoundingBox();
            S = computeGap(small, large.values, large.offset);
        } else {
            small = (ArrayBox) node.getBoundingBox();
            S = computeGap(small, node.expanded(), 0);
        }

        double probOfCut = (sumOfNewRange == 0.0) ? 0.0 : S / sumOfNewRange;
        if (probOfCut <= 0) {
            pointInsideBox = true;
            return;
        }

        // note field and influence are uniform at this moment
        double fieldVal = fieldExt(node, depthOfNode, centerOfMass, savedMass, null);
        double influenceVal = influenceExt(node, depthOfNode, centerOfMass, savedMass, null);
        double heightVal = heightExt(node, depthOfNode, savedMass);
        double invSumNew = (sumOfNewRange == 0.0) ? 0.0 : 1.0 / sumOfNewRange;
        VectorSupport.probAndDistInto(gap, distComp, small.values, small.offset, dim, invSumNew);
        deposit(gap, distComp, 1.0 - probOfCut, node.getMass());
        recur(fieldVal, influenceVal, 1.0 - probOfCut);

        savedScore = probOfCut * fieldVal + (1.0 - probOfCut) * savedScore;
        savedHeight = probOfCut * heightVal + (1.0 - probOfCut) * savedHeight;
    }

    @Override
    public void acceptLeaf(INodeView leafNode, int depthOfNode) {

        float[] leaf = leafNode.getLeafPoint();
        float[] expandedPoint = leafNode.expanded();
        double S = VectorSupport.signedGapInto(expandedPoint, 0, +1f, leaf, 0, gap, 0, dim)
                + VectorSupport.signedGapInto(expandedPoint, dim, -1f, leaf, 0, gap, dim, dim);
        sumOfNewRange = S; // leaf rangeSum ≡ 0
        if (S <= 0) {
            savedMass = pointMass + leafNode.getMass();
            pointEqualsLeaf = true;
            double selfF = 0.5 * selfField(leafNode, savedMass) / dim;
            double selfI = 0.5 * selfInfluence(leafNode, savedMass) / dim;
            Arrays.fill(measure, selfF);
            Arrays.fill(probMass, selfI);
            depositDegenerate(selfI);
            savedScore = selfF * 2 * dim;
            savedHeight = heightSeen(leafNode, depthOfNode, leafNode.getMass());
        } else {
            savedMass = pointMass;
            double fieldVal = fieldPoint(leafNode, depthOfNode, savedMass, null);
            double influenceVal = influencePoint(leafNode, depthOfNode, savedMass, null);
            double invSumNew = (sumOfNewRange == 0.0) ? 0.0 : 1.0 / sumOfNewRange;
            VectorSupport.probOnlyInto(gap, distComp, len, invSumNew);
            deposit(gap, distComp, 0.0, leafNode.getMass());
            recur(fieldVal, influenceVal, 0.0); // sumOfNewRange == S here
            savedScore = (sumOfNewRange == 0) ? 0.0 : fieldVal * (S / sumOfNewRange);
            savedHeight = heightExt(leafNode, depthOfNode, savedMass);
        }
    }

    public void foldOut() {
        sampleSize = treeMass;
        foldedSampleSize += sampleSize;
        foldedTreeCount++;
        VectorSupport.axpyInto(foldedMeasure, measure, 1.0);
        VectorSupport.axpyInto(foldedDistances, distances, 1.0);
        VectorSupport.axpyInto(foldedProbMass, probMass, 1.0);
        foldedScore += savedScore;
        foldedHeight += savedHeight * normalizer.scale(treeMass);
    }

    @Override
    public InterpolationMeasure getResult() {
        return toMeasure(measure, distances, probMass, sampleSize, 1.0, savedScore,
                savedHeight * normalizer.scale(treeMass));
    }

    @Override
    public InterpolationMeasure getFoldResult() {
        return toMeasure(foldedMeasure, foldedDistances, foldedProbMass, foldedSampleSize, foldedTreeCount, foldedScore,
                foldedHeight);
    }

    public void resetAcrossQueries(float[] point) {
        reset();
        Arrays.fill(foldedMeasure, 0.0);
        Arrays.fill(foldedDistances, 0.0);
        Arrays.fill(foldedProbMass, 0.0);
        foldedSampleSize = 0;
        foldedTreeCount = 0;
        foldedScore = 0.0;
        foldedHeight = 0.0;
    }

    /**
     * Subclasses that add accumulators override this to materialize a subclass of
     * InterpolationMeasure. A visitor's result type is pinned by its
     * Visitor&lt;R&gt; parameter and cannot be narrowed, so extra state travels
     * inside the measure and merges through InterpolationMeasure.mergeExtras.
     */
    protected InterpolationMeasure toMeasure(double[] m, double[] d, double[] p, double ss, double trees, double sc,
            double h) {
        return new InterpolationMeasure(ss, new DiVector(m), new DiVector(d), new DiVector(p), trees, pointMass, sc, h);
    }

    // ------------------------------------------------------------------
    // hooks
    // ------------------------------------------------------------------

    /**
     * Fired once per contributing node, after prob and distComp are filled and
     * before the recurrence consumes them. prob[j] is π_j, lenComp[j] is π_j·len_j
     * (so the length is lenComp[j]/prob[j], with no box access needed), decay is 1
     * - probOfCut and 0 at a leaf, and mass is the node mass separated from -- the
     * quantity whose log-slope against log length is the local exponent.
     */
    protected void deposit(float[] prob, float[] lenComp, double decay, double mass) {
    }

    /** Fired at a duplicate leaf, where the gap is zero so no direction exists. */
    protected void depositDegenerate(double perComponentInfluence) {
    }

    double fieldExt(INodeView n, int depth, boolean c, double m, float[] loc) {
        return n.getMass() + m;
    }

    double influenceExt(INodeView n, int depth, boolean c, double m, float[] loc) {
        return 1.0;
    }

    double fieldPoint(INodeView n, int depth, double m, float[] loc) {
        return n.getMass() + m;
    }

    double influencePoint(INodeView n, int depth, double m, float[] loc) {
        return 1.0;
    }

    /** Expected inverse height for a point not present in the tree. */
    double heightExt(INodeView n, int depth, double m) {
        return scoreUnseenFn.of(depth, n.getMass());
    }

    /** The seen variant, damped by duplicate mass exactly as ScoreVisitor does. */
    double heightSeen(INodeView n, int depth, double mass) {
        return dampFn.of((int) (1 + mass), treeMass) * scoreSeenFn.of(depth, (int) (1 + mass));
    }

    double selfField(INodeView n, double m) {
        return m;
    }

    double selfInfluence(INodeView n, double m) {
        return 1.0;
    }

    InterpolationMeasure observeResult() {
        return getResult();
    }

    public static IVisitorFactory<InterpolationMeasure> reusableFactory(double pointMass, boolean centerOfMass) {
        return new IVisitorFactory<InterpolationMeasure>() {
            @Override
            public boolean isReusable() {
                return true;
            }

            @Override
            public boolean isReusableAcrossQueries() {
                return true;
            }

            @Override
            public boolean isFoldable() {
                return true;
            }

            @Override
            public IRFVisitor<InterpolationMeasure> newReusableVisitor(float[] point) {
                return new InterpolationVisitor(point.length, pointMass, centerOfMass);
            }

            @Override
            public Visitor<InterpolationMeasure> newVisitor(ITree<?, ?> tree, float[] point) {
                return new InterpolationVisitor(tree.projectToTree(point), tree.getMass(), pointMass, centerOfMass);
            }
        };
    }
}
