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
 * Gap is computed ONCE per node via a single gapInto over the SMALL box: point
 * mode -> newValues = expandedPoint [p,-p] shadow mode -> newValues = node box
 * slice sumOfNewRange = small.rangeSum + S (the rangeSum field already holds Σ
 * oldRange). oldRange for the distance term is read inline from the small box
 * in probAndDistInto.
 */

public class InterpolationVisitor extends RFVisitor<InterpolationMeasure> {

    private final double pointMass;
    private final boolean centerOfMass;

    private final int dim;
    private final int len; // 2 * dim

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

    private double savedScore;
    private double foldedScore; // NOT cleared in reset()

    private boolean pointEqualsLeaf;
    private double savedMass;

    // float scratch, fully overwritten per node — never cleared
    private final float[] gap; // expandedPoint gap -> prob, in place
    private final float[] distComp; // prob * (gap + oldRange)

    private double sumOfNewRange;
    private double sumOfDifferenceInRange;

    InterpolationVisitor(int dimension, double pointMass, boolean centerOfMass) {
        this.pointMass = pointMass;
        this.centerOfMass = centerOfMass;
        this.dim = dimension;
        this.len = 2 * dimension;
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
        pointEqualsLeaf = false;
        pointInsideBox = false;
        shadowBoxActive = false;
        sampleSize = 0;
        Arrays.fill(measure, 0.0);
        Arrays.fill(distances, 0.0);
        Arrays.fill(probMass, 0.0);
        // folded*, foldedScore deliberately NOT cleared
    }

    @Override
    protected void reset() {
        setDefaults();
    }

    /**
     * One gapInto over the small box. Sets sumOfDifferenceInRange (S),
     * sumOfNewRange; returns S.
     */
    private double computeGap(ArrayBox small, float[] nv, int nvOff) {
        double S = VectorSupport.gapInto(nv, nvOff, small.values, small.offset, gap, 0, len);
        sumOfDifferenceInRange = S;
        sumOfNewRange = small.getRangeSum() + S; // rangeSum field == Σ oldRange
        return S;
    }

    /**
     * gap -> prob + distComp (reads small box for oldRange), then the three
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

        ArrayBox small, large;
        boolean pointMode;

        double S;

        if (pointEqualsLeaf) {
            small = growShadow((ArrayBox) node.getSiblingBoundingBox());
            large = (ArrayBox) node.getBoundingBox();
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
        double fieldVal = fieldExt(node, centerOfMass, savedMass, null);
        double influenceVal = influenceExt(node, centerOfMass, savedMass, null);
        double invSumNew = (sumOfNewRange == 0.0) ? 0.0 : 1.0 / sumOfNewRange;
        VectorSupport.probAndDistInto(gap, distComp, small.values, small.offset, dim, invSumNew);
        recur(fieldVal, influenceVal, 1.0 - probOfCut);

        savedScore = probOfCut * fieldVal + (1.0 - probOfCut) * savedScore;
    }

    @Override
    public void acceptLeaf(INodeView leafNode, int depthOfNode) {

        float[] leaf = leafNode.getLeafPoint();
        float[] expandedPoint = leafNode.expanded();
        double S = VectorSupport.signedGapInto(expandedPoint, 0, +1f, leaf, 0, gap, 0, dim)
                + VectorSupport.signedGapInto(expandedPoint, dim, -1f, leaf, 0, gap, dim, dim);
        sumOfDifferenceInRange = S;
        sumOfNewRange = S; // leaf rangeSum ≡ 0
        if (S <= 0) {
            savedMass = pointMass + leafNode.getMass();
            pointEqualsLeaf = true;
            double selfF = 0.5 * selfField(leafNode, savedMass) / dim;
            double selfI = 0.5 * selfInfluence(leafNode, savedMass) / dim;
            Arrays.fill(measure, selfF);
            Arrays.fill(probMass, selfI);
            savedScore = selfF * 2 * dim;
        } else {
            savedMass = pointMass;
            double fieldVal = fieldPoint(leafNode, savedMass, null);
            double influenceVal = influencePoint(leafNode, savedMass, null);
            double invSumNew = (sumOfNewRange == 0.0) ? 0.0 : 1.0 / sumOfNewRange;
            VectorSupport.probOnlyInto(gap, distComp, len, invSumNew);
            recur(fieldVal, influenceVal, 0.0); // sumOfNewRange == S here
            savedScore = (sumOfNewRange == 0) ? 0.0 : fieldVal * (S / sumOfNewRange);
        }
    }

    public void foldOut() {
        sampleSize = treeMass;
        foldedSampleSize += sampleSize;
        VectorSupport.axpyInto(foldedMeasure, measure, 1.0);
        VectorSupport.axpyInto(foldedDistances, distances, 1.0);
        VectorSupport.axpyInto(foldedProbMass, probMass, 1.0);
        foldedScore += savedScore;
    }

    @Override
    public InterpolationMeasure getResult() {
        return toMeasure(measure, distances, probMass, sampleSize);
    }

    @Override
    public InterpolationMeasure getFoldResult() {
        return toMeasure(foldedMeasure, foldedDistances, foldedProbMass, foldedSampleSize);
    }

    public void resetAcrossQueries(float[] point) {
        reset();
        Arrays.fill(foldedMeasure, 0.0);
        Arrays.fill(foldedDistances, 0.0);
        Arrays.fill(foldedProbMass, 0.0);
        foldedSampleSize = 0;
        foldedScore = 0.0;
    }

    private InterpolationMeasure toMeasure(double[] m, double[] d, double[] p, int ss) {
        return new InterpolationMeasure(ss, new DiVector(m), new DiVector(d), new DiVector(p));
    }

    InterpolationMeasure observeResult() {
        return getResult();
    }

    double fieldExt(INodeView n, boolean c, double m, float[] loc) {
        return n.getMass() + m;
    }

    double influenceExt(INodeView n, boolean c, double m, float[] loc) {
        return 1.0;
    }

    double fieldPoint(INodeView n, double m, float[] loc) {
        return n.getMass() + m;
    }

    double influencePoint(INodeView n, double m, float[] loc) {
        return 1.0;
    }

    double selfField(INodeView n, double m) {
        return m;
    }

    double selfInfluence(INodeView n, double m) {
        return 1.0;
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