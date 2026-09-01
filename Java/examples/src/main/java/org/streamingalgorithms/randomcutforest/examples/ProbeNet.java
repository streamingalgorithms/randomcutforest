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

package org.streamingalgorithms.randomcutforest.examples;

import java.util.ArrayList;
import java.util.List;

/**
 * An online k-centre cover, used to place probes where the data actually is.
 *
 * <p>
 * Every arriving point is either within the cover radius of an existing probe
 * -- in which case it is already represented and is ignored beyond refreshing
 * that probe -- or it is not, and becomes a probe itself. When the count
 * exceeds the budget the radius doubles and probes now within the new radius of
 * each other are merged, which is the standard doubling construction for
 * streaming k-centre. One pass, no points retained, O(k) per arrival.
 *
 * <p>
 * <b>Why this rather than a fixed lattice or a fixed parametrization.</b>
 * Probes spaced uniformly along a spiral angle are uniform in log-radius, so on
 * an exponential arm most of them crowd into the inner turn and the sparse
 * outer arm gets one or two. A lattice over a mostly-empty volume is worse: an
 * earlier version had 72 of 75 cells empty every frame. A cover places probes
 * in proportion to the covering number of the data at scale R, which is the
 * only spacing that is neither arbitrary nor tied to a parametrization the
 * estimator cannot see.
 *
 * <p>
 * <b>Probes expire.</b> A probe not covered by any arrival for a full decay
 * window is dropped, so the set tracks what the model still holds rather than
 * accumulating everywhere the data has ever been. Without it the comet would
 * leave a permanent trail of probes behind.
 *
 * <p>
 * The radius is also the natural glyph pitch: probes are at least R apart by
 * construction, so nothing needs to be inferred from pairwise distances.
 */
public class ProbeNet {

    private final int budget;
    private final long ttl;

    private final List<double[]> centres = new ArrayList<>();
    private final List<Integer> sources = new ArrayList<>();
    private final List<Long> lastHit = new ArrayList<>();

    private double radius;
    private long clock;

    /**
     * @param budget how many probes to keep; the radius grows until this holds
     * @param ttl    arrivals a probe may go uncovered before it is forgotten; the
     *               decay window is the natural choice, so probes and data expire
     *               together
     */
    public ProbeNet(int budget, long ttl) {
        this.budget = budget;
        this.ttl = ttl;
    }

    /** Feed one arrival. Cheap enough to call on the update path. */
    public void offer(float[] p, int source) {
        clock++;
        int nearest = -1;
        double best = Double.MAX_VALUE;
        for (int i = 0; i < centres.size(); i++) {
            double dd = distance(centres.get(i), p);
            if (dd < best) {
                best = dd;
                nearest = i;
            }
        }
        if (nearest >= 0 && best <= radius) {
            lastHit.set(nearest, clock); // already covered; the point is ignored
            return;
        }
        double[] c = new double[p.length];
        for (int i = 0; i < p.length; i++) {
            c[i] = p[i];
        }
        centres.add(c);
        sources.add(source);
        lastHit.add(clock);
        while (centres.size() > budget) {
            grow();
        }
    }

    /**
     * Double the radius and merge. On the first overflow there is no radius yet, so
     * it starts at the closest pair -- the smallest radius that would have merged
     * anything.
     */
    private void grow() {
        if (radius <= 0.0) {
            double min = Double.MAX_VALUE;
            for (int i = 0; i < centres.size(); i++) {
                for (int j = i + 1; j < centres.size(); j++) {
                    min = Math.min(min, distance(centres.get(i), centres.get(j)));
                }
            }
            radius = (min < Double.MAX_VALUE && min > 0) ? min : 1e-6;
        } else {
            radius *= 2;
        }

        List<double[]> keptCentres = new ArrayList<>();
        List<Integer> keptSources = new ArrayList<>();
        List<Long> keptHits = new ArrayList<>();
        for (int i = 0; i < centres.size(); i++) {
            boolean covered = false;
            for (int j = 0; j < keptCentres.size(); j++) {
                if (distance(keptCentres.get(j), centres.get(i)) <= radius) {
                    // the survivor inherits the more recent activity
                    keptHits.set(j, Math.max(keptHits.get(j), lastHit.get(i)));
                    covered = true;
                    break;
                }
            }
            if (!covered) {
                keptCentres.add(centres.get(i));
                keptSources.add(sources.get(i));
                keptHits.add(lastHit.get(i));
            }
        }
        centres.clear();
        centres.addAll(keptCentres);
        sources.clear();
        sources.addAll(keptSources);
        lastHit.clear();
        lastHit.addAll(keptHits);
    }

    /** Drop probes nothing has covered for a full window. Call once per frame. */
    public void expire() {
        for (int i = centres.size() - 1; i >= 0; i--) {
            if (clock - lastHit.get(i) > ttl) {
                centres.remove(i);
                sources.remove(i);
                lastHit.remove(i);
            }
        }
    }

    public int size() {
        return centres.size();
    }

    public double radius() {
        return radius;
    }

    public double[] centre(int i) {
        return centres.get(i);
    }

    /** Which stream the point that created this probe came from. */
    public int source(int i) {
        return sources.get(i);
    }

    private static double distance(double[] a, float[] b) {
        double sum = 0;
        for (int i = 0; i < a.length; i++) {
            double d = a[i] - b[i];
            sum += d * d;
        }
        return Math.sqrt(sum);
    }

    private static double distance(double[] a, double[] b) {
        double sum = 0;
        for (int i = 0; i < a.length; i++) {
            double d = a[i] - b[i];
            sum += d * d;
        }
        return Math.sqrt(sum);
    }
}
