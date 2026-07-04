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

package org.streamingalgorithms.randomcutforest.benchmark.operations;

import org.openjdk.jol.info.GraphLayout;
import org.streamingalgorithms.randomcutforest.RandomCutForest;
import org.streamingalgorithms.randomcutforest.executor.SamplerPlusTree;
import org.streamingalgorithms.randomcutforest.parkservices.RCFCaster;
import org.streamingalgorithms.randomcutforest.parkservices.ThresholdedRandomCutForest;

/**
 * Requires the JOL JVM flags on whichever driver calls it:
 * -Djdk.attach.allowAttachSelf=true -Djol.magicFieldOffset=true
 */

public final class JolBreakdown {

    public final double wholeMb;
    public final double rcfMb;
    public final double storeMb;
    public final double treeMb;
    public final double samplerMb;

    private JolBreakdown(double wholeMb, double rcfMb, double storeMb, double treeMb, double samplerMb) {
        this.wholeMb = wholeMb;
        this.rcfMb = rcfMb;
        this.storeMb = storeMb;
        this.treeMb = treeMb;
        this.samplerMb = samplerMb;
    }

    /** Thresholder/forecast wrapper overhead above the raw RCF (0 for Kind.RCF). */
    public double wrapperMb() {
        return wholeMb - rcfMb;
    }

    public static JolBreakdown of(Models.Kind kind, Object model) {
        RandomCutForest rcf = rcfOf(kind, model);
        double whole = mb(model);
        double rcfMb = mb(rcf);
        double storeMb = mb(rcf.getUpdateCoordinator().getStore());
        double treeMb = 0, samplerMb = 0;
        for (Object c : rcf.getComponents()) {
            SamplerPlusTree<?, ?> spt = (SamplerPlusTree<?, ?>) c;
            treeMb += mb(spt.getTree());
            samplerMb += mb(spt.getSampler());
        }
        return new JolBreakdown(whole, rcfMb, storeMb, treeMb, samplerMb);
    }

    private static RandomCutForest rcfOf(Models.Kind kind, Object model) {
        switch (kind) {
        case RCF:
            return (RandomCutForest) model;
        case TRCF:
            return ((ThresholdedRandomCutForest) model).getForest();
        case CASTER:
            return ((RCFCaster) model).getForest(); // CONFIRM Caster exposes getForest()
        default:
            throw new IllegalStateException("unknown model " + kind);
        }
    }

    private static double mb(Object o) {
        return GraphLayout.parseInstance(o).totalSize() / 1048576.0;
    }
}
