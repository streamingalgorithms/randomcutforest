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
 * This file is substantially modified from the file which had the following notice.
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

package org.streamingalgorithms.randomcutforest.tree;

import java.util.Arrays;
import java.util.function.Function;

public class HyperTree extends RandomCutTree {

    private final SeparationOracle gVecBuild;
    private final Function<IBoundingBoxView, double[]> gVec;

    public SeparationOracle getOracle() {
        return gVecBuild;
    }

    public Function<IBoundingBoxView, double[]> getgVec() {
        return gVec;
    }

    public static Builder builder() {
        return new Builder();
    }

    protected HyperTree(HyperTree.Builder builder) {
        super(builder);
        this.gVec = builder.gVec;
        this.gVecBuild = (box, scratch) -> {
            Arrays.fill(scratch, 0.0);
            double[] v = gVec.apply(box);
            System.arraycopy(v, 0, scratch, 0, v.length);
            return Arrays.stream(v).sum();
        };
    }

    public static class Builder extends RandomCutTree.Builder<Builder> {
        private Function<IBoundingBoxView, double[]> gVec;

        public Builder buildGVec(Function<IBoundingBoxView, double[]> gVec) {
            this.gVec = gVec;
            return this;
        }

        public HyperTree build() {
            return new HyperTree(this);
        }
    }
}
