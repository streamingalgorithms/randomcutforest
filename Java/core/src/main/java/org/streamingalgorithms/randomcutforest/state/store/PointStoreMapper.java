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
 * This file has been substantially modified from its original which had the
 * following notice.
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

package org.streamingalgorithms.randomcutforest.state.store;

import static org.streamingalgorithms.randomcutforest.CommonUtils.*;
import static org.streamingalgorithms.randomcutforest.util.ArrayEncoder.*;

import org.streamingalgorithms.randomcutforest.config.Precision;
import org.streamingalgorithms.randomcutforest.state.IStateMapper;
import org.streamingalgorithms.randomcutforest.state.Version;
import org.streamingalgorithms.randomcutforest.store.PointStore;
import org.streamingalgorithms.randomcutforest.tree.Column;
import org.streamingalgorithms.randomcutforest.util.ArrayEncoder;

import it.unimi.dsi.fastutil.ints.Int2IntOpenHashMap;
import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
public class PointStoreMapper implements IStateMapper<PointStore, PointStoreState> {

    /**
     * If true, then the arrays are compressed via simple data dependent scheme
     */
    private boolean compressionEnabled = true;

    private int numberOfTrees = 255; // byte encoding as default

    @Override
    public PointStore toModel(PointStoreState state, long seed) {
        checkNotNull(state.getRefCount(), "refCount must not be null");
        checkNotNull(state.getPointData(), "pointData must not be null");
        checkArgument(Precision.valueOf(state.getPrecision()) == Precision.FLOAT_32,
                "precision must be " + Precision.FLOAT_32);
        int indexCapacity = state.getIndexCapacity();
        int dimensions = state.getDimensions();
        float[] store = ArrayEncoder.unpackFloats(state.getPointData(), state.getCurrentStoreCapacity() * dimensions);
        int startOfFreeSegment = state.getStartOfFreeSegment();

        int baseDimension = dimensions / state.getShingleSize();
        long bound = PointStore.locationBound(state.getCapacity(), state.getShingleSize(), state.isRotationEnabled());
        Column location = Column.of(indexCapacity, bound, true); // born all-sentinel; no fill needed

        byte[] refCountBytes = new byte[indexCapacity];
        Int2IntOpenHashMap overflow = ArrayEncoder.unpackRefCounts(state.getRefCount(), refCountBytes, indexCapacity,
                state.isCompressed());

        if (!state.getVersion().equals(Version.V3_0)) {
            // legacy: positional, every index encoded, value is a RAW location.
            // Guard on occupancy so dead slots stay at the column sentinel — otherwise the
            // legacy -1 sentinel divides to 0 and masquerades as a live quotient.
            int length = unpackedLength(state.getLocationList(), state.isCompressed());
            ArrayEncoder.decodeCore(state.getLocationList(), length, state.isCompressed(), (int i, int v) -> {
                if (refCountBytes[i] != 0)
                    location.set(i, v / baseDimension);
            });
        } else {
            if (state.getDuplicateRefs() != null) {
                if (overflow == null) {
                    overflow = new Int2IntOpenHashMap();
                    overflow.defaultReturnValue(0);
                }
                mergeDuplicateRefs(state.getDuplicateRefs(), state.isCompressed(), refCountBytes, overflow);
            }
            // V3: dense stream (live slots only, ascending index), values already
            // quotients.
            int liveCount = 0;
            for (int i = 0; i < indexCapacity; i++)
                if (refCountBytes[i] != 0)
                    liveCount++;
            final int[] cursor = { 0 }, placed = { 0 };
            ArrayEncoder.decodeCore(state.getLocationList(), indexCapacity, state.isCompressed(), (int v) -> {
                int i = cursor[0];
                while (i < indexCapacity && refCountBytes[i] == 0)
                    i++;
                checkState(i < indexCapacity, "location count exceeds occupied slots");
                location.set(i, v);
                cursor[0] = i + 1;
                placed[0]++;
            });
            checkState(placed[0] == liveCount, "location count underflows occupied slots");
        }

        return PointStore.builder().internalRotationEnabled(state.isRotationEnabled())
                .internalShinglingEnabled(state.isInternalShinglingEnabled()).indexCapacity(indexCapacity)
                .currentStoreCapacity(state.getCurrentStoreCapacity()).capacity(state.getCapacity())
                .shingleSize(state.getShingleSize()).dimensions(state.getDimensions()).location(location)
                .nextTimeStamp(state.getLastTimeStamp()).startOfFreeSegment(startOfFreeSegment)
                .refCountBytes(refCountBytes).overflow(overflow).knownShingle(state.getInternalShingle()).store(store)
                .build();
    }

    @Override
    public PointStoreState toState(PointStore model) {
        model.compact();
        PointStoreState state = new PointStoreState();
        state.setVersion(Version.V3_0);
        state.setCompressed(compressionEnabled);
        state.setDimensions(model.getDimensions());
        state.setCapacity(model.getCapacity());
        state.setShingleSize(model.getShingleSize());
        state.setDirectLocationMap(false);
        state.setInternalShinglingEnabled(model.isInternalShinglingEnabled());
        state.setLastTimeStamp(model.getNextSequenceIndex());
        if (model.isInternalShinglingEnabled()) {
            state.setInternalShingle(toDoubleArray(model.getInternalShingle()));
            state.setRotationEnabled(model.isInternalRotationEnabled());
        }
        state.setDynamicResizingEnabled(true);

        state.setCurrentStoreCapacity(model.getCurrentStoreCapacity());
        state.setIndexCapacity(model.getIndexCapacity());

        state.setStartOfFreeSegment(model.getStartOfFreeSegment());
        state.setPrecision(Precision.FLOAT_32.name());
        state.setRefCount(model.getPackedRefCount(state.isCompressed()));
        state.setDuplicateRefs(model.getPackedDuplicates(state.isCompressed()));
        state.setLocationList(model.getPackedLocation(state.isCompressed()));
        state.setPointData(pack(model.getStore(), model.getStartOfFreeSegment()));
        return state;
    }

}
