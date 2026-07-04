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

import java.io.IOException;

import org.apache.fory.Fory;
import org.apache.fory.config.Language;

import io.protostuff.LinkedBuffer;
import io.protostuff.ProtostuffIOUtil;
import io.protostuff.Schema;
import io.protostuff.runtime.RuntimeSchema;
import tools.jackson.databind.DeserializationFeature;
import tools.jackson.databind.SerializationFeature;
import tools.jackson.databind.json.JsonMapper;

/**
 * The op4 codec set, generalized so ONE codec instance serves every state type
 * (RandomCutForestState / ThresholdedRandomCutForestState / RCFCasterState) --
 *
 */
public interface Codec {

    enum Id {
        PROTOSTUFF, JACKSON2, JACKSON3, FORY, STATE, REBUILT
    }

    String name();

    /** Build reusable serializer state once, before any encode/decode. */
    default void init() {
    }

    byte[] encode(Object state);

    <S> S decode(byte[] wire, Class<S> type);

    default boolean isControl() {
        return false;
    }

    /** STATE probes retained size. Only meaningful for controls. */
    default boolean probesSize() {
        return false;
    }

    /** REBUILT uses TreeMode.REBUILD (RCF-only); everyone else uses SAVE. */
    default Models.TreeMode treeMode() {
        return Models.TreeMode.SAVE;
    }

    static Codec of(Id id) {
        Codec c;
        switch (id) {
        case PROTOSTUFF:
            c = new ProtostuffCodec("Protostuff");
            break;
        case JACKSON2:
            c = new Jackson2Codec();
            break;
        case JACKSON3:
            c = new Jackson3Codec();
            break;
        case FORY:
            c = new ForyCodec();
            break;
        case STATE:
            c = new Control("State", true);
            break;
        case REBUILT:
            c = new ProtostuffCodec("Rebuilt") {
                @Override
                public Models.TreeMode treeMode() {
                    return Models.TreeMode.REBUILD;
                }
            };
            break;
        default:
            throw new IllegalStateException("unknown codec " + id);
        }
        c.init();
        return c;
    }
}

/**
 * Protostuff runtime schema per state class; the wire format the serialization
 * module already speaks.
 */
class ProtostuffCodec implements Codec {
    private final LinkedBuffer buffer = LinkedBuffer.allocate(512);
    private final String name;

    ProtostuffCodec(String name) {
        this.name = name;
    }

    public String name() {
        return name;
    }

    @SuppressWarnings({ "unchecked", "rawtypes" })
    public byte[] encode(Object state) {
        Schema schema = RuntimeSchema.getSchema(state.getClass());
        try {
            return ProtostuffIOUtil.toByteArray(state, schema, buffer);
        } finally {
            buffer.clear();
        }
    }

    public <S> S decode(byte[] wire, Class<S> type) {
        Schema<S> schema = RuntimeSchema.getSchema(type);
        S msg = schema.newMessage();
        ProtostuffIOUtil.mergeFrom(wire, msg, schema);
        return msg;
    }
}

/**
 * Jackson 2 (com.fasterxml.jackson) -- checked exceptions, the CVE-era line,
 * kept as a comparison point.
 */
final class Jackson2Codec implements Codec {
    private final com.fasterxml.jackson.databind.ObjectMapper json = new com.fasterxml.jackson.databind.ObjectMapper();

    public String name() {
        return "Jackson2";
    }

    public byte[] encode(Object state) {
        try {
            return json.writeValueAsBytes(state);
        } catch (com.fasterxml.jackson.core.JsonProcessingException e) {
            throw new RuntimeException(e);
        }
    }

    public <S> S decode(byte[] wire, Class<S> type) {
        try {
            return json.readValue(wire, type);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }
}

/**
 * Jackson 3 (tools.jackson) -- this is finicky, and this package is not a
 * serialization packe. But the encoding/decoding is simple for 3.2.0
 */
final class Jackson3Codec implements Codec {

    private final tools.jackson.databind.ObjectMapper json = JsonMapper.builder()
            .disable(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES)
            .disable(SerializationFeature.WRITE_EMPTY_JSON_ARRAYS).build();

    public String name() {
        return "Jackson3";
    }

    public byte[] encode(Object state) {
        return json.writeValueAsBytes(state);
    }

    public <S> S decode(byte[] wire, Class<S> type) {
        return json.readValue(wire, type);
    }
}

/**
 * Apache Fory 1.3.0, direct-typed. Single-threaded instance
 */
final class ForyCodec implements Codec {
    private Fory fory;

    public String name() {
        return "Fory";
    }

    public void init() {
        fory = Fory.builder().withLanguage(Language.JAVA).requireClassRegistration(false).build();
    }

    public byte[] encode(Object state) {
        return fory.serialize(state);
    }

    @SuppressWarnings("unchecked")
    public <S> S decode(byte[] wire, Class<S> type) {
        return (S) fory.deserialize(wire);
    }
}

/**
 * STATE / NONE controls: no wire. Drivers price only toState -> toModel for
 * these.
 */
final class Control implements Codec {
    private final String name;
    private final boolean probes;

    Control(String name, boolean probes) {
        this.name = name;
        this.probes = probes;
    }

    public String name() {
        return name;
    }

    public boolean isControl() {
        return true;
    }

    public boolean probesSize() {
        return probes;
    }

    public byte[] encode(Object state) {
        throw new UnsupportedOperationException("control codec has no wire form");
    }

    public <S> S decode(byte[] wire, Class<S> type) {
        throw new UnsupportedOperationException("control codec has no wire form");
    }
}
