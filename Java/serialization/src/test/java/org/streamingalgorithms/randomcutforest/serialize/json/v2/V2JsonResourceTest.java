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

package org.streamingalgorithms.randomcutforest.serialize.json.v2;

import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.fail;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.Random;

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import org.streamingalgorithms.randomcutforest.RandomCutForest;
import org.streamingalgorithms.randomcutforest.preprocessor.IPreprocessor;
import org.streamingalgorithms.randomcutforest.state.RandomCutForestMapper;
import org.streamingalgorithms.randomcutforest.state.RandomCutForestState;
import org.streamingalgorithms.randomcutforest.state.preprocessor.PreprocessorMapper;
import org.streamingalgorithms.randomcutforest.state.preprocessor.PreprocessorState;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.MapperFeature;
import com.fasterxml.jackson.databind.ObjectMapper;
// + whatever package PreprocessorMapper / PreprocessorState live in

public class V2JsonResourceTest {

    @ParameterizedTest
    @EnumSource(V2RCFJsonResource.class)
    public void testJson(V2RCFJsonResource jsonResource) throws JsonProcessingException {
        RandomCutForestMapper rcfMapper = new RandomCutForestMapper();
        String json = getStateFromFile(jsonResource.getResource());
        assertNotNull(json);
        ObjectMapper mapper = new ObjectMapper();
        mapper.configure(MapperFeature.ACCEPT_CASE_INSENSITIVE_PROPERTIES, true);
        RandomCutForestState state = mapper.readValue(json, RandomCutForestState.class);
        RandomCutForest forest = rcfMapper.toModel(state);
        Random r = new Random(0);
        for (int i = 0; i < 20000; i++) {
            double[] point = r.ints(forest.getDimensions(), 0, 50).asDoubleStream().toArray();
            forest.getAnomalyScore(point);
            forest.update(point, 0L);
        }
        assertNotNull(forest);
    }

    @ParameterizedTest
    @EnumSource(V2PreProcessorJsonResource.class)
    public void testPreprocessorJson(V2PreProcessorJsonResource jsonResource) throws JsonProcessingException {
        PreprocessorMapper preMapper = new PreprocessorMapper();
        String json = getStateFromFile(jsonResource.getResource());
        assertNotNull(json);
        ObjectMapper mapper = new ObjectMapper();
        mapper.configure(MapperFeature.ACCEPT_CASE_INSENSITIVE_PROPERTIES, true);
        PreprocessorState state = mapper.readValue(json, PreprocessorState.class);
        IPreprocessor preprocessor = preMapper.toModel(state);
        assertNotNull(preprocessor);
    }

    private String getStateFromFile(String resourceFile) {
        try (InputStream is = V2JsonResourceTest.class.getResourceAsStream(resourceFile);
                BufferedReader rr = new BufferedReader(new InputStreamReader(is, StandardCharsets.UTF_8))) {
            StringBuilder b = new StringBuilder();
            String line;
            while ((line = rr.readLine()) != null) {
                b.append(line);
            }
            return b.toString();
        } catch (IOException e) {
            fail("Unable to load resource");
        }
        return null;
    }
}
