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

package org.streamingalgorithms.randomcutforest.examples.plot;

import java.awt.image.BufferedImage;
import java.io.Closeable;
import java.io.File;
import java.io.IOException;

import javax.imageio.IIOImage;
import javax.imageio.ImageIO;
import javax.imageio.ImageTypeSpecifier;
import javax.imageio.ImageWriteParam;
import javax.imageio.ImageWriter;
import javax.imageio.metadata.IIOMetadata;
import javax.imageio.metadata.IIOMetadataNode;
import javax.imageio.stream.ImageOutputStream;

/**
 * Minimal animated-GIF sequence writer built on the JDK's ImageIO. No external
 * deps.
 */
public final class GifWriter implements Closeable {

    private final ImageWriter writer;
    private final ImageWriteParam params;
    private final IIOMetadata metadata;
    private final ImageOutputStream output;

    public GifWriter(File file, int frameDelayMs, boolean loop) throws IOException {
        writer = ImageIO.getImageWritersBySuffix("gif").next();
        output = ImageIO.createImageOutputStream(file);
        writer.setOutput(output);
        params = writer.getDefaultWriteParam();

        ImageTypeSpecifier type = ImageTypeSpecifier.createFromBufferedImageType(BufferedImage.TYPE_INT_RGB);
        metadata = writer.getDefaultImageMetadata(type, params);
        String fmt = metadata.getNativeMetadataFormatName();
        IIOMetadataNode root = (IIOMetadataNode) metadata.getAsTree(fmt);

        IIOMetadataNode gce = node(root, "GraphicControlExtension");
        gce.setAttribute("disposalMethod", "none");
        gce.setAttribute("userInputFlag", "FALSE");
        gce.setAttribute("transparentColorFlag", "FALSE");
        gce.setAttribute("delayTime", Integer.toString(Math.max(1, frameDelayMs / 10))); // hundredths of a sec
        gce.setAttribute("transparentColorIndex", "0");

        IIOMetadataNode appExt = node(root, "ApplicationExtensions");
        IIOMetadataNode netscape = new IIOMetadataNode("ApplicationExtension");
        netscape.setAttribute("applicationID", "NETSCAPE");
        netscape.setAttribute("authenticationCode", "2.0");
        int loopCount = loop ? 0 : 1; // 0 == loop forever
        netscape.setUserObject(new byte[] { 0x1, (byte) (loopCount & 0xFF), (byte) ((loopCount >> 8) & 0xFF) });
        appExt.appendChild(netscape);

        metadata.setFromTree(fmt, root);
        writer.prepareWriteSequence(null);
    }

    public void writeFrame(BufferedImage image) throws IOException {
        writer.writeToSequence(new IIOImage(image, null, metadata), params);
    }

    @Override
    public void close() throws IOException {
        writer.endWriteSequence();
        output.close();
        writer.dispose();
    }

    private static IIOMetadataNode node(IIOMetadataNode root, String name) {
        for (int i = 0; i < root.getLength(); i++) {
            if (root.item(i).getNodeName().equalsIgnoreCase(name)) {
                return (IIOMetadataNode) root.item(i);
            }
        }
        IIOMetadataNode created = new IIOMetadataNode(name);
        root.appendChild(created);
        return created;
    }
}
