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
 */

package org.streamingalgorithms.randomcutforest.examples.plot;

import java.awt.*;
import java.util.ArrayList;
import java.util.List;
import java.util.function.BooleanSupplier;
import java.util.function.Supplier;

/**
 * A small key card in a corner of the plot: one row per key, with what it does
 * and, where it has one, its current state. The states are read when painting,
 * like the toggles themselves, so the card is right while paused.
 */
public final class KeysLayer implements Layer {

    public enum Corner {
        BOTTOM_LEFT, BOTTOM_RIGHT, TOP_LEFT, TOP_RIGHT
    }

    private final Corner corner;
    private final List<String> keys = new ArrayList<>();
    private final List<String> actions = new ArrayList<>();
    private final List<Supplier<String>> states = new ArrayList<>();

    public KeysLayer(Corner corner) {
        this.corner = corner;
    }

    /** Adds a row; state may be null for a key without one. */
    public KeysLayer add(String key, String action, Supplier<String> state) {
        keys.add(key);
        actions.add(action);
        states.add(state);
        return this;
    }

    /** "on" or "off", read when painting. */
    public static Supplier<String> onOff(BooleanSupplier flag) {
        return () -> flag.getAsBoolean() ? "on" : "off";
    }

    @Override
    public void draw(Graphics2D g, Plot2D.Viewport vp) {
        Graphics2D g2 = (Graphics2D) g.create();
        try {
            g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
            g2.setFont(g2.getFont().deriveFont(Font.PLAIN, 11f));
            FontMetrics fm = g2.getFontMetrics();
            String[] now = new String[keys.size()];
            int keyW = 0, actionW = 0, stateW = 0;
            for (int i = 0; i < keys.size(); i++) {
                now[i] = (states.get(i) == null) ? "" : states.get(i).get();
                keyW = Math.max(keyW, fm.stringWidth(keys.get(i)));
                actionW = Math.max(actionW, fm.stringWidth(actions.get(i)));
                stateW = Math.max(stateW, fm.stringWidth(now[i]));
            }
            int pad = 7, gap = 8, line = fm.getHeight() + 1;
            int w = 2 * pad + keyW + gap + actionW + (stateW > 0 ? gap + stateW : 0);
            int h = 2 * pad + keys.size() * line - 1;
            boolean right = (corner == Corner.BOTTOM_RIGHT || corner == Corner.TOP_RIGHT);
            boolean top = (corner == Corner.TOP_LEFT || corner == Corner.TOP_RIGHT);
            int x = right ? (int) Math.round(vp.px(vp.xmax())) - w - 12 : (int) Math.round(vp.px(vp.xmin())) + 12;
            int y = top ? (int) Math.round(vp.py(vp.ymax())) + 12 : (int) Math.round(vp.py(vp.ymin())) - h - 12;

            g2.setColor(new Color(255, 255, 255, 215));
            g2.fillRoundRect(x, y, w, h, 6, 6);
            g2.setColor(new Color(0, 0, 0, 40));
            g2.drawRoundRect(x, y, w, h, 6, 6);
            for (int i = 0; i < keys.size(); i++) {
                int ty = y + pad + i * line + fm.getAscent();
                g2.setColor(new Color(40, 40, 40));
                g2.setFont(g2.getFont().deriveFont(Font.BOLD));
                g2.drawString(keys.get(i), x + pad, ty);
                g2.setFont(g2.getFont().deriveFont(Font.PLAIN));
                g2.setColor(new Color(70, 70, 70));
                g2.drawString(actions.get(i), x + pad + keyW + gap, ty);
                if (!now[i].isEmpty()) {
                    boolean off = now[i].equals("off");
                    g2.setColor(off ? new Color(150, 150, 150) : new Color(40, 40, 40));
                    g2.drawString(now[i], x + w - pad - fm.stringWidth(now[i]), ty);
                }
            }
        } finally {
            g2.dispose();
        }
    }
}
