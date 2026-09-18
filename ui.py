"""
ui.py -- FitCheck AI HUD drawing.

Every overlay element drawn on the webcam feed lives here: the animated
scan frame that traces the detected person, the live SCANNING / ANALYZING /
RESULT READY status chip, the prominent animated score card, and the
anchored category labels with thin pointer lines to the exact analyzed
body regions.

Pure OpenCV drawing primitives. No new dependencies and no extra model
calls -- the drawing cost is negligible next to pose estimation.
"""

from __future__ import annotations

import math
import os
import time

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Palette (BGR)
# ---------------------------------------------------------------------------
ACCENT = (214, 228, 6)        # cyan / teal accent
GREEN = (124, 242, 158)       # good score / result ready
AMBER = (72, 194, 255)        # busy / analyzing
RED = (100, 100, 240)         # low score
WHITE = (235, 238, 245)
MUTED = (150, 158, 170)
INK = (12, 14, 18)
TRACK = (34, 38, 46)

FONT = cv2.FONT_HERSHEY_SIMPLEX

REGION_TITLES = (("head", "hair", "HEAD"), ("upper", "top", "UPPER"),
                 ("lower", "bottom", "LOWER"), ("shoes", "shoes", "SHOES"))
COMP_ORDER = (("color_coordination", "COLOR"), ("top_bottom_combo", "COMBO"),
              ("style_consistency", "STYLE"), ("outfit_balance", "BALANCE"),
              ("footwear_coordination", "SHOES"), ("hair_grooming", "HAIR"))


def _blend(c1, c2, f):
    return tuple(int(a + (b - a) * f) for a, b in zip(c1, c2))


def _text_size(text, scale, thick=1):
    return cv2.getTextSize(text, FONT, scale, thick)[0]


def _text(img, text, org, scale, color, thick=1):
    cv2.putText(img, text, org, FONT, scale, color, thick, cv2.LINE_AA)


def _score_color(v):
    return GREEN if v >= 7.5 else ACCENT if v >= 5.5 else RED


def rects_overlap(a, b, m=0):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return not (ax2 + m < bx1 or bx2 + m < ax1 or ay2 + m < by1 or by2 + m < ay1)


def _display_text(res):
    """Natural label like 'Black T-Shirt'; None when nothing is supported."""
    label = (res.get("label") or "").strip()
    if not label or label.lower() in ("unknown", "not clear"):
        return None
    color = res.get("color") or "unknown"
    if color == "unknown":
        return label.title()
    return f"{color.title()} {label.title()}"


def fill_rounded(img, x1, y1, x2, y2, r, color, alpha):
    r = max(1, min(r, (x2 - x1) // 2, (y2 - y1) // 2))
    overlay = img.copy()
    cv2.rectangle(overlay, (x1 + r, y1), (x2 - r, y2), color, -1)
    cv2.rectangle(overlay, (x1, y1 + r), (x2, y2 - r), color, -1)
    for cx, cy in ((x1 + r, y1 + r), (x2 - r, y1 + r),
                   (x1 + r, y2 - r), (x2 - r, y2 - r)):
        cv2.circle(overlay, (cx, cy), r, color, -1)
    cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0, img)


def rounded_rect(img, x1, y1, x2, y2, r, color, thickness=1):
    r = max(1, min(r, (x2 - x1) // 2, (y2 - y1) // 2))
    cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, thickness, cv2.LINE_AA)
    for cx, cy, a0, a1 in ((x1 + r, y1 + r, 180, 270), (x2 - r, y1 + r, 270, 360),
                           (x2 - r, y2 - r, 0, 90), (x1 + r, y2 - r, 90, 180)):
        cv2.ellipse(img, (cx, cy), (r, r), 0, a0, a1, color, thickness, cv2.LINE_AA)


class HUD:
    """Stateful overlay renderer. One instance per window, per frame loop."""

    def __init__(self, w, h):
        self.w, self.h = w, h
        self._last = time.perf_counter()
        self.dt = 1.0 / 30.0
        self.phase = 0.0
        self._bbox = None
        self.disp_score = 0.0
        self.disp_comps = {}
        self.chip_rect = None
        self.card_rect = None

    def update(self):
        now = time.perf_counter()
        self.dt = min(now - self._last, 0.1)
        self._last = now
        self.phase = (self.phase + self.dt * 0.55) % 1.0

    @staticmethod
    def _smooth(v, target, dt, k=6.0):
        return v + (target - v) * (1.0 - math.exp(-k * dt))

    @staticmethod
    def _perim_pt(x1, y1, x2, y2, f):
        f %= 1.0
        w, h = x2 - x1, y2 - y1
        p = f * 2.0 * (w + h)
        if p < w:
            return int(x1 + p), y1
        p -= w
        if p < h:
            return x2, int(y1 + p)
        p -= h
        if p < w:
            return int(x2 - p), y2
        p -= w
        return x1, int(y2 - p)

    # ------------------------------------------------------------------
    # Person scan frame
    # ------------------------------------------------------------------

    def draw_scan_box(self, img, bbox, intensity):
        prev = self._bbox
        if prev is None:
            self._bbox = [float(v) for v in bbox]
        else:
            f = 1.0 - math.exp(-8.0 * self.dt)
            for i in range(4):
                prev[i] += (bbox[i] - prev[i]) * f
        x1, y1, x2, y2 = [int(round(v)) for v in self._bbox]
        if x2 - x1 < 4 or y2 - y1 < 4:
            return None

        l = max(16, min(30, (x2 - x1) // 9))
        bright = _blend(ACCENT, WHITE, 0.15)
        cv2.rectangle(img, (x1, y1), (x2, y2), (58, 66, 78), 1, cv2.LINE_AA)
        for cx, cy, dx, dy in ((x1, y1, 1, 1), (x2, y1, -1, 1),
                               (x1, y2, 1, -1), (x2, y2, -1, -1)):
            cv2.line(img, (cx, cy), (cx + dx * l, cy), bright, 2, cv2.LINE_AA)
            cv2.line(img, (cx, cy), (cx, cy + dy * l), bright, 2, cv2.LINE_AA)

        # gliding scanline (ping-pong) with a soft glow
        sy = y1 + int((y2 - y1) * (0.5 - 0.5 * math.cos(2 * math.pi * self.phase)))
        glow = img.copy()
        cv2.line(glow, (x1, sy), (x2, sy), ACCENT, 9, cv2.LINE_AA)
        cv2.addWeighted(glow, 0.05 + 0.12 * intensity, img, 1.0 - (0.05 + 0.12 * intensity), 0, img)
        cv2.line(img, (x1, sy), (x2, sy), ACCENT, 2, cv2.LINE_AA)

        # perimeter sweep with bright tip
        p1 = self._perim_pt(x1, y1, x2, y2, self.phase)
        p2 = self._perim_pt(x1, y1, x2, y2, self.phase - 0.06)
        cv2.line(img, p2, p1, _blend(ACCENT, WHITE, 0.4), 2, cv2.LINE_AA)
        cv2.circle(img, p1, 3, _blend(ACCENT, WHITE, 0.4), -1, cv2.LINE_AA)
        return [x1, y1, x2, y2]

    def draw_ambient_scan(self, img):
        """Faint full-frame sweep used while searching for a subject."""
        sy = int(self.h * (0.5 - 0.5 * math.cos(2.0 * math.pi * self.phase * 0.6)))
        glow = img.copy()
        cv2.line(glow, (0, sy), (self.w, sy), ACCENT, 6, cv2.LINE_AA)
        cv2.addWeighted(glow, 0.08, img, 0.92, 0, img)
        for cx, cy, dx, dy in ((10, 10, 1, 1), (self.w - 10, 10, -1, 1),
                               (10, self.h - 10, 1, -1), (self.w - 10, self.h - 10, -1, -1)):
            cv2.line(img, (cx, cy), (cx + dx * 16, cy), (42, 48, 58), 1, cv2.LINE_AA)
            cv2.line(img, (cx, cy), (cx, cy + dy * 16), (42, 48, 58), 1, cv2.LINE_AA)

    # ------------------------------------------------------------------
    # Status chip
    # ------------------------------------------------------------------

    def draw_status_chip(self, img, text, color, pulse=False, sub=None):
        f = 0.5 + 0.5 * math.cos(2.0 * math.pi * self.phase)
        dot_r = 4 + (2 * int(f) if pulse else 0)
        main_w = _text_size(text, 0.5, 1)[0]
        sub_w = _text_size(sub or "", 0.36, 1)[0]
        tw = max(main_w, sub_w)
        pad = 12
        w = tw + 52
        h = 30 + (16 if sub else 0)
        x, y = 16, 16
        cy = y + h // 2
        fill_rounded(img, x, y, x + w, y + h, h // 2, INK, 0.82)
        rounded_rect(img, x, y, x + w, y + h, h // 2, _blend(MUTED, color, 0.45), 1)
        dot_alpha = (0.45 + 0.55 * f) if pulse else 1.0
        glow = img.copy()
        cv2.circle(glow, (x + 24, cy), dot_r + (4 if pulse else 0), color, -1)
        cv2.addWeighted(glow, 0.5 * dot_alpha, img, 1.0 - 0.5 * dot_alpha, 0, img)
        cv2.circle(img, (x + 24, cy), dot_r, color, -1, cv2.LINE_AA)
        _text(img, text, (x + 38, y + 19), 0.5, WHITE, 1)
        if sub:
            _text(img, sub, (x + 38, y + 36), 0.36, MUTED, 1)
        self.chip_rect = (x, y, x + w, y + h)

    # ------------------------------------------------------------------
    # Score card
    # ------------------------------------------------------------------

    def draw_score_card(self, img, scores, style_label):
        wc, hc = 250, 232
        x, y = self.w - wc - 16, 16
        self.card_rect = (x, y, x + wc, y + hc)
        px = x + 16

        if scores is not None:
            target = float(scores.get("overall", 0.0))
            self.disp_score = self._smooth(self.disp_score, target, self.dt, k=6.0)
            for key, _ in COMP_ORDER:
                tgt = float(scores["components"].get(key, 0.0))
                if key not in self.disp_comps:
                    self.disp_comps[key] = tgt
                else:
                    self.disp_comps[key] = self._smooth(self.disp_comps[key], tgt, self.dt, k=8.0)

        fill_rounded(img, x, y, x + wc, y + hc, 12, INK, 0.82)
        cv2.line(img, (x + 3, y + 3), (x + wc - 3, y + 3), ACCENT, 2, cv2.LINE_AA)
        rounded_rect(img, x, y, x + wc, y + hc, 12, (58, 66, 78), 1)

        _text(img, "FITCHECK", (px, y + 32), 0.62, WHITE, 1)

        score_color = WHITE if scores is None else _score_color(self.disp_score)
        score_txt = f"{self.disp_score:.1f}" if scores is not None else "--.-"
        sw = _text_size(score_txt, 1.9, 3)[0]
        _text(img, score_txt, (px, y + 78), 1.9, score_color, 3)
        _text(img, "/ 10", (px + sw + 8, y + 78), 0.5, MUTED, 1)

        cap = (style_label or "awaiting analysis").title()
        if len(cap) > 22:
            cap = cap[:21] + ".."
        _text(img, cap, (px, y + 100), 0.4, MUTED, 1)
        cv2.line(img, (px, y + 112), (x + wc - 16, y + 112), (46, 52, 62), 1, cv2.LINE_AA)

        bx = px + 66
        bw = (x + wc - 74) - bx
        for i, (key, short) in enumerate(COMP_ORDER):
            ry = y + 128 + i * 17
            val = self.disp_comps.get(key, 0.0)
            col = _score_color(val)
            _text(img, short, (px, ry + 12), 0.42, (190, 196, 206), 1)
            cv2.rectangle(img, (bx, ry + 8), (bx + bw, ry + 12), TRACK, -1)
            if val > 0.02:
                fw = max(3, int(bw * val / 10.0))
                cv2.rectangle(img, (bx, ry + 8), (bx + fw, ry + 12), col, -1)
            vt = f"{val:.1f}"
            vw = _text_size(vt, 0.4, 1)[0]
            _text(img, vt, (x + wc - 16 - vw, ry + 12), 0.4, col, 1)

    # ------------------------------------------------------------------
    # Region labels + pointer lines
    # ------------------------------------------------------------------

    def draw_regions(self, img, regions, result, scan_box=None):
        col = (88, 98, 112) if result is not None else (58, 66, 78)
        for name in ("head", "upper", "lower", "shoes"):
            x1, y1, x2, y2 = regions[name]
            cv2.rectangle(img, (x1, y1), (x2, y2), col, 1, cv2.LINE_AA)
        if result is not None:
            return self._draw_labels(img, regions, result, scan_box)
        return []

    def _draw_labels(self, img, regions, result, scan_box):
        blocked = [b for b in (self.card_rect, self.chip_rect) if b is not None]
        placed = []
        for name, rkey, title in REGION_TITLES:
            res = result.get(rkey) or {}
            line2 = _display_text(res)
            if line2 is None:
                continue
            bw = max(_text_size(title, 0.42, 1)[0], _text_size(line2, 0.52, 1)[0]) + 22
            bh = 44

            x1, y1, x2, y2 = regions[name]
            rx, ry = (x1 + x2) // 2, (y1 + y2) // 2
            gap = 16
            if scan_box is None:
                scan_box = [x1, y1, x2, y2]
            prefer_right = rx < self.w // 2
            cands = [(x2 + gap, ry - bh // 2), (x1 - gap - bw, ry - bh // 2),
                     (x1 - gap - bw - 6, y1 - bh - gap), (x2 + gap + 6, y1 - bh - gap),
                     (rx - bw // 2, y2 + gap)]
            if not prefer_right:
                cands[0], cands[1] = cands[1], cands[0]

            box = None
            for cx, cy in cands:
                cb = self._clamp_box(cx, cy, bw, bh)
                if self._fits(cb, scan_box, blocked):
                    box = cb
                    break
            if box is None:
                box = self._clamp_box(cands[0][0], cands[0][1], bw, bh)

            self._draw_label_box(img, box, (title, line2))
            self._draw_pointer(img, box, (rx, ry))
            blocked.append(box)
            placed.append((box, (rx, ry)))
        return placed

    def _clamp_box(self, cx, cy, bw, bh):
        x1 = max(6, min(int(cx), self.w - bw - 6))
        y1 = max(6, min(int(cy), self.h - bh - 6))
        return (x1, y1, x1 + bw, y1 + bh)

    @staticmethod
    def _fits(box, scan, blocked):
        if rects_overlap(box, scan, 4):
            return False
        for b in blocked:
            if rects_overlap(box, b, 6):
                return False
        return True

    def _draw_label_box(self, img, box, lines):
        x1, y1, x2, y2 = box
        fill_rounded(img, x1, y1, x2, y2, 6, INK, 0.72)
        rounded_rect(img, x1, y1, x2, y2, 6, (62, 70, 82), 1)
        _text(img, lines[0], (x1 + 10, y1 + 15), 0.42, ACCENT, 1)
        _text(img, lines[1], (x1 + 10, y1 + 33), 0.52, WHITE, 1)

    def _draw_pointer(self, img, box, target):
        bx1, by1, bx2, by2 = box
        tx, ty = target
        if bx1 <= tx <= bx2 and by1 <= ty <= by2:
            return
        cx = max(bx1, min(tx, bx2))
        cy = max(by1, min(ty, by2))
        col = _blend(ACCENT, WHITE, 0.35)
        cv2.line(img, (cx, cy), (tx, ty), col, 1, cv2.LINE_AA)
        cv2.circle(img, (tx, ty), 2, col, -1, cv2.LINE_AA)
        cv2.circle(img, (tx, ty), 5, col, 1, cv2.LINE_AA)


if __name__ == "__main__":
    # Smoke check: renders a synthetic scene and saves a preview image.
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    os.makedirs(out, exist_ok=True)
    frame = np.full((720, 1280, 3), (24, 26, 30), np.uint8)
    hud = HUD(1280, 720)
    bbox = (420, 160, 860, 640)
    fake_regions = {
        "head": (430, 170, 850, 300),
        "upper": (430, 300, 850, 470),
        "lower": (430, 470, 850, 600),
        "shoes": (430, 600, 850, 655),
        "full": (420, 160, 860, 640),
    }
    fake_result = {
        "hair": {"label": "textured hair", "confidence": 0.6},
        "top": {"label": "hoodie", "confidence": 0.85, "color": "grey"},
        "bottom": {"label": "jeans", "confidence": 0.8, "color": "navy"},
        "shoes": {"label": "sneakers", "confidence": 0.75, "color": "white"},
        "style": {"label": "smart casual", "confidence": 0.9},
        "accessories": {"label": "wearing a watch", "confidence": 0.8},
    }
    fake_scores = {
        "overall": 8.7,
        "components": {"color_coordination": 8.0, "top_bottom_combo": 8.5,
                       "style_consistency": 9.0, "outfit_balance": 8.2,
                       "footwear_coordination": 8.8, "hair_grooming": 7.9},
    }
    preview = None
    label_boxes = []
    for _ in range(240):
        hud.update()
        preview = frame.copy()
        sb = hud.draw_scan_box(preview, bbox, 1.0)
        label_boxes = hud.draw_regions(preview, fake_regions, fake_result, sb)
        hud.draw_status_chip(preview, "RESULT READY", GREEN, pulse=False, sub="Smart Casual")
        hud.draw_score_card(preview, fake_scores, "smart casual")

    assert len(label_boxes) == 4, "expected 4 placed labels"
    for item in label_boxes:
        box, _pt = item
        assert not rects_overlap(box, sb, 4), f"label overlaps person: {box}"
        assert not rects_overlap(box, hud.chip_rect, 4), f"label overlaps chip: {box}"
        assert not rects_overlap(box, hud.card_rect, 4), f"label overlaps card: {box}"
        x1, y1, x2, y2 = box
        assert x1 >= 0 and y1 >= 0 and x2 <= 1280 and y2 <= 720, f"label off-frame: {box}"
    for i, (box, pt) in enumerate(label_boxes):
        cx, cy = pt
        assert not rects_overlap(box, (cx - 5, cy - 5, cx + 5, cy + 5)), f"pointer target under label {i}"

    # Multi-outfit display verification: natural labels, never "Unknown"/empty.
    outfits = [
        ({"label": "t-shirt", "color": "black"}, "Black T-Shirt"),
        ({"label": "jeans", "color": "blue"}, "Blue Jeans"),
        ({"label": "sneakers", "color": "white"}, "White Sneakers"),
        ({"label": "textured hair"}, "Textured Hair"),
        ({"label": "hoodie", "color": "unknown"}, "Hoodie"),
        ({"label": "polo shirt", "color": "navy"}, "Navy Polo Shirt"),
        ({"label": "Unknown", "confidence": 0.1}, None),
        ({"label": "", "confidence": 0.0}, None),
        ({}, None),
    ]
    for res, expected in outfits:
        got = _display_text(res)
        assert got == expected, f"display mismatch: {res!r} -> {got!r}, want {expected!r}"
        if got is not None:
            assert "unknown" not in got.lower() and "unk" not in got.lower(), got
    print("display labels ok")

    path = os.path.join(out, "ui_preview.png")
    cv2.imwrite(path, preview)
    print("layout ok; preview ->", path)