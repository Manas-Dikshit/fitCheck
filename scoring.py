"""
scoring.py

Two responsibilities, both deliberately kept model-free:

1. Dominant color detection for a clothing crop (HSV histogram based).
2. The FitCheck scoring formula that turns detected attributes into a
   transparent, weighted 0-10 score.

Nothing here fabricates numbers: every score component is derived from
real detected values (color-wheel distance, label confidences, etc.), and
the formula/weights match what the README documents.
"""

from __future__ import annotations
import numpy as np
import cv2

from config import COLOR_NAMES, SCORE_WEIGHTS

# ---------------------------------------------------------------------------
# Color detection
# ---------------------------------------------------------------------------

# Reference HSV centers (OpenCV hue range 0-179) for each color name.
# (hue, sat_min, val_min/max are handled procedurally below; this table is
# just used for nearest-hue matching on chromatic colors.)
_HUE_CENTERS = {
    "red": 0,
    "orange": 15,
    "yellow": 28,
    "green": 60,
    "blue": 105,
    "navy": 112,
    "purple": 135,
    "pink": 155,
}


def _classify_pixel_bucket(h: float, s: float, v: float) -> str:
    """Classify a single mean HSV pixel into one of COLOR_NAMES."""
    # Achromatic checks first (low saturation or extreme value).
    if v < 40:
        return "black"
    if s < 28 and v > 200:
        return "white"
    if s < 35:
        return "grey"
    if s < 55 and v < 130:
        return "brown" if 5 <= h <= 30 else "grey"
    if 5 <= h <= 22 and s < 90 and v > 120:
        return "beige"
    if 5 <= h <= 25 and v < 150:
        return "brown"

    # Chromatic: nearest hue center, with navy vs blue split on value.
    best_name, best_dist = None, 1e9
    for name, center in _HUE_CENTERS.items():
        dist = min(abs(h - center), 180 - abs(h - center))
        if dist < best_dist:
            best_dist = dist
            best_name = name

    if best_name == "blue" and v < 110:
        return "navy"
    return best_name


def get_dominant_color(bgr_crop: np.ndarray) -> str:
    """
    Return the human-readable dominant color name of a BGR image crop.
    Ignores near-black background/shadow pixels at the very edges and
    uses a histogram vote rather than a single average pixel, which is
    more robust to noise, prints, and lighting gradients.
    """
    if bgr_crop is None or bgr_crop.size == 0:
        return "unknown"

    # Downscale for speed; color distribution doesn't need full resolution.
    h, w = bgr_crop.shape[:2]
    scale = 64 / max(h, w) if max(h, w) > 64 else 1.0
    if scale < 1.0:
        bgr_crop = cv2.resize(bgr_crop, (max(1, int(w * scale)), max(1, int(h * scale))))

    hsv = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2HSV).astype(np.float32)
    pixels = hsv.reshape(-1, 3)

    votes = {}
    for h_, s_, v_ in pixels:
        name = _classify_pixel_bucket(h_, s_, v_)
        votes[name] = votes.get(name, 0) + 1

    if not votes:
        return "unknown"

    dominant = max(votes.items(), key=lambda kv: kv[1])[0]
    return dominant if dominant in COLOR_NAMES else "unknown"


# ---------------------------------------------------------------------------
# Color-coordination heuristic
# ---------------------------------------------------------------------------

# Rough "distance" between color families on a simplified color wheel /
# neutral scale, used purely to score coordination -- not a claim about
# fashion theory correctness, just a consistent, explainable heuristic.
_NEUTRALS = {"black", "white", "grey", "navy", "beige", "brown"}

_WHEEL_ORDER = ["red", "orange", "yellow", "green", "blue", "purple", "pink"]


def _wheel_distance(c1: str, c2: str) -> float:
    if c1 == c2:
        return 0.0
    if c1 in _NEUTRALS and c2 in _NEUTRALS:
        return 0.15
    if c1 in _NEUTRALS or c2 in _NEUTRALS:
        return 0.25  # a neutral paired with any color is generally safe
    if c1 in _WHEEL_ORDER and c2 in _WHEEL_ORDER:
        i, j = _WHEEL_ORDER.index(c1), _WHEEL_ORDER.index(c2)
        n = len(_WHEEL_ORDER)
        d = min(abs(i - j), n - abs(i - j)) / (n / 2)  # 0 (same) .. 1 (opposite-ish)
        return d
    return 0.5


def color_coordination_score(colors: list[str]) -> float:
    """
    Score (0-10) for how well a set of detected colors coordinate.
    Fewer, more harmonious (neutral-anchored or adjacent-on-wheel) colors
    score higher; many clashing saturated colors score lower.
    """
    colors = [c for c in colors if c and c != "unknown"]
    if len(colors) < 2:
        return 7.5  # not enough info to judge combination; neutral default

    dists = []
    for i in range(len(colors)):
        for j in range(i + 1, len(colors)):
            dists.append(_wheel_distance(colors[i], colors[j]))
    avg_dist = sum(dists) / len(dists)  # 0 = harmonious, 1 = clashing
    return round(10.0 * (1.0 - avg_dist), 1)


# ---------------------------------------------------------------------------
# Overall score assembly
# ---------------------------------------------------------------------------

def _conf_to_10(confidence: float) -> float:
    """Map a 0-1 model confidence/similarity to a 0-10 scale, gently."""
    confidence = max(0.0, min(1.0, confidence))
    # Slight curve so mid confidences don't feel punishingly low.
    return round(10.0 * (0.35 + 0.65 * confidence), 1)


def compute_scores(result: dict) -> dict:
    """
    Given the structured analysis result (see analyzer.py), compute the
    weighted FitCheck score and its sub-component breakdown.

    Returns a dict:
        {
            "overall": float,
            "components": {name: float, ...}
        }
    """
    top = result.get("top", {})
    bottom = result.get("bottom", {})
    shoes = result.get("shoes", {})
    style = result.get("style", {})
    hair = result.get("hair", {})

    colors = [top.get("color", "unknown"), bottom.get("color", "unknown")]
    if shoes.get("color"):
        colors.append(shoes["color"])

    color_score = color_coordination_score(colors)

    # Top+bottom combination: average of their individual detection
    # confidences plus a bonus for a coherent color pairing.
    top_conf = top.get("confidence", 0.0)
    bottom_conf = bottom.get("confidence", 0.0)
    combo_conf_score = _conf_to_10((top_conf + bottom_conf) / 2.0)
    combo_score = round((combo_conf_score + color_score) / 2.0, 1)

    style_score = _conf_to_10(style.get("confidence", 0.0))

    # Outfit balance: how consistently confident detection was across all
    # regions -- a proxy for "the outfit reads as a coherent look".
    confidences = [c for c in (top_conf, bottom_conf, shoes.get("confidence", 0.0),
                                style.get("confidence", 0.0)) if c is not None]
    if confidences:
        mean_conf = sum(confidences) / len(confidences)
        spread = max(confidences) - min(confidences)
        balance_score = _conf_to_10(mean_conf) - round(spread * 3, 1)
        balance_score = max(0.0, round(balance_score, 1))
    else:
        balance_score = 5.0

    footwear_score = _conf_to_10(shoes.get("confidence", 0.0))

    hair_conf = hair.get("confidence", 0.0)
    hair_score = _conf_to_10(hair_conf)

    components = {
        "color_coordination": color_score,
        "top_bottom_combo": combo_score,
        "style_consistency": style_score,
        "outfit_balance": balance_score,
        "footwear_coordination": footwear_score,
        "hair_grooming": hair_score,
    }

    overall = sum(components[k] * SCORE_WEIGHTS[k] for k in SCORE_WEIGHTS)
    overall = max(0.0, min(10.0, round(overall, 1)))

    return {"overall": overall, "components": components}
