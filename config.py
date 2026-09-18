"""
config.py

All tunable configuration for FitCheck AI lives here: model names,
zero-shot candidate labels, scoring weights, and timing constants.
Edit this file to change vocabulary or behavior without touching app logic.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

FASHION_CLIP_DIR = os.path.join(MODELS_DIR, "fashion-clip")

# ---------------------------------------------------------------------------
# Model identifiers
# ---------------------------------------------------------------------------
FASHION_CLIP_REPO = "patrickjohncyh/fashion-clip"

# Ultralytics pose model gives us a person bounding box AND body keypoints
# in a single lightweight pass (nose, shoulders, hips, ankles, etc.), which
# we use to build head / upper-body / lower-body / shoe crops.
YOLO_POSE_MODEL = "yolov8n-pose.pt"

# ---------------------------------------------------------------------------
# Timing / performance
# ---------------------------------------------------------------------------
ANALYSIS_INTERVAL_SECONDS = 1.5   # minimum gap between FashionCLIP analyses
PERSON_MOVE_THRESHOLD = 0.12      # fraction of frame width; bbox center shift
                                    # large enough to trigger an early re-analysis
CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# ---------------------------------------------------------------------------
# Zero-shot candidate labels (FashionCLIP compares the crop against these)
# ---------------------------------------------------------------------------
TOP_LABELS = [
    "t-shirt", "polo shirt", "button-down shirt", "flannel shirt",
    "long sleeve shirt", "tank top", "crop top", "hoodie", "sweatshirt",
    "sweater", "cardigan", "jacket", "denim jacket", "leather jacket",
    "bomber jacket", "blazer", "coat", "kurta", "traditional top",
]

BOTTOM_LABELS = [
    "jeans", "trousers", "chinos", "dress pants", "shorts", "cargo pants",
    "joggers", "sweatpants", "leggings", "skirt", "traditional bottom",
]

SHOE_LABELS = [
    "sneakers", "running shoes", "formal shoes", "loafers", "boots",
    "sandals", "flip-flops", "high heels", "sports shoes",
]

STYLE_LABELS = [
    "casual", "smart casual", "formal", "streetwear",
    "sporty", "traditional", "minimalist", "party wear",
]

HAIR_LABELS = [
    "short hair", "medium length hair", "long hair", "curly hair",
    "wavy hair", "straight hair", "textured hair", "buzz cut",
    "bald head", "ponytail", "bun",
]

ACCESSORY_LABELS = [
    "wearing glasses", "wearing a cap or hat", "wearing a watch",
    "wearing a necklace", "wearing earrings", "no visible accessories",
]

# Simple readable color palette. Each entry maps to an HSV hue range used
# by scoring.py's dominant-color detector (see that file for the logic).
COLOR_NAMES = [
    "black", "white", "grey", "navy", "blue", "beige",
    "brown", "red", "green", "yellow", "pink", "purple", "orange",
]

# ---------------------------------------------------------------------------
# Scoring weights (must sum to 1.0)
# ---------------------------------------------------------------------------
SCORE_WEIGHTS = {
    "color_coordination": 0.25,
    "top_bottom_combo": 0.25,
    "style_consistency": 0.20,
    "outfit_balance": 0.15,
    "footwear_coordination": 0.10,
    "hair_grooming": 0.05,
}

assert abs(sum(SCORE_WEIGHTS.values()) - 1.0) < 1e-6, "SCORE_WEIGHTS must sum to 1.0"

# ---------------------------------------------------------------------------
# Region layout fallback ratios (used only if a keypoint is missing)
# Fractions of the person bounding box height, measured from the top.
# ---------------------------------------------------------------------------
REGION_FALLBACK = {
    "head_end": 0.20,
    "upper_end": 0.55,
    "lower_end": 0.90,
}
