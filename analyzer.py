"""
analyzer.py

Wraps FashionCLIP (patrickjohncyh/fashion-clip) for zero-shot image/text
similarity classification of body-region crops (hair, top, bottom, shoes,
overall style, accessories).

The model is loaded exactly once (see FashionAnalyzer.__init__) and reused
for every classification call -- it is never reloaded per frame.

All returned confidences are the model's own softmax-normalized similarity
scores. The top-scoring candidate is always reported so the UI never shows
"Unknown": low confidence is kept internally (it still scales the score),
but a best-supported label is available instead of a guess. The model never
invents attributes that were not candidates in the first place.
"""

from __future__ import annotations
import os
from typing import Optional

import numpy as np
from PIL import Image
import torch

from config import FASHION_CLIP_DIR, FASHION_CLIP_REPO
from scoring import get_dominant_color


class FashionAnalyzer:
    """Loads FashionCLIP once and exposes zero-shot classification helpers."""

    def __init__(self, device: Optional[str] = None):
        from transformers import CLIPModel, CLIPProcessor

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        model_path = FASHION_CLIP_DIR if os.path.isdir(FASHION_CLIP_DIR) else FASHION_CLIP_REPO
        if not os.path.isdir(FASHION_CLIP_DIR):
            print(f"[analyzer] Local FashionCLIP weights not found at {FASHION_CLIP_DIR}; "
                  f"falling back to downloading '{FASHION_CLIP_REPO}' from the Hub. "
                  f"Run download.py first to avoid this at startup.")

        self.model = CLIPModel.from_pretrained(model_path)
        self.processor = CLIPProcessor.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

        # Cache text embeddings per unique label set so repeated calls with
        # the same candidate labels (e.g. every analysis cycle) don't re-run
        # the text tower each time.
        self._text_cache: dict[tuple, torch.Tensor] = {}

    @torch.inference_mode()
    def _encode_text(self, labels: list[str]) -> torch.Tensor:
        key = tuple(labels)
        if key in self._text_cache:
            return self._text_cache[key]
        inputs = self.processor(text=labels, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        text_features = self.model.get_text_features(**inputs)
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        self._text_cache[key] = text_features
        return text_features

    @torch.inference_mode()
    def classify(self, bgr_crop: np.ndarray, labels: list[str]) -> dict:
        """
        Zero-shot classify a BGR (OpenCV) crop against a list of candidate
        text labels. Returns {"label": str, "confidence": float} using the
        model's own softmax over cosine similarities.

        The best-supported candidate is always returned (never "Unknown"):
        confidence is kept for scoring/quality control, while the label
        reflects whichever candidate the model actually supports most.
        """
        if bgr_crop is None or bgr_crop.size == 0 or not labels:
            return {"label": "", "confidence": 0.0}

        rgb = bgr_crop[:, :, ::-1]
        image = Image.fromarray(rgb)

        image_inputs = self.processor(images=image, return_tensors="pt")
        image_inputs = {k: v.to(self.device) for k, v in image_inputs.items()}
        image_features = self.model.get_image_features(**image_inputs)
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)

        text_features = self._encode_text(labels)

        logit_scale = self.model.logit_scale.exp()
        similarity = (logit_scale * image_features @ text_features.T).softmax(dim=-1)
        probs = similarity.squeeze(0).detach().cpu().numpy()

        best_idx = int(np.argmax(probs))
        best_conf = float(probs[best_idx])

        return {"label": labels[best_idx], "confidence": round(best_conf, 3)}

    def classify_clothing(self, bgr_crop: np.ndarray, labels: list[str]) -> dict:
        """Classify a clothing crop and also attach its dominant color."""
        result = self.classify(bgr_crop, labels)
        result["color"] = get_dominant_color(bgr_crop) if bgr_crop is not None and bgr_crop.size else "unknown"
        result["type"] = result["label"]
        return result
