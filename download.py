"""
download.py

Downloads every model FitCheck AI needs, once, into ./models/.

Usage:
    python download.py

Requires a .env file with:
    HF_TOKEN=your_huggingface_token_here
"""

from __future__ import annotations
import os
import sys

from dotenv import load_dotenv

import config


def fail(msg: str) -> None:
    print(f"\nERROR: {msg}")
    sys.exit(1)


def download_fashion_clip(token: str) -> None:
    from huggingface_hub import snapshot_download

    print(f"\n[download] FashionCLIP ({config.FASHION_CLIP_REPO})")

    if os.path.isdir(config.FASHION_CLIP_DIR) and os.listdir(config.FASHION_CLIP_DIR):
        print("[download] Already present locally, skipping.")
        return

    try:
        snapshot_download(
            repo_id=config.FASHION_CLIP_REPO,
            local_dir=config.FASHION_CLIP_DIR,
            token=token,
            # safetensors + config/json/text files are all we need; skip
            # any legacy .bin/.h5 duplicate weight formats to save space.
            ignore_patterns=["*.bin", "*.h5", "*.msgpack", "*.ot"],
        )
    except Exception as e:  # noqa: BLE001
        fail(f"Failed to download FashionCLIP: {e}")

    print("[download] FashionCLIP downloaded successfully.")


def download_yolo_pose() -> None:
    """
    YOLOv8-pose is distributed by Ultralytics, not the Hugging Face Hub.
    Instantiating YOLO(<name>) triggers Ultralytics' own download-and-cache
    mechanism; we then copy the resulting weights file into ./models/ so
    everything the app needs lives in one predictable place.
    """
    print(f"\n[download] YOLO pose model ({config.YOLO_POSE_MODEL})")

    target_path = os.path.join(config.MODELS_DIR, config.YOLO_POSE_MODEL)
    if os.path.exists(target_path):
        print("[download] Already present locally, skipping.")
        return

    try:
        from ultralytics import YOLO
    except ImportError:
        fail("The 'ultralytics' package is not installed. Run: pip install -r requirements.txt")

    try:
        model = YOLO(config.YOLO_POSE_MODEL)  # downloads to Ultralytics cache on first use
        cached_path = getattr(model, "ckpt_path", None) or config.YOLO_POSE_MODEL
        if os.path.exists(cached_path):
            import shutil
            shutil.copyfile(cached_path, target_path)
        elif os.path.exists(config.YOLO_POSE_MODEL):
            import shutil
            shutil.move(config.YOLO_POSE_MODEL, target_path)
        else:
            print("[download] Note: could not locate cached weights file to copy into models/; "
                  "Ultralytics will still use its own cache at runtime.")
    except Exception as e:  # noqa: BLE001
        fail(f"Failed to download YOLO pose model: {e}")

    print("[download] YOLO pose model ready.")


def main():
    load_dotenv()

    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        fail(
            "HF_TOKEN not found. Create a .env file (see .env.example) with:\n"
            "    HF_TOKEN=your_huggingface_token_here"
        )

    os.makedirs(config.MODELS_DIR, exist_ok=True)

    print("FITCHECK AI - Model downloader")
    print(f"Models will be saved to: {config.MODELS_DIR}")

    download_fashion_clip(token)
    download_yolo_pose()

    print("\nAll models ready. You can now run: python app.py")


if __name__ == "__main__":
    main()
