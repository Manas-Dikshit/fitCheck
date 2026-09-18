"""
app.py -- FitCheck AI

Real-time webcam outfit/style analyzer.

Fast loop  (every frame):   webcam capture + YOLOv8-pose person/keypoint
                             detection + overlay drawing.
Analysis loop (background):  FashionCLIP zero-shot classification of
                             head/top/bottom/shoe crops, run every
                             ANALYSIS_INTERVAL_SECONDS (or on big movement),
                             never on every frame.

Run:
    python app.py

Controls:
    Q / ESC  quit
    R        re-analyze immediately
    SPACE    freeze / unfreeze frame
    S        save current annotated frame to outputs/
"""

from __future__ import annotations
import os
import sys
import time
import threading
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
import torch

import ui
import config
from analyzer import FashionAnalyzer
from scoring import compute_scores

try:
    from ultralytics import YOLO
except ImportError:
    print("ERROR: ultralytics is not installed. Run: pip install -r requirements.txt")
    sys.exit(1)


# COCO keypoint indices used by YOLOv8-pose
KP_NOSE = 0
KP_L_SHOULDER, KP_R_SHOULDER = 5, 6
KP_L_HIP, KP_R_HIP = 11, 12
KP_L_ANKLE, KP_R_ANKLE = 15, 16
POSE_CONF_MIN = 0.35


# ---------------------------------------------------------------------------
# Shared analysis state
# ---------------------------------------------------------------------------

@dataclass
class AnalysisState:
    lock: threading.Lock = field(default_factory=threading.Lock)
    result: Optional[dict] = None
    scores: Optional[dict] = None
    analyzing: bool = False
    last_run_time: float = 0.0
    last_bbox_center: Optional[tuple] = None
    request_now: bool = False
    stop: bool = False


def device_str() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Region extraction from pose keypoints
# ---------------------------------------------------------------------------

def _clamp_box(x1, y1, x2, y2, w, h):
    x1 = max(0, min(int(x1), w - 1))
    x2 = max(0, min(int(x2), w))
    y1 = max(0, min(int(y1), h - 1))
    y2 = max(0, min(int(y2), h))
    if x2 <= x1:
        x2 = min(w, x1 + 1)
    if y2 <= y1:
        y2 = min(h, y1 + 1)
    return x1, y1, x2, y2


def get_regions(frame_shape, bbox, keypoints, kp_conf) -> dict:
    """
    Given a person bounding box and (optional) pose keypoints, compute
    pixel-space crop boxes for head, upper body, lower body, and shoes.
    Falls back to fixed proportions of the bbox when keypoints are missing
    or low-confidence.
    """
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = bbox
    box_h = y2 - y1
    box_w = x2 - x1

    def kp_ok(idx):
        return keypoints is not None and kp_conf is not None and kp_conf[idx] > POSE_CONF_MIN

    # Shoulder line
    if kp_ok(KP_L_SHOULDER) and kp_ok(KP_R_SHOULDER):
        shoulder_y = int((keypoints[KP_L_SHOULDER][1] + keypoints[KP_R_SHOULDER][1]) / 2)
    else:
        shoulder_y = int(y1 + box_h * config.REGION_FALLBACK["head_end"])

    # Hip line
    if kp_ok(KP_L_HIP) and kp_ok(KP_R_HIP):
        hip_y = int((keypoints[KP_L_HIP][1] + keypoints[KP_R_HIP][1]) / 2)
    else:
        hip_y = int(y1 + box_h * config.REGION_FALLBACK["upper_end"])

    # Ankle line
    if kp_ok(KP_L_ANKLE) and kp_ok(KP_R_ANKLE):
        ankle_y = int((keypoints[KP_L_ANKLE][1] + keypoints[KP_R_ANKLE][1]) / 2)
    else:
        ankle_y = int(y1 + box_h * config.REGION_FALLBACK["lower_end"])

    head_pad = int(box_h * 0.05)
    regions = {
        "head": _clamp_box(x1, y1 - head_pad, x2, shoulder_y, w, h),
        "upper": _clamp_box(x1, shoulder_y, x2, hip_y, w, h),
        "lower": _clamp_box(x1, hip_y, x2, ankle_y, w, h),
        "shoes": _clamp_box(x1, ankle_y, x2, min(y2, ankle_y + int(box_h * 0.12) + 20), w, h),
        "full": _clamp_box(x1, y1, x2, y2, w, h),
    }
    return regions


# ---------------------------------------------------------------------------
# Analysis worker (background thread)
# ---------------------------------------------------------------------------

def analysis_worker(state: AnalysisState, fashion: FashionAnalyzer, frame_source: dict):
    """
    Runs in a background thread. Periodically (or on request) pulls the
    latest available frame + region crops from `frame_source` and runs the
    expensive FashionCLIP classification, storing results in `state`.
    """
    while not state.stop:
        snapshot = frame_source.get("latest")
        now = time.time()

        due = (now - state.last_run_time) >= config.ANALYSIS_INTERVAL_SECONDS
        moved = False
        if snapshot is not None and state.last_bbox_center is not None:
            cx, cy = snapshot["bbox_center"]
            lcx, lcy = state.last_bbox_center
            frame_w = snapshot["frame_w"]
            if abs(cx - lcx) / frame_w > config.PERSON_MOVE_THRESHOLD:
                moved = True

        should_run = snapshot is not None and (due or moved or state.request_now)

        if not should_run:
            time.sleep(0.05)
            continue

        state.request_now = False
        state.analyzing = True

        regions = snapshot["regions"]
        frame = snapshot["frame"]

        def crop(name):
            x1, y1, x2, y2 = regions[name]
            return frame[y1:y2, x1:x2]

        try:
            hair_res = fashion.classify(crop("head"), config.HAIR_LABELS)
            top_res = fashion.classify_clothing(crop("upper"), config.TOP_LABELS)
            bottom_res = fashion.classify_clothing(crop("lower"), config.BOTTOM_LABELS)
            shoe_res = fashion.classify_clothing(crop("shoes"), config.SHOE_LABELS)
            style_res = fashion.classify(crop("full"), config.STYLE_LABELS)
            accessory_res = fashion.classify(crop("full"), config.ACCESSORY_LABELS)

            result = {
                "hair": hair_res,
                "top": top_res,
                "bottom": bottom_res,
                "shoes": shoe_res,
                "style": style_res,
                "accessories": accessory_res,
            }
            scores = compute_scores(result)

            with state.lock:
                state.result = result
                state.scores = scores
                state.last_run_time = time.time()
                state.last_bbox_center = snapshot["bbox_center"]
        except Exception as e:  # noqa: BLE001
            print(f"[analysis_worker] analysis failed: {e}")
        finally:
            state.analyzing = False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def detect_primary_person(pose_model, frame):
    """
    Runs YOLOv8-pose on the frame. Returns (bbox, keypoints, kp_conf, num_people)
    for the largest detected person, or (None, None, None, 0) if nobody found.
    """
    results = pose_model.predict(frame, verbose=False, conf=0.5)
    if not results:
        return None, None, None, 0

    r = results[0]
    if r.boxes is None or len(r.boxes) == 0:
        return None, None, None, 0

    boxes = r.boxes.xyxy.cpu().numpy()
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    primary_idx = int(np.argmax(areas))
    bbox = tuple(boxes[primary_idx])

    keypoints, kp_conf = None, None
    if r.keypoints is not None and len(r.keypoints) > primary_idx:
        kp_xy = r.keypoints.xy.cpu().numpy()[primary_idx]
        kp_c = r.keypoints.conf
        kp_conf = kp_c.cpu().numpy()[primary_idx] if kp_c is not None else np.ones(len(kp_xy))
        keypoints = kp_xy

    return bbox, keypoints, kp_conf, len(boxes)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def pick_video_file():
    """Native file picker for choosing a video file (stdlib tkinter)."""
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception:  # noqa: BLE001
        return None
    root = tk.Tk()
    root.withdraw()
    try:
        return filedialog.askopenfilename(
            title="Select a video to analyze",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.webm"), ("All files", "*.*")])
    finally:
        root.destroy()


MENU_WIN = "FitCheck AI - select source"


def choose_source():
    """Show a startup menu: live webcam or upload a video file."""
    w, h = 660, 340
    menu = np.full((h, w, 3), (16, 18, 24), np.uint8)
    rows = [("1", "LIVE WEBCAM", "Analyze your camera feed in real time"),
            ("2", "UPLOAD VIDEO", "Pick a video file and analyze it on screen"),
            ("ESC", "QUIT", "")]
    while True:
        frame = menu.copy()
        cv2.line(frame, (0, 0), (w, 0), ui.ACCENT, 4)
        cv2.putText(frame, "FITCHECK AI", (30, 64), cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                    ui.WHITE, 2, cv2.LINE_AA)
        cv2.putText(frame, "SELECT ANALYSIS SOURCE", (30, 94), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    ui.MUTED, 1, cv2.LINE_AA)
        y = 150
        for key, title, desc in rows:
            cv2.rectangle(frame, (24, y - 26), (w - 24, y + 24), ui.INK, -1)
            cv2.rectangle(frame, (24, y - 26), (w - 24, y + 24), (58, 66, 78), 1, cv2.LINE_AA)
            cv2.putText(frame, f"[{key}]", (42, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        ui.ACCENT, 1, cv2.LINE_AA)
            cv2.putText(frame, title, (118, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        ui.WHITE, 1, cv2.LINE_AA)
            if desc:
                cv2.putText(frame, desc, (42, y + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                            ui.MUTED, 1, cv2.LINE_AA)
            y += 66

        cv2.imshow(MENU_WIN, frame)
        key = cv2.waitKey(30) & 0xFF

        if key in (ord('1'),):
            cv2.destroyWindow(MENU_WIN)
            return ("camera", config.CAMERA_INDEX)
        if key in (ord('2'),):
            path = pick_video_file()
            if path:
                cv2.destroyWindow(MENU_WIN)
                return ("video", path)
        if key in (ord('q'), ord('Q'), 27):  # ESC / Q
            cv2.destroyWindow(MENU_WIN)
            return None


def main():
    os.makedirs(config.OUTPUTS_DIR, exist_ok=True)

    dev = device_str()
    print("FITCHECK AI")
    print("Initializing models...")
    print(f"Using {'CUDA' if dev == 'cuda' else 'CPU'}")

    pose_model_path = os.path.join(config.MODELS_DIR, config.YOLO_POSE_MODEL)
    if not os.path.exists(pose_model_path):
        print(f"[app] {config.YOLO_POSE_MODEL} not found in models/, "
              f"letting Ultralytics fetch it (run download.py to pre-fetch next time).")
        pose_model_path = config.YOLO_POSE_MODEL

    try:
        pose_model = YOLO(pose_model_path)
        if dev == "cuda":
            pose_model.to("cuda")
    except Exception as e:  # noqa: BLE001
        print(f"ERROR: failed to load pose model: {e}")
        sys.exit(1)

    try:
        fashion = FashionAnalyzer(device=dev)
    except Exception as e:  # noqa: BLE001
        print(f"ERROR: failed to load FashionCLIP: {e}")
        print("Did you run 'python download.py' and set HF_TOKEN in .env?")
        sys.exit(1)

    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg.lower() == "camera" or arg.isdigit():
            source = ("camera", config.CAMERA_INDEX)
        elif os.path.isfile(arg):
            source = ("video", arg)
        else:
            print(f"[app] ignoring unknown argument: {arg}")
            source = choose_source()
    else:
        source = choose_source()
    if source is None:
        print("[app] no source selected, exiting.")
        return

    kind, src = source
    display_title = (os.path.basename(src) if kind == "video" else "LIVE WEBCAM")

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"ERROR: could not open {src}. Check that it exists and is a readable video "
              f"({'webcam' if kind == 'camera' else 'file'}); release it if another app is using it.")
        sys.exit(1)
    if kind == "camera":
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    state = AnalysisState()
    frame_source = {"latest": None}

    worker = threading.Thread(target=analysis_worker, args=(state, fashion, frame_source), daemon=True)
    worker.start()

    print(f"{display_title} ready")

    frozen = False
    frozen_frame = None
    window_name = f"FitCheck AI - {display_title}"
    hud = None

    try:
        while True:
            if not frozen:
                ok, frame = cap.read()
                if not ok and kind == "video":
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop the video
                    ok, frame = cap.read()
                if not ok:
                    print("ERROR: lost video feed.")
                    break
            else:
                frame = frozen_frame.copy()

            display = frame.copy()
            h, w = frame.shape[:2]
            if hud is None:
                hud = ui.HUD(w, h)
            hud.update()

            bbox, keypoints, kp_conf, num_people = detect_primary_person(pose_model, frame)

            if bbox is None:
                hud.draw_ambient_scan(display)
                with state.lock:
                    scores_snapshot = state.scores
                hud.draw_status_chip(display, "SCANNING", ui.ACCENT, pulse=True,
                                     sub="Step into the frame")
                hud.draw_score_card(display, scores_snapshot, None)
            else:
                x1, y1, x2, y2 = [int(v) for v in bbox]
                regions = get_regions(frame.shape, (x1, y1, x2, y2), keypoints, kp_conf)

                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                frame_source["latest"] = {
                    "frame": frame.copy(),
                    "regions": regions,
                    "bbox_center": (cx, cy),
                    "frame_w": w,
                }

                with state.lock:
                    result_snapshot = state.result
                    scores_snapshot = state.scores
                    analyzing = state.analyzing

                scan_box = hud.draw_scan_box(display, (x1, y1, x2, y2),
                                             1.0 if analyzing else 0.45)

                style_label = result_snapshot["style"]["label"] if result_snapshot else None
                working = analyzing or result_snapshot is None
                status_text = "ANALYZING" if working else "RESULT READY"
                status_col = ui.AMBER if working else ui.GREEN
                sub = None
                if num_people > 1:
                    sub = f"{num_people} subjects - focusing primary"
                elif not working and style_label:
                    sub = style_label.title()
                hud.draw_status_chip(display, status_text, status_col,
                                     pulse=working, sub=sub)

                hud.draw_score_card(display, scores_snapshot, style_label)
                hud.draw_regions(display, regions, result_snapshot, scan_box)

            if frozen:
                cv2.putText(display, "FROZEN", (16, h - 16), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 200, 255), 2, cv2.LINE_AA)

            cv2.imshow(window_name, display)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord('q'), ord('Q'), 27):  # Q / ESC
                break
            elif key in (ord('r'), ord('R')):
                state.request_now = True
            elif key == ord(' '):
                frozen = not frozen
                if frozen:
                    frozen_frame = frame.copy()
            elif key in (ord('s'), ord('S')):
                fname = os.path.join(config.OUTPUTS_DIR, f"fitcheck_{int(time.time())}.png")
                cv2.imwrite(fname, display)
                print(f"[app] saved {fname}")

    finally:
        state.stop = True
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
