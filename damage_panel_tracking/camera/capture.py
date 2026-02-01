from __future__ import annotations

import time
from typing import Any, Dict, Tuple

import cv2

from .v4l2ctl import dev_to_path, v4l2_set


def setup_camera(device: Any, capture_cfg: Dict[str, Any], init_ctrls: Dict[str, Any]) -> Tuple[cv2.VideoCapture, str]:
    cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera: {device}")

    fourcc = str(capture_cfg.get("fourcc", "MJPG"))
    width = int(capture_cfg.get("width", 800))
    height = int(capture_cfg.get("height", 600))
    fps = int(capture_cfg.get("fps", 120))

    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    except Exception:
        pass
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)

    dev_path = dev_to_path(device)
    apply_camera_init(dev_path, cap, init_ctrls)
    return cap, dev_path


def apply_camera_init(dev_path: str, cap: cv2.VideoCapture, init_ctrls: Dict[str, Any]) -> None:
    """Apply camera init controls once at startup."""

    # Order-sensitive controls first
    if "auto_exposure" in init_ctrls:
        v4l2_set(dev_path, "auto_exposure", init_ctrls["auto_exposure"])
    if "white_balance_automatic" in init_ctrls:
        v4l2_set(dev_path, "white_balance_automatic", init_ctrls["white_balance_automatic"])
    if "focus_automatic_continuous" in init_ctrls:
        v4l2_set(dev_path, "focus_automatic_continuous", init_ctrls["focus_automatic_continuous"])

    # Apply rest
    for k, v in init_ctrls.items():
        if k in ("auto_exposure", "white_balance_automatic", "focus_automatic_continuous"):
            continue
        v4l2_set(dev_path, k, v)

    # Fallback (best-effort)
    try:
        if "brightness" in init_ctrls:
            cap.set(cv2.CAP_PROP_BRIGHTNESS, float(init_ctrls["brightness"]))
        if "contrast" in init_ctrls:
            cap.set(cv2.CAP_PROP_CONTRAST, float(init_ctrls["contrast"]))
        if "saturation" in init_ctrls:
            cap.set(cv2.CAP_PROP_SATURATION, float(init_ctrls["saturation"]))
        if "gain" in init_ctrls:
            cap.set(cv2.CAP_PROP_GAIN, float(init_ctrls["gain"]))
        if "gamma" in init_ctrls:
            cap.set(cv2.CAP_PROP_GAMMA, float(init_ctrls["gamma"]))
        if "sharpness" in init_ctrls:
            cap.set(cv2.CAP_PROP_SHARPNESS, float(init_ctrls["sharpness"]))
        if "exposure_time_absolute" in init_ctrls:
            cap.set(cv2.CAP_PROP_EXPOSURE, float(init_ctrls["exposure_time_absolute"]))
        if "white_balance_temperature" in init_ctrls:
            try:
                cap.set(cv2.CAP_PROP_WB_TEMPERATURE, float(init_ctrls["white_balance_temperature"]))
            except Exception:
                pass
    except Exception:
        pass

    time.sleep(0.05)
