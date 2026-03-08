from __future__ import annotations

import time
from typing import Any, Dict, Tuple

import cv2

from .v4l2ctl import dev_to_path, v4l2_set


def _prefer_v4l2_backend(device: Any) -> bool:
    # video4linuxデバイスを指している場合はCAP_V4L2を優先する。
    if isinstance(device, int):
        return True
    if isinstance(device, str):
        return bool(dev_to_path(device))
    return False


def _open_capture(device: Any) -> cv2.VideoCapture:
    """Prefer V4L2 for Linux camera devices, then fall back to default backend."""
    # バックエンドのフォールバック戦略つきでカメラを開く。
    resolved_dev = dev_to_path(device)
    if isinstance(device, int):
        open_device = device
    else:
        open_device = resolved_dev or device

    if _prefer_v4l2_backend(device):
        cap = cv2.VideoCapture(open_device, cv2.CAP_V4L2)
        if cap.isOpened():
            return cap
        cap.release()
    cap = cv2.VideoCapture(open_device)
    if cap.isOpened():
        return cap
    raise RuntimeError(f"Failed to open camera: {device}")


def _fourcc_to_str(v: float) -> str:
    # OpenCVのFOURCC整数値を可読なコーデック文字列へ変換する。
    iv = int(v)
    if iv <= 0:
        return "N/A"
    return "".join(chr((iv >> (8 * i)) & 0xFF) for i in range(4))


def setup_camera(device: Any, capture_cfg: Dict[str, Any], init_ctrls: Dict[str, Any]) -> Tuple[cv2.VideoCapture, str]:
    # カメラを開き、キャプチャ形式を要求し、起動時制御を適用する。
    cap = _open_capture(device)

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

    try:
        backend = cap.getBackendName()
    except Exception:
        backend = "UNKNOWN"
    actual_fourcc = _fourcc_to_str(cap.get(cv2.CAP_PROP_FOURCC))
    actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(
        "[INFO] camera open: "
        f"backend={backend} requested={width}x{height}@{fps} {fourcc} "
        f"actual={int(actual_w)}x{int(actual_h)}@{actual_fps:.1f} {actual_fourcc}"
    )
    if actual_fps > 0 and actual_fps < fps * 0.5:
        print(
            "[WARN] camera negotiated low fps. "
            "Check backend and pixel format (e.g. MJPG vs YUYV)."
        )
    return cap, dev_path


def apply_camera_init(dev_path: str, cap: cv2.VideoCapture, init_ctrls: Dict[str, Any]) -> None:
    """Apply camera init controls once at startup."""

    # 順序依存のある制御を先に適用する。
    if "auto_exposure" in init_ctrls:
        v4l2_set(dev_path, "auto_exposure", init_ctrls["auto_exposure"])
    if "white_balance_automatic" in init_ctrls:
        v4l2_set(dev_path, "white_balance_automatic", init_ctrls["white_balance_automatic"])
    if "focus_automatic_continuous" in init_ctrls:
        v4l2_set(dev_path, "focus_automatic_continuous", init_ctrls["focus_automatic_continuous"])

    # 残りの制御を適用する。
    for k, v in init_ctrls.items():
        if k in ("auto_exposure", "white_balance_automatic", "focus_automatic_continuous"):
            continue
        v4l2_set(dev_path, k, v)

    # v4l2制御が使えない場合はOpenCVプロパティへフォールバックする。
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
