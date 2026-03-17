from __future__ import annotations

import time
from typing import Any, Dict, Tuple

import cv2

from .v4l2ctl import dev_to_path, has_v4l2_ctl, v4l2_set


def _prefer_v4l2_backend(device: Any) -> bool:
    # video4linuxデバイスを指している場合はCAP_V4L2を優先する。
    if isinstance(device, int):
        return True
    if isinstance(device, str):
        return bool(dev_to_path(device))
    return False


def _open_capture(device: Any, resolved_dev: str) -> cv2.VideoCapture:
    """Prefer V4L2 for Linux camera devices, then fall back to default backend."""
    # バックエンドのフォールバック戦略つきでカメラを開く。
    open_device = resolved_dev or device

    if bool(resolved_dev) or _prefer_v4l2_backend(device):
        cap = cv2.VideoCapture(open_device, cv2.CAP_V4L2)
        if cap.isOpened():
            return cap
        cap.release()
    cap = cv2.VideoCapture(open_device)
    if cap.isOpened():
        return cap
    raise RuntimeError(f"Failed to open camera: {device} (resolved: {open_device})")


def _fourcc_to_str(v: float) -> str:
    # OpenCVのFOURCC整数値を可読なコーデック文字列へ変換する。
    iv = int(v)
    if iv <= 0:
        return "N/A"
    return "".join(chr((iv >> (8 * i)) & 0xFF) for i in range(4))


def setup_camera(device: Any, capture_cfg: Dict[str, Any], init_ctrls: Dict[str, Any]) -> Tuple[cv2.VideoCapture, str]:
    # カメラを開き、キャプチャ形式を要求し、起動時制御を適用する。
    dev_path = dev_to_path(device)
    cap = _open_capture(device, dev_path)
    if isinstance(device, str):
        req = device.strip()
        if req and dev_path and req != dev_path:
            print(f"[INFO] camera device resolved: {req} -> {dev_path}")

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


def _v4l2_set_with_retry(
    dev_path: str,
    name: str,
    value: Any,
    *,
    retries: int = 3,
    delay_s: float = 0.03,
) -> bool:
    # 一時的な未反映に備えて、短い間隔で再試行する。
    attempts = max(1, int(retries))
    for _ in range(attempts):
        if v4l2_set(dev_path, name, value):
            return True
        time.sleep(delay_s)
    return False


def _apply_opencv_fallback(cap: cv2.VideoCapture, name: str, value: Any) -> None:
    # v4l2-ctlが使えない/失敗した制御のみ、OpenCVプロパティへフォールバックする。
    try:
        if name == "brightness":
            cap.set(cv2.CAP_PROP_BRIGHTNESS, float(value))
        elif name == "contrast":
            cap.set(cv2.CAP_PROP_CONTRAST, float(value))
        elif name == "saturation":
            cap.set(cv2.CAP_PROP_SATURATION, float(value))
        elif name == "gain":
            cap.set(cv2.CAP_PROP_GAIN, float(value))
        elif name == "gamma":
            cap.set(cv2.CAP_PROP_GAMMA, float(value))
        elif name == "sharpness":
            cap.set(cv2.CAP_PROP_SHARPNESS, float(value))
        elif name == "exposure_time_absolute":
            cap.set(cv2.CAP_PROP_EXPOSURE, float(value))
        elif name == "white_balance_temperature":
            wb_prop = getattr(cv2, "CAP_PROP_WB_TEMPERATURE", None)
            if wb_prop is not None:
                cap.set(wb_prop, float(value))
    except Exception:
        pass


def apply_camera_init(dev_path: str, cap: cv2.VideoCapture, init_ctrls: Dict[str, Any]) -> None:
    """Apply camera init controls once at startup."""

    fallback_names: set[str] = set()
    v4l2_available = bool(dev_path) and has_v4l2_ctl()
    ordered_controls = ("auto_exposure", "white_balance_automatic", "focus_automatic_continuous")

    if v4l2_available:
        # 順序依存のある制御を先に適用する。
        for ctrl_name in ordered_controls:
            if ctrl_name not in init_ctrls:
                continue
            if not _v4l2_set_with_retry(dev_path, ctrl_name, init_ctrls[ctrl_name]):
                fallback_names.add(ctrl_name)

        # 残りの制御を適用する。
        for ctrl_name, value in init_ctrls.items():
            if ctrl_name in ordered_controls:
                continue
            if ctrl_name == "exposure_time_absolute" and "auto_exposure" in init_ctrls:
                _v4l2_set_with_retry(dev_path, "auto_exposure", init_ctrls["auto_exposure"])
                time.sleep(0.02)
            if ctrl_name == "white_balance_temperature" and "white_balance_automatic" in init_ctrls:
                _v4l2_set_with_retry(dev_path, "white_balance_automatic", init_ctrls["white_balance_automatic"])
                time.sleep(0.02)
            if not _v4l2_set_with_retry(dev_path, ctrl_name, value):
                fallback_names.add(ctrl_name)

        # 依存関係のある組を最後にもう一度適用し、起動直後の取りこぼしを減らす。
        if "auto_exposure" in init_ctrls and "exposure_time_absolute" in init_ctrls:
            _v4l2_set_with_retry(dev_path, "auto_exposure", init_ctrls["auto_exposure"])
            time.sleep(0.02)
            if not _v4l2_set_with_retry(dev_path, "exposure_time_absolute", init_ctrls["exposure_time_absolute"]):
                fallback_names.add("exposure_time_absolute")
        if "white_balance_automatic" in init_ctrls and "white_balance_temperature" in init_ctrls:
            _v4l2_set_with_retry(dev_path, "white_balance_automatic", init_ctrls["white_balance_automatic"])
            time.sleep(0.02)
            if not _v4l2_set_with_retry(dev_path, "white_balance_temperature", init_ctrls["white_balance_temperature"]):
                fallback_names.add("white_balance_temperature")
    else:
        fallback_names = set(init_ctrls.keys())

    if v4l2_available and fallback_names:
        failed_controls = ", ".join(sorted(fallback_names))
        print(f"[WARN] v4l2 init controls partially failed; fallback to OpenCV props: {failed_controls}")

    # v4l2-ctlが使えない/失敗した制御のみ、OpenCVプロパティへフォールバックする。
    for ctrl_name, value in init_ctrls.items():
        if ctrl_name not in fallback_names:
            continue
        _apply_opencv_fallback(cap, ctrl_name, value)

    time.sleep(0.05)
