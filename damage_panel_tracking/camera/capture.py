from __future__ import annotations

import time
from typing import Any, Dict, Tuple

import cv2

from .v4l2ctl import dev_to_path, v4l2_set

_WARNED_CTRL_FAILS: set[str] = set()


_CTRL_TO_CAP_PROP: Dict[str, int] = {
    "brightness": cv2.CAP_PROP_BRIGHTNESS,
    "contrast": cv2.CAP_PROP_CONTRAST,
    "saturation": cv2.CAP_PROP_SATURATION,
    "hue": cv2.CAP_PROP_HUE,
    "gain": cv2.CAP_PROP_GAIN,
    "gamma": cv2.CAP_PROP_GAMMA,
    "sharpness": cv2.CAP_PROP_SHARPNESS,
    "exposure_time_absolute": cv2.CAP_PROP_EXPOSURE,
    "white_balance_temperature": cv2.CAP_PROP_WB_TEMPERATURE,
    "white_balance_automatic": cv2.CAP_PROP_AUTO_WB,
    "focus_automatic_continuous": cv2.CAP_PROP_AUTOFOCUS,
}


def _to_cap_prop_value(name: str, value: Any) -> float:
    # OpenCVの値スケール差分（特にauto_exposure）を吸収する。
    if name == "auto_exposure":
        iv = int(value)
        if iv == 1:
            return 0.25  # Manual mode (OpenCV/V4L2 convention)
        if iv == 3:
            return 0.75  # Auto mode (OpenCV/V4L2 convention)
        return float(iv)
    return float(value)


def _cap_prop_for(name: str) -> int | None:
    if name == "auto_exposure":
        return cv2.CAP_PROP_AUTO_EXPOSURE
    return _CTRL_TO_CAP_PROP.get(name)


def _set_cap_prop(cap: cv2.VideoCapture | None, name: str, value: Any) -> bool:
    # v4l2-ctl失敗時のフォールバックとしてOpenCVプロパティでも適用を試みる。
    if cap is None:
        return False
    prop_id = _cap_prop_for(name)
    if prop_id is None:
        return False
    try:
        return bool(cap.set(prop_id, _to_cap_prop_value(name, value)))
    except Exception:
        return False


def set_camera_control(dev_path: str, cap: cv2.VideoCapture | None, name: str, value: Any) -> bool:
    """Apply camera control using v4l2-ctl first, then CAP_PROP fallback."""
    v4l2_ok = v4l2_set(dev_path, name, value)
    cap_ok = _set_cap_prop(cap, name, value)
    return bool(v4l2_ok or cap_ok)


def _apply_with_retry(
    dev_path: str,
    cap: cv2.VideoCapture | None,
    name: str,
    value: Any,
    *,
    retries: int = 2,
    delay_sec: float = 0.02,
) -> bool:
    # 一部UVC機器では自動制御切替の直後に値が書けないため、短時間だけ再試行する。
    for i in range(max(0, int(retries)) + 1):
        if set_camera_control(dev_path, cap, name, value):
            return True
        if i < retries:
            time.sleep(max(0.0, float(delay_sec)))
    key = f"{dev_path}:{name}"
    if key not in _WARNED_CTRL_FAILS:
        _WARNED_CTRL_FAILS.add(key)
        print(f"[WARN] failed to apply camera control: {name}={value} device={dev_path}")
    return False


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
    auto_exposure = init_ctrls.get("auto_exposure")
    wb_auto = init_ctrls.get("white_balance_automatic")

    if "auto_exposure" in init_ctrls:
        _apply_with_retry(dev_path, cap, "auto_exposure", init_ctrls["auto_exposure"], retries=3, delay_sec=0.03)
        time.sleep(0.06)
    if "white_balance_automatic" in init_ctrls:
        _apply_with_retry(
            dev_path,
            cap,
            "white_balance_automatic",
            init_ctrls["white_balance_automatic"],
            retries=3,
            delay_sec=0.03,
        )
        time.sleep(0.04)
    if "focus_automatic_continuous" in init_ctrls:
        _apply_with_retry(
            dev_path,
            cap,
            "focus_automatic_continuous",
            init_ctrls["focus_automatic_continuous"],
            retries=2,
            delay_sec=0.02,
        )

    # 残りの制御を適用する。
    deferred: list[tuple[str, Any]] = []
    for k, v in init_ctrls.items():
        if k in ("auto_exposure", "white_balance_automatic", "focus_automatic_continuous"):
            continue
        if k == "exposure_time_absolute" and auto_exposure is not None and int(auto_exposure) != 1:
            # auto_exposure=manual(1) 以外では通常inactiveになるため後段適用をスキップする。
            continue
        if k == "white_balance_temperature" and wb_auto is not None and int(wb_auto) != 0:
            # white_balance_automatic=0(手動) でない場合は通常inactiveになる。
            continue
        if k in ("exposure_time_absolute", "white_balance_temperature"):
            deferred.append((k, v))
            continue
        _apply_with_retry(dev_path, cap, k, v, retries=2, delay_sec=0.02)

    # auto系切替直後に反映されにくい制御は最後にまとめて再試行する。
    for k, v in deferred:
        _apply_with_retry(dev_path, cap, k, v, retries=4, delay_sec=0.03)

    time.sleep(0.05)
