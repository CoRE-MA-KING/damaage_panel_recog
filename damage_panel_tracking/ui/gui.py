from __future__ import annotations

import time
from typing import Any, Callable, Dict

import cv2

from ..camera.capture import set_camera_control
from ..camera.v4l2ctl import v4l2_list_ctrls


FALLBACK_CTRL_RANGES: Dict[str, tuple[int, int]] = {
    "brightness": (-64, 64),
    "contrast": (0, 64),
    "saturation": (0, 128),
    "hue": (-40, 40),
    "white_balance_automatic": (0, 1),
    "gamma": (72, 500),
    "gain": (0, 100),
    "power_line_frequency": (0, 2),
    "white_balance_temperature": (2800, 6500),
    "sharpness": (0, 6),
    "backlight_compensation": (0, 2),
    "auto_exposure": (0, 3),
    "exposure_time_absolute": (1, 5000),
    "exposure_dynamic_framerate": (0, 1),
    "focus_automatic_continuous": (0, 1),
}

CTRL_ORDER: tuple[str, ...] = (
    "auto_exposure",
    "exposure_time_absolute",
    "exposure_dynamic_framerate",
    "white_balance_automatic",
    "white_balance_temperature",
    "focus_automatic_continuous",
    "power_line_frequency",
    "gain",
    "brightness",
    "contrast",
    "saturation",
    "hue",
    "gamma",
    "sharpness",
    "backlight_compensation",
)


def create_setting_gui(
    win_name: str,
    dev_path: str,
    cap: cv2.VideoCapture,
    hsv_cfg: Dict[str, Any],
    init_ctrls: Dict[str, Any],
) -> Callable[[], None]:
    """Create OpenCV trackbars and return a poller that syncs values into runtime config."""

    ctrl_info = v4l2_list_ctrls(dev_path)
    last_positions: Dict[str, int] = {}
    poll_targets: Dict[str, Callable[[int], None]] = {}

    def register_trackbar(
        tb_name: str,
        init_pos: int,
        tb_max: int,
        on_change: Callable[[int], None],
    ) -> None:
        # callbackが届かない環境向けに、最後に見た位置も保存してポーリング同期する。
        init_pos = int(max(0, min(int(tb_max), int(init_pos))))

        def _wrapped(pos: int) -> None:
            last_positions[tb_name] = int(pos)
            on_change(int(pos))

        cv2.createTrackbar(tb_name, win_name, init_pos, int(tb_max), _wrapped)
        poll_targets[tb_name] = _wrapped
        _wrapped(init_pos)

    def ctrl_range(name: str) -> tuple[int, int]:
        # 実機値の範囲を優先し、取得できない場合は既知のフォールバックを使う。
        if name in ctrl_info and ctrl_info[name].get("min") is not None and ctrl_info[name].get("max") is not None:
            mn = int(ctrl_info[name]["min"])
            mx = int(ctrl_info[name]["max"])
            return (mn, mx) if mn <= mx else (mx, mn)
        return FALLBACK_CTRL_RANGES.get(name, (0, 255))

    def clamp(name: str, v: int) -> int:
        mn, mx = ctrl_range(name)
        return max(mn, min(mx, int(v)))

    def apply_ctrl(name: str, value: int) -> None:
        value = clamp(name, value)
        init_ctrls[name] = value

        if name == "auto_exposure":
            set_camera_control(dev_path, cap, "auto_exposure", value)
            time.sleep(0.02)
            if value == 1 and "exposure_time_absolute" in init_ctrls:
                set_camera_control(dev_path, cap, "exposure_time_absolute", int(init_ctrls["exposure_time_absolute"]))
            return

        if name == "white_balance_automatic":
            set_camera_control(dev_path, cap, "white_balance_automatic", value)
            time.sleep(0.02)
            if value == 0 and "white_balance_temperature" in init_ctrls:
                set_camera_control(dev_path, cap, "white_balance_temperature", int(init_ctrls["white_balance_temperature"]))
            return

        set_camera_control(dev_path, cap, name, value)

    def make_ctrl_trackbar(name: str) -> None:
        mn, mx = ctrl_range(name)
        init = clamp(name, int(init_ctrls.get(name, mn)))
        tb_max = max(1, mx - mn)

        def on_change(pos: int) -> None:
            apply_ctrl(name, int(pos) + mn)

        register_trackbar(name, int(init - mn), int(tb_max), on_change)

    created: set[str] = set()
    for ctrl_name in CTRL_ORDER:
        if ctrl_name in init_ctrls:
            make_ctrl_trackbar(ctrl_name)
            created.add(ctrl_name)

    for ctrl_name in init_ctrls.keys():
        if ctrl_name in created:
            continue
        if ctrl_name not in ctrl_info and ctrl_name not in FALLBACK_CTRL_RANGES:
            continue
        make_ctrl_trackbar(ctrl_name)

    # HSV blue
    def bset(name: str) -> Callable[[int], None]:
        # blue閾値をその場で更新するコールバックを作る。
        return lambda v: hsv_cfg["blue"].__setitem__(name, max(0, min(255, int(v))))

    register_trackbar("B_H_low", int(hsv_cfg["blue"]["H_low"]), 179, bset("H_low"))
    register_trackbar("B_H_high", int(hsv_cfg["blue"]["H_high"]), 179, bset("H_high"))
    register_trackbar("B_S_low", int(hsv_cfg["blue"]["S_low"]), 255, bset("S_low"))
    register_trackbar("B_S_high", int(hsv_cfg["blue"]["S_high"]), 255, bset("S_high"))
    register_trackbar("B_V_low", int(hsv_cfg["blue"]["V_low"]), 255, bset("V_low"))
    register_trackbar("B_V_high", int(hsv_cfg["blue"]["V_high"]), 255, bset("V_high"))

    # HSV red
    def r1set(name: str) -> Callable[[int], None]:
        # red1色相レンジ更新コールバックを作る。
        return lambda v: hsv_cfg["red1"].__setitem__(name, max(0, min(179, int(v))))

    def r2set(name: str) -> Callable[[int], None]:
        # red2色相レンジ更新コールバックを作る。
        return lambda v: hsv_cfg["red2"].__setitem__(name, max(0, min(179, int(v))))

    def rsvset(name: str) -> Callable[[int], None]:
        # red共通の彩度/明度レンジ更新コールバックを作る。
        return lambda v: hsv_cfg["redSV"].__setitem__(name, max(0, min(255, int(v))))

    register_trackbar("R1_H_low", int(hsv_cfg["red1"]["H_low"]), 179, r1set("H_low"))
    register_trackbar("R1_H_high", int(hsv_cfg["red1"]["H_high"]), 179, r1set("H_high"))
    register_trackbar("R2_H_low", int(hsv_cfg["red2"]["H_low"]), 179, r2set("H_low"))
    register_trackbar("R2_H_high", int(hsv_cfg["red2"]["H_high"]), 179, r2set("H_high"))

    register_trackbar("R_S_low", int(hsv_cfg["redSV"]["S_low"]), 255, rsvset("S_low"))
    register_trackbar("R_S_high", int(hsv_cfg["redSV"]["S_high"]), 255, rsvset("S_high"))
    register_trackbar("R_V_low", int(hsv_cfg["redSV"]["V_low"]), 255, rsvset("V_low"))
    register_trackbar("R_V_high", int(hsv_cfg["redSV"]["V_high"]), 255, rsvset("V_high"))

    def poll_trackbars() -> None:
        # 一部WM環境でコールバックが欠落することがあるため、値変化を毎フレーム確認する。
        for tb_name, on_change in poll_targets.items():
            try:
                pos = int(cv2.getTrackbarPos(tb_name, win_name))
            except cv2.error:
                continue
            if last_positions.get(tb_name) == pos:
                continue
            on_change(pos)

    return poll_trackbars
