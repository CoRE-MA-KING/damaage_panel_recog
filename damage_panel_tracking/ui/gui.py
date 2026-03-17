from __future__ import annotations

import time
from typing import Any, Callable, Dict

import cv2

from ..camera.v4l2ctl import v4l2_list_ctrls, v4l2_set


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


def create_setting_gui(win_name: str, dev_path: str, hsv_cfg: Dict[str, Any], init_ctrls: Dict[str, Any]) -> None:
    """Create OpenCV trackbars to update v4l2 controls and HSV thresholds."""

    ctrl_info = v4l2_list_ctrls(dev_path)

    def ctrl_spec(name: str) -> tuple[int, int, int]:
        # 実機値の範囲を優先し、取得できない場合は既知のフォールバックを使う。
        if name in ctrl_info and ctrl_info[name].get("min") is not None and ctrl_info[name].get("max") is not None:
            mn = int(ctrl_info[name]["min"])
            mx = int(ctrl_info[name]["max"])
            if mn > mx:
                mn, mx = mx, mn
            step_raw = int(ctrl_info[name].get("step", 1))
            step = max(1, abs(step_raw))
            return mn, mx, step
        mn, mx = FALLBACK_CTRL_RANGES.get(name, (0, 255))
        if mn > mx:
            mn, mx = mx, mn
        return mn, mx, 1

    def clamp(name: str, v: int) -> int:
        mn, mx, step = ctrl_spec(name)
        clamped = max(mn, min(mx, int(v)))
        if step <= 1:
            return clamped
        aligned = mn + int(round((clamped - mn) / float(step))) * step
        return max(mn, min(mx, aligned))

    def apply_ctrl(name: str, value: int) -> None:
        value = clamp(name, value)
        init_ctrls[name] = value

        if name == "auto_exposure":
            v4l2_set(dev_path, "auto_exposure", value)
            time.sleep(0.02)
            if value == 1 and "exposure_time_absolute" in init_ctrls:
                v4l2_set(dev_path, "exposure_time_absolute", int(init_ctrls["exposure_time_absolute"]))
            return

        if name == "exposure_time_absolute":
            if "auto_exposure" in init_ctrls:
                v4l2_set(dev_path, "auto_exposure", int(init_ctrls["auto_exposure"]))
                time.sleep(0.02)
            v4l2_set(dev_path, "exposure_time_absolute", value)
            return

        if name == "white_balance_automatic":
            v4l2_set(dev_path, "white_balance_automatic", value)
            time.sleep(0.02)
            if value == 0 and "white_balance_temperature" in init_ctrls:
                v4l2_set(dev_path, "white_balance_temperature", int(init_ctrls["white_balance_temperature"]))
            return

        if name == "white_balance_temperature":
            if "white_balance_automatic" in init_ctrls:
                v4l2_set(dev_path, "white_balance_automatic", int(init_ctrls["white_balance_automatic"]))
                time.sleep(0.02)
            v4l2_set(dev_path, "white_balance_temperature", value)
            return

        v4l2_set(dev_path, name, value)

    def make_ctrl_trackbar(name: str) -> None:
        mn, mx, step = ctrl_spec(name)
        init = clamp(name, int(init_ctrls.get(name, mn)))
        tb_max = max(1, int((mx - mn) // step))
        init_pos = int((init - mn) // step)

        def on_change(pos: int) -> None:
            apply_ctrl(name, mn + int(pos) * step)

        cv2.createTrackbar(name, win_name, int(init_pos), int(tb_max), on_change)
        on_change(int(init_pos))

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

    cv2.createTrackbar("B_H_low", win_name, int(hsv_cfg["blue"]["H_low"]), 179, bset("H_low"))
    cv2.createTrackbar("B_H_high", win_name, int(hsv_cfg["blue"]["H_high"]), 179, bset("H_high"))
    cv2.createTrackbar("B_S_low", win_name, int(hsv_cfg["blue"]["S_low"]), 255, bset("S_low"))
    cv2.createTrackbar("B_S_high", win_name, int(hsv_cfg["blue"]["S_high"]), 255, bset("S_high"))
    cv2.createTrackbar("B_V_low", win_name, int(hsv_cfg["blue"]["V_low"]), 255, bset("V_low"))
    cv2.createTrackbar("B_V_high", win_name, int(hsv_cfg["blue"]["V_high"]), 255, bset("V_high"))

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

    cv2.createTrackbar("R1_H_low", win_name, int(hsv_cfg["red1"]["H_low"]), 179, r1set("H_low"))
    cv2.createTrackbar("R1_H_high", win_name, int(hsv_cfg["red1"]["H_high"]), 179, r1set("H_high"))
    cv2.createTrackbar("R2_H_low", win_name, int(hsv_cfg["red2"]["H_low"]), 179, r2set("H_low"))
    cv2.createTrackbar("R2_H_high", win_name, int(hsv_cfg["red2"]["H_high"]), 179, r2set("H_high"))

    cv2.createTrackbar("R_S_low", win_name, int(hsv_cfg["redSV"]["S_low"]), 255, rsvset("S_low"))
    cv2.createTrackbar("R_S_high", win_name, int(hsv_cfg["redSV"]["S_high"]), 255, rsvset("S_high"))
    cv2.createTrackbar("R_V_low", win_name, int(hsv_cfg["redSV"]["V_low"]), 255, rsvset("V_low"))
    cv2.createTrackbar("R_V_high", win_name, int(hsv_cfg["redSV"]["V_high"]), 255, rsvset("V_high"))
