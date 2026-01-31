from __future__ import annotations

from typing import Any, Callable, Dict

import cv2

from ..camera.v4l2ctl import v4l2_set


def create_setting_gui(win_name: str, dev_path: str, hsv_cfg: Dict[str, Any], init_ctrls: Dict[str, Any]) -> None:
    """Create OpenCV trackbars to update v4l2 controls and HSV thresholds."""

    def tb_set_v4l2(ctrl_name: str, minv: int = 0) -> Callable[[int], None]:
        def _f(v: int) -> None:
            val = int(v) + int(minv)
            v4l2_set(dev_path, ctrl_name, val)
            init_ctrls[ctrl_name] = val
        return _f

    # Camera controls (only if present in config)
    if "exposure_time_absolute" in init_ctrls:
        cv2.createTrackbar("ExposureAbs", win_name, int(init_ctrls["exposure_time_absolute"]), 10000, tb_set_v4l2("exposure_time_absolute", 0))
    if "gain" in init_ctrls:
        cv2.createTrackbar("Gain", win_name, int(init_ctrls["gain"]), 1023, tb_set_v4l2("gain", 0))
    if "white_balance_temperature" in init_ctrls:
        cv2.createTrackbar("WhiteBalance", win_name, max(0, int(init_ctrls["white_balance_temperature"]) - 2800), 6500 - 2800, tb_set_v4l2("white_balance_temperature", 2800))
    if "brightness" in init_ctrls:
        cv2.createTrackbar("Brightness", win_name, int(init_ctrls["brightness"]) + 64, 128, tb_set_v4l2("brightness", -64))
    if "contrast" in init_ctrls:
        cv2.createTrackbar("Contrast", win_name, int(init_ctrls["contrast"]), 95, tb_set_v4l2("contrast", 0))
    if "sharpness" in init_ctrls:
        cv2.createTrackbar("Sharpness", win_name, int(init_ctrls["sharpness"]), 7, tb_set_v4l2("sharpness", 0))
    if "saturation" in init_ctrls:
        cv2.createTrackbar("Saturation", win_name, int(init_ctrls["saturation"]), 255, tb_set_v4l2("saturation", 0))
    if "hue" in init_ctrls:
        cv2.createTrackbar("Hue", win_name, int(init_ctrls["hue"]) + 2000, 4000, tb_set_v4l2("hue", -2000))
    if "gamma" in init_ctrls:
        cv2.createTrackbar("Gamma", win_name, max(0, int(init_ctrls["gamma"]) - 64), 300 - 64, tb_set_v4l2("gamma", 64))

    # HSV blue
    def bset(name: str):
        return lambda v: hsv_cfg["blue"].__setitem__(name, int(v))

    cv2.createTrackbar("B_H_low", win_name, int(hsv_cfg["blue"]["H_low"]), 179, bset("H_low"))
    cv2.createTrackbar("B_H_high", win_name, int(hsv_cfg["blue"]["H_high"]), 179, bset("H_high"))
    cv2.createTrackbar("B_S_low", win_name, int(hsv_cfg["blue"]["S_low"]), 255, bset("S_low"))
    cv2.createTrackbar("B_S_high", win_name, int(hsv_cfg["blue"]["S_high"]), 255, bset("S_high"))
    cv2.createTrackbar("B_V_low", win_name, int(hsv_cfg["blue"]["V_low"]), 255, bset("V_low"))
    cv2.createTrackbar("B_V_high", win_name, int(hsv_cfg["blue"]["V_high"]), 255, bset("V_high"))

    # HSV red
    def r1set(name: str):
        return lambda v: hsv_cfg["red1"].__setitem__(name, int(v))

    def r2set(name: str):
        return lambda v: hsv_cfg["red2"].__setitem__(name, int(v))

    def rsvset(name: str):
        return lambda v: hsv_cfg["redSV"].__setitem__(name, int(v))

    cv2.createTrackbar("R1_H_low", win_name, int(hsv_cfg["red1"]["H_low"]), 179, r1set("H_low"))
    cv2.createTrackbar("R1_H_high", win_name, int(hsv_cfg["red1"]["H_high"]), 179, r1set("H_high"))
    cv2.createTrackbar("R2_H_low", win_name, int(hsv_cfg["red2"]["H_low"]), 179, r2set("H_low"))
    cv2.createTrackbar("R2_H_high", win_name, int(hsv_cfg["red2"]["H_high"]), 179, r2set("H_high"))

    cv2.createTrackbar("R_S_low", win_name, int(hsv_cfg["redSV"]["S_low"]), 255, rsvset("S_low"))
    cv2.createTrackbar("R_S_high", win_name, int(hsv_cfg["redSV"]["S_high"]), 255, rsvset("S_high"))
    cv2.createTrackbar("R_V_low", win_name, int(hsv_cfg["redSV"]["V_low"]), 255, rsvset("V_low"))
    cv2.createTrackbar("R_V_high", win_name, int(hsv_cfg["redSV"]["V_high"]), 255, rsvset("V_high"))
