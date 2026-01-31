from __future__ import annotations

from typing import Any, Dict, List, Tuple
import numpy as np
import cv2


def get_led_mask(hsv: np.ndarray, color: str, hsv_cfg: Dict[str, Any]) -> np.ndarray:
    if color == "blue":
        b = hsv_cfg["blue"]
        lo = np.array([b["H_low"], b["S_low"], b["V_low"]], dtype=np.uint8)
        hi = np.array([b["H_high"], b["S_high"], b["V_high"]], dtype=np.uint8)
        return cv2.inRange(hsv, lo, hi)

    if color == "red":
        r1 = hsv_cfg["red1"]; r2 = hsv_cfg["red2"]; rsv = hsv_cfg["redSV"]
        m1 = cv2.inRange(
            hsv,
            np.array([r1["H_low"], rsv["S_low"], rsv["V_low"]], dtype=np.uint8),
            np.array([r1["H_high"], rsv["S_high"], rsv["V_high"]], dtype=np.uint8),
        )
        m2 = cv2.inRange(
            hsv,
            np.array([r2["H_low"], rsv["S_low"], rsv["V_low"]], dtype=np.uint8),
            np.array([r2["H_high"], rsv["S_high"], rsv["V_high"]], dtype=np.uint8),
        )
        return cv2.bitwise_or(m1, m2)

    raise ValueError(f"Unknown color: {color}")


def find_boxes(mask: np.ndarray, kernel_sz: int, min_box_w: int, min_box_h: int) -> List[Tuple[int, int, int, int]]:
    kernel = np.ones((kernel_sz, kernel_sz), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes: List[Tuple[int, int, int, int]] = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w < min_box_w or h < min_box_h:
            continue
        boxes.append((int(x), int(y), int(w), int(h)))
    return boxes
