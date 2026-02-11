from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from .detection.hsv import find_boxes, get_led_mask
from .detection.pairing import build_pair_meta, pair_boxes_same_color
from .detection.types import PairMeta, xyxy_center
from .tracking.base import Detection, MultiObjectTracker, Track
from .tracking.distance_tracker import DistanceConfig, DistanceTracker
from .tracking.motpy_tracker import MotpyConfig, MotpyTracker
from .tracking.noop_tracker import NoopTracker


@dataclass(frozen=True)
class FrameResult:
    pairs: List[PairMeta]
    target: Tuple[int, int]
    selected_pair: PairMeta | None
    tracks: List[Track]
    selected_track: Track | None
    chosen_from_tracks: bool


def normalize_device_arg(dev: Any) -> Any:
    if isinstance(dev, str) and dev.isdigit():
        return int(dev)
    return dev


def pairs_from_frame(frame_bgr: np.ndarray, det_cfg: Dict[str, Any]) -> List[PairMeta]:
    hsv_cfg = det_cfg["hsv"]
    kernel_sz = int(det_cfg["kernel_sz"])
    min_box_w = int(det_cfg["min_box_w"])
    min_box_h = int(det_cfg["min_box_h"])
    width_tol = float(det_cfg["width_tol"])
    min_h_overlap = float(det_cfg["min_h_overlap"])
    min_v_gap = int(det_cfg["min_v_gap"])

    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    pairs: List[PairMeta] = []
    for color in ("blue", "red"):
        mask = get_led_mask(hsv, color, hsv_cfg)
        boxes = find_boxes(mask, kernel_sz=kernel_sz, min_box_w=min_box_w, min_box_h=min_box_h)
        for (top, bottom) in pair_boxes_same_color(boxes, width_tol, min_h_overlap, min_v_gap):
            pairs.append(build_pair_meta(color=color, top=top, bottom=bottom))
    return pairs


def detections_from_pairs(pairs: List[PairMeta], frame_shape: Tuple[int, int, int]) -> List[Detection]:
    h, w = frame_shape[0], frame_shape[1]
    detections: List[Detection] = []
    for p in pairs:
        _, _, uw, uh = p.union_xywh
        score = min(1.0, (uw * uh) / (w * h / 8.0) + 0.1)
        detections.append(Detection(box_xyxy=p.union_xyxy, score=float(score), class_id=0))
    return detections


def select_target_pair(pairs: List[PairMeta], frame_w: int, frame_h: int) -> Tuple[Tuple[int, int], PairMeta | None]:
    cx0 = frame_w / 2.0
    best_xy = (int(frame_w / 2), int(frame_h / 2))
    best_abs = float("inf")
    best_pair: PairMeta | None = None
    for p in pairs:
        cx, cy = xyxy_center(p.union_xyxy)
        dx = abs(cx - cx0)
        if dx < best_abs:
            best_abs = dx
            best_xy = (int(cx), int(cy))
            best_pair = p
    return best_xy, best_pair


def select_target_track(tracks: List[Track], frame_w: int, frame_h: int) -> Tuple[Tuple[int, int], Track | None]:
    cx0 = frame_w / 2.0
    best_xy = (int(frame_w / 2), int(frame_h / 2))
    best_abs = float("inf")
    best_track: Track | None = None
    for t in tracks:
        cx, cy = xyxy_center(t.box_xyxy)
        dx = abs(cx - cx0)
        if dx < best_abs:
            best_abs = dx
            best_xy = (int(cx), int(cy))
            best_track = t
    return best_xy, best_track


def build_tracker(track_cfg: Dict[str, Any], fps: float) -> MultiObjectTracker:
    backend = str(track_cfg.get("backend", "motpy")).lower()
    if backend == "noop":
        return NoopTracker()

    if backend == "motpy":
        mot = track_cfg.get("motpy", {})
        cfg = MotpyConfig(
            order_pos=int(mot.get("order_pos", 2)),
            dim_pos=int(mot.get("dim_pos", 2)),
            order_size=int(mot.get("order_size", 0)),
            dim_size=int(mot.get("dim_size", 2)),
            q_var_pos=float(mot.get("q_var_pos", 5000.0)),
            r_var_pos=float(mot.get("r_var_pos", 0.1)),
            min_iou=mot.get("min_iou", None),
            max_staleness=mot.get("max_staleness", None),
            min_steps_alive=int(track_cfg.get("min_steps_alive", 2)),
        )
        return MotpyTracker(cfg, dt=1.0 / float(fps))

    if backend == "distance":
        dist = track_cfg.get("distance", {})
        cfg = DistanceConfig(
            gate_px=float(dist.get("gate_px", 80.0)),
            normalize=str(dist.get("normalize", "diag")),
            size_weight=float(dist.get("size_weight", 0.15)),
            use_prediction=bool(dist.get("use_prediction", True)),
            vel_alpha=float(dist.get("vel_alpha", 0.6)),
            max_speed_px_s=float(dist.get("max_speed_px_s", 8000.0)),
            max_age=int(
                dist.get(
                    "max_age",
                    track_cfg.get("max_staleness", 4) if isinstance(track_cfg.get("max_staleness", 4), int) else 4,
                )
            ),
            min_steps_alive=int(track_cfg.get("min_steps_alive", 2)),
        )
        return DistanceTracker(cfg)

    raise ValueError(f"Unknown tracking backend: {backend}")


def process_frame(
    frame_bgr: np.ndarray,
    det_cfg: Dict[str, Any],
    tracking_cfg: Dict[str, Any],
    tracker: MultiObjectTracker | None,
    dt: float,
) -> FrameResult:
    pairs = pairs_from_frame(frame_bgr, det_cfg)
    frame_h, frame_w = frame_bgr.shape[0], frame_bgr.shape[1]

    target, selected_pair = select_target_pair(pairs, frame_w=frame_w, frame_h=frame_h)
    selected_track: Track | None = None
    tracks: List[Track] = []
    chosen_from_tracks = False

    if tracking_cfg["enabled"] and tracker is not None:
        detections = detections_from_pairs(pairs, frame_bgr.shape)
        tracks = tracker.step(detections, dt=dt)
        if len(tracks) > 0:
            target, selected_track = select_target_track(tracks, frame_w=frame_w, frame_h=frame_h)
            chosen_from_tracks = True

    return FrameResult(
        pairs=pairs,
        target=target,
        selected_pair=selected_pair,
        tracks=tracks,
        selected_track=selected_track,
        chosen_from_tracks=chosen_from_tracks,
    )
