from __future__ import annotations

import argparse
import time
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from .config import load_config, build_effective_config
from .camera.capture import setup_camera
from .detection.hsv import get_led_mask, find_boxes
from .detection.pairing import pair_boxes_same_color, build_pair_meta
from .detection.types import PairMeta, xyxy_center
from .tracking.base import Detection, Track
from .tracking.noop_tracker import NoopTracker
from .tracking.motpy_tracker import MotpyTracker, MotpyConfig
from .ui.draw import TrackVizState, draw_detection_pair, draw_tracks, draw_target, draw_fps, draw_mode
from .ui.gui import create_setting_gui
from .publish.zenoh_pub import ZenohPublisher, ZenohConfig
from .domain.message import DamagePanelRecognition


DEFAULTS: Dict[str, Any] = {
    "camera": {
        "device": "/dev/video0",
        "capture": {"width": 800, "height": 600, "fps": 120, "fourcc": "MJPG"},
        "init_controls": {
            "brightness": 0,
            "contrast": 32,
            "saturation": 90,
            "hue": 0,
            "white_balance_automatic": 0,
            "gamma": 100,
            "gain": 0,
            "power_line_frequency": 1,
            "white_balance_temperature": 4600,
            "sharpness": 3,
            "backlight_compensation": 0,
            "auto_exposure": 1,
            "exposure_time_absolute": 4,
        },
    },
    "detection": {
        "kernel_sz": 3,
        "width_tol": 0.6,
        "min_h_overlap": 0.05,
        "min_v_gap": 1,
        "min_box_h": 1,
        "min_box_w": 4,
        "hsv": {
            "blue":  {"H_low": 100, "H_high": 135, "S_low": 180, "S_high": 255, "V_low": 120, "V_high": 255},
            "red1":  {"H_low": 0, "H_high": 15},
            "red2":  {"H_low": 165, "H_high": 179},
            "redSV": {"S_low": 180, "S_high": 255, "V_low": 120, "V_high": 255},
        },
    },
    "tracking": {
        "enabled": False,
        "backend": "motpy",  # motpy | noop (future: sort)
        "min_steps_alive": 2,
        "history_len": 20,
        "color_bgr": [0, 255, 255],
        "motpy": {
            "order_pos": 2,
            "dim_pos": 2,
            "order_size": 0,
            "dim_size": 2,
            "q_var_pos": 5000.0,
            "r_var_pos": 0.1,
            "min_iou": None,
            "max_staleness": 4,
        },
    },
    "publish": {
        "enabled": False,
        "key_prefix": "",
        "publish_key": "damagepanel",
    },
    "ui": {
        "window_name": "Panel (paired by same-color top & bottom)",
        "fps_ema_alpha": 0.2,
    },
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--publish", action="store_true", help="Zenohにpublishする")
    ap.add_argument("-n", "--no-display", action="store_true", help="映像表示を行わない（GUIも無効）")
    ap.add_argument("-s", "--setting", action="store_true", help="トラックバーでカメラ/HSVを調整する")
    ap.add_argument("-t", "--track", action="store_true", help="多目標トラッキングを有効化（backendはconfigで選択）")
    ap.add_argument("-d", "--device", default=None, help="キャプチャデバイスのパスまたは番号（例: 0, /dev/video0）")
    ap.add_argument("--config", default="config/default.yaml", help="設定ファイル（YAML/JSON）")
    return ap.parse_args()


def _normalize_device_arg(dev: Any) -> Any:
    if isinstance(dev, str) and dev.isdigit():
        return int(dev)
    return dev


def _pairs_from_frame(frame_bgr: np.ndarray, det_cfg: Dict[str, Any]) -> List[PairMeta]:
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


def _detections_from_pairs(pairs: List[PairMeta], frame_shape: Tuple[int, int, int]) -> List[Detection]:
    H, W = frame_shape[0], frame_shape[1]
    dets: List[Detection] = []
    for p in pairs:
        ux, uy, uw, uh = p.union_xywh
        score = min(1.0, (uw * uh) / (W * H / 8.0) + 0.1)
        dets.append(Detection(box_xyxy=p.union_xyxy, score=float(score), class_id=0))
    return dets


def _select_target_from_pairs(pairs: List[PairMeta], frame_w: int, frame_h: int) -> Tuple[int, int]:
    cx0 = frame_w / 2.0
    best = (int(frame_w / 2), int(frame_h / 2))
    best_abs = float("inf")
    for p in pairs:
        cx, cy = xyxy_center(p.union_xyxy)
        dx = abs(cx - cx0)
        if dx < best_abs:
            best_abs = dx
            best = (int(cx), int(cy))
    return best


def _select_target_from_tracks(tracks: List[Track], frame_w: int, frame_h: int) -> Tuple[int, int]:
    cx0 = frame_w / 2.0
    best = (int(frame_w / 2), int(frame_h / 2))
    best_abs = float("inf")
    for t in tracks:
        cx, cy = xyxy_center(t.box_xyxy)
        dx = abs(cx - cx0)
        if dx < best_abs:
            best_abs = dx
            best = (int(cx), int(cy))
    return best


def _build_tracker(track_cfg: Dict[str, Any], fps: float) -> Any:
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

    raise ValueError(f"Unknown tracking backend: {backend}")


def main() -> int:
    args = parse_args()

    override = load_config(args.config) if args.config else {}
    cfg = build_effective_config(DEFAULTS, override)

    # CLI overrides
    if args.device is not None:
        cfg["camera"]["device"] = args.device
    cfg["publish"]["enabled"] = bool(args.publish or cfg["publish"].get("enabled", False))
    cfg["tracking"]["enabled"] = bool(args.track or cfg["tracking"].get("enabled", False))

    do_display = not bool(args.no_display)
    use_gui = bool(args.setting) and do_display

    device = _normalize_device_arg(cfg["camera"]["device"])
    cap, dev_path = setup_camera(device, cfg["camera"]["capture"], cfg["camera"]["init_controls"])

    win_name = str(cfg["ui"]["window_name"])
    if do_display or use_gui:
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    if use_gui:
        create_setting_gui(win_name, dev_path, cfg["detection"]["hsv"], cfg["camera"]["init_controls"])

    publisher = None
    if cfg["publish"]["enabled"]:
        publisher = ZenohPublisher(ZenohConfig(key_prefix=str(cfg["publish"]["key_prefix"]), publish_key=str(cfg["publish"]["publish_key"])))

    tracker = None
    viz = TrackVizState(history_len=int(cfg["tracking"]["history_len"]))
    if cfg["tracking"]["enabled"]:
        try:
            tracker = _build_tracker(cfg["tracking"], fps=float(cfg["camera"]["capture"]["fps"]))
        except Exception as e:
            print(f"[WARN] tracking backend init failed: {e}")
            tracker = None

    last_t = time.time()
    fps = 0.0
    alpha = float(cfg["ui"].get("fps_ema_alpha", 0.2))

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            now = time.time()
            dt = max(1e-6, now - last_t)
            last_t = now
            inst_fps = 1.0 / dt
            fps = alpha * inst_fps + (1.0 - alpha) * fps

            pairs = _pairs_from_frame(frame, cfg["detection"])
            H, W = frame.shape[0], frame.shape[1]

            target = _select_target_from_pairs(pairs, frame_w=W, frame_h=H)
            chosen_from_tracks = False
            tracks: List[Track] = []

            if cfg["tracking"]["enabled"] and tracker is not None:
                dets = _detections_from_pairs(pairs, frame.shape)
                tracks = tracker.step(dets, dt=dt)
                if len(tracks) > 0:
                    target = _select_target_from_tracks(tracks, frame_w=W, frame_h=H)
                    chosen_from_tracks = True

            if do_display:
                if not cfg["tracking"]["enabled"] or tracker is None:
                    for p in pairs:
                        draw_detection_pair(frame, p)
                else:
                    track_color = tuple(int(x) for x in cfg["tracking"]["color_bgr"])
                    draw_tracks(frame, tracks, pairs, track_color=track_color, history=viz)

                draw_target(frame, target, from_tracks=chosen_from_tracks)
                mode = "TRACK=ON" if (cfg["tracking"]["enabled"] and tracker is not None) else ("TRACK=ON (backend missing!)" if cfg["tracking"]["enabled"] else "TRACK=OFF")
                draw_mode(frame, mode)
                draw_fps(frame, fps)

                cv2.imshow(win_name, frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                time.sleep(0.01)

            if publisher is not None:
                tx, ty = target
                publisher.put(
                    DamagePanelRecognition(
                        target_x=int(tx),
                        target_y=int(ty),
                        target_distance=0,
                    ).model_dump_json()
                )

    finally:
        try:
            if publisher is not None:
                publisher.close()
        except Exception:
            pass
        cap.release()
        cv2.destroyAllWindows()

    return 0
