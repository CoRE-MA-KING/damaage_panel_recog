from __future__ import annotations

import argparse
import time
from typing import Any, Dict

import cv2

from .camera.capture import setup_camera
from .config import build_effective_config, load_config
from .defaults import DEFAULTS
from .pipeline import build_tracker, normalize_device_arg, process_frame
from .runtime import (
    close_motion_logger,
    close_publisher,
    create_motion_logger,
    create_publisher,
    log_motion_sample,
    publish_target,
    render_frame,
)
from .ui.draw import TrackVizState
from .ui.gui import create_setting_gui


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--publish", action="store_true", help="Zenohにpublishする")
    ap.add_argument("-n", "--no-display", action="store_true", help="映像表示を行わない（GUIも無効）")
    ap.add_argument("-s", "--setting", action="store_true", help="トラックバーでカメラ/HSVを調整する")
    ap.add_argument("-t", "--track", action="store_true", help="多目標トラッキングを有効化（backendはconfigで選択）")
    ap.add_argument("-d", "--device", default=None, help="キャプチャデバイスのパスまたは番号（例: 0, /dev/video0）")
    ap.add_argument("-l", "--log", action="store_true", help="(計測用) ターゲット座標の時系列をCSVへ記録する")
    ap.add_argument("--log-path", default=None, help="(計測用) ログCSVの出力先（省略時は logs/motion_log_YYYYmmdd_HHMMSS.csv）")
    ap.add_argument("--log-flush-every", default=None, type=int, help="(計測用) 何フレームごとにflushするか（デフォルト60）")
    ap.add_argument("--config", default="config/default.yaml", help="設定ファイル（YAML/JSON）")
    return ap.parse_args()


def _apply_cli_overrides(cfg: Dict[str, Any], args: argparse.Namespace) -> None:
    if args.device is not None:
        cfg["camera"]["device"] = args.device

    cfg["publish"]["enabled"] = bool(args.publish or cfg["publish"].get("enabled", False))
    cfg["tracking"]["enabled"] = bool(args.track or cfg["tracking"].get("enabled", False))
    cfg["logging"]["enabled"] = bool(args.log or cfg.get("logging", {}).get("enabled", False))

    if args.log_path is not None:
        cfg["logging"]["path"] = str(args.log_path)
    if args.log_flush_every is not None:
        cfg["logging"]["flush_every"] = int(args.log_flush_every)


def main() -> int:
    args = parse_args()

    override = load_config(args.config) if args.config else {}
    cfg = build_effective_config(DEFAULTS, override)

    _apply_cli_overrides(cfg, args)

    do_display = not bool(args.no_display)
    use_gui = bool(args.setting) and do_display

    device = normalize_device_arg(cfg["camera"]["device"])
    cap, dev_path = setup_camera(device, cfg["camera"]["capture"], cfg["camera"]["init_controls"])

    win_name = str(cfg["ui"]["window_name"])
    if do_display or use_gui:
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    if use_gui:
        create_setting_gui(win_name, dev_path, cfg["detection"]["hsv"], cfg["camera"]["init_controls"])

    publisher = create_publisher(cfg["publish"])
    motion_logger = create_motion_logger(cfg.get("logging", {}))

    tracker = None
    viz = TrackVizState(history_len=int(cfg["tracking"]["history_len"]))
    if cfg["tracking"]["enabled"]:
        try:
            tracker = build_tracker(cfg["tracking"], fps=float(cfg["camera"]["capture"]["fps"]))
        except Exception as e:
            print(f"[WARN] tracking backend init failed: {e}")
            tracker = None

    last_t = time.time()
    fps = 0.0
    alpha = float(cfg["ui"].get("fps_ema_alpha", 0.2))

    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            frame_idx += 1

            now = time.time()
            dt = max(1e-6, now - last_t)
            last_t = now
            inst_fps = 1.0 / dt
            fps = alpha * inst_fps + (1.0 - alpha) * fps

            frame_result = process_frame(
                frame_bgr=frame,
                det_cfg=cfg["detection"],
                tracking_cfg=cfg["tracking"],
                tracker=tracker,
                dt=dt,
            )

            log_motion_sample(
                motion_logger=motion_logger,
                frame_idx=frame_idx,
                now=now,
                dt=dt,
                result=frame_result,
            )

            if do_display:
                should_quit = render_frame(
                    frame,
                    result=frame_result,
                    tracking_enabled=bool(cfg["tracking"]["enabled"]),
                    tracker_available=tracker is not None,
                    track_color_bgr=tuple(int(x) for x in cfg["tracking"]["color_bgr"]),
                    history=viz,
                    fps=fps,
                    win_name=win_name,
                )
                if should_quit:
                    break
            else:
                time.sleep(0.01)

            publish_target(publisher, frame_result.target)

    finally:
        close_motion_logger(motion_logger)
        close_publisher(publisher)
        cap.release()
        cv2.destroyAllWindows()

    return 0
