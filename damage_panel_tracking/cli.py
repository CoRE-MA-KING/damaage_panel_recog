from __future__ import annotations

import argparse
import math
import threading
import time
from typing import Any, Dict

import cv2

from msg import (
    DamagePanelColorMessage,
)

from damage_panel_tracking.publish.zenoh_pub import LatestFramePublisher, ZenohSession

from .camera.capture import setup_camera
from .config import build_effective_config, load_config
from .defaults import DEFAULTS
from .detection.types import ColorName
from .pipeline import build_tracker, normalize_device_arg, process_frame
from .runtime import (
    close_motion_logger,
    close_publisher,
    create_motion_logger,
    log_motion_sample,
    render_frame,
    result_to_publish_payload,
)
from .transform import (
    ProjectionModel,
    build_projection_model,
    draw_projected_overlay,
    parse_size,
    project_pair_to_main_camera,
    projected_to_publish_payload,
)
from .ui.draw import TrackVizState
from .ui.gui import create_setting_gui


VALID_TARGET_COLORS: tuple[ColorName, ColorName] = ("blue", "red")


class TargetColorState:
    def __init__(self, initial: ColorName) -> None:
        self._color = initial
        self._lock = threading.Lock()

    def get(self) -> ColorName:
        with self._lock:
            return self._color

    def set(self, next_color: ColorName) -> bool:
        with self._lock:
            if self._color == next_color:
                return False
            self._color = next_color
            return True


def _normalize_target_color(value: Any, *, source: str) -> ColorName:
    color = str(value).strip().lower()
    if color not in VALID_TARGET_COLORS:
        raise ValueError(f"{source} must be one of {VALID_TARGET_COLORS}: got {value!r}")
    if color == "blue":
        return "blue"
    return "red"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--publish", action="store_true", help="Zenohにpublishする")
    ap.add_argument("--publish-max-hz", default=None, type=float, help="publishの最大レート(Hz)。0以下で無制限")
    ap.add_argument("--subscribe", action="store_true", help="Zenohからtarget colorをsubscribeする")
    ap.add_argument("--default-target", default=None, help="subscribe無効時/受信前に使う色（blue|red）")
    ap.add_argument("-n", "--no-display", action="store_true", help="映像表示を行わない（GUIも無効）")
    ap.add_argument("-s", "--setting", action="store_true", help="トラックバーでカメラ/HSVを調整する")
    ap.add_argument("-t", "--track", action="store_true", help="多目標トラッキングを有効化（backendはconfigで選択）")
    ap.add_argument("-d", "--device", default=None, help="キャプチャデバイスのパスまたは番号（例: 0, /dev/video0）")
    ap.add_argument("-l", "--log", action="store_true", help="(計測用) ターゲット座標の時系列をCSVへ記録する")
    ap.add_argument("--log-path", default=None, help="(計測用) ログCSVの出力先（省略時は logs/motion_log_YYYYmmdd_HHMMSS.csv）")
    ap.add_argument("--log-flush-every", default=None, type=int, help="(計測用) 何フレームごとにflushするか（デフォルト60）")
    ap.add_argument("--coord-transform", action="store_true", help="panel_recog_camera -> main_camera 座標変換を有効化")
    ap.add_argument("--main-overlay", action="store_true", help="main_camera重畳表示デバッグを有効化")
    ap.add_argument("--main-camera-device", default=None, help="デバッグ表示用main_cameraデバイス（例: /dev/video0）")
    ap.add_argument("--config", default="config/default.yaml", help="設定ファイル（YAML/JSON）")
    return ap.parse_args()


def _apply_cli_overrides(cfg: Dict[str, Any], args: argparse.Namespace) -> None:
    if args.device is not None:
        cfg["camera"]["device"] = args.device

    cfg["publish"]["enabled"] = bool(args.publish or cfg["publish"].get("enabled", False))
    cfg["subscribe"]["enabled"] = bool(args.subscribe or cfg["subscribe"].get("enabled", False))
    cfg["tracking"]["enabled"] = bool(args.track or cfg["tracking"].get("enabled", False))
    cfg["logging"]["enabled"] = bool(args.log or cfg.get("logging", {}).get("enabled", False))

    if args.log_path is not None:
        cfg["logging"]["path"] = str(args.log_path)
    if args.log_flush_every is not None:
        cfg["logging"]["flush_every"] = int(args.log_flush_every)

    if args.default_target is not None:
        cfg["subscribe"]["default_target"] = str(args.default_target)
    if args.publish_max_hz is not None:
        cfg["publish"]["max_hz"] = float(args.publish_max_hz)

    if args.coord_transform:
        cfg["coordinate_transform"]["enabled"] = True
    if args.main_overlay:
        cfg["coordinate_transform"]["enabled"] = True
        cfg["coordinate_transform"]["debug_overlay"]["enabled"] = True
    if args.main_camera_device is not None:
        cfg["coordinate_transform"]["debug_overlay"]["camera"]["device"] = args.main_camera_device

    cfg["publish"]["publish_key"] = str(cfg["publish"].get("publish_key", "damagepanel/target"))
    cfg["publish"]["max_hz"] = float(cfg["publish"].get("max_hz", 0.0))
    if not math.isfinite(cfg["publish"]["max_hz"]):
        raise ValueError(f"publish.max_hz must be a finite number: got {cfg['publish']['max_hz']!r}")
    cfg["publish"]["drop_if_congested"] = bool(cfg["publish"].get("drop_if_congested", True))
    cfg["publish"]["express"] = bool(cfg["publish"].get("express", True))
    cfg["subscribe"]["subscribe_key"] = str(cfg["subscribe"].get("subscribe_key", "damagepanel/color"))
    cfg["subscribe"]["default_target"] = _normalize_target_color(
        cfg["subscribe"].get("default_target", "blue"),
        source="subscribe.default_target",
    )


def _parse_optional_size(value: Any, *, field_name: str) -> tuple[int, int] | None:
    try:
        return parse_size(value)
    except ValueError as e:
        raise ValueError(f"{field_name}: {e}") from e


def _read_first_frame(cap: cv2.VideoCapture, *, camera_label: str) -> Any:
    attempts = 0
    while True:
        ok, frame = cap.read()
        if ok:
            return frame
        attempts += 1
        if attempts % 100 == 0:
            print(f"[WARN] waiting for first frame from {camera_label}...")
        time.sleep(0.01)


def main() -> int:
    args = parse_args()

    override = load_config(args.config) if args.config else {}
    cfg = build_effective_config(DEFAULTS, override)
    try:
        _apply_cli_overrides(cfg, args)
    except ValueError as e:
        print(f"[ERROR] {e}")
        return 2

    do_display = not bool(args.no_display)
    use_gui = bool(args.setting) and do_display

    transform_cfg = cfg.get("coordinate_transform", {})
    transform_enabled = bool(transform_cfg.get("enabled", False))
    debug_overlay_cfg = transform_cfg.get("debug_overlay", {})
    debug_overlay_enabled = bool(debug_overlay_cfg.get("enabled", False))
    if debug_overlay_enabled and not transform_enabled:
        transform_enabled = True
    if debug_overlay_enabled and not do_display:
        print("[WARN] coordinate_transform.debug_overlay is enabled but --no-display was set. debug overlay is disabled.")
        debug_overlay_enabled = False

    device = normalize_device_arg(cfg["camera"]["device"])
    cap, dev_path = setup_camera(device, cfg["camera"]["capture"], cfg["camera"]["init_controls"])

    win_name = str(cfg["ui"]["window_name"])
    if do_display or use_gui:
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    if use_gui:
        create_setting_gui(win_name, dev_path, cfg["detection"]["hsv"], cfg["camera"]["init_controls"])

    publish_enabled = bool(cfg["publish"]["enabled"])
    subscribe_enabled = bool(cfg["subscribe"]["enabled"])
    publish_key = str(cfg["publish"]["publish_key"])
    publish_max_hz = float(cfg["publish"]["max_hz"])
    publish_drop_if_congested = bool(cfg["publish"]["drop_if_congested"])
    publish_express = bool(cfg["publish"]["express"])
    subscribe_key = str(cfg["subscribe"]["subscribe_key"])
    target_color_state = TargetColorState(cfg["subscribe"]["default_target"])
    if not subscribe_enabled:
        print(f"[INFO] subscribe disabled: target color={target_color_state.get()}")

    session: ZenohSession | None = None
    latest_publisher: LatestFramePublisher | None = None
    motion_logger = None
    tracker = None
    viz = TrackVizState(history_len=int(cfg["tracking"]["history_len"]))
    main_cap: cv2.VideoCapture | None = None
    main_window_name = str(debug_overlay_cfg.get("window_name", "Main Camera (projected target)"))
    publish_projection_model: ProjectionModel | None = None
    debug_projection_model: ProjectionModel | None = None

    next_panel_frame = _read_first_frame(cap, camera_label="panel_recog_camera")
    panel_frame_size = (int(next_panel_frame.shape[1]), int(next_panel_frame.shape[0]))
    next_main_frame = None

    try:
        if publish_enabled:
            latest_publisher = LatestFramePublisher(
                key=publish_key,
                drop_if_congested=publish_drop_if_congested,
                express=publish_express,
                max_hz=publish_max_hz,
            )
            if publish_max_hz > 0.0:
                print(f"[INFO] publish async(process) enabled: key={publish_key} max_hz={publish_max_hz:.1f}")
            else:
                print(f"[INFO] publish async(process) enabled: key={publish_key} max_hz=unlimited")

        if subscribe_enabled:
            session = ZenohSession()

            def _on_color_message(data: Any) -> None:
                try:
                    msg = DamagePanelColorMessage.FromString(data.payload.to_bytes())
                    next_color = _normalize_target_color(
                        msg.color,
                        source="DamagePanelColorMessage.color",
                    )
                except Exception as e:
                    print(f"[WARN] failed to parse color message: {e}")
                    return

                if target_color_state.set(next_color):
                    print(f"[INFO] target color updated by subscribe: {next_color}")

            session.create_subscriber(subscribe_key, _on_color_message)

        motion_logger = create_motion_logger(cfg.get("logging", {}))

        if cfg["tracking"]["enabled"]:
            try:
                tracker = build_tracker(cfg["tracking"], fps=float(cfg["camera"]["capture"]["fps"]))
            except Exception as e:
                print(f"[WARN] tracking backend init failed: {e}")
                tracker = None

        if transform_enabled:
            try:
                panel_cfg = transform_cfg.get("panel_recog_camera", {})
                publish_main_cfg = transform_cfg.get("publish_main_camera", {})

                panel_intrinsics_path = str(panel_cfg.get("intrinsics_path", "")).strip()
                if not panel_intrinsics_path:
                    raise ValueError("coordinate_transform.panel_recog_camera.intrinsics_path is required")
                panel_calib_size = _parse_optional_size(
                    panel_cfg.get("calib_size"),
                    field_name="coordinate_transform.panel_recog_camera.calib_size",
                )

                publish_main_intrinsics_path = str(publish_main_cfg.get("intrinsics_path", "")).strip()
                if not publish_main_intrinsics_path:
                    raise ValueError("coordinate_transform.publish_main_camera.intrinsics_path is required")
                publish_main_extrinsics_path = str(publish_main_cfg.get("extrinsics_from_panel_recog_path", "")).strip()
                if not publish_main_extrinsics_path:
                    raise ValueError(
                        "coordinate_transform.publish_main_camera.extrinsics_from_panel_recog_path is required"
                    )
                publish_main_calib_size = _parse_optional_size(
                    publish_main_cfg.get("calib_size"),
                    field_name="coordinate_transform.publish_main_camera.calib_size",
                )
                publish_main_frame_size = _parse_optional_size(
                    publish_main_cfg.get("frame_size"),
                    field_name="coordinate_transform.publish_main_camera.frame_size",
                ) or publish_main_calib_size
                if publish_main_frame_size is None:
                    raise ValueError(
                        "coordinate_transform.publish_main_camera.frame_size or calib_size is required"
                    )

                panel_vertical_span_m = float(transform_cfg.get("panel_vertical_span_m", 0.180))
                publish_projection_model = build_projection_model(
                    panel_intrinsics_path=panel_intrinsics_path,
                    main_intrinsics_path=publish_main_intrinsics_path,
                    extrinsics_panel_to_main_path=publish_main_extrinsics_path,
                    panel_vertical_span_m=panel_vertical_span_m,
                    panel_frame_size=panel_frame_size,
                    main_frame_size=publish_main_frame_size,
                    panel_calib_size=panel_calib_size,
                    main_calib_size=publish_main_calib_size,
                )
                print(
                    "[INFO] coordinate transform enabled for publish: "
                    f"panel_recog_camera->{publish_projection_model.main_frame_size[0]}x{publish_projection_model.main_frame_size[1]} main_camera"
                )

                if debug_overlay_enabled:
                    debug_camera_cfg = debug_overlay_cfg.get("camera", {})
                    debug_device = normalize_device_arg(debug_camera_cfg.get("device", "/dev/video0"))
                    main_cap, _ = setup_camera(
                        debug_device,
                        debug_camera_cfg.get("capture", {}),
                        debug_camera_cfg.get("init_controls", {}),
                    )
                    next_main_frame = _read_first_frame(main_cap, camera_label="main_camera")
                    debug_main_frame_size = (int(next_main_frame.shape[1]), int(next_main_frame.shape[0]))
                    if do_display:
                        cv2.namedWindow(main_window_name, cv2.WINDOW_NORMAL)

                    use_publish_params = bool(debug_overlay_cfg.get("use_publish_main_camera_params", True))
                    if use_publish_params:
                        debug_intrinsics_path = publish_main_intrinsics_path
                        debug_extrinsics_path = publish_main_extrinsics_path
                        debug_main_calib_size = publish_main_calib_size
                    else:
                        debug_main_cfg = debug_overlay_cfg.get("main_camera", {})
                        debug_intrinsics_path = str(debug_main_cfg.get("intrinsics_path", "")).strip()
                        if not debug_intrinsics_path:
                            raise ValueError("coordinate_transform.debug_overlay.main_camera.intrinsics_path is required")
                        debug_extrinsics_path = str(debug_main_cfg.get("extrinsics_from_panel_recog_path", "")).strip()
                        if not debug_extrinsics_path:
                            raise ValueError(
                                "coordinate_transform.debug_overlay.main_camera.extrinsics_from_panel_recog_path is required"
                            )
                        debug_main_calib_size = _parse_optional_size(
                            debug_main_cfg.get("calib_size"),
                            field_name="coordinate_transform.debug_overlay.main_camera.calib_size",
                        )

                    debug_projection_model = build_projection_model(
                        panel_intrinsics_path=panel_intrinsics_path,
                        main_intrinsics_path=debug_intrinsics_path,
                        extrinsics_panel_to_main_path=debug_extrinsics_path,
                        panel_vertical_span_m=panel_vertical_span_m,
                        panel_frame_size=panel_frame_size,
                        main_frame_size=debug_main_frame_size,
                        panel_calib_size=panel_calib_size,
                        main_calib_size=debug_main_calib_size,
                    )
                    print(
                        "[INFO] debug overlay enabled: "
                        f"main camera window={main_window_name!r} size={debug_main_frame_size[0]}x{debug_main_frame_size[1]}"
                    )
            except Exception as e:
                print(f"[ERROR] coordinate transform init failed: {e}")
                return 2

        last_t = time.time()
        fps = 0.0
        alpha = float(cfg["ui"].get("fps_ema_alpha", 0.2))

        frame_idx = 0
        last_target_color = target_color_state.get()

        while True:
            if next_panel_frame is not None:
                frame = next_panel_frame
                next_panel_frame = None
            else:
                ret, frame = cap.read()
                if not ret:
                    continue

            main_frame = None
            if main_cap is not None:
                if next_main_frame is not None:
                    main_frame = next_main_frame
                    next_main_frame = None
                else:
                    ok_main, frame_main = main_cap.read()
                    if ok_main:
                        main_frame = frame_main

            frame_idx += 1

            now = time.time()
            dt = max(1e-6, now - last_t)
            last_t = now
            inst_fps = 1.0 / dt
            fps = alpha * inst_fps + (1.0 - alpha) * fps

            target_color = target_color_state.get()
            if target_color != last_target_color:
                last_target_color = target_color
                if cfg["tracking"]["enabled"]:
                    try:
                        tracker = build_tracker(cfg["tracking"], fps=float(cfg["camera"]["capture"]["fps"]))
                        print(f"[INFO] tracker reset: target color switched to {target_color}")
                    except Exception as e:
                        print(f"[WARN] tracker reset failed: {e}")
                        tracker = None

            frame_result = process_frame(
                frame_bgr=frame,
                det_cfg=cfg["detection"],
                tracking_cfg=cfg["tracking"],
                tracker=tracker,
                dt=dt,
                target_color=target_color,
            )

            publish_projected = None
            if publish_projection_model is not None and frame_result.selected_pair is not None:
                publish_projected = project_pair_to_main_camera(frame_result.selected_pair, publish_projection_model)

            debug_projected = None
            if debug_projection_model is not None and frame_result.selected_pair is not None:
                debug_projected = project_pair_to_main_camera(frame_result.selected_pair, debug_projection_model)

            log_motion_sample(
                motion_logger=motion_logger,
                frame_idx=frame_idx,
                now=now,
                dt=dt,
                result=frame_result,
            )

            should_quit = False
            if do_display:
                if main_cap is not None:
                    should_quit = render_frame(
                        frame,
                        result=frame_result,
                        tracking_enabled=bool(cfg["tracking"]["enabled"]),
                        tracker_available=tracker is not None,
                        track_color_bgr=tuple(int(x) for x in cfg["tracking"]["color_bgr"]),
                        history=viz,
                        fps=fps,
                        win_name=win_name,
                        poll_key=False,
                    )
                    if main_frame is not None:
                        draw_projected_overlay(main_frame, debug_projected, label="panel")
                        cv2.imshow(main_window_name, main_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        should_quit = True
                else:
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
            else:
                time.sleep(0.01)

            if should_quit:
                break

            if publish_enabled and latest_publisher is not None:
                payload = result_to_publish_payload(frame_result)
                if publish_projection_model is not None:
                    payload = projected_to_publish_payload(
                        publish_projected,
                        main_frame_size=publish_projection_model.main_frame_size,
                    )
                latest_publisher.submit(payload)

    finally:
        close_motion_logger(motion_logger)
        if latest_publisher is not None:
            try:
                latest_publisher.close()
            except Exception as e:
                print(f"[WARN] publisher process close failed: {e}")
        close_publisher(session)
        if main_cap is not None:
            main_cap.release()
        cap.release()
        cv2.destroyAllWindows()

    return 0
