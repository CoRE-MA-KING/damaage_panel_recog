from __future__ import annotations

import argparse
import math
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict

from .ui.qt_compat import configure_qt_fontdir

configure_qt_fontdir()

import cv2

configure_qt_fontdir()

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
    TransformSession,
    draw_projected_overlay,
    projected_to_publish_payload,
)
from .ui.draw import TrackVizState
from .ui.gui import create_setting_gui


VALID_TARGET_COLORS: tuple[ColorName, ColorName] = ("blue", "red")
ROBOAPP_CONFIG_DIR = "roboapp"
ROBOAPP_CONFIG_FILE = "damage_panel_recog_config.yaml"


class TargetColorState:
    def __init__(self, initial: ColorName) -> None:
        # subscribeコールバックから安全に参照できるよう色状態をロック付きで保持する。
        self._color = initial
        self._lock = threading.Lock()

    def get(self) -> ColorName:
        # ターゲット色を排他的に読み取る。
        with self._lock:
            return self._color

    def set(self, next_color: ColorName) -> bool:
        # ターゲット色を排他的に更新し、変更有無を返す。
        with self._lock:
            if self._color == next_color:
                return False
            self._color = next_color
            return True


def _normalize_target_color(value: Any, *, source: str) -> ColorName:
    # ターゲット色を検証し、サポート値へ正規化する。
    color = str(value).strip().lower()
    if color not in VALID_TARGET_COLORS:
        raise ValueError(f"{source} must be one of {VALID_TARGET_COLORS}: got {value!r}")
    if color == "blue":
        return "blue"
    return "red"


def parse_args() -> argparse.Namespace:
    # CLIオプションを定義して実行時引数を解釈する。
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
    return ap.parse_args()


def _resolve_roboapp_config_path() -> Path:
    # XDG_CONFIG_HOME があればそれを使い、なければ ~/.config へフォールバックする。
    xdg_config_home = os.getenv("XDG_CONFIG_HOME", "").strip()
    if xdg_config_home:
        return Path(xdg_config_home).expanduser() / ROBOAPP_CONFIG_DIR / ROBOAPP_CONFIG_FILE
    return Path.home() / ".config" / ROBOAPP_CONFIG_DIR / ROBOAPP_CONFIG_FILE


def _load_roboapp_config_override() -> Dict[str, Any]:
    # 既定の外部設定を読み込む。失敗時は例外を投げて起動を中止させる。
    path = _resolve_roboapp_config_path()
    try:
        loaded = load_config(path)
    except FileNotFoundError:
        print(f"[エラー] 設定ファイルが見つかりません: {path}")
        print("[エラー] 起動できません。")
        print(
            "[案内] `damage_panel_recog/config/default.yaml` を "
            "`damage_panel_recog_config.yaml` にリネームし、"
            "`~/.config/roboapp/` 配下に配置してください。"
        )
        raise
    except ValueError as e:
        print(f"[エラー] 設定ファイルの形式が不正です: {path}")
        print(f"[エラー] 理由: {e}")
        raise
    except Exception as e:
        print(f"[エラー] 設定ファイルの読み込みに失敗しました: {path}")
        print(f"[エラー] 理由: {type(e).__name__}: {e}")
        raise

    print(f"[情報] 設定ファイルを読み込みました: {path}")
    return loaded


def _apply_cli_overrides(cfg: Dict[str, Any], args: argparse.Namespace) -> None:
    # 実行初期化前にCLI上書きを適用し、設定値を正規化する。
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


def _read_first_frame(cap: cv2.VideoCapture, *, camera_label: str) -> Any:
    # 有効な最初の1フレームが取得できるまで待機する。
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
    # 入力を解釈し、実効設定（defaults <- file <- CLI）を組み立てる。
    args = parse_args()

    try:
        override = _load_roboapp_config_override()
    except Exception:
        return 2

    cfg = build_effective_config(DEFAULTS, override)
    try:
        _apply_cli_overrides(cfg, args)
    except ValueError as e:
        print(f"[エラー] 設定値が不正なため起動できません: {e}")
        return 2

    # 表示/GUIの動作を確定する。
    do_display = not bool(args.no_display)
    use_gui = bool(args.setting) and do_display

    transform_cfg = cfg.get("coordinate_transform", {})

    # 確定済み設定からpublish/subscribe用の実行パラメータを取り出す。
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

    # try/finally前にランタイムハンドルを安全な初期値で用意する。
    session: ZenohSession | None = None
    latest_publisher: LatestFramePublisher | None = None
    motion_logger = None
    tracker = None
    viz: TrackVizState | None = None
    transform_session: TransformSession | None = None
    cap: cv2.VideoCapture | None = None
    win_name = str(cfg["ui"]["window_name"])

    try:
        # 認識カメラを開いて表示系を初期化する。
        device = normalize_device_arg(cfg["camera"]["device"])
        cap, dev_path = setup_camera(device, cfg["camera"]["capture"], cfg["camera"]["init_controls"])

        # 表示ウィンドウと任意の調整用GUIを準備する。
        if do_display or use_gui:
            cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

        if use_gui:
            create_setting_gui(win_name, dev_path, cfg["detection"]["hsv"], cfg["camera"]["init_controls"])

        viz = TrackVizState(history_len=int(cfg["tracking"]["history_len"]))

        # 実際にネゴシエーションされた結果の画像サイズを得る（サイズ取得専用）
        negotiated_panel_frame = _read_first_frame(cap, camera_label="panel_recog_camera")
        panel_frame_size = (int(negotiated_panel_frame.shape[1]), int(negotiated_panel_frame.shape[0]))

        # publish有効時は非同期publisherプロセスを起動する。
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

        # subscribeコールバックを開始し、ターゲット色を動的更新できるようにする。
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

        # 任意機能（ログ/追跡/座標変換）を初期化する。
        motion_logger = create_motion_logger(cfg.get("logging", {}))

        if cfg["tracking"]["enabled"]:
            try:
                tracker = build_tracker(cfg["tracking"], fps=float(cfg["camera"]["capture"]["fps"]))
            except Exception as e:
                print(f"[WARN] tracking backend init failed: {e}")
                tracker = None

        try:
            transform_session = TransformSession.create(
                transform_cfg,
                panel_frame_size=panel_frame_size,
                do_display=do_display,
            )
        except Exception as e:
            print(f"[ERROR] coordinate transform init failed: {e}")
            return 2

        # FPS計算状態とループ内カウンタを初期化する。
        last_t = time.time()
        fps = 0.0
        alpha = float(cfg["ui"].get("fps_ema_alpha", 0.2))

        frame_idx = 0
        last_target_color = target_color_state.get()
        previous_target_track_id: str | None = None

        # メインループで取得・検出追跡・描画・必要ならpublishを行う。
        while True:
            # フレーム取得
            ret, frame = cap.read()
            if not ret:
                continue            

            # 座標変換セッションが有効な場合、デバッグ用main_cameraフレームを取得
            main_frame = None
            if transform_session is not None:
                main_frame = transform_session.read_debug_main_frame()

            frame_idx += 1

            now = time.time()
            # ゼロ除算を防ぐためにdtに最小値を設ける
            dt = max(1e-6, now - last_t)
            last_t = now
            inst_fps = 1.0 / dt
            fps = alpha * inst_fps + (1.0 - alpha) * fps

            # 現在のターゲット色（blue/red）をスレッドセーフに取得する。
            target_color = target_color_state.get()
            # 前フレーム時点からターゲット色が切り替わったか確認する。
            if target_color != last_target_color:
                # 比較基準を最新の色に更新する。
                last_target_color = target_color
                # 追跡機能が有効なときのみトラッカーを再初期化する。
                if cfg["tracking"]["enabled"]:
                    try:
                        # 色切替後の誤対応を避けるため、トラッカー内部状態をリセットする。
                        tracker = build_tracker(cfg["tracking"], fps=float(cfg["camera"]["capture"]["fps"]))
                        previous_target_track_id = None
                        print(f"[INFO] tracker reset: target color switched to {target_color}")
                    except Exception as e:
                        # 再初期化に失敗した場合は追跡を一時無効化する。
                        print(f"[WARN] tracker reset failed: {e}")
                        tracker = None
                        previous_target_track_id = None

            # 現在のターゲット色で1フレーム分の検出/追跡を実行する。
            frame_result = process_frame(
                frame_bgr=frame,
                det_cfg=cfg["detection"],
                tracking_cfg=cfg["tracking"],
                tracker=tracker,
                dt=dt,
                target_color=target_color,
                previous_target_track_id=previous_target_track_id,
            )
            if frame_result.chosen_from_tracks and frame_result.selected_track is not None:
                previous_target_track_id = frame_result.selected_track.track_id
            else:
                previous_target_track_id = None

            # 座標変換有効時は選択ペアをmain_camera座標へ投影する。
            publish_projected = None
            debug_projected = None
            if transform_session is not None:
                publish_projected, debug_projected = transform_session.project_selected_pair(frame_result.selected_pair)

            # ログ有効時はモーションサンプルを1行追記する。
            log_motion_sample(
                motion_logger=motion_logger,
                frame_idx=frame_idx,
                now=now,
                dt=dt,
                result=frame_result,
            )

            # パネル/デバッグ画面を描画し、終了キー入力を処理する。
            should_quit = False
            if do_display:
                if transform_session is not None and transform_session.debug_main_cap is not None:
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
                        cv2.imshow(transform_session.debug_main_window_name, main_frame)
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

            # 最新ターゲットpayloadを非同期プロセスへ渡してpublishする。
            if publish_enabled and latest_publisher is not None:
                payload = result_to_publish_payload(frame_result)
                if transform_session is not None and transform_session.publish_model is not None:
                    payload = projected_to_publish_payload(
                        publish_projected,
                        main_frame_size=transform_session.publish_model.main_frame_size,
                    )
                latest_publisher.submit(payload)
    except ValueError as e:
        print("[エラー] 設定ファイルの値が不正なため起動できません。")
        print(f"[エラー] 理由: {e}")
        return 2
    except TypeError as e:
        print("[エラー] 設定ファイルの値の型が不正なため起動できません。")
        print(f"[エラー] 理由: {e}")
        return 2
    except RuntimeError as e:
        print("[エラー] 設定ファイルの内容により初期化に失敗しました。")
        print(f"[エラー] 理由: {e}")
        return 2

    # 例外時も含めて全リソースを確実にクローズする。
    finally:
        close_motion_logger(motion_logger)
        if latest_publisher is not None:
            try:
                latest_publisher.close()
            except Exception as e:
                print(f"[WARN] publisher process close failed: {e}")
        close_publisher(session)
        if transform_session is not None:
            transform_session.close()
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()

    return 0
