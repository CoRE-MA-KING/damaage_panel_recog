from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Dict

import cv2
import numpy as np

from ..camera.capture import setup_camera
from ..detection.types import PairMeta
from ..ui.qt_compat import start_highgui_event_thread
from .projection import (
    ProjectedPanel,
    ProjectionModel,
    build_projection_model,
    parse_size,
    project_pair_to_main_camera,
)


def _normalize_device_arg(dev: Any) -> Any:
    # 数字文字列のデバイス指定を整数インデックスへ変換する。
    if isinstance(dev, str) and dev.isdigit():
        return int(dev)
    return dev


def _parse_optional_size(value: Any, *, field_name: str) -> tuple[int, int] | None:
    # サイズ設定を解釈し、エラー時に設定項目名を付与する。
    try:
        return parse_size(value)
    except ValueError as e:
        raise ValueError(f"{field_name}: {e}") from e


def _read_first_frame(cap: cv2.VideoCapture, *, camera_label: str) -> np.ndarray:
    # 読み取り可能な最初のフレームが来るまで待機する。
    attempts = 0
    while True:
        ok, frame = cap.read()
        if ok:
            return frame
        attempts += 1
        if attempts % 100 == 0:
            print(f"[WARN] waiting for first frame from {camera_label}...")
        time.sleep(0.01)


@dataclass
class TransformSession:
    publish_model: ProjectionModel | None
    debug_model: ProjectionModel | None
    debug_main_cap: cv2.VideoCapture | None
    debug_main_window_name: str
    _next_debug_main_frame: np.ndarray | None

    @classmethod
    def create(
        cls,
        transform_cfg: Dict[str, Any],
        *,
        panel_frame_size: tuple[int, int],
        do_display: bool,
    ) -> TransformSession:
        # 投影モデルと、必要ならデバッグ用mainカメラを初期化する。
        enabled = bool(transform_cfg.get("enabled", False))
        debug_overlay_cfg = transform_cfg.get("debug_overlay", {})
        debug_overlay_enabled = bool(debug_overlay_cfg.get("enabled", False))
        if debug_overlay_enabled and not enabled:
            enabled = True
        if debug_overlay_enabled and not do_display:
            print("[WARN] coordinate_transform.debug_overlay is enabled but --no-display was set. debug overlay is disabled.")
            debug_overlay_enabled = False

        debug_main_window_name = str(debug_overlay_cfg.get("window_name", "Main Camera (projected target)"))
        if not enabled:
            return cls(
                publish_model=None,
                debug_model=None,
                debug_main_cap=None,
                debug_main_window_name=debug_main_window_name,
                _next_debug_main_frame=None,
            )

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
            raise ValueError("coordinate_transform.publish_main_camera.frame_size or calib_size is required")

        panel_vertical_span_m = float(transform_cfg.get("panel_vertical_span_m", 0.180))
        publish_model = build_projection_model(
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
            f"panel_recog_camera->{publish_model.main_frame_size[0]}x{publish_model.main_frame_size[1]} main_camera"
        )

        if not debug_overlay_enabled:
            return cls(
                publish_model=publish_model,
                debug_model=None,
                debug_main_cap=None,
                debug_main_window_name=debug_main_window_name,
                _next_debug_main_frame=None,
            )

        debug_camera_cfg = debug_overlay_cfg.get("camera", {})
        debug_device = _normalize_device_arg(debug_camera_cfg.get("device", "/dev/video0"))
        debug_main_cap, _dev_path = setup_camera(
            debug_device,
            debug_camera_cfg.get("capture", {}),
            debug_camera_cfg.get("init_controls", {}),
        )
        next_debug_main_frame = _read_first_frame(debug_main_cap, camera_label="main_camera")
        debug_main_frame_size = (int(next_debug_main_frame.shape[1]), int(next_debug_main_frame.shape[0]))
        if do_display:
            cv2.namedWindow(debug_main_window_name, cv2.WINDOW_NORMAL)
            start_highgui_event_thread()

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

        debug_model = build_projection_model(
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
            f"main camera window={debug_main_window_name!r} size={debug_main_frame_size[0]}x{debug_main_frame_size[1]}"
        )
        return cls(
            publish_model=publish_model,
            debug_model=debug_model,
            debug_main_cap=debug_main_cap,
            debug_main_window_name=debug_main_window_name,
            _next_debug_main_frame=next_debug_main_frame,
        )

    def read_debug_main_frame(self) -> np.ndarray | None:
        # 初回はキャッシュ済みフレームを返し、以後は通常読み取りを行う。
        if self.debug_main_cap is None:
            return None
        if self._next_debug_main_frame is not None:
            frame = self._next_debug_main_frame
            self._next_debug_main_frame = None
            return frame
        ok, frame = self.debug_main_cap.read()
        if not ok:
            return None
        return frame

    def project_selected_pair(
        self,
        selected_pair: PairMeta | None,
    ) -> tuple[ProjectedPanel | None, ProjectedPanel | None]:
        # 選択されたパネルペアをpublish/デバッグ各カメラ座標へ投影する。
        publish_projected: ProjectedPanel | None = None
        debug_projected: ProjectedPanel | None = None
        if selected_pair is None:
            return publish_projected, debug_projected

        if self.publish_model is not None:
            publish_projected = project_pair_to_main_camera(selected_pair, self.publish_model)
        if self.debug_model is not None:
            debug_projected = project_pair_to_main_camera(selected_pair, self.debug_model)
        return publish_projected, debug_projected

    def close(self) -> None:
        # 開いていればデバッグ用mainカメラを解放する。
        if self.debug_main_cap is not None:
            self.debug_main_cap.release()
