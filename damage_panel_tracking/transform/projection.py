from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple

import cv2
import numpy as np

from ..detection.types import PairMeta


@dataclass(frozen=True)
class IntrinsicsData:
    K: np.ndarray
    dist: np.ndarray
    size: tuple[int, int] | None


@dataclass(frozen=True)
class ProjectionModel:
    panel_K: np.ndarray
    main_K: np.ndarray
    R_panel_to_main: np.ndarray
    t_panel_to_main: np.ndarray
    panel_vertical_span_m: float
    main_frame_size: tuple[int, int]


@dataclass(frozen=True)
class ProjectedPanel:
    center_uv: tuple[float, float]
    bbox_xywh: tuple[int, int, int, int] | None
    depth_m: float
    vertical_span_px: float


def _fs_read_mat(fs: cv2.FileStorage, key: str) -> np.ndarray:
    node = fs.getNode(key)
    if node.empty():
        raise RuntimeError(f"YAML missing key: {key}")
    mat = node.mat()
    if mat is None:
        raise RuntimeError(f"YAML key '{key}' is not a matrix")
    return mat


def parse_size(value: Any) -> tuple[int, int] | None:
    if value is None:
        return None
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return int(value[0]), int(value[1])
    if isinstance(value, str):
        s = value.strip().lower()
        if not s:
            return None
        if "x" not in s:
            raise ValueError(f"Invalid size format: {value!r} (expected like '1280x720')")
        w_s, h_s = s.split("x", 1)
        return int(w_s), int(h_s)
    raise ValueError(f"Invalid size value type: {type(value).__name__}")


def load_intrinsics(path: str) -> IntrinsicsData:
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise RuntimeError(f"Failed to open intrinsics: {path}")
    try:
        K = _fs_read_mat(fs, "K")
        dist = _fs_read_mat(fs, "dist")

        width = None
        height = None
        n_w = fs.getNode("width")
        n_h = fs.getNode("height")
        if not n_w.empty() and not n_h.empty():
            width = int(n_w.real())
            height = int(n_h.real())
    finally:
        fs.release()

    size = (width, height) if width is not None and height is not None and width > 0 and height > 0 else None
    return IntrinsicsData(K=K, dist=dist, size=size)


def load_extrinsics(path: str) -> tuple[np.ndarray, np.ndarray]:
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise RuntimeError(f"Failed to open extrinsics: {path}")
    try:
        R = _fs_read_mat(fs, "R")
        t = _fs_read_mat(fs, "t")
    finally:
        fs.release()
    return R, t


def _resolve_calib_size(intr: IntrinsicsData, explicit_size: tuple[int, int] | None, *, label: str) -> tuple[int, int]:
    size = explicit_size or intr.size
    if size is None:
        raise RuntimeError(
            f"{label} calibration size is missing. "
            "Set `calib_size` in config or include width/height in the intrinsics YAML."
        )
    return size


def scale_K(K: np.ndarray, from_size: tuple[int, int], to_size: tuple[int, int]) -> np.ndarray:
    fw, fh = from_size
    tw, th = to_size
    sx = tw / float(fw)
    sy = th / float(fh)
    K2 = K.copy()
    K2[0, 0] *= sx
    K2[0, 2] *= sx
    K2[1, 1] *= sy
    K2[1, 2] *= sy
    return K2


def build_projection_model(
    *,
    panel_intrinsics_path: str,
    main_intrinsics_path: str,
    extrinsics_panel_to_main_path: str,
    panel_vertical_span_m: float,
    panel_frame_size: tuple[int, int],
    main_frame_size: tuple[int, int],
    panel_calib_size: tuple[int, int] | None = None,
    main_calib_size: tuple[int, int] | None = None,
) -> ProjectionModel:
    panel_intr = load_intrinsics(panel_intrinsics_path)
    main_intr = load_intrinsics(main_intrinsics_path)

    panel_from_size = _resolve_calib_size(panel_intr, panel_calib_size, label="panel_recog_camera intrinsics")
    main_from_size = _resolve_calib_size(main_intr, main_calib_size, label="main_camera intrinsics")

    panel_K = scale_K(panel_intr.K, panel_from_size, panel_frame_size)
    main_K = scale_K(main_intr.K, main_from_size, main_frame_size)

    R, t = load_extrinsics(extrinsics_panel_to_main_path)
    if R.shape != (3, 3):
        raise RuntimeError(f"Invalid R shape in extrinsics: {R.shape}")
    if t.shape not in ((3, 1), (1, 3), (3,)):
        raise RuntimeError(f"Invalid t shape in extrinsics: {t.shape}")
    t_col = np.asarray(t, dtype=np.float64).reshape(3, 1)

    if panel_vertical_span_m <= 0.0:
        raise RuntimeError(f"panel_vertical_span_m must be > 0: {panel_vertical_span_m}")

    return ProjectionModel(
        panel_K=np.asarray(panel_K, dtype=np.float64),
        main_K=np.asarray(main_K, dtype=np.float64),
        R_panel_to_main=np.asarray(R, dtype=np.float64),
        t_panel_to_main=t_col,
        panel_vertical_span_m=float(panel_vertical_span_m),
        main_frame_size=main_frame_size,
    )


def _center_from_xywh(box_xywh: tuple[int, int, int, int]) -> tuple[float, float]:
    x, y, w, h = box_xywh
    return x + w / 2.0, y + h / 2.0


def _estimate_depth_from_vertical(top_xy: tuple[float, float], bottom_xy: tuple[float, float], fy_px: float, span_m: float) -> tuple[float, float] | None:
    h_px = float(np.hypot(top_xy[0] - bottom_xy[0], top_xy[1] - bottom_xy[1]))
    if h_px < 1.0:
        return None
    Z = (fy_px * span_m) / h_px
    return float(Z), h_px


def _backproject_pixel_to_3d(u: float, v: float, Z: float, K: np.ndarray) -> np.ndarray:
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])
    x = (u - cx) / fx
    y = (v - cy) / fy
    return np.array([[Z * x], [Z * y], [Z]], dtype=np.float64)


def _project_3d_to_pixel(P: np.ndarray, K: np.ndarray) -> tuple[float, float] | None:
    X, Y, Z = float(P[0, 0]), float(P[1, 0]), float(P[2, 0])
    if Z <= 1e-6:
        return None
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])
    u = fx * (X / Z) + cx
    v = fy * (Y / Z) + cy
    return float(u), float(v)


def project_pair_to_main_camera(pair: PairMeta, model: ProjectionModel) -> ProjectedPanel | None:
    top_center = _center_from_xywh(pair.top_xywh)
    bottom_center = _center_from_xywh(pair.bottom_xywh)

    depth = _estimate_depth_from_vertical(
        top_center,
        bottom_center,
        fy_px=float(model.panel_K[1, 1]),
        span_m=model.panel_vertical_span_m,
    )
    if depth is None:
        return None
    Z, h_px = depth

    uc = 0.5 * (top_center[0] + bottom_center[0])
    vc = 0.5 * (top_center[1] + bottom_center[1])
    P_panel = _backproject_pixel_to_3d(uc, vc, Z, model.panel_K)
    P_main = model.R_panel_to_main @ P_panel + model.t_panel_to_main
    uv_main = _project_3d_to_pixel(P_main, model.main_K)
    if uv_main is None:
        return None

    ux, uy, uw, uh = pair.union_xywh
    corners_panel = [
        (ux, uy),
        (ux + uw, uy),
        (ux + uw, uy + uh),
        (ux, uy + uh),
    ]
    proj_corners: list[tuple[float, float]] = []
    for (u, v) in corners_panel:
        P_panel_c = _backproject_pixel_to_3d(float(u), float(v), Z, model.panel_K)
        P_main_c = model.R_panel_to_main @ P_panel_c + model.t_panel_to_main
        uv_main_c = _project_3d_to_pixel(P_main_c, model.main_K)
        if uv_main_c is not None:
            proj_corners.append(uv_main_c)

    bbox: tuple[int, int, int, int] | None = None
    if len(proj_corners) == 4:
        xs = [p[0] for p in proj_corners]
        ys = [p[1] for p in proj_corners]
        x1, y1 = int(min(xs)), int(min(ys))
        x2, y2 = int(max(xs)), int(max(ys))
        if x2 > x1 and y2 > y1:
            bbox = (x1, y1, int(x2 - x1), int(y2 - y1))

    return ProjectedPanel(
        center_uv=uv_main,
        bbox_xywh=bbox,
        depth_m=Z,
        vertical_span_px=h_px,
    )


def projected_to_publish_payload(
    projected: ProjectedPanel | None,
    *,
    main_frame_size: tuple[int, int],
) -> tuple[bool, int, int, int, int, int]:
    if projected is None:
        return (False, 0, 0, 0, 0, 0)

    frame_w, frame_h = main_frame_size
    u, v = projected.center_uv
    if u < 0.0 or v < 0.0 or u >= frame_w or v >= frame_h:
        return (False, 0, 0, 0, 0, 0)

    x = int(round(u))
    y = int(round(v))
    x = max(0, min(frame_w - 1, x))
    y = max(0, min(frame_h - 1, y))

    if projected.bbox_xywh is not None:
        bx, by, bw, bh = projected.bbox_xywh
        x1 = max(0, min(frame_w - 1, int(bx)))
        y1 = max(0, min(frame_h - 1, int(by)))
        x2 = max(x1 + 1, min(frame_w, int(bx + bw)))
        y2 = max(y1 + 1, min(frame_h, int(by + bh)))
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)
    else:
        w = 1
        h = 1

    distance_mm = max(0, int(round(projected.depth_m * 1000.0)))
    return (True, x, y, int(w), int(h), distance_mm)


def draw_projected_overlay(
    frame_bgr: np.ndarray,
    projected: ProjectedPanel | None,
    *,
    label: str = "projected panel",
) -> None:
    text = "projection: no target"
    if projected is None:
        cv2.putText(frame_bgr, text, (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        return

    u, v = projected.center_uv
    if np.isfinite(u) and np.isfinite(v):
        cv2.circle(frame_bgr, (int(round(u)), int(round(v))), 10, (0, 0, 255), 3)
        cv2.putText(
            frame_bgr,
            label,
            (int(round(u)) + 12, int(round(v)) - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )

    if projected.bbox_xywh is not None:
        x, y, w, h = projected.bbox_xywh
        cv2.rectangle(frame_bgr, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)

    text = f"projection: Z={projected.depth_m:.3f}m span={projected.vertical_span_px:.1f}px"
    cv2.putText(frame_bgr, text, (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
