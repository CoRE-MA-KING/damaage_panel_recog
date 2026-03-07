from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from .detection.hsv import find_boxes, get_led_mask
from .detection.pairing import build_pair_meta, pair_boxes_same_color
from .detection.types import ColorName, PairMeta, iou_xyxy, xyxy_center
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
    # 数字文字列のデバイス指定を整数インデックスへ変換する。
    if isinstance(dev, str) and dev.isdigit():
        return int(dev)
    return dev


def pairs_from_frame(frame_bgr: np.ndarray, det_cfg: Dict[str, Any], target_color: ColorName) -> List[PairMeta]:
    # 指定色LED領域のみを検出し、同色の上下LEDをペア化する。
    hsv_cfg = det_cfg["hsv"]
    kernel_sz = int(det_cfg["kernel_sz"])
    min_box_w = int(det_cfg["min_box_w"])
    min_box_h = int(det_cfg["min_box_h"])
    width_tol = float(det_cfg["width_tol"])
    min_h_overlap = float(det_cfg["min_h_overlap"])
    min_v_gap = int(det_cfg["min_v_gap"])

    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    mask = get_led_mask(hsv, target_color, hsv_cfg)
    boxes = find_boxes(mask, kernel_sz=kernel_sz, min_box_w=min_box_w, min_box_h=min_box_h)

    pairs: List[PairMeta] = []
    for (top, bottom) in pair_boxes_same_color(boxes, width_tol, min_h_overlap, min_v_gap):
        pairs.append(build_pair_meta(color=target_color, top=top, bottom=bottom))
    return pairs

# ペア検出を面積ベースの簡易scoreつきでtracker入力へ変換する。
def detections_from_pairs(pairs: List[PairMeta], frame_shape: Tuple[int, int, int]) -> List[Detection]:    
    # フレーム形状から高さ・幅を取り出す（shapeはH,W,Cの順）。
    h, w = frame_shape[0], frame_shape[1]
    # 返却用のDetectionリストを初期化する。
    detections: List[Detection] = []
    # 1ペアずつtracker入力形式へ変換する。
    for p in pairs:
        # 統合bbox（xywh）から幅・高さだけ取り出す。
        _, _, uw, uh = p.union_xywh
        
        # 画面の1/8面積を基準にヒューリスティックで正規化し（値が小さくなりすぎるのを抑える）、
        # 加えて、+0.1の下駄を履かせた簡易scoreを1.0でクリップする。
        score = min(1.0, (uw * uh) / (w * h / 8.0) + 0.1)
        # tracker入力形式（xyxy, score, class_id）で追加する。class_idは現状単一クラスなので0固定（将来拡張用）。
        detections.append(Detection(box_xyxy=p.union_xyxy, score=float(score), class_id=0))
    # 変換後のDetection一覧を返す。
    return detections


def select_target_pair(pairs: List[PairMeta], frame_w: int, frame_h: int) -> Tuple[Tuple[int, int], PairMeta | None]:
    # 画像中心に対してx方向で最も近い検出を選ぶ。
    cx0 = frame_w / 2.0
    # フレーム中心を理想の座標と置く
    best_xy = (int(frame_w / 2), int(frame_h / 2))
    best_abs = float("inf")
    # 理想的なターゲットの座標情報を決定する
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
    # 画像中心に対してx方向で最も近い追跡対象を選ぶ。
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


def find_track_by_id(tracks: List[Track], track_id: str) -> Track | None:
    # 指定IDのtrackが現在フレームに存在すれば返す。
    for t in tracks:
        if t.track_id == track_id:
            return t
    return None


def track_center_xy(track: Track) -> Tuple[int, int]:
    # track bbox中心を整数ピクセル座標へ変換する。
    cx, cy = xyxy_center(track.box_xyxy)
    return int(cx), int(cy)


def select_pair_for_track(track: Track, pairs: List[PairMeta], *, iou_thresh: float = 0.1) -> PairMeta | None:
    # 選択trackに最も重なる検出ペアを返す（重なり不足ならNone）。
    best_pair: PairMeta | None = None
    best_iou = 0.0
    for p in pairs:
        iou = iou_xyxy(track.box_xyxy, p.union_xyxy)
        if iou > best_iou:
            best_iou = iou
            best_pair = p
    if best_pair is None or best_iou < float(iou_thresh):
        return None
    return best_pair


def build_tracker(track_cfg: Dict[str, Any], fps: float) -> MultiObjectTracker:
    # 設定に従って追跡バックエンドを生成する。
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
    target_color: ColorName,
    previous_target_track_id: str | None = None,
) -> FrameResult:
    # 1フレーム分の検出/追跡を実行し、publish対象座標を決定する。
    pairs = pairs_from_frame(frame_bgr, det_cfg, target_color)
    target_pairs = pairs
    frame_h, frame_w = frame_bgr.shape[0], frame_bgr.shape[1]
    target, selected_pair = select_target_pair(target_pairs, frame_w=frame_w, frame_h=frame_h)
    
    selected_track: Track | None = None
    tracks: List[Track] = []
    chosen_from_tracks = False

    if tracking_cfg["enabled"] and tracker is not None:
        # トラッキングに渡す形式にデータを加工（スコアとして、bboxサイズをヒューリスティックに正規化）
        detections = detections_from_pairs(target_pairs, frame_bgr.shape)
        
        tracks = tracker.step(detections, dt=dt)
        if len(tracks) > 0:
            if previous_target_track_id:
                selected_track = find_track_by_id(tracks, previous_target_track_id)
            if selected_track is None:
                target, selected_track = select_target_track(tracks, frame_w=frame_w, frame_h=frame_h)
            else:
                target = track_center_xy(selected_track)
            if selected_track is not None:
                selected_pair = select_pair_for_track(selected_track, target_pairs, iou_thresh=0.1)
            chosen_from_tracks = True

    return FrameResult(
        pairs=pairs,
        target=target,
        selected_pair=selected_pair,
        tracks=tracks,
        selected_track=selected_track,
        chosen_from_tracks=chosen_from_tracks,
    )
