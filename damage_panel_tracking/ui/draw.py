from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import collections

import cv2

from ..detection.types import PairMeta, iou_xyxy, xyxy_center
from ..tracking.base import Track


def put_text(img, text: str, org: Tuple[int, int], scale=0.6, color=(255, 255, 255), thickness=1) -> None:
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_detection_pair(img, meta: PairMeta) -> None:
    box_col = (255, 0, 0) if meta.color == "blue" else (0, 0, 255)
    for (x, y, w, h) in (meta.top_xywh, meta.bottom_xywh):
        cv2.rectangle(img, (x, y), (x + w, y + h), box_col, 2)

    ux, uy, uw, uh = meta.union_xywh
    cv2.rectangle(img, (ux, uy), (ux + uw, uy + uh), (0, 255, 0), 2)

    cx, cy = xyxy_center(meta.union_xyxy)
    cv2.circle(img, (int(cx), int(cy)), 3, (0, 255, 0), -1)
    put_text(
        img,
        f"( {int(meta.union_xyxy[0])},{int(meta.union_xyxy[1])} )-( {int(meta.union_xyxy[2])},{int(meta.union_xyxy[3])} )",
        (ux, max(0, uy - 6)),
        0.5,
        (0, 255, 0),
        1,
    )


def associate_tracks_to_pairs(tracks: List[Track], pairs: List[PairMeta], iou_thresh: float = 0.1) -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    for t in tracks:
        best_i, best_iou = -1, 0.0
        for i, p in enumerate(pairs):
            iou = iou_xyxy(t.box_xyxy, p.union_xyxy)
            if iou > best_iou:
                best_iou = iou
                best_i = i
        if best_i >= 0 and best_iou >= iou_thresh:
            mapping[t.track_id] = best_i
    return mapping


class TrackVizState:
    def __init__(self, history_len: int = 20):
        self.history_len = int(history_len)
        self._hist: Dict[str, collections.deque] = {}
        self._alias: Dict[str, int] = {}
        self._next_alias = 1

    def alias(self, track_id: str) -> int:
        if track_id not in self._alias:
            self._alias[track_id] = self._next_alias
            self._next_alias += 1
        return self._alias[track_id]

    def push(self, track_id: str, cx: int, cy: int) -> None:
        if track_id not in self._hist:
            self._hist[track_id] = collections.deque(maxlen=self.history_len)
        self._hist[track_id].append((cx, cy))

    def get(self, track_id: str):
        return list(self._hist.get(track_id, []))


def draw_tracks(
    img,
    tracks: List[Track],
    pairs: List[PairMeta],
    track_color=(0, 255, 255),
    history: Optional[TrackVizState] = None,
) -> None:
    assoc = associate_tracks_to_pairs(tracks, pairs, iou_thresh=0.1)

    for t in tracks:
        x1, y1, x2, y2 = map(int, t.box_xyxy.tolist())
        cx, cy = xyxy_center(t.box_xyxy)
        cx_i, cy_i = int(cx), int(cy)

        cv2.rectangle(img, (x1, y1), (x2, y2), track_color, 2)

        show_id = history.alias(t.track_id) if history is not None else t.track_id
        put_text(img, f"ID {show_id} ({cx_i},{cy_i})", (x1, max(0, y1 - 10)), 0.6, track_color, 2)

        if history is not None:
            history.push(t.track_id, cx_i, cy_i)
            pts = history.get(t.track_id)
            for k in range(1, len(pts)):
                cv2.line(img, pts[k - 1], pts[k], track_color, 2)

        if t.track_id in assoc:
            p = pairs[assoc[t.track_id]]
            box_col = (255, 0, 0) if p.color == "blue" else (0, 0, 255)
            for (x, y, w, h) in (p.top_xywh, p.bottom_xywh):
                cv2.rectangle(img, (x, y), (x + w, y + h), box_col, 2)


def draw_target(img, target_xy: Tuple[int, int], from_tracks: bool) -> None:
    tx, ty = target_xy
    col = (255, 255, 0) if from_tracks else (255, 255, 255)
    cv2.drawMarker(img, (int(tx), int(ty)), col, markerType=cv2.MARKER_CROSS, markerSize=16, thickness=2)
    put_text(img, f"target=({int(tx)},{int(ty)})", (10, 26), 0.7, col, 2)


def draw_fps(img, fps: float) -> None:
    text = f"FPS: {fps:.1f}"
    (tw, _), _2 = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    x_right = img.shape[1] - tw - 10
    y_top = 30
    put_text(img, text, (x_right, y_top), 0.7, (255, 255, 255), 2)


def draw_mode(img, mode: str) -> None:
    put_text(img, mode, (10, 50), 0.6, (200, 200, 200), 1)
