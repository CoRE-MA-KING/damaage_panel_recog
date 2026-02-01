from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .base import Detection, Track


@dataclass
class DistanceConfig:
    """Config for distance-based association tracker (bbox-only).

    This backend is designed for cases where:
      - appearance features are unavailable (or useless),
      - the number of objects is small (e.g., <= 4),
      - abrupt direction changes (large jerk) can happen,
      - you want to avoid "IoU=0 -> immediate split" failure modes.

    Key idea:
      - match detections to tracks by *center distance* (optionally normalized),
        plus a small penalty for bbox size change.
      - use a simple constant-velocity state as a hint, but keep matching robust
        by allowing fallback to the last observed center (helps on hard reversals).
    """

    # Association / gating
    gate_px: float = 80.0                 # maximum allowed center movement per step to match (pixels)
    normalize: str = "diag"               # "diag" | "sqrt_area" | "none"
    size_weight: float = 0.15             # weight for bbox size change penalty (log-ratio)

    # Motion model
    use_prediction: bool = True           # use constant-velocity prediction as a hint
    vel_alpha: float = 0.6                # EMA factor for velocity update (0..1, larger = more responsive)
    max_speed_px_s: float = 8000.0        # clamp velocity magnitude (pixels/second)

    # Lifecycle
    max_age: int = 4                      # allowed missed steps before deleting a track
    min_steps_alive: int = 2              # minimum hits before reporting a track to the caller


class _InternalTrack:
    __slots__ = (
        "id", "box", "center", "wh", "vel", "age", "hits", "misses"
    )

    def __init__(self, tid: int, box_xyxy: np.ndarray) -> None:
        self.id = tid
        self.box = box_xyxy.astype(float)
        self.center = _xyxy_center(self.box)
        self.wh = _xyxy_wh(self.box)
        self.vel = np.zeros(2, dtype=float)   # pixels/second
        self.age = 1
        self.hits = 1
        self.misses = 0

    def predict_center(self, dt: float) -> np.ndarray:
        return self.center + self.vel * float(dt)

    def predict_box(self, dt: float) -> np.ndarray:
        c = self.predict_center(dt)
        w, h = self.wh
        x1 = c[0] - w / 2.0
        y1 = c[1] - h / 2.0
        x2 = c[0] + w / 2.0
        y2 = c[1] + h / 2.0
        return np.array([x1, y1, x2, y2], dtype=float)

    def mark_missed(self, dt: float, use_prediction: bool) -> None:
        self.age += 1
        self.misses += 1
        if use_prediction and dt > 0:
            # carry state forward (useful for short gaps)
            self.center = self.predict_center(dt)
            self.box = self.predict_box(dt)


def _xyxy_center(box: np.ndarray) -> np.ndarray:
    return np.array([(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0], dtype=float)


def _xyxy_wh(box: np.ndarray) -> np.ndarray:
    return np.array([max(1e-6, box[2] - box[0]), max(1e-6, box[3] - box[1])], dtype=float)


def _scale_for_norm(wh: np.ndarray, normalize: str) -> float:
    w, h = float(wh[0]), float(wh[1])
    if normalize == "none":
        return 1.0
    if normalize == "sqrt_area":
        return max(1.0, float(np.sqrt(max(1e-6, w * h))))
    # default: diag
    return max(1.0, float(np.sqrt(w * w + h * h)))


def _size_penalty(track_wh: np.ndarray, det_wh: np.ndarray) -> float:
    # penalty = |log(w2/w1)| + |log(h2/h1)|
    eps = 1e-6
    w1, h1 = float(track_wh[0]) + eps, float(track_wh[1]) + eps
    w2, h2 = float(det_wh[0]) + eps, float(det_wh[1]) + eps
    return abs(np.log(w2 / w1)) + abs(np.log(h2 / h1))


def _greedy_assignment(cost: np.ndarray, invalid_cost: float) -> List[Tuple[int, int]]:
    """Return list of (track_idx, det_idx) assignments."""
    n_trk, n_det = cost.shape
    pairs: List[Tuple[int, int]] = []
    if n_trk == 0 or n_det == 0:
        return pairs

    # Build candidate list
    cand: List[Tuple[float, int, int]] = []
    for i in range(n_trk):
        for j in range(n_det):
            c = float(cost[i, j])
            if c < invalid_cost:
                cand.append((c, i, j))
    cand.sort(key=lambda x: x[0])

    used_i = set()
    used_j = set()
    for c, i, j in cand:
        if i in used_i or j in used_j:
            continue
        used_i.add(i)
        used_j.add(j)
        pairs.append((i, j))
    return pairs


class DistanceTracker:
    """Distance-based multi-object tracker backend.

    Interface:
      step(detections, dt) -> List[Track]
    """

    def __init__(self, cfg: DistanceConfig):
        self._cfg = cfg
        self._tracks: List[_InternalTrack] = []
        self._next_id = 1

        # If SciPy is available, we can use Hungarian for optimal assignment.
        self._use_scipy = False
        try:
            from scipy.optimize import linear_sum_assignment  # type: ignore
            self._linear_sum_assignment = linear_sum_assignment
            self._use_scipy = True
        except Exception:
            self._linear_sum_assignment = None

    def step(self, detections: List[Detection], dt: float) -> List[Track]:
        dt = float(dt)
        dt = max(1e-6, dt)

        det_boxes = [d.box_xyxy.astype(float) for d in detections]
        det_centers = [ _xyxy_center(b) for b in det_boxes ]
        det_wh = [ _xyxy_wh(b) for b in det_boxes ]

        n_trk = len(self._tracks)
        n_det = len(det_boxes)

        # Build cost matrix
        invalid = 1e9
        cost = np.full((n_trk, n_det), invalid, dtype=float)

        for i, tr in enumerate(self._tracks):
            tr_pred = tr.predict_center(dt) if self._cfg.use_prediction else tr.center
            tr_last = tr.center
            tr_scale = _scale_for_norm(tr.wh, self._cfg.normalize)

            for j in range(n_det):
                # Robust distance: allow fallback to last center (helps on hard reversals)
                d_pred = float(np.linalg.norm(det_centers[j] - tr_pred))
                d_last = float(np.linalg.norm(det_centers[j] - tr_last))
                d = min(d_pred, d_last) if self._cfg.use_prediction else d_last

                if d > float(self._cfg.gate_px):
                    continue

                sp = _size_penalty(tr.wh, det_wh[j])
                c = (d / tr_scale) + float(self._cfg.size_weight) * sp
                cost[i, j] = c

        # Solve assignment
        matches: List[Tuple[int, int]] = []
        if n_trk > 0 and n_det > 0:
            if self._use_scipy and self._linear_sum_assignment is not None:
                row_ind, col_ind = self._linear_sum_assignment(cost)
                for i, j in zip(row_ind.tolist(), col_ind.tolist()):
                    if float(cost[i, j]) < invalid:
                        matches.append((i, j))
            else:
                matches = _greedy_assignment(cost, invalid_cost=invalid)

        matched_trk = {i for i, _ in matches}
        matched_det = {j for _, j in matches}

        # Update matched tracks
        for i, j in matches:
            tr = self._tracks[i]
            new_box = det_boxes[j]
            new_center = det_centers[j]
            new_wh = det_wh[j]

            # velocity update (pixels/second)
            meas_v = (new_center - tr.center) / dt
            # clip speed
            spd = float(np.linalg.norm(meas_v))
            max_spd = float(self._cfg.max_speed_px_s)
            if spd > max_spd and spd > 1e-6:
                meas_v = meas_v * (max_spd / spd)

            a = float(self._cfg.vel_alpha)
            tr.vel = (1.0 - a) * tr.vel + a * meas_v

            tr.center = new_center
            tr.wh = new_wh
            tr.box = new_box
            tr.age += 1
            tr.hits += 1
            tr.misses = 0

        # Predict / age unmatched tracks
        survivors: List[_InternalTrack] = []
        for i, tr in enumerate(self._tracks):
            if i in matched_trk:
                survivors.append(tr)
            else:
                tr.mark_missed(dt=dt, use_prediction=self._cfg.use_prediction)
                if tr.misses <= int(self._cfg.max_age):
                    survivors.append(tr)
                # else drop

        self._tracks = survivors

        # Create new tracks from unmatched detections
        for j in range(n_det):
            if j in matched_det:
                continue
            tr = _InternalTrack(self._next_id, det_boxes[j])
            self._next_id += 1
            self._tracks.append(tr)

        # Output tracks: only those updated this step (misses==0) and matured enough
        out: List[Track] = []
        for tr in self._tracks:
            if tr.misses != 0:
                continue
            if tr.hits < int(self._cfg.min_steps_alive):
                continue
            out.append(
                Track(
                    track_id=str(tr.id),
                    box_xyxy=tr.box.copy(),
                    age=int(tr.age),
                    hits=int(tr.hits),
                )
            )

        return out
