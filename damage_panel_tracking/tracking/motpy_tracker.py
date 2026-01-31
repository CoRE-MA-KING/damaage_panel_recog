from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .base import Detection, Track


@dataclass
class MotpyConfig:
    order_pos: int = 2
    dim_pos: int = 2
    order_size: int = 0
    dim_size: int = 2
    q_var_pos: float = 5000.0
    r_var_pos: float = 0.1
    min_iou: Optional[float] = None
    max_staleness: Optional[int] = 4
    min_steps_alive: int = 2


class MotpyTracker:
    """motpy wrapper implementing our tracker interface."""

    def __init__(self, cfg: MotpyConfig, dt: float):
        try:
            from motpy import MultiObjectTracker  # type: ignore
        except Exception as e:
            raise RuntimeError("motpy is not installed. `pip install motpy`") from e

        model_spec = {
            "order_pos": cfg.order_pos,
            "dim_pos": cfg.dim_pos,
            "order_size": cfg.order_size,
            "dim_size": cfg.dim_size,
            "q_var_pos": cfg.q_var_pos,
            "r_var_pos": cfg.r_var_pos,
        }
        self._cfg = cfg
        self._tracker = MultiObjectTracker(dt=dt, model_spec=model_spec)

        if cfg.min_iou is not None and hasattr(self._tracker, "min_iou"):
            self._tracker.min_iou = cfg.min_iou
        if cfg.max_staleness is not None and hasattr(self._tracker, "max_staleness"):
            self._tracker.max_staleness = cfg.max_staleness

    def step(self, detections: List[Detection], dt: float) -> List[Track]:
        from motpy import Detection as MotpyDet  # type: ignore

        try:
            self._tracker.dt = float(dt)
        except Exception:
            pass

        mot_dets = [
            MotpyDet(box=d.box_xyxy.astype(float), score=float(d.score), class_id=int(d.class_id))
            for d in detections
        ]
        self._tracker.step(detections=mot_dets)

        tracks = self._tracker.active_tracks(min_steps_alive=int(self._cfg.min_steps_alive))
        out: List[Track] = []
        for t in tracks:
            out.append(
                Track(
                    track_id=str(t.id),
                    box_xyxy=np.array(t.box, dtype=float),
                    age=int(getattr(t, "age", 0)),
                    hits=int(getattr(t, "steps_alive", 0)),
                )
            )
        return out
