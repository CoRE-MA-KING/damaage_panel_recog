from __future__ import annotations

from typing import List
from .base import Detection, Track


class NoopTracker:
    """Tracker that always returns no tracks (tracking disabled)."""

    def step(self, detections: List[Detection], dt: float) -> List[Track]:
        # インターフェース互換を保ちつつ、追跡は意図的に無効化する。
        return []
