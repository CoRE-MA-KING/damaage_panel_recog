from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Protocol
import numpy as np


@dataclass(frozen=True)
class Detection:
    """Detection input for trackers (bbox only, no pixels)."""
    box_xyxy: np.ndarray  # float (x1,y1,x2,y2)
    score: float = 1.0
    class_id: int = 0


@dataclass(frozen=True)
class Track:
    """Tracker output."""
    track_id: str
    box_xyxy: np.ndarray  # float
    age: int
    hits: int


class MultiObjectTracker(Protocol):
    """Interface to allow swapping motpy with your own SORT implementation later."""
    def step(self, detections: List[Detection], dt: float) -> List[Track]:
        ...
