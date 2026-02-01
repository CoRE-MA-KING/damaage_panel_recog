from __future__ import annotations

import csv
import os
import time
from typing import Any, List, Tuple


def default_motion_log_path(prefix_dir: str = "logs", basename_prefix: str = "motion_log") -> str:
    """Return default CSV log path like `logs/motion_log_YYYYmmdd_HHMMSS.csv`."""
    ts = time.strftime("%Y%m%d_%H%M%S")
    return os.path.join(prefix_dir, f"{basename_prefix}_{ts}.csv")


class MotionLogger:
    """Low-overhead CSV logger to estimate per-frame motion without video recording.

    Design goals:
    - *Very low overhead* (no video encoding)
    - *Portable* (std-lib only)
    - Easy to post-process in Python/Excel

    What it logs:
    - per-frame selected target (x,y) and its bbox size (w,h) if available
    - dx, dy, and instantaneous speed (px/s)
    - counts of detections/tracks to help sanity-check

    Notes:
    - It buffers rows and flushes every `flush_every` frames.
    - If target is not available for a frame, x/y/w/h become empty.
    """

    def __init__(self, path: str, flush_every: int = 60) -> None:
        self.path = path
        self.flush_every = max(1, int(flush_every))
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        self._fp = open(path, "w", newline="", encoding="utf-8")
        self._wr = csv.writer(self._fp)
        self._wr.writerow(
            [
                "frame_idx",
                "t_sec",
                "dt_sec",
                "n_pairs",
                "n_tracks",
                "source",
                "x",
                "y",
                "w",
                "h",
                "dx",
                "dy",
                "speed_px_per_s",
            ]
        )

        self._buf: List[List[Any]] = []
        self._prev_xy: Tuple[int, int] | None = None
        self._count = 0
        self._sum_speed = 0.0
        self._sum_abs_dx = 0.0
        self._sum_abs_dy = 0.0

    def log(
        self,
        *,
        frame_idx: int,
        t_sec: float,
        dt_sec: float,
        n_pairs: int,
        n_tracks: int,
        source: str,
        xy: Tuple[int, int] | None,
        wh: Tuple[int, int] | None,
    ) -> None:
        if xy is None:
            x = y = None
        else:
            x, y = int(xy[0]), int(xy[1])

        if wh is None:
            w = h = None
        else:
            w, h = int(wh[0]), int(wh[1])

        dx = dy = None
        speed = None
        if xy is not None and self._prev_xy is not None:
            dx = int(xy[0]) - int(self._prev_xy[0])
            dy = int(xy[1]) - int(self._prev_xy[1])
            if dt_sec > 0:
                speed = (float(dx * dx + dy * dy) ** 0.5) / float(dt_sec)
                self._sum_speed += float(speed)
                self._sum_abs_dx += abs(float(dx))
                self._sum_abs_dy += abs(float(dy))
                self._count += 1

        self._prev_xy = (int(xy[0]), int(xy[1])) if xy is not None else None

        self._buf.append(
            [
                int(frame_idx),
                float(t_sec),
                float(dt_sec),
                int(n_pairs),
                int(n_tracks),
                str(source),
                x,
                y,
                w,
                h,
                dx,
                dy,
                speed,
            ]
        )
        if len(self._buf) >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        if not self._buf:
            return
        self._wr.writerows(self._buf)
        self._buf.clear()
        self._fp.flush()

    def close(self) -> None:
        try:
            self.flush()
        finally:
            self._fp.close()

    def summary_text(self) -> str:
        if self._count <= 0:
            return "(no valid motion samples)"
        mean_speed = self._sum_speed / float(self._count)
        mean_abs_dx = self._sum_abs_dx / float(self._count)
        mean_abs_dy = self._sum_abs_dy / float(self._count)
        return (
            f"samples={self._count}, mean_speed={mean_speed:.2f}px/s, "
            f"mean|dx|={mean_abs_dx:.3f}px, mean|dy|={mean_abs_dy:.3f}px"
        )
