"""Utility helpers (small, dependency-light).

- motion_logger: CSV logging utilities for estimating per-frame motion without video recording.
"""

from .motion_logger import MotionLogger, default_motion_log_path

__all__ = [
    "MotionLogger",
    "default_motion_log_path",
]
