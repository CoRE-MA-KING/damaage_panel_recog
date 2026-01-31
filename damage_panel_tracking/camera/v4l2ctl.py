from __future__ import annotations

import re
import shutil
import subprocess
from typing import Any


def dev_to_path(dev: Any) -> str:
    """Normalize device spec (0, '0', '/dev/video0') to '/dev/videoX' for v4l2-ctl."""
    if isinstance(dev, int):
        return f"/dev/video{dev}"
    if isinstance(dev, str):
        s = dev.strip()
        if s.isdigit():
            return f"/dev/video{int(s)}"
        if re.match(r"^/dev/video\d+$", s):
            return s
    return ""


def has_v4l2_ctl() -> bool:
    return shutil.which("v4l2-ctl") is not None


def v4l2_set(dev_path: str, name: str, value: Any) -> bool:
    """Set a V4L2 control. Returns True on success, False otherwise."""
    if not dev_path or not has_v4l2_ctl():
        return False
    try:
        r = subprocess.run(
            ["v4l2-ctl", "-d", dev_path, f"--set-ctrl={name}={value}"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return r.returncode == 0
    except Exception:
        return False
