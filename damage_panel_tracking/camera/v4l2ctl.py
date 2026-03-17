from __future__ import annotations

import re
import os
import shutil
import stat
import subprocess
from typing import Any, Dict


_CTRL_NUM_PATTERN = re.compile(
    r"^\s*([a-zA-Z0-9_]+)\s+0x[0-9a-fA-F]+\s+\(([^)]+)\)\s*:\s*"
    r".*min=([-]?\d+)\s+max=([-]?\d+)\s+step=([-]?\d+)\s+default=([-]?\d+)\s+value=([-]?\d+)"
    r"(?:\s+\(([^)]+)\))?"
    r"(?:\s+flags=(.+))?\s*$"
)
_CTRL_BOOL_PATTERN = re.compile(
    r"^\s*([a-zA-Z0-9_]+)\s+0x[0-9a-fA-F]+\s+\((bool)\)\s*:\s*"
    r".*default=([-]?\d+)\s+value=([-]?\d+)"
    r"(?:\s+flags=(.+))?\s*$"
)
_VIDEO_NODE_PATH_PATTERN = re.compile(r"^/dev/video\d+$")


def dev_to_path(dev: Any) -> str:
    """Normalize device spec (0, '0', '/dev/video0') to '/dev/videoX' for v4l2-ctl."""
    # 複数のデバイス表記を、実在するvideo4linuxノードへ正規化する。
    def _is_video4linux_node(path: str) -> bool:
        try:
            st = os.stat(path)
        except OSError:
            return False
        if not stat.S_ISCHR(st.st_mode):
            return False
        return os.major(st.st_rdev) == 81

    def _canonical_video_node(path: str) -> str:
        # /dev/recog_camera など同一ノードの別名入力を、/dev/videoXへ正規化する。
        if not _is_video4linux_node(path):
            return ""
        if _VIDEO_NODE_PATH_PATTERN.match(path):
            return path

        try:
            names = [n for n in os.listdir("/dev") if n.startswith("video") and n[5:].isdigit()]
        except OSError:
            return path

        for name in sorted(names, key=lambda n: int(n[5:])):
            candidate = f"/dev/{name}"
            try:
                if os.path.samefile(candidate, path):
                    return candidate
            except OSError:
                continue
        return path

    if isinstance(dev, int):
        return f"/dev/video{dev}"
    if isinstance(dev, str):
        s = dev.strip()
        if not s:
            return ""
        if s.isdigit():
            s = f"/dev/video{int(s)}"

        # /dev/camera_front や /dev/v4l/by-id/* のようなシンボリックリンクを解決する。
        real = os.path.realpath(s)
        canonical = _canonical_video_node(real)
        if canonical:
            return canonical

        # 既存挙動の後方互換: /dev/videoX 形式は未解決でもそのまま返す。
        if _VIDEO_NODE_PATH_PATTERN.match(s):
            return s
    return ""


def has_v4l2_ctl() -> bool:
    # この環境でv4l2-ctlが利用可能か確認する。
    return shutil.which("v4l2-ctl") is not None


def v4l2_list_ctrls(dev_path: str) -> Dict[str, Dict[str, Any]]:
    """List V4L2 controls with range/default/current values."""
    # v4l2-ctlの出力を解析し、トラックバー範囲計算に使う。
    if not dev_path or not has_v4l2_ctl():
        return {}
    try:
        result = subprocess.run(
            ["v4l2-ctl", "-d", dev_path, "--list-ctrls"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
        )
    except Exception:
        return {}

    out = result.stdout or ""
    info: Dict[str, Dict[str, Any]] = {}
    for line in out.splitlines():
        match_num = _CTRL_NUM_PATTERN.match(line)
        if match_num:
            name = match_num.group(1)
            flags = (match_num.group(9) or "").strip()
            info[name] = {
                "type": match_num.group(2),
                "min": int(match_num.group(3)),
                "max": int(match_num.group(4)),
                "step": int(match_num.group(5)),
                "default": int(match_num.group(6)),
                "value": int(match_num.group(7)),
                "flags": flags,
                "inactive": "inactive" in flags,
            }
            continue

        match_bool = _CTRL_BOOL_PATTERN.match(line)
        if match_bool:
            name = match_bool.group(1)
            flags = (match_bool.group(5) or "").strip()
            info[name] = {
                "type": match_bool.group(2),
                "min": 0,
                "max": 1,
                "step": 1,
                "default": int(match_bool.group(3)),
                "value": int(match_bool.group(4)),
                "flags": flags,
                "inactive": "inactive" in flags,
            }
    return info


def v4l2_set(dev_path: str, name: str, value: Any) -> bool:
    """Set a V4L2 control. Returns True on success, False otherwise."""
    # v4l2-ctlで1つのカメラ制御をベストエフォートで適用する。
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
