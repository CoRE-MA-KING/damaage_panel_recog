from __future__ import annotations

import re
import os
import shutil
import stat
import subprocess
import time
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

_V4L2_CTL_CANDIDATES: tuple[str, ...] = (
    "/usr/bin/v4l2-ctl",
    "/usr/local/bin/v4l2-ctl",
    "/bin/v4l2-ctl",
    "/usr/sbin/v4l2-ctl",
    "/sbin/v4l2-ctl",
)


def _resolve_v4l2_ctl_path() -> str:
    # PATHが限定される環境でも実行できるよう、一般的な絶対パスを順に試す。
    env_path = os.getenv("V4L2_CTL_PATH", "").strip()
    if env_path and os.path.isfile(env_path) and os.access(env_path, os.X_OK):
        return env_path

    from_path = shutil.which("v4l2-ctl")
    if from_path:
        return from_path

    for path in _V4L2_CTL_CANDIDATES:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    return ""


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
        if _is_video4linux_node(real):
            return real

        # 既存挙動の後方互換: /dev/videoX 形式は未解決でもそのまま返す。
        if re.match(r"^/dev/video\d+$", s):
            return s
    return ""


def has_v4l2_ctl() -> bool:
    # この環境でv4l2-ctlが利用可能か確認する。
    return bool(_resolve_v4l2_ctl_path())


def v4l2_list_ctrls(dev_path: str) -> Dict[str, Dict[str, Any]]:
    """List V4L2 controls with range/default/current values."""
    # v4l2-ctlの出力を解析し、トラックバー範囲計算に使う。
    v4l2_ctl_path = _resolve_v4l2_ctl_path()
    if not dev_path or not v4l2_ctl_path:
        return {}
    try:
        result = subprocess.run(
            [v4l2_ctl_path, "-d", dev_path, "--list-ctrls"],
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


def v4l2_get(dev_path: str, names: list[str] | tuple[str, ...]) -> Dict[str, int]:
    """Read current V4L2 control values for the requested names."""
    v4l2_ctl_path = _resolve_v4l2_ctl_path()
    if not dev_path or not v4l2_ctl_path:
        return {}

    req_names = [str(name).strip() for name in names if str(name).strip()]
    if not req_names:
        return {}

    try:
        result = subprocess.run(
            [v4l2_ctl_path, "-d", dev_path, f"--get-ctrl={','.join(req_names)}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
        )
    except Exception:
        return {}

    values: Dict[str, int] = {}
    for line in (result.stdout or "").splitlines():
        if ":" not in line:
            continue
        name, raw = line.split(":", 1)
        match = re.search(r"[-]?\d+", raw)
        if not match:
            continue
        values[name.strip()] = int(match.group(0))
    return values


def v4l2_set(dev_path: str, name: str, value: Any) -> bool:
    """Set a V4L2 control. Returns True on success, False otherwise."""
    # v4l2-ctlで1つのカメラ制御をベストエフォートで適用する。
    v4l2_ctl_path = _resolve_v4l2_ctl_path()
    if not dev_path or not v4l2_ctl_path:
        return False
    cmd = [v4l2_ctl_path, "-d", dev_path, f"--set-ctrl={name}={value}"]
    retries = 2
    for attempt in range(retries + 1):
        try:
            r = subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            if r.returncode == 0:
                return True
        except Exception:
            pass
        if attempt < retries:
            time.sleep(0.01)
    return False
