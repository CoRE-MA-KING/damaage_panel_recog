from __future__ import annotations

import glob
import os
import sys


def _first_existing_font_dir() -> str:
    # Qtが参照可能なLinux標準フォントディレクトリを探索する。
    candidates = (
        "/usr/share/fonts/truetype/dejavu",
        "/usr/share/fonts/truetype/noto",
        "/usr/share/fonts/opentype/noto",
        "/usr/share/fonts/truetype",
        "/usr/share/fonts",
    )
    for path in candidates:
        if _font_files_in_dir(path):
            return path
    return ""


def _font_files_in_dir(path: str) -> list[str]:
    # ディレクトリ直下にフォントファイルがあるか確認する。
    if not path or not os.path.isdir(path):
        return []

    patterns = (
        "*.ttf",
        "*.ttc",
        "*.otf",
        "*.pfa",
        "*.pfb",
    )
    found: list[str] = []
    for pat in patterns:
        found.extend(glob.glob(os.path.join(path, pat)))
        if found:
            break
    return found


def configure_qt_fontdir() -> bool:
    """Set QT_QPA_FONTDIR when missing (Linux only)."""
    # OpenCV(Qt)のトラックバーラベル欠落を回避するためフォント探索先を補う。
    if not sys.platform.startswith("linux"):
        return False
    current = os.environ.get("QT_QPA_FONTDIR", "")
    if current and _font_files_in_dir(current):
        return False
    fallback = _first_existing_font_dir()
    if not fallback:
        return False
    os.environ["QT_QPA_FONTDIR"] = fallback
    return True
