import os
from pathlib import Path


def get_config_path() -> Path:
    """設定ファイルのパスを取得する"""
    return Path(os.getenv("XDG_CONFIG_HOME", Path.home() / ".config")) / "roboapp"
