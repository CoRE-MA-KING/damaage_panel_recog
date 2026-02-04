from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


@dataclass
class ZenohConfig:
    publish_key: str = "damagepanel"


class ZenohPublisher:
    """Small wrapper to keep zenoh optional."""

    def __init__(self, cfg: ZenohConfig):
        try:
            import zenoh  # type: ignore
        except Exception as e:
            raise RuntimeError("zenoh is not installed. `pip install zenoh`") from e

        self._session = zenoh.open(
            zenoh.Config.from_file(
                Path(os.getenv("XDG_CONFIG_HOME", Path.home() / ".config"))
                / "roboapp"
                / "zenoh.json5"
            )
        )
        self._pub = self._session.declare_publisher(cfg.publish_key)

    def put(self, payload: str) -> None:
        self._pub.put(payload)

    def close(self) -> None:
        try:
            self._session.close()
        except Exception:
            pass
