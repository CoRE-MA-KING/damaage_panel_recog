from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable

from google.protobuf.message import Message


class ZenohSession:
    """Small wrapper to keep zenoh usage isolated."""

    def __init__(self) -> None:
        try:
            import zenoh  # type: ignore
        except Exception as e:
            raise RuntimeError("Zenoh is required when publish/subscribe is enabled.") from e

        self._session = zenoh.open(
            zenoh.Config.from_file(
                Path(os.getenv("XDG_CONFIG_HOME", Path.home() / ".config"))
                / "roboapp"
                / "zenoh.json5"
            )
        )

        self._pub: dict[str, Any] = {}
        self._sub: list[Any] = []

    def create_publisher(self, key: str) -> None:
        self._pub[key] = self._session.declare_publisher(key)

    def create_subscriber(self, key: str, callback: Callable[[Any], None]) -> None:
        self._sub.append(self._session.declare_subscriber(key, callback))

    def put(self, key: str, payload: Message) -> None:
        if key in self._pub:
            self._pub[key].put(payload.SerializeToString())

    def close(self) -> None:
        try:
            self._session.close()  # type: ignore
        except Exception:
            pass
