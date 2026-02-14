import os
from pathlib import Path
from typing import Callable

import zenoh
from google.protobuf.message import Message


class ZenohSession:
    """Small wrapper to keep zenoh optional."""

    def __init__(self) -> None:
        self._session = zenoh.open(
            zenoh.Config.from_file(
                Path(os.getenv("XDG_CONFIG_HOME", Path.home() / ".config"))
                / "roboapp"
                / "zenoh.json5"
            )
        )

        self._pub: dict[str, zenoh.Publisher] = {}

    def create_publisher(self, key: str) -> None:
        self._pub[key] = self._session.declare_publisher(key)

    def create_subscriber(
        self, key: str, callback: Callable[[zenoh.Sample], None]
    ) -> None:
        self._session.declare_subscriber(key, callback)

    def put(self, key: str, payload: Message) -> None:
        if key in self._pub:
            self._pub[key].put(payload.SerializeToString())

    def close(self) -> None:
        try:
            self._session.close()  # type: ignore
        except Exception:
            pass
