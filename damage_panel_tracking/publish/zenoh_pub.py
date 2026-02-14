from __future__ import annotations

import os
import threading
import time
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

        self._zenoh = zenoh
        self._session = zenoh.open(
            zenoh.Config.from_file(
                Path(os.getenv("XDG_CONFIG_HOME", Path.home() / ".config"))
                / "roboapp"
                / "zenoh.json5"
            )
        )

        self._pub: dict[str, Any] = {}
        self._sub: list[Any] = []

    def create_publisher(
        self,
        key: str,
        *,
        drop_if_congested: bool = False,
        express: bool = False,
    ) -> None:
        kwargs: dict[str, Any] = {}
        if drop_if_congested:
            kwargs["congestion_control"] = self._zenoh.CongestionControl.DROP
        if express:
            kwargs["express"] = True
        self._pub[key] = self._session.declare_publisher(key, **kwargs)

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


class LatestFramePublisher:
    """Background publisher that keeps only the latest pending item."""

    def __init__(
        self,
        *,
        session: ZenohSession,
        key: str,
        build_payload: Callable[[Any], Message],
        max_hz: float = 0.0,
    ) -> None:
        self._session = session
        self._key = key
        self._build_payload = build_payload
        self._interval_sec = (1.0 / max_hz) if max_hz > 0.0 else 0.0

        self._cond = threading.Condition()
        self._latest: Any | None = None
        self._has_item = False
        self._stopping = False
        self._last_pub_t = 0.0

        self._thread = threading.Thread(
            target=self._run,
            name="latest-frame-publisher",
            daemon=True,
        )
        self._thread.start()

    def submit(self, item: Any) -> None:
        with self._cond:
            self._latest = item
            self._has_item = True
            self._cond.notify()

    def close(self) -> None:
        with self._cond:
            self._stopping = True
            self._cond.notify_all()
        self._thread.join(timeout=3.0)

    def _run(self) -> None:
        while True:
            with self._cond:
                while not self._has_item and not self._stopping:
                    self._cond.wait()

                if self._stopping and not self._has_item:
                    return

                if self._stopping:
                    item = self._latest
                    self._latest = None
                    self._has_item = False
                elif self._interval_sec > 0.0 and self._last_pub_t > 0.0:
                    now = time.monotonic()
                    remain = self._interval_sec - (now - self._last_pub_t)
                    if remain > 0.0:
                        self._cond.wait(timeout=remain)
                        continue
                    item = self._latest
                    self._latest = None
                    self._has_item = False
                else:
                    item = self._latest
                    self._latest = None
                    self._has_item = False

            if item is None:
                continue
            try:
                payload = self._build_payload(item)
                self._session.put(self._key, payload)
            except Exception as e:
                print(f"[WARN] async publish failed: {e}")
                continue

            self._last_pub_t = time.monotonic()
            if self._stopping:
                return
