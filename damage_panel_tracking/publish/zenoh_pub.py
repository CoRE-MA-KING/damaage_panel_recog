from __future__ import annotations

import multiprocessing as mp
import os
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


_STOP_TOKEN = ("__STOP__",)


def _build_payload_from_payload(payload_data: tuple[bool, int, int, int, int, int]) -> Message:
    from msg import DamagePanelTargetMessage, Target
    from protovalidate import validate

    has_target, x, y, width, height, distance = payload_data
    if not has_target:
        return DamagePanelTargetMessage(target=None)

    msg = DamagePanelTargetMessage(
        target=Target(
            x=int(x),
            y=int(y),
            width=int(width),
            height=int(height),
            distance=int(distance),
        )
    )
    validate(msg)
    return msg


def _publisher_process_main(
    recv_conn: Any,
    *,
    key: str,
    max_hz: float,
    drop_if_congested: bool,
    express: bool,
) -> None:
    session: ZenohSession | None = None
    interval_sec = (1.0 / max_hz) if max_hz > 0.0 else 0.0
    last_pub_t = 0.0

    try:
        session = ZenohSession()
        session.create_publisher(
            key,
            drop_if_congested=drop_if_congested,
            express=express,
        )
    except Exception as e:
        print(f"[WARN] publisher process init failed: {e}")
        return

    while True:
        try:
            item = recv_conn.recv()
        except EOFError:
            break
        if item == _STOP_TOKEN:
            break

        # Keep only the latest payload waiting in the pipe.
        while recv_conn.poll():
            try:
                newer = recv_conn.recv()
            except EOFError:
                item = _STOP_TOKEN
                break
            if newer == _STOP_TOKEN:
                item = _STOP_TOKEN
                break
            item = newer

        if item == _STOP_TOKEN:
            break

        if interval_sec > 0.0 and last_pub_t > 0.0:
            now = time.monotonic()
            remain = interval_sec - (now - last_pub_t)
            if remain > 0.0:
                time.sleep(remain)

        try:
            payload = _build_payload_from_payload(item)
            session.put(key, payload)
            last_pub_t = time.monotonic()
        except Exception as e:
            print(f"[WARN] async publish failed: {e}")

    if session is not None:
        session.close()
    try:
        recv_conn.close()
    except Exception:
        pass


class LatestFramePublisher:
    """Process-based publisher that keeps only the latest pending payload."""

    def __init__(
        self,
        *,
        key: str,
        drop_if_congested: bool = True,
        express: bool = True,
        max_hz: float = 0.0,
    ) -> None:
        self._ctx = mp.get_context("spawn")
        self._recv_conn, self._send_conn = self._ctx.Pipe(duplex=False)
        try:
            os.set_blocking(self._send_conn.fileno(), False)
        except (AttributeError, OSError):
            pass
        self._closed = False
        self._proc = self._ctx.Process(
            target=_publisher_process_main,
            kwargs={
                "recv_conn": self._recv_conn,
                "key": key,
                "max_hz": float(max_hz),
                "drop_if_congested": bool(drop_if_congested),
                "express": bool(express),
            },
            name="latest-frame-publisher-process",
            daemon=True,
        )
        self._proc.start()

    def _put_latest(self, payload_data: Any) -> None:
        try:
            self._send_conn.send(payload_data)
        except (BlockingIOError, BrokenPipeError, EOFError, OSError):
            pass

    def submit(self, payload_data: tuple[bool, int, int, int, int, int]) -> None:
        if self._closed:
            return
        if not self._proc.is_alive():
            return
        self._put_latest(payload_data)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True

        self._put_latest(_STOP_TOKEN)
        try:
            self._send_conn.close()
        except Exception:
            pass
        self._proc.join(timeout=3.0)
        if self._proc.is_alive():
            self._proc.terminate()
            self._proc.join(timeout=1.0)

        try:
            self._recv_conn.close()
        except Exception:
            pass
