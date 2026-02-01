from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ZenohConfig:
    key_prefix: str = ""
    publish_key: str = "damagepanel"


class ZenohPublisher:
    """Small wrapper to keep zenoh optional."""
    def __init__(self, cfg: ZenohConfig):
        try:
            import zenoh  # type: ignore
        except Exception as e:
            raise RuntimeError("zenoh is not installed. `pip install zenoh`") from e

        self._session = zenoh.open(zenoh.Config())
        prefix = cfg.key_prefix.strip("/")
        key_expr = f"{prefix}/{cfg.publish_key}" if prefix else cfg.publish_key
        self._pub = self._session.declare_publisher(key_expr)

    def put(self, payload: str) -> None:
        self._pub.put(payload)

    def close(self) -> None:
        try:
            self._session.close()
        except Exception:
            pass
