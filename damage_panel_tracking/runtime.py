from __future__ import annotations

from typing import Any, Dict, Tuple

import cv2
from protovalidate import validate
from msg import Target, DamagePanelTargetMessage

from .pipeline import FrameResult
from .publish.zenoh_pub import ZenohSession
from .ui.draw import (
    TrackVizState,
    draw_detection_pair,
    draw_fps,
    draw_mode,
    draw_target,
    draw_tracks,
)
from .utils.motion_logger import MotionLogger, default_motion_log_path


def create_motion_logger(logging_cfg: Dict[str, Any]) -> MotionLogger | None:
    if not logging_cfg.get("enabled", False):
        return None

    log_path = logging_cfg.get("path") or default_motion_log_path()
    flush_every = int(logging_cfg.get("flush_every", 60))
    motion_logger = MotionLogger(str(log_path), flush_every=flush_every)
    print(f"[INFO] motion log enabled: {motion_logger.path}")
    return motion_logger


def log_motion_sample(
    motion_logger: MotionLogger | None,
    frame_idx: int,
    now: float,
    dt: float,
    result: FrameResult,
) -> None:
    if motion_logger is None:
        return

    src = "none"
    xy = None
    wh = None
    if result.chosen_from_tracks and result.selected_track is not None:
        src = "track"
        xy = result.target
        x1, y1, x2, y2 = result.selected_track.box_xyxy
        wh = (int(x2 - x1), int(y2 - y1))
    elif (not result.chosen_from_tracks) and result.selected_pair is not None:
        src = "det"
        xy = result.target
        _, _, uw, uh = result.selected_pair.union_xywh
        wh = (int(uw), int(uh))

    motion_logger.log(
        frame_idx=frame_idx,
        t_sec=now,
        dt_sec=dt,
        n_pairs=len(result.pairs),
        n_tracks=len(result.tracks),
        source=src,
        xy=xy,
        wh=wh,
    )


def render_frame(
    frame,
    *,
    result: FrameResult,
    tracking_enabled: bool,
    tracker_available: bool,
    track_color_bgr: Tuple[int, int, int],
    history: TrackVizState,
    fps: float,
    win_name: str,
) -> bool:
    if not tracking_enabled or not tracker_available:
        for pair in result.pairs:
            draw_detection_pair(frame, pair)
    else:
        draw_tracks(
            frame,
            result.tracks,
            result.pairs,
            track_color=track_color_bgr,
            history=history,
        )

    draw_target(frame, result.target, from_tracks=result.chosen_from_tracks)
    if tracking_enabled and tracker_available:
        mode = "TRACK=ON"
    elif tracking_enabled:
        mode = "TRACK=ON (backend missing!)"
    else:
        mode = "TRACK=OFF"
    draw_mode(frame, mode)
    draw_fps(frame, fps)

    cv2.imshow(win_name, frame)
    return bool(cv2.waitKey(1) & 0xFF == ord("q"))


def result_to_target(result: FrameResult) -> DamagePanelTargetMessage:
    msg = DamagePanelTargetMessage(target=None)
    
    if result.chosen_from_tracks and result.selected_track is not None:
        tx, ty = result.target
        x1, y1, x2, y2 = result.selected_track.box_xyxy
        msg.target=Target(
            x=int(tx),
            y=int(ty),
            distance=0,
            width=max(0, int(x2 - x1)),
            height=max(0, int(y2 - y1)),
        )

    if result.selected_pair is not None:
        tx, ty = result.target
        _, _, uw, uh = result.selected_pair.union_xywh
        msg.target=Target(
            x=int(tx),
            y=int(ty),
            distance=0,
            width=int(uw),
            height=int(uh),
        )

    try:
        validate(msg)

    except Exception as e:
        print(f"[WARN] target validation failed: {e}")
        return DamagePanelTargetMessage(target=None)
    return msg

def close_motion_logger(motion_logger: MotionLogger | None) -> None:
    if motion_logger is None:
        return
    try:
        motion_logger.close()
        print(
            f"[INFO] motion log saved: {motion_logger.path} ({motion_logger.summary_text()})"
        )
    except Exception as e:
        print(f"[WARN] motion logger close failed: {e}")


def close_publisher(session: ZenohSession | None) -> None:
    if session is None:
        return
    try:
        session.close()
    except Exception:
        pass
