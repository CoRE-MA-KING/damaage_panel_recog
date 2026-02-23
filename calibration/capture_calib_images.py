#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import os
import re

import cv2


DEFAULT_FPS = 90


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def open_cap(device: str, width: int | None, height: int | None, fps: int | None, fourcc: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera: {device}")
    if fourcc:
        try:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        except Exception:
            pass
    if width is not None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
    if height is not None:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
    if fps is not None:
        cap.set(cv2.CAP_PROP_FPS, int(fps))
    return cap


def next_index(out_dir: str, regex: str) -> int:
    max_n = 0
    if not os.path.isdir(out_dir):
        return 1
    pat = re.compile(regex)
    for name in os.listdir(out_dir):
        m = pat.match(name)
        if m:
            max_n = max(max_n, int(m.group(1)))
    return max_n + 1


def pick_common_or_specific(common: int | None, specific: int | None) -> int | None:
    return common if specific is None else specific


def pick_fourcc(common: str, specific: str | None) -> str:
    if specific in (None, ""):
        return common
    return specific


def run_pair(
    main_device: str,
    panel_recog_device: str,
    out_main: str,
    out_panel_recog: str,
    width_main: int | None,
    height_main: int | None,
    fps_main: int | None,
    fourcc_main: str,
    width_panel_recog: int | None,
    height_panel_recog: int | None,
    fps_panel_recog: int | None,
    fourcc_panel_recog: str,
) -> None:
    ensure_dir(out_main)
    ensure_dir(out_panel_recog)

    cap_main = open_cap(main_device, width_main, height_main, fps_main, fourcc_main)
    cap_panel = open_cap(panel_recog_device, width_panel_recog, height_panel_recog, fps_panel_recog, fourcc_panel_recog)

    idx_main = next_index(out_main, r"pair_(\d+)_main_camera\.png")
    idx_panel = next_index(out_panel_recog, r"pair_(\d+)_panel_recog_camera\.png")
    idx = max(idx_main, idx_panel)
    shown_size = False
    try:
        while True:
            ok_main, frame_main = cap_main.read()
            ok_panel, frame_panel = cap_panel.read()
            if ok_main:
                cv2.imshow("pair:main_camera (s=save, q=quit)", frame_main)
            if ok_panel:
                cv2.imshow("pair:panel_recog_camera (s=save, q=quit)", frame_panel)

            if not shown_size and ok_main and ok_panel:
                h1, w1 = frame_main.shape[:2]
                h2, w2 = frame_panel.shape[:2]
                print(f"[INFO] actual size main_camera: {w1}x{h1}, panel_recog_camera: {w2}x{h2}")
                shown_size = True

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s") and ok_main and ok_panel:
                p_main = os.path.join(out_main, f"pair_{idx:04d}_main_camera.png")
                p_panel = os.path.join(out_panel_recog, f"pair_{idx:04d}_panel_recog_camera.png")
                cv2.imwrite(p_main, frame_main)
                cv2.imwrite(p_panel, frame_panel)
                print(f"saved: {p_main}")
                print(f"saved: {p_panel}")
                idx += 1
    finally:
        cap_main.release()
        cap_panel.release()
        cv2.destroyAllWindows()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["pair"], default="pair", help=argparse.SUPPRESS)

    ap.add_argument("--main-device", required=True, help="main_camera device (e.g. /dev/video4)")
    ap.add_argument("--panel-recog-device", required=True, help="panel_recog_camera device (e.g. /dev/video6)")
    ap.add_argument("--out-main", default="calib/main_camera")
    ap.add_argument("--out-panel-recog", default="calib/panel_recog_camera")
    ap.add_argument("--width", type=int, default=None, help="common width for both cameras")
    ap.add_argument("--height", type=int, default=None, help="common height for both cameras")
    ap.add_argument(
        "--fps",
        type=int,
        default=DEFAULT_FPS,
        help=f"common fps for both cameras (default: {DEFAULT_FPS})",
    )
    ap.add_argument("--fourcc", default="", help="common fourcc for both cameras")

    ap.add_argument("--width-main", type=int, default=None)
    ap.add_argument("--height-main", type=int, default=None)
    ap.add_argument("--fps-main", type=int, default=None)
    ap.add_argument("--fourcc-main", default=None)
    ap.add_argument("--width-panel-recog", type=int, default=None)
    ap.add_argument("--height-panel-recog", type=int, default=None)
    ap.add_argument("--fps-panel-recog", type=int, default=None)
    ap.add_argument("--fourcc-panel-recog", default=None)

    args = ap.parse_args()

    print("[press key]\n 's': save the picture\n 'q': quit")

    width_main = pick_common_or_specific(args.width, args.width_main)
    width_panel = pick_common_or_specific(args.width, args.width_panel_recog)
    height_main = pick_common_or_specific(args.height, args.height_main)
    height_panel = pick_common_or_specific(args.height, args.height_panel_recog)
    fps_main = pick_common_or_specific(args.fps, args.fps_main)
    fps_panel = pick_common_or_specific(args.fps, args.fps_panel_recog)
    fourcc_main = pick_fourcc(args.fourcc, args.fourcc_main)
    fourcc_panel = pick_fourcc(args.fourcc, args.fourcc_panel_recog)

    # This tool assumes both cameras use the same image size in pair mode.
    if width_main is not None and width_panel is not None and int(width_main) != int(width_panel):
        ap.error(
            "pair mode assumes the same resolution for both cameras: "
            f"--width-main({width_main}) != --width-panel-recog({width_panel})"
        )
    if height_main is not None and height_panel is not None and int(height_main) != int(height_panel):
        ap.error(
            "pair mode assumes the same resolution for both cameras: "
            f"--height-main({height_main}) != --height-panel-recog({height_panel})"
        )

    run_pair(
        main_device=args.main_device,
        panel_recog_device=args.panel_recog_device,
        out_main=args.out_main,
        out_panel_recog=args.out_panel_recog,
        width_main=width_main,
        height_main=height_main,
        fps_main=fps_main,
        fourcc_main=fourcc_main,
        width_panel_recog=width_panel,
        height_panel_recog=height_panel,
        fps_panel_recog=fps_panel,
        fourcc_panel_recog=fourcc_panel,
    )


if __name__ == "__main__":
    main()
