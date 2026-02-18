#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import os
import re

import cv2


CAMERA_ROLES = ("main_camera", "panel_recog_camera")


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


def run_single(device: str, out_dir: str, camera_role: str, width: int | None, height: int | None, fps: int | None, fourcc: str) -> None:
    ensure_dir(out_dir)
    cap = open_cap(device, width=width, height=height, fps=fps, fourcc=fourcc)
    idx = next_index(out_dir, rf"single_(\d+)_{camera_role}\.png")
    shown_size = False
    win = f"single:{camera_role} (s=save, q=quit)"
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            if not shown_size:
                h, w = frame.shape[:2]
                print(f"[INFO] actual size ({camera_role}): {w}x{h}")
                shown_size = True
            cv2.imshow(win, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                path = os.path.join(out_dir, f"single_{idx:04d}_{camera_role}.png")
                cv2.imwrite(path, frame)
                print(f"saved: {path}")
                idx += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()


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
    ap.add_argument("--mode", choices=["single", "pair"], required=True)

    # single
    ap.add_argument("--device", default=None, help="single: camera device (e.g. /dev/video0)")
    ap.add_argument("--camera-role", choices=list(CAMERA_ROLES), default="panel_recog_camera")
    ap.add_argument("--out-dir", default=None, help="single: output dir (default: calib/<camera-role>)")
    ap.add_argument("--width", type=int, default=None)
    ap.add_argument("--height", type=int, default=None)
    ap.add_argument("--fps", type=int, default=None)
    ap.add_argument("--fourcc", default="")

    # pair (new names)
    ap.add_argument("--main-device", default=None, help="pair: main_camera device")
    ap.add_argument("--panel-recog-device", default=None, help="pair: panel_recog_camera device")
    ap.add_argument("--out-main", default="calib/main_camera")
    ap.add_argument("--out-panel-recog", default="calib/panel_recog_camera")
    ap.add_argument("--width-main", type=int, default=None)
    ap.add_argument("--height-main", type=int, default=None)
    ap.add_argument("--fps-main", type=int, default=None)
    ap.add_argument("--fourcc-main", default="")
    ap.add_argument("--width-panel-recog", type=int, default=None)
    ap.add_argument("--height-panel-recog", type=int, default=None)
    ap.add_argument("--fps-panel-recog", type=int, default=None)
    ap.add_argument("--fourcc-panel-recog", default="")

    # pair (compat aliases)
    ap.add_argument("--deviceA", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--deviceB", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--outA", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--outB", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--widthA", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--heightA", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--fpsA", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--fourccA", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--widthB", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--heightB", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--fpsB", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--fourccB", default=None, help=argparse.SUPPRESS)

    args = ap.parse_args()

    print("[press key]\n 's': save the picture\n 'q': quit")

    if args.mode == "single":
        if args.device is None:
            ap.error("single mode requires --device")
        out_dir = args.out_dir or os.path.join("calib", args.camera_role)
        run_single(
            device=args.device,
            out_dir=out_dir,
            camera_role=args.camera_role,
            width=args.width,
            height=args.height,
            fps=args.fps,
            fourcc=args.fourcc,
        )
        return

    main_device = args.main_device or args.deviceA
    panel_device = args.panel_recog_device or args.deviceB
    out_main = args.out_main if args.outA is None else args.outA
    out_panel = args.out_panel_recog if args.outB is None else args.outB
    width_main = args.width_main if args.widthA is None else args.widthA
    height_main = args.height_main if args.heightA is None else args.heightA
    fps_main = args.fps_main if args.fpsA is None else args.fpsA
    fourcc_main = args.fourcc_main if args.fourccA is None else args.fourccA
    width_panel = args.width_panel_recog if args.widthB is None else args.widthB
    height_panel = args.height_panel_recog if args.heightB is None else args.heightB
    fps_panel = args.fps_panel_recog if args.fpsB is None else args.fpsB
    fourcc_panel = args.fourcc_panel_recog if args.fourccB is None else args.fourccB

    if not main_device or not panel_device:
        ap.error("pair mode requires --main-device and --panel-recog-device")

    run_pair(
        main_device=main_device,
        panel_recog_device=panel_device,
        out_main=out_main,
        out_panel_recog=out_panel,
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
