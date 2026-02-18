#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import glob
import os

import cv2
import numpy as np


def load_intrinsics(path: str) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise RuntimeError(f"Failed to open intrinsics: {path}")
    K = fs.getNode("K").mat()
    dist = fs.getNode("dist").mat()
    width = int(fs.getNode("width").real())
    height = int(fs.getNode("height").real())
    fs.release()
    return K, dist, (width, height)


def save_extrinsics(path: str, R: np.ndarray, t: np.ndarray, rms: float = 0.0, note: str = "") -> None:
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
    fs.write("R", R)
    fs.write("t", t)
    fs.write("rms", float(rms))
    if note:
        fs.write("note", note)
    fs.release()


def rot_x(rad: float) -> np.ndarray:
    c, s = np.cos(rad), np.sin(rad)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def rot_y(rad: float) -> np.ndarray:
    c, s = np.cos(rad), np.sin(rad)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def rot_z(rad: float) -> np.ndarray:
    c, s = np.cos(rad), np.sin(rad)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def make_ideal_extrinsics(
    baseline_m: float,
    baseline_sign: int,
    roll_deg: float,
    pitch_deg: float,
    yaw_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    roll = np.deg2rad(roll_deg)
    pitch = np.deg2rad(pitch_deg)
    yaw = np.deg2rad(yaw_deg)

    # panel_recog_camera -> main_camera rotation
    R = rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)
    t = np.array([[0.0], [baseline_sign * baseline_m], [0.0]], dtype=np.float64)
    return R, t


def stereo_mode(args: argparse.Namespace) -> None:
    K_main, dist_main, size_main = load_intrinsics(args.intr_main)
    K_panel, dist_panel, size_panel = load_intrinsics(args.intr_panel_recog)
    if size_main != size_panel:
        print("WARN: image size differs main_camera vs panel_recog_camera. same resolution is recommended.")
    image_size = size_main

    paths_main = sorted(glob.glob(os.path.join(args.dir_main, "pair_*_main_camera.png")))
    if len(paths_main) == 0:
        raise RuntimeError("No pair_*_main_camera.png found. Capture pair images first.")

    board_size = (args.board_cols, args.board_rows)
    objp = np.zeros((args.board_rows * args.board_cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0 : args.board_cols, 0 : args.board_rows].T.reshape(-1, 2)
    objp *= args.square_size

    objpoints = []
    imgpoints_main = []
    imgpoints_panel = []

    for p_main in paths_main:
        stem = os.path.basename(p_main).replace("_main_camera.png", "")
        p_panel = os.path.join(args.dir_panel_recog, f"{stem}_panel_recog_camera.png")
        if not os.path.exists(p_panel):
            continue

        img_main = cv2.imread(p_main)
        img_panel = cv2.imread(p_panel)
        if img_main is None or img_panel is None:
            continue

        g_main = cv2.cvtColor(img_main, cv2.COLOR_BGR2GRAY)
        g_panel = cv2.cvtColor(img_panel, cv2.COLOR_BGR2GRAY)

        found_main, corners_main = cv2.findChessboardCorners(g_main, board_size)
        found_panel, corners_panel = cv2.findChessboardCorners(g_panel, board_size)
        if not (found_main and found_panel):
            continue

        corners_main = cv2.cornerSubPix(
            g_main,
            corners_main,
            (11, 11),
            (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
        )
        corners_panel = cv2.cornerSubPix(
            g_panel,
            corners_panel,
            (11, 11),
            (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
        )

        objpoints.append(objp)
        imgpoints_main.append(corners_main)
        imgpoints_panel.append(corners_panel)

        if args.show:
            vis_main = img_main.copy()
            vis_panel = img_panel.copy()
            cv2.drawChessboardCorners(vis_main, board_size, corners_main, True)
            cv2.drawChessboardCorners(vis_panel, board_size, corners_panel, True)
            cv2.imshow("main_camera", vis_main)
            cv2.imshow("panel_recog_camera", vis_panel)
            cv2.waitKey(50)

    if args.show:
        cv2.destroyAllWindows()

    if len(objpoints) < 10:
        raise RuntimeError(f"Not enough valid pairs. valid={len(objpoints)}")

    result = cv2.stereoCalibrate(
        objpoints,
        imgpoints_main,
        imgpoints_panel,
        K_main,
        dist_main,
        K_panel,
        dist_panel,
        image_size,
        criteria=(cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-6),
        flags=cv2.CALIB_FIX_INTRINSIC,
    )
    if len(result) == 5:
        rms, R_main_to_panel, t_main_to_panel, _E, _F = result
    elif len(result) == 9:
        rms, _K1, _d1, _K2, _d2, R_main_to_panel, t_main_to_panel, _E, _F = result
    else:
        raise RuntimeError(f"Unexpected stereoCalibrate return length: {len(result)}")

    # OpenCV returns main->panel. We store panel->main.
    R_panel_to_main = R_main_to_panel.T
    t_panel_to_main = -R_main_to_panel.T @ t_main_to_panel

    save_extrinsics(args.out, R_panel_to_main, t_panel_to_main, rms=float(rms), note="stereoCalibrate (pair images)")
    print(f"saved panel_recog_camera->main_camera extrinsics: {args.out}")


def ideal_mode(args: argparse.Namespace) -> None:
    if args.baseline is None:
        raise RuntimeError("--baseline is required in ideal mode.")
    baseline_sign = +1 if args.baseline_sign >= 0 else -1
    R, t = make_ideal_extrinsics(
        baseline_m=args.baseline,
        baseline_sign=baseline_sign,
        roll_deg=args.roll,
        pitch_deg=args.pitch,
        yaw_deg=args.yaw,
    )
    note = (
        "IDEAL model (no pair images). "
        "panel_recog_camera->main_camera, OpenCV camera coordinates (x right, y down, z forward). "
        f"t_y = baseline_sign({baseline_sign}) * baseline({args.baseline} m)."
    )
    save_extrinsics(args.out, R, t, rms=0.0, note=note)
    print(f"saved IDEAL panel_recog_camera->main_camera extrinsics: {args.out}")
    print("R:\n", R)
    print("t:\n", t.ravel())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["ideal", "stereo"], default="ideal")
    ap.add_argument("--out", default="calib/extrinsics_panel_recog_camera_to_main_camera.yaml")

    # ideal
    ap.add_argument("--baseline", type=float, default=None, help="Distance between camera centers [m]")
    ap.add_argument("--baseline-sign", type=int, default=+1, help="+1 or -1 for t_y sign")
    ap.add_argument("--roll", type=float, default=0.0)
    ap.add_argument("--pitch", type=float, default=0.0)
    ap.add_argument("--yaw", type=float, default=0.0)

    # stereo
    ap.add_argument("--dir-main", default="calib/main_camera")
    ap.add_argument("--dir-panel-recog", default="calib/panel_recog_camera")
    ap.add_argument("--intr-main", default="calib/intrinsics_main_camera.yaml")
    ap.add_argument("--intr-panel-recog", default="calib/intrinsics_panel_recog_camera.yaml")
    ap.add_argument("--board-cols", type=int, default=9)
    ap.add_argument("--board-rows", type=int, default=6)
    ap.add_argument("--square-size", type=float, default=0.025)
    ap.add_argument("--show", action="store_true")

    # compat aliases
    ap.add_argument("--dirA", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--dirB", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--intrA", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--intrB", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--baseline_sign", dest="baseline_sign_compat", type=int, default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.dirA:
        args.dir_main = args.dirA
    if args.dirB:
        args.dir_panel_recog = args.dirB
    if args.intrA:
        args.intr_main = args.intrA
    if args.intrB:
        args.intr_panel_recog = args.intrB
    if args.baseline_sign_compat is not None:
        args.baseline_sign = args.baseline_sign_compat

    if args.mode == "ideal":
        ideal_mode(args)
    else:
        stereo_mode(args)


if __name__ == "__main__":
    main()
