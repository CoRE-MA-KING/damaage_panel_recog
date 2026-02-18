#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import glob
import os

import cv2
import numpy as np


def save_intrinsics(path: str, K: np.ndarray, dist: np.ndarray, image_size: tuple[int, int], rms: float) -> None:
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
    fs.write("K", K)
    fs.write("dist", dist)
    fs.write("width", int(image_size[0]))
    fs.write("height", int(image_size[1]))
    fs.write("rms", float(rms))
    fs.release()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--img-dir", required=True, help="Calibration images directory")
    ap.add_argument("--out", required=True, help="Output YAML path")
    ap.add_argument("--board-cols", type=int, required=True, help="Chessboard inner corners (columns)")
    ap.add_argument("--board-rows", type=int, required=True, help="Chessboard inner corners (rows)")
    ap.add_argument("--square-size", type=float, required=True, help="Square size [m]")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    pattern = os.path.join(args.img_dir, "*.png")
    paths = sorted(glob.glob(pattern))
    if len(paths) < 10:
        raise RuntimeError(f"Need at least 10 images. found={len(paths)}")

    board_size = (args.board_cols, args.board_rows)
    objp = np.zeros((args.board_rows * args.board_cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0 : args.board_cols, 0 : args.board_rows].T.reshape(-1, 2)
    objp *= args.square_size

    objpoints = []
    imgpoints = []
    image_size = None

    for path in paths:
        img = cv2.imread(path)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if image_size is None:
            image_size = (gray.shape[1], gray.shape[0])

        found, corners = cv2.findChessboardCorners(
            gray,
            board_size,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE,
        )
        if not found:
            continue

        corners2 = cv2.cornerSubPix(
            gray,
            corners,
            (11, 11),
            (-1, -1),
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
        )
        objpoints.append(objp)
        imgpoints.append(corners2)

        if args.show:
            vis = img.copy()
            cv2.drawChessboardCorners(vis, board_size, corners2, found)
            cv2.imshow("corners", vis)
            cv2.waitKey(50)

    if args.show:
        cv2.destroyAllWindows()

    if len(objpoints) < 10:
        raise RuntimeError(f"Not enough valid detections. valid={len(objpoints)}")
    if image_size is None:
        raise RuntimeError("No valid input images found.")

    rms, K, dist, _rvecs, _tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_size, None, None)
    print(f"RMS: {rms}")
    print("K:\n", K)
    print("dist:\n", dist.ravel())

    save_intrinsics(args.out, K, dist, image_size, float(rms))
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
