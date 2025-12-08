#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RealSense で RGB + Depth を .bag に録画しつつ、
RGB 画像だけをリアルタイム表示するシンプルなスクリプト。

終了は 'q' キー または ESC で。
"""

import sys
import time

import numpy as np
import cv2
import pyrealsense2 as rs

# --- 録画設定 -------------------------------------------------
BAG_PATH = "realsense_record.bag"  # 出力する .bag ファイル名

# --- RealSense カメラ設定パラメータ -------------------------
RS_EXPOSURE     = 3
RS_GAIN         = 0
RS_WHITEBALANCE = 4600
RS_BRIGHTNESS   = 0
RS_CONTRAST     = 50
RS_SHARPNESS    = 50
RS_SATURATION   = 50
RS_GAMMA        = 100


def safe_set(sensor, option, value):
    """そのセンサでサポートされていれば option を設定するヘルパー。"""
    try:
        if sensor.supports(option):
            sensor.set_option(option, value)
            print(f"[INFO] set {option.name} = {value}")
        else:
            print(f"[WARN] sensor does not support option: {option.name}")
    except Exception as e:
        print(f"[WARN] failed to set {option.name}: {e}")


def main():
    # --- パイプラインとコンフィグの準備 ----------------------
    pipeline = rs.pipeline()
    config = rs.config()

    # RGB + Depth ストリームを有効化（解像度や fps は必要に応じて変更可）
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    # .bag ファイルへの録画を有効化（RGB, Depth など有効ストリームが全部入る）
    config.enable_record_to_file(BAG_PATH)

    try:
        print(f"[INFO] starting pipeline, recording to: {BAG_PATH}")
        profile = pipeline.start(config)
    except Exception as e:
        print(f"[ERROR] failed to start pipeline: {e}")
        sys.exit(1)

    # --- カメラパラメータの設定（RGB カメラに対して） --------
    device = profile.get_device()
    color_sensor = None
    for s in device.sensors:
        try:
            name = s.get_info(rs.camera_info.name)
        except Exception:
            continue
        if "RGB" in name:  # 'RGB Camera' など
            color_sensor = s
            break

    if color_sensor is None:
        print("[WARN] RGB Camera sensor not found. cannot set RGB options.")
    else:
        # マニュアル露光にする（自動露光が有効だと EXPO が上書きされる）
        if color_sensor.supports(rs.option.enable_auto_exposure):
            color_sensor.set_option(rs.option.enable_auto_exposure, 0)
            print("[INFO] disable auto exposure")

        # ご指定のパラメータを設定
        safe_set(color_sensor, rs.option.exposure,      RS_EXPOSURE)
        safe_set(color_sensor, rs.option.gain,          RS_GAIN)
        safe_set(color_sensor, rs.option.white_balance, RS_WHITEBALANCE)
        safe_set(color_sensor, rs.option.brightness,    RS_BRIGHTNESS)
        safe_set(color_sensor, rs.option.contrast,      RS_CONTRAST)
        safe_set(color_sensor, rs.option.sharpness,     RS_SHARPNESS)
        safe_set(color_sensor, rs.option.saturation,    RS_SATURATION)
        safe_set(color_sensor, rs.option.gamma,         RS_GAMMA)

    print("[INFO] recording... press 'q' or ESC to stop.")

    # --- メインループ：録画しつつ RGB だけ表示 ---------------
    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()

            if not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())

            # RGB だけ画面表示
            cv2.imshow("RealSense RGB", color_image)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):  # ESC or 'q'
                print("[INFO] stop requested by user.")
                break

    except KeyboardInterrupt:
        print("\n[INFO] KeyboardInterrupt, stopping...")
    finally:
        # 後始末
        pipeline.stop()
        cv2.destroyAllWindows()
        print(f"[INFO] pipeline stopped. bag saved to: {BAG_PATH}")


if __name__ == "__main__":
    main()
