#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RealSenseで赤/青の上下ペア矩形を検出し、中心座標を計算。
-p/--publish を付けたときのみ Zenoh に target_x/target_y/depth/dummy を publish します。
引数なしのときは Zenoh を一切使わず、純粋な画像処理プログラムとして動作します。
"""

import argparse
import sys
import numpy as np
import cv2
import pyrealsense2 as rs

# ========= Zenoh セッション設定（キー接頭）=========
KEY_PREFIX = "robot/command"
# ================================================

# ========= RealSense カラーカメラ パラメータ（仮で 0 に設定） =========
RS_EXPOSURE     = 10   # 1～10000          
RS_GAIN         = 0    # 0～128
RS_WHITEBALANCE = 4600 # 800～6500 [K]
RS_BRIGHTNESS   = 64-64
RS_CONTRAST     = 50
RS_SHARPNESS    = 50
RS_SATURATION   = 50
RS_GAMMA        = 0+100
# ================================================================

# ========= 検出ロジックの可調整パラメータ =========
AREA_MIN       = 100    # px^2
KERNEL_SZ      = 3
WIDTH_TOL      = 0.25
MIN_H_OVERLAP  = 0.50
MIN_V_GAP      = 2

# ★ 色ごとの矩形候補抽出段階での最小サイズ制限
MIN_BOX_H      = 10     # px
MIN_BOX_W      = 50     # px
# ================================================

# ========= HSV 範囲 =========
COLOR_RANGES = {
    'blue': ([105, 180, 120], [125, 255, 255]),
    'red1': ([0,   180, 120], [10,  255, 255]),
    'red2': ([160, 180, 120], [180, 255, 255]),
}
# ==========================


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Panel detector with optional Zenoh publishing")
    ap.add_argument("-p", "--publish", action="store_true",
                    help="Zenoh に publish する（付けない場合は画像処理のみ）")
    return ap.parse_args()


def get_color_sensor(dev: rs.device) -> rs.sensor | None:
    """RGB/Color センサを推定して返す（見つからなければ先頭センサを返す）"""
    sensors = dev.query_sensors()
    for s in sensors:
        try:
            name = s.get_info(rs.camera_info.name)
            if ('RGB' in name) or ('Color' in name):
                return s
        except Exception:
            pass
    return sensors[0] if len(sensors) > 0 else None


def setup_realsense():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    profile = pipeline.start(config)

    # カラーセンサに各種パラメータ設定
    try:
        color_sensor = get_color_sensor(profile.get_device())
        if color_sensor is not None:
            if color_sensor.supports(rs.option.enable_auto_exposure):
                color_sensor.set_option(rs.option.enable_auto_exposure, 0)
            if color_sensor.supports(rs.option.enable_auto_white_balance):
                color_sensor.set_option(rs.option.enable_auto_white_balance, 0)

            if color_sensor.supports(rs.option.exposure):
                color_sensor.set_option(rs.option.exposure, RS_EXPOSURE)
            if color_sensor.supports(rs.option.gain):
                color_sensor.set_option(rs.option.gain, RS_GAIN)
            if color_sensor.supports(rs.option.white_balance):
                color_sensor.set_option(rs.option.white_balance, RS_WHITEBALANCE)
            if color_sensor.supports(rs.option.brightness):
                color_sensor.set_option(rs.option.brightness, RS_BRIGHTNESS)
            if color_sensor.supports(rs.option.contrast):
                color_sensor.set_option(rs.option.contrast, RS_CONTRAST)
            if color_sensor.supports(rs.option.sharpness):
                color_sensor.set_option(rs.option.sharpness, RS_SHARPNESS)
            if color_sensor.supports(rs.option.saturation):
                color_sensor.set_option(rs.option.saturation, RS_SATURATION)
            if color_sensor.supports(rs.option.gamma):
                color_sensor.set_option(rs.option.gamma, RS_GAMMA)
        else:
            print("[WARN] カラーセンサが見つかりませんでした。パラメータ設定をスキップします。")
    except Exception as e:
        print("[WARN] RealSense パラメータ設定中に例外:", e)

    align = rs.align(rs.stream.color)
    return pipeline, align


def get_led_mask(hsv, color):
    if color == 'blue':
        lower, upper = COLOR_RANGES['blue']
        return cv2.inRange(hsv, np.array(lower), np.array(upper))
    elif color == 'red':
        l1, u1 = COLOR_RANGES['red1']
        l2, u2 = COLOR_RANGES['red2']
        m1 = cv2.inRange(hsv, np.array(l1), np.array(u1))
        m2 = cv2.inRange(hsv, np.array(l2), np.array(u2))
        return cv2.bitwise_or(m1, m2)
    return None


def find_boxes(mask):
    """マスク→輪郭→(x,y,w,h) のリストを返す（最小高さ・最小幅・最小面積で足切り）。"""
    kernel = np.ones((KERNEL_SZ, KERNEL_SZ), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if (h < MIN_BOX_H) or (w < MIN_BOX_W):
            continue
        if (w * h) < AREA_MIN:
            continue
        boxes.append((x, y, w, h))
    return boxes


def horiz_overlap_ratio(b1, b2):
    """2 矩形の水平方向重なり幅 / min(w1,w2)"""
    x1, _, w1, _ = b1
    x2, _, w2, _ = b2
    left = max(x1, x2)
    right = min(x1 + w1, x2 + w2)
    overlap = max(0, right - left)
    return 0.0 if min(w1, w2) == 0 else overlap / float(min(w1, w2))


def pair_boxes_same_color(boxes):
    """同色ボックス群から上下ペアを作る（1対1, 最も近い下ボックスを選択）。"""
    boxes_sorted = sorted(boxes, key=lambda b: b[1])  # y昇順
    paired = []
    used_bottom = set()

    for i, top in enumerate(boxes_sorted):
        x_t, y_t, w_t, h_t = top
        best_j = -1
        best_dy = None

        for j, bottom in enumerate(boxes_sorted):
            if j == i or j in used_bottom:
                continue
            x_b, y_b, w_b, h_b = bottom

            dy = (y_b) - (y_t + h_t)
            if dy < MIN_V_GAP:
                continue
            if abs(w_t - w_b) > WIDTH_TOL * max(w_t, w_b):
                continue
            if horiz_overlap_ratio(top, bottom) < MIN_H_OVERLAP:
                continue

            if (best_dy is None) or (dy < best_dy):
                best_dy = dy
                best_j = j

        if best_j >= 0:
            paired.append((top, boxes_sorted[best_j]))
            used_bottom.add(best_j)

    return paired


def bbox_union(top, bottom):
    """上下 2 矩形を含む最小外接矩形 (x_min, y_min, w, h) と中心 (cx, cy)"""
    x1, y1, w1, h1 = top
    x2, y2, w2, h2 = bottom
    x_min = min(x1, x2)
    y_min = y1
    x_max = max(x1 + w1, x2 + w2)
    y_max = y2 + h2
    w = x_max - x_min
    h = y_max - y_min
    cx = x_min + w / 2.0
    cy = y_min + h / 2.0
    return (x_min, y_min, w, h), (cx, cy)


def declare_publishers(session):
    """
    RobotCommand のフィールド名を使って publisher を生成。
    取得できない場合は既定フィールドでフォールバック。
    """
    try:
        from domain.message import RobotCommand  # 必要時のみ import
        keys = list(RobotCommand.model_fields.keys())
    except Exception as e:
        print(f"[WARN] RobotCommand の取得に失敗: {e}")
        print("[INFO] 既定フィールド ['target_x','target_y','depth','dummy'] を使用します。")
        keys = ["target_x", "target_y", "depth", "dummy"]

    pubs = {key: session.declare_publisher(f"{KEY_PREFIX}/{key}") for key in keys}
    return pubs, keys


def main():
    args = parse_args()
    do_publish = args.publish

    # RealSense 準備
    pipeline, align = setup_realsense()

    # 画像処理のみ（非公開モード）のときは Zenoh を import すらしない
    if not do_publish:
        print("[MODE] 非公開モード（Zenoh未使用）で起動しました。")
        try:
            while True:
                frames = pipeline.wait_for_frames()
                aligned = align.process(frames)

                color_frame = aligned.get_color_frame()
                depth_frame = aligned.get_depth_frame()
                if not color_frame or not depth_frame:
                    continue

                color_image = np.asanyarray(color_frame.get_data())
                hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

                boxes_by_color = {}
                for color in ['blue', 'red']:
                    mask = get_led_mask(hsv, color)
                    boxes_by_color[color] = find_boxes(mask)

                target_x, target_y = 640, 360  # 初期値（画面中心）
                target_to_center_distance = 9999

                for color in ['blue', 'red']:
                    pairs = pair_boxes_same_color(boxes_by_color[color])
                    for (top, bottom) in pairs:
                        box_col = (255, 0, 0) if color == 'blue' else (0, 0, 255)
                        for (x, y, w, h) in (top, bottom):
                            cv2.rectangle(color_image, (x, y), (x + w, y + h), box_col, 1)

                        (ux, uy, uw, uh), (cx, cy) = bbox_union(top, bottom)
                        cv2.rectangle(color_image, (ux, uy), (ux + uw, uy + uh), (0, 255, 0), 2)
                        cv2.circle(color_image, (int(cx), int(cy)), 3, (0, 255, 0), -1)
                        cv2.putText(
                            color_image,
                            f"{color} cx,cy=({int(cx)},{int(cy)})",
                            (ux, max(0, uy - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA
                        )

                        if abs(cx - 640) < abs(target_to_center_distance):
                            target_to_center_distance = cx - 640
                            target_x = int(cx)
                            target_y = int(cy)

                cv2.imshow('Panel (paired by same-color top & bottom)', color_image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            pipeline.stop()
            cv2.destroyAllWindows()
        return

    # ここから公開モード（Zenohあり）
    print("[MODE] 公開モード（Zenohにpublish）で起動しました。")

    # 必要時のみ Zenoh を import
    try:
        import zenoh
    except ImportError:
        print("[ERROR] zenoh が見つかりません。`pip install zenoh` などで導入してください。")
        pipeline.stop()
        cv2.destroyAllWindows()
        sys.exit(1)

    try:
        with zenoh.open(zenoh.Config()) as session:
            publishers, pub_keys = declare_publishers(session)
            print(f"[INFO] Publish keys: {', '.join(pub_keys)}")

            while True:
                frames = pipeline.wait_for_frames()
                aligned = align.process(frames)

                color_frame = aligned.get_color_frame()
                depth_frame = aligned.get_depth_frame()
                if not color_frame or not depth_frame:
                    continue

                color_image = np.asanyarray(color_frame.get_data())
                hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

                boxes_by_color = {}
                for color in ['blue', 'red']:
                    mask = get_led_mask(hsv, color)
                    boxes_by_color[color] = find_boxes(mask)

                target_x, target_y = 640, 360
                depth = 0.0
                dummy = 0
                target_to_center_distance = 9999

                for color in ['blue', 'red']:
                    pairs = pair_boxes_same_color(boxes_by_color[color])
                    for (top, bottom) in pairs:
                        box_col = (255, 0, 0) if color == 'blue' else (0, 0, 255)
                        for (x, y, w, h) in (top, bottom):
                            cv2.rectangle(color_image, (x, y), (x + w, y + h), box_col, 1)

                        (ux, uy, uw, uh), (cx, cy) = bbox_union(top, bottom)
                        cv2.rectangle(color_image, (ux, uy), (ux + uw, uy + uh), (0, 255, 0), 2)
                        cv2.circle(color_image, (int(cx), int(cy)), 3, (0, 255, 0), -1)
                        cv2.putText(
                            color_image,
                            f"{color} cx,cy=({int(cx)},{int(cy)})",
                            (ux, max(0, uy - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA
                        )

                        if abs(cx - 640) < abs(target_to_center_distance):
                            target_to_center_distance = cx - 640
                            target_x = int(cx)
                            target_y = int(cy)
                            depth = depth_frame.get_distance(target_x, target_y)

                cv2.imshow('Panel (paired by same-color top & bottom)', color_image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                # Publish（RobotCommand がなければ既定キー）
                for key, pub in publishers.items():
                    if key == "target_x":
                        value = target_x
                    elif key == "target_y":
                        value = target_y
                    elif key == "depth":
                        value = depth
                    elif key == "dummy":
                        value = dummy
                    else:
                        # 予期せぬフィールドはスキップ
                        continue
                    pub.put(str(value))  # 文字列で送信

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
