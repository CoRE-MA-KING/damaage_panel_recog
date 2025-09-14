#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
import numpy as np
import cv2
import pyrealsense2 as rs

KEY_PREFIX = "robot/command"
MAIN_WIN = 'Panel (paired by same-color top & bottom)'

RS_EXPOSURE     = 10
RS_GAIN         = 0
RS_WHITEBALANCE = 4600
RS_BRIGHTNESS   = 0
RS_CONTRAST     = 50
RS_SHARPNESS    = 50
RS_SATURATION   = 50
RS_GAMMA        = 100

AREA_MIN       = 100
KERNEL_SZ      = 3
WIDTH_TOL      = 0.25
MIN_H_OVERLAP  = 0.50
MIN_V_GAP      = 2
MIN_BOX_H      = 10
MIN_BOX_W      = 50

HSV_INIT = {
    "blue":  {"H_low":105, "H_high":125, "S_low":180, "S_high":255, "V_low":120, "V_high":255},
    "red1":  {"H_low":  0, "H_high": 10},
    "red2":  {"H_low":160, "H_high":179},
    "redSV": {"S_low":180, "S_high":255, "V_low":120, "V_high":255},
}

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--publish", action="store_true", help="Zenohにpublishする")
    ap.add_argument("-s", "--setting", action="store_true", help="トラックバーでカメラ/HSVを調整する（同じウインドウに表示）")
    return ap.parse_args()

def get_color_sensor(dev: rs.device):
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
    try:
        color_sensor = get_color_sensor(profile.get_device())
        if color_sensor is not None:
            if color_sensor.supports(rs.option.enable_auto_exposure):
                color_sensor.set_option(rs.option.enable_auto_exposure, 0)
            if color_sensor.supports(rs.option.enable_auto_white_balance):
                color_sensor.set_option(rs.option.enable_auto_white_balance, 0)
            def safe_set(opt, val):
                if color_sensor.supports(opt):
                    try: color_sensor.set_option(opt, float(val))
                    except Exception as e: print(f"[WARN] set_option({opt}) failed: {e}")
            safe_set(rs.option.exposure,      RS_EXPOSURE)
            safe_set(rs.option.gain,          RS_GAIN)
            safe_set(rs.option.white_balance, RS_WHITEBALANCE)
            safe_set(rs.option.brightness,    RS_BRIGHTNESS)
            safe_set(rs.option.contrast,      RS_CONTRAST)
            safe_set(rs.option.sharpness,     RS_SHARPNESS)
            safe_set(rs.option.saturation,    RS_SATURATION)
            safe_set(rs.option.gamma,         RS_GAMMA)
        else:
            print("[WARN] Color sensor not found.")
    except Exception as e:
        print("[WARN] Parameter init error:", e)
    align = rs.align(rs.stream.color)
    return pipeline, align, profile

def get_led_mask(hsv, color, hsv_cfg):
    if color == 'blue':
        lo = [hsv_cfg["blue"]["H_low"], hsv_cfg["blue"]["S_low"], hsv_cfg["blue"]["V_low"]]
        hi = [hsv_cfg["blue"]["H_high"],hsv_cfg["blue"]["S_high"],hsv_cfg["blue"]["V_high"]]
        return cv2.inRange(hsv, np.array(lo), np.array(hi))
    elif color == 'red':
        r1 = hsv_cfg["red1"]; r2 = hsv_cfg["red2"]; rsv = hsv_cfg["redSV"]
        m1 = cv2.inRange(hsv, np.array([r1["H_low"], rsv["S_low"], rsv["V_low"]]),
                              np.array([r1["H_high"],rsv["S_high"],rsv["V_high"]]))
        m2 = cv2.inRange(hsv, np.array([r2["H_low"], rsv["S_low"], rsv["V_low"]]),
                              np.array([r2["H_high"],rsv["S_high"],rsv["V_high"]]))
        return cv2.bitwise_or(m1, m2)
    return None

def find_boxes(mask):
    kernel = np.ones((KERNEL_SZ, KERNEL_SZ), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if (h < MIN_BOX_H) or (w < MIN_BOX_W): continue
        if (w * h) < AREA_MIN:                 continue
        boxes.append((x, y, w, h))
    return boxes

def horiz_overlap_ratio(b1, b2):
    x1, _, w1, _ = b1; x2, _, w2, _ = b2
    left = max(x1, x2); right = min(x1 + w1, x2 + w2)
    overlap = max(0, right - left)
    return 0.0 if min(w1, w2) == 0 else overlap / float(min(w1, w2))

def pair_boxes_same_color(boxes):
    boxes_sorted = sorted(boxes, key=lambda b: b[1])  # y昇順
    paired = []; used_bottom = set()
    for i, top in enumerate(boxes_sorted):
        x_t, y_t, w_t, h_t = top
        best_j = -1; best_dy = None
        for j, bottom in enumerate(boxes_sorted):
            if j == i or j in used_bottom: continue
            x_b, y_b, w_b, h_b = bottom
            dy = (y_b) - (y_t + h_t)
            if dy < MIN_V_GAP: continue
            if abs(w_t - w_b) > WIDTH_TOL * max(w_t, w_b): continue
            if horiz_overlap_ratio(top, bottom) < MIN_H_OVERLAP: continue
            if (best_dy is None) or (dy < best_dy):
                best_dy = dy; best_j = j
        if best_j >= 0:
            paired.append((top, boxes_sorted[best_j])); used_bottom.add(best_j)
    return paired

def bbox_union(top, bottom):
    x1, y1, w1, h1 = top; x2, y2, w2, h2 = bottom
    x_min = min(x1, x2); y_min = y1
    x_max = max(x1 + w1, x2 + w2); y_max = y2 + h2
    w = x_max - x_min; h = y_max - y_min
    cx = x_min + w / 2.0; cy = y_min + h / 2.0
    return (x_min, y_min, w, h), (cx, cy)

def declare_publishers(session):
    try:
        from domain.message import RobotCommand
        keys = list(RobotCommand.model_fields.keys())
    except Exception as e:
        print(f"[WARN] RobotCommand 取得失敗: {e}")
        keys = ["target_x", "target_y", "depth", "dummy"]
    pubs = {k: session.declare_publisher(f"{KEY_PREFIX}/{k}") for k in keys}
    return pubs, keys

def main():
    args = parse_args()
    do_publish = args.publish
    use_gui = args.setting

    hsv_cfg = {
        "blue":  dict(HSV_INIT["blue"]),
        "red1":  dict(HSV_INIT["red1"]),
        "red2":  dict(HSV_INIT["red2"]),
        "redSV": dict(HSV_INIT["redSV"]),
    }
    hsv_cfg["blue"]["H_high"] = min(hsv_cfg["blue"]["H_high"], 179)
    hsv_cfg["red2"]["H_high"] = min(hsv_cfg["red2"]["H_high"], 179)

    pipeline, align, profile = setup_realsense()
    color_sensor = get_color_sensor(profile.get_device())

    # メイン映像ウインドウを先に作成（ここにトラックバーを付ける）
    cv2.namedWindow(MAIN_WIN, cv2.WINDOW_NORMAL)

    # 設定GUI：トラックバーはメインウインドウにアタッチ（別ウインドウは作らない）
    if use_gui and color_sensor is not None:
        def try_set(opt, val):
            if color_sensor.supports(opt):
                try: color_sensor.set_option(opt, float(val))
                except Exception as e: print(f"[WARN] set_option({opt}) failed: {e}")
        def gopt(opt, default):
            try:
                if color_sensor.supports(opt): return int(color_sensor.get_option(opt))
            except Exception:
                pass
            return int(default)
        init = dict(
            exposure   = max(1, gopt(rs.option.exposure,      RS_EXPOSURE)),
            gain       = gopt(rs.option.gain,                 RS_GAIN),
            wb         = gopt(rs.option.white_balance,        RS_WHITEBALANCE),
            brightness = gopt(rs.option.brightness,           RS_BRIGHTNESS),
            contrast   = gopt(rs.option.contrast,             RS_CONTRAST),
            sharpness  = gopt(rs.option.sharpness,            RS_SHARPNESS),
            saturation = gopt(rs.option.saturation,           RS_SATURATION),
            gamma      = gopt(rs.option.gamma,                RS_GAMMA),
        )
        cv2.createTrackbar('Exposure',     MAIN_WIN, init["exposure"]-1,   10000-1, lambda v: try_set(rs.option.exposure,      v+1))
        cv2.createTrackbar('Gain',         MAIN_WIN, init["gain"],         128,     lambda v: try_set(rs.option.gain,          v))
        cv2.createTrackbar('WhiteBalance', MAIN_WIN, max(0, init["wb"]-2800), 6500-2800, lambda v: try_set(rs.option.white_balance, v+2800))
        cv2.createTrackbar('Brightness',   MAIN_WIN, init["brightness"]+64,   128,  lambda v: try_set(rs.option.brightness,    v-64))
        cv2.createTrackbar('Contrast',     MAIN_WIN, init["contrast"],     100,     lambda v: try_set(rs.option.contrast,      v))
        cv2.createTrackbar('Sharpness',    MAIN_WIN, init["sharpness"],    100,     lambda v: try_set(rs.option.sharpness,     v))
        cv2.createTrackbar('Saturation',   MAIN_WIN, init["saturation"],   100,     lambda v: try_set(rs.option.saturation,    v))
        cv2.createTrackbar('Gamma',        MAIN_WIN, max(0, init["gamma"]-100), 500-100, lambda v: try_set(rs.option.gamma,     v+100))
        # HSV blue
        def bset(name): return lambda v: hsv_cfg["blue"].__setitem__(name, int(v))
        cv2.createTrackbar('B_H_low',  MAIN_WIN, hsv_cfg["blue"]["H_low"],  179, bset("H_low"))
        cv2.createTrackbar('B_H_high', MAIN_WIN, hsv_cfg["blue"]["H_high"], 179, bset("H_high"))
        cv2.createTrackbar('B_S_low',  MAIN_WIN, hsv_cfg["blue"]["S_low"],  255, bset("S_low"))
        cv2.createTrackbar('B_S_high', MAIN_WIN, hsv_cfg["blue"]["S_high"], 255, bset("S_high"))
        cv2.createTrackbar('B_V_low',  MAIN_WIN, hsv_cfg["blue"]["V_low"],  255, bset("V_low"))
        cv2.createTrackbar('B_V_high', MAIN_WIN, hsv_cfg["blue"]["V_high"], 255, bset("V_high"))
        # HSV red
        def r1set(name): return lambda v: hsv_cfg["red1"].__setitem__(name, int(v))
        def r2set(name): return lambda v: hsv_cfg["red2"].__setitem__(name, int(v))
        def rsvset(name): return lambda v: hsv_cfg["redSV"].__setitem__(name, int(v))
        cv2.createTrackbar('R1_H_low', MAIN_WIN, hsv_cfg["red1"]["H_low"],  179, r1set("H_low"))
        cv2.createTrackbar('R1_H_high',MAIN_WIN, hsv_cfg["red1"]["H_high"], 179, r1set("H_high"))
        cv2.createTrackbar('R2_H_low', MAIN_WIN, hsv_cfg["red2"]["H_low"],  179, r2set("H_low"))
        cv2.createTrackbar('R2_H_high',MAIN_WIN, hsv_cfg["red2"]["H_high"], 179, r2set("H_high"))
        cv2.createTrackbar('R_S_low',  MAIN_WIN, hsv_cfg["redSV"]["S_low"], 255, rsvset("S_low"))
        cv2.createTrackbar('R_S_high', MAIN_WIN, hsv_cfg["redSV"]["S_high"],255, rsvset("S_high"))
        cv2.createTrackbar('R_V_low',  MAIN_WIN, hsv_cfg["redSV"]["V_low"], 255, rsvset("V_low"))
        cv2.createTrackbar('R_V_high', MAIN_WIN, hsv_cfg["redSV"]["V_high"],255, rsvset("V_high"))

    # Zenoh（必要時のみ）
    publishers = None
    if do_publish:
        try:
            import zenoh
        except ImportError:
            print("[ERROR] zenoh が見つかりません。`pip install zenoh` を実施してください。")
            pipeline.stop(); cv2.destroyAllWindows(); sys.exit(1)
        session = zenoh.open(zenoh.Config())
        publishers, pub_keys = declare_publishers(session)
        print(f"[INFO] Publish keys: {', '.join(pub_keys)}")

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
            for c in ['blue', 'red']:
                mask = get_led_mask(hsv, c, hsv_cfg)
                boxes_by_color[c] = find_boxes(mask)

            target_x, target_y = 640, 360
            depth_val = 0.0
            dummy = 0
            target_to_center_distance = 9999

            for c in ['blue', 'red']:
                pairs = pair_boxes_same_color(boxes_by_color[c])
                for (top, bottom) in pairs:
                    box_col = (255, 0, 0) if c == 'blue' else (0, 0, 255)
                    for (x, y, w, h) in (top, bottom):
                        cv2.rectangle(color_image, (x, y), (x + w, y + h), box_col, 1)
                    (ux, uy, uw, uh), (cx, cy) = bbox_union(top, bottom)
                    cv2.rectangle(color_image, (ux, uy), (ux + uw, uy + uh), (0, 255, 0), 2)
                    cv2.circle(color_image, (int(cx), int(cy)), 3, (0, 255, 0), -1)
                    cv2.putText(color_image, f"{c} cx,cy=({int(cx)},{int(cy)})",
                                (ux, max(0, uy - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
                    if abs(cx - 640) < abs(target_to_center_distance):
                        target_to_center_distance = cx - 640
                        target_x, target_y = int(cx), int(cy)
                        # depth_val = depth_frame.get_distance(target_x, target_y)

            cv2.imshow(MAIN_WIN, color_image)

            if publishers is not None:
                for key, pub in publishers.items():
                    if   key == "target_x": value = target_x
                    elif key == "target_y": value = target_y
                    elif key == "depth":    value = depth_val
                    elif key == "dummy":    value = dummy
                    else: continue
                    pub.put(str(value))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        if 'session' in locals():
            try: session.close()
            except Exception: pass
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
