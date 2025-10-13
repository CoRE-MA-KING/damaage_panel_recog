#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
import time
import collections
import numpy as np
import cv2
import pyrealsense2 as rs

# --- 追加（motpyはオプション依存） ---
_MOTPY_AVAILABLE = False
try:
    from motpy import Detection, MultiObjectTracker
    _MOTPY_AVAILABLE = True
except Exception:
    # --track を使わなければそのまま動く
    pass

KEY_PREFIX = "robot/command"
MAIN_WIN = 'Panel (paired by same-color top & bottom)'

# --- RealSense カメラ設定パラメータ -------------------------
RS_EXPOSURE     = 10
RS_GAIN         = 0
RS_WHITEBALANCE = 4600
RS_BRIGHTNESS   = 0
RS_CONTRAST     = 50
RS_SHARPNESS    = 50
RS_SATURATION   = 50
RS_GAMMA        = 100

# --- LED検出パラメータ -------------------------
KERNEL_SZ      = 3
WIDTH_TOL      = 0.6    # 幅の類似性の許容誤差（大きい矩形基準）
MIN_H_OVERLAP  = 0.05   # 横方向オーバーラップ率（小さい矩形基準）
MIN_V_GAP      = 1
MIN_BOX_H      = 2
MIN_BOX_W      = 8

# --- motpy（トラッキング）調整パラメータ -------------------------
MOT_ORDER_POS   = 2      # 1: 定速度, 2: 定加速度
MOT_DIM_POS     = 2
MOT_ORDER_SIZE  = 0
MOT_DIM_SIZE    = 2

# ノイズ（px^2 目安）
MOT_Q_VAR_POS   = 5000.0
MOT_R_VAR_POS   = 0.1

# マッチングとライフサイクル
MOT_MIN_IOU     = None
MOT_MAX_STALE   = 4

# --- 追跡表示用 ---
TRACK_MIN_STEPS = 2
TRACK_HISTORY   = 20
TRACK_COLOR     = (0, 255, 255)  # 黄

HSV_INIT = {
    "blue":  {"H_low":105, "H_high":135, "S_low":180, "S_high":255, "V_low":120, "V_high":255},
    "red1":  {"H_low":  0, "H_high": 15},
    "red2":  {"H_low":165, "H_high":179},
    "redSV": {"S_low":180, "S_high":255, "V_low":120, "V_high":255},
}

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--publish", action="store_true", help="Zenohにpublishする")
    ap.add_argument("-s", "--setting", action="store_true", help="トラックバーでカメラ/HSVを調整する（同じウインドウに表示）")
    ap.add_argument("-t", "--track", action="store_true", help="motpyを用いて多目標トラッキングを有効化する")
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
        boxes.append((x, y, w, h))
    return boxes

def horiz_overlap_ratio(b1, b2):
    x1, _, w1, _ = b1
    x2, _, w2, _ = b2
    left = max(x1, x2)
    right = min(x1 + w1, x2 + w2)
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

# --- motpyユーティリティ ---
def _xywh_to_xyxy(box_xywh):
    x, y, w, h = box_xywh
    return np.array([x, y, x + w, y + h], dtype=float)

def _xyxy_to_center(box_xyxy):
    x1, y1, x2, y2 = box_xyxy
    return ( (x1 + x2) / 2.0, (y1 + y2) / 2.0 )

def _iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    union = area_a + area_b - inter + 1e-6
    return inter / union

def _put_text_rgba(img, text, org, scale=0.6, color=(255,255,255), thickness=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

def main():
    args = parse_args()
    do_publish = args.publish
    use_gui = args.setting
    use_track = args.track

    if use_track and not _MOTPY_AVAILABLE:
        print("[ERROR] --track が指定されましたが motpy が見つかりません。`pip install motpy` を実行するか、--track を外してください。")
        # 追跡無しで継続（検出・描画は動作）

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

    # メイン映像ウインドウ
    cv2.namedWindow(MAIN_WIN, cv2.WINDOW_NORMAL)

    # 設定GUI
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

    # --- motpy トラッカー初期化 ---
    tracker = None
    track_history = {}  # t.id -> deque([(cx,cy), ...])

    # 追加: 連番IDマッピング
    id_alias = {}       # motpyのt.id -> 1,2,3,...
    next_alias = 1

    last_t = time.time()
    fps = 0.0
    fps_alpha = 0.2  # EMA用

    if use_track and _MOTPY_AVAILABLE:
        model_spec = {
            'order_pos':  MOT_ORDER_POS,
            'dim_pos':    MOT_DIM_POS,
            'order_size': MOT_ORDER_SIZE,
            'dim_size':   MOT_DIM_SIZE,
            'q_var_pos':  MOT_Q_VAR_POS,
            'r_var_pos':  MOT_R_VAR_POS,
        }
        tracker = MultiObjectTracker(dt=1/30.0, model_spec=model_spec)
        if (MOT_MIN_IOU is not None) and hasattr(tracker, 'min_iou'):
            tracker.min_iou = MOT_MIN_IOU
        if (MOT_MAX_STALE is not None) and hasattr(tracker, 'max_staleness'):
            tracker.max_staleness = MOT_MAX_STALE

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            # dt / FPS更新
            now = time.time()
            dt = max(1e-6, now - last_t)
            last_t = now
            if tracker is not None:
                tracker.dt = dt
            inst_fps = 1.0 / dt
            fps = fps_alpha * inst_fps + (1 - fps_alpha) * fps

            color_image = np.asanyarray(color_frame.get_data())
            hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

            boxes_by_color = {}
            for c in ['blue', 'red']:
                mask = get_led_mask(hsv, c, hsv_cfg)
                boxes_by_color[c] = find_boxes(mask)

            # 検出（上下ペア→unionを作り、同時にtop/bottomも保持）
            union_boxes_xywh = []
            detection_meta = []  # {xyxy, union_xywh, top, bottom, color}
            show_raw = not (use_track and _MOTPY_AVAILABLE)

            target_x, target_y = 640, 360
            depth_val = 0.0
            dummy = 0
            target_to_center_distance = 9999

            for c in ['blue', 'red']:
                pairs = pair_boxes_same_color(boxes_by_color[c])
                for (top, bottom) in pairs:
                    (ux, uy, uw, uh), (cx, cy) = bbox_union(top, bottom)
                    union_boxes_xywh.append((ux, uy, uw, uh))

                    xyxy = _xywh_to_xyxy((ux, uy, uw, uh))
                    detection_meta.append({
                        'xyxy': xyxy,
                        'union': (ux, uy, uw, uh),
                        'top': top,
                        'bottom': bottom,
                        'color': c
                    })

                    if show_raw:
                        box_col = (255, 0, 0) if c == 'blue' else (0, 0, 255)
                        # top/bottomの矩形
                        for (x, y, w, h) in (top, bottom):
                            cv2.rectangle(color_image, (x, y), (x + w, y + h), box_col, 2)
                        # union
                        cv2.rectangle(color_image, (ux, uy), (ux + uw, uy + uh), (0, 255, 0), 2)
                        cv2.circle(color_image, (int(cx), int(cy)), 3, (0, 255, 0), -1)
                        # ラベル（座標）
                        x1, y1, x2, y2 = map(int, xyxy)
                        _put_text_rgba(color_image,
                                       f"( {x1},{y1} )-( {x2},{y2} )",
                                       (ux, max(0, uy - 6)), 0.5, (0,255,0), 1)

                    if not (use_track and _MOTPY_AVAILABLE):
                        if abs(cx - 640) < abs(target_to_center_distance):
                            target_to_center_distance = cx - 640
                            target_x, target_y = int(cx), int(cy)

            # --- 追跡フェーズ ---
            chosen_from_tracks = False
            if use_track and _MOTPY_AVAILABLE and tracker is not None:
                detections = []
                for m in detection_meta:
                    ux, uy, uw, uh = m['union']
                    score = min(1.0, (uw * uh) / (1280*720/8.0) + 0.1)
                    detections.append(Detection(box=m['xyxy'], score=float(score), class_id=0))

                tracker.step(detections=detections)
                tracks = tracker.active_tracks(min_steps_alive=TRACK_MIN_STEPS)

                # 各トラックに最もIoUが高い検出を対応付け（上下LEDの矩形表示のため）
                matched_detection_idx = {}
                for ti, t in enumerate(tracks):
                    best_i, best_iou = -1, 0.0
                    for di, m in enumerate(detection_meta):
                        iou = _iou_xyxy(t.box, m['xyxy'])
                        if iou > best_iou:
                            best_iou = iou; best_i = di
                    if best_i >= 0 and best_iou > 0.1:
                        matched_detection_idx[ti] = best_i

                # 可視化・軌跡・ラベル
                for ti, t in enumerate(tracks):
                    x1, y1, x2, y2 = map(int, t.box)
                    cx, cy = _xyxy_to_center(t.box)

                    # 連番IDを割り当て
                    if t.id not in id_alias:
                        id_alias[t.id] = next_alias
                        next_alias += 1
                    show_id = id_alias[t.id]

                    # 軌跡の更新
                    if t.id not in track_history:
                        track_history[t.id] = collections.deque(maxlen=TRACK_HISTORY)
                    track_history[t.id].append((int(cx), int(cy)))

                    # パネル枠（黄）
                    cv2.rectangle(color_image, (x1, y1), (x2, y2), TRACK_COLOR, 2)

                    # 上下LEDの矩形も描画（対応検出がある場合）
                    if ti in matched_detection_idx:
                        mi = matched_detection_idx[ti]
                        top = detection_meta[mi]['top']
                        bottom = detection_meta[mi]['bottom']
                        c = detection_meta[mi]['color']
                        box_col = (255, 0, 0) if c == 'blue' else (0, 0, 255)
                        for (x, y, w, h) in (top, bottom):
                            cv2.rectangle(color_image, (x, y), (x + w, y + h), box_col, 2)

                    # 直前で求めている中心（float）を整数化して使う
                    cx_i, cy_i = map(int, _xyxy_to_center(t.box))

                    # ラベル: ID と中心座標（←ここを変更）
                    _put_text_rgba(
                        color_image,
                        f"ID {show_id} ({cx_i},{cy_i})",
                        (x1, max(0, y1 - 10)),
                        0.6, TRACK_COLOR, 2
                    )

                    # 軌跡（折れ線）
                    pts = list(track_history[t.id])
                    for k in range(1, len(pts)):
                        cv2.line(color_image, pts[k-1], pts[k], TRACK_COLOR, 2)

                # ターゲット：中心に最も近いトラック
                if len(tracks) > 0:
                    center_x = color_image.shape[1] // 2
                    best = None; best_dx = None
                    for t in tracks:
                        cx, cy = _xyxy_to_center(t.box)
                        dx = abs(cx - center_x)
                        if (best_dx is None) or (dx < best_dx):
                            best = (int(cx), int(cy)); best_dx = dx
                    if best is not None:
                        target_x, target_y = best
                        chosen_from_tracks = True

            # ターゲット表示（トラック由来ならシアン、検出由来なら白）
            tgt_col = (255, 255, 0) if chosen_from_tracks else (255, 255, 255)
            cv2.drawMarker(color_image, (int(target_x), int(target_y)), tgt_col,
                           markerType=cv2.MARKER_CROSS, markerSize=16, thickness=2)
            _put_text_rgba(color_image, f"target=({target_x},{target_y})",
                           (10, 26), 0.7, tgt_col, 2)

            # モード表示
            mode = "TRACK=ON (motpy)" if (use_track and _MOTPY_AVAILABLE) else ("TRACK=ON (motpy missing!)" if use_track else "TRACK=OFF")
            _put_text_rgba(color_image, mode, (10, 50), 0.6, (200,200,200), 1)

            # 右上にFPS表示
            fps_text = f"FPS: {fps:.1f}"
            (tw, th), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            x_right = color_image.shape[1] - tw - 10
            y_top = 30
            _put_text_rgba(color_image, fps_text, (x_right, y_top), 0.7, (255,255,255), 2)

            cv2.imshow(MAIN_WIN, color_image)

            # Publish（必要時）
            if publishers is not None:
                for key, pub in publishers.items():
                    if   key == "target_x": value = target_x
                    elif key == "target_y": value = target_y
                    elif key == "depth":    value = depth_val
                    elif key == "dummy":    value = 0
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
