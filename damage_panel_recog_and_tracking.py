#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
import time
import collections
import subprocess
import shutil
import re

import numpy as np
import cv2
from domain.message import DamagePanelRecognition

# --- 追加（motpyはオプション依存） ---
_MOTPY_AVAILABLE = False
try:
    from motpy import Detection, MultiObjectTracker
    _MOTPY_AVAILABLE = True
except Exception:
    # --track を使わなければそのまま動く
    pass

KEY_PREFIX = ""
PUBLISH_KEY = "damagepanel"
MAIN_WIN = 'Panel (paired by same-color top & bottom)'

# ============================================================
# Webカメラ設定パラメータ（起動時に -s の有無に関係なく適用）
# ※使用する /dev/video* の v4l2-ctl --list-ctrls の項目に合わせる
#
# 各種パラメータの変数は v4l2-ctl に設定する値に準拠
# トラックバーで調整する時は負の値分をオフセットに付けた値で扱うことに注意
# ============================================================

# --- User Controls ---
CAM_BRIGHTNESS               = 0       # brightness [-64..64]
CAM_CONTRAST                 = 32      # contrast [0..64] 0がいいかも？
CAM_SATURATION               = 90     # saturation [0..128]
CAM_HUE                      = 0       # hue [-40..40]
CAM_WHITE_BALANCE_AUTOMATIC  = 0       # white_balance_automatic [0/1]
CAM_GAMMA                    = 100     # gamma [72..500]
CAM_GAIN                     = 0       # gain [0..100]
CAM_POWER_LINE_FREQUENCY     = 1       # power_line_frequency: 0=Disabled,1=50Hz,2=60Hz
CAM_WHITE_BALANCE_TEMPERATURE= 4600    # white_balance_temperature [2800..6500]
CAM_SHARPNESS                = 3       # sharpness [0..6]
CAM_BACKLIGHT_COMPENSATION   = 0      # backlight_compensation [0,1,2]

# --- Camera Controls ---
CAM_AUTO_EXPOSURE            = 1       # auto_exposure: 1=Manual Mode 3=Aperture Priority Mode
CAM_EXPOSURE_TIME_ABSOLUTE   = 4     # exposure_time_absolute [1..5000]  ※一般に 100µs単位 → 100=10ms

# 現在のカメラには存在しない
# CAM_PAN_ABSOLUTE             = 0       # pan_absolute
# CAM_TILT_ABSOLUTE            = 0       # tilt_absolute
# CAM_FOCUS_AUTO_CONTINUOUS    = 0       # focus_automatic_continuous (固定したいなら0)
# CAM_FOCUS_ABSOLUTE           = 0       # focus_absolute（inactiveなら無視される）
# CAM_ZOOM_ABSOLUTE            = 0       # zoom_absolute

# ============================================================
# キャプチャ仕様（オプション指定はしない：固定値/変数で管理）
# ============================================================
CAM_FRAME_WIDTH  = 800
CAM_FRAME_HEIGHT = 600
CAM_FPS          = 120
CAM_FOURCC       = "MJPG"    # "MJPEG" ではなく FourCC は "MJPG"

# --- LED検出パラメータ -------------------------
KERNEL_SZ      = 3
WIDTH_TOL      = 0.6    # 幅の類似性の許容誤差（大きい矩形基準）
MIN_H_OVERLAP  = 0.05   # 横方向オーバーラップ率（小さい矩形基準）
MIN_V_GAP      = 1
MIN_BOX_H      = 1
MIN_BOX_W      = 4

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
    "blue":  {"H_low":100, "H_high":135, "S_low":180, "S_high":255, "V_low":120, "V_high":255},
    "red1":  {"H_low":  0, "H_high": 15},
    "red2":  {"H_low":165, "H_high":179},
    "redSV": {"S_low":180, "S_high":255, "V_low":120, "V_high":255},
}

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--publish", action="store_true", help="Zenohにpublishする")
    ap.add_argument("-n", "--no-display", action="store_true", help="映像表示を行わない")
    ap.add_argument("-s", "--setting", action="store_true", help="トラックバーでカメラ/HSVを調整する（同じウインドウに表示）")
    ap.add_argument("-t", "--track", action="store_true", help="motpyを用いて多目標トラッキングを有効化する")
    ap.add_argument("-d", "--device", default="/dev/video0",
                    help="キャプチャデバイスのパスまたは番号（例: 0, /dev/video0）")
    return ap.parse_args()

def _dev_to_path(dev) -> str:
    """v4l2-ctl 用に /dev/videoX 形式へ正規化."""
    if isinstance(dev, int):
        return f"/dev/video{dev}"
    if isinstance(dev, str):
        s = dev.strip()
        if s.isdigit():
            return f"/dev/video{int(s)}"
        # "/dev/video4" など
        if re.match(r"^/dev/video\d+$", s):
            return s
    # それ以外（例: gstreamer文字列など）は v4l2-ctl で触れないので空
    return ""

def _has_v4l2_ctl() -> bool:
    return shutil.which("v4l2-ctl") is not None

def v4l2_set(dev_path: str, name: str, value) -> bool:
    """v4l2-ctl で設定。失敗しても落とさず False を返す."""
    if not dev_path or not _has_v4l2_ctl():
        return False
    try:
        r = subprocess.run(
            ["v4l2-ctl", "-d", dev_path, f"--set-ctrl={name}={value}"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        return (r.returncode == 0)
    except Exception:
        return False

def apply_camera_init(dev_path: str, cap: cv2.VideoCapture):
    """
    -s の有無に関係なく、起動時に一回だけ初期値をカメラへ反映する。
    基本は v4l2-ctl。だめなら cap.set にフォールバック。
    """

    # まずオート系の順番（重要）
    # 露光: manual にしてから exposure_time_absolute
    v4l2_set(dev_path, "auto_exposure", CAM_AUTO_EXPOSURE)
    v4l2_set(dev_path, "white_balance_automatic", CAM_WHITE_BALANCE_AUTOMATIC)

    # フォーカス: まず AF を止めてから focus_absolute
    # v4l2_set(dev_path, "focus_automatic_continuous", CAM_FOCUS_AUTO_CONTINUOUS)

    # 主要項目
    v4l2_set(dev_path, "exposure_time_absolute", CAM_EXPOSURE_TIME_ABSOLUTE)
    v4l2_set(dev_path, "gain", CAM_GAIN)
    v4l2_set(dev_path, "brightness", CAM_BRIGHTNESS)
    v4l2_set(dev_path, "contrast", CAM_CONTRAST)
    v4l2_set(dev_path, "saturation", CAM_SATURATION)
    v4l2_set(dev_path, "hue", CAM_HUE)
    v4l2_set(dev_path, "gamma", CAM_GAMMA)
    v4l2_set(dev_path, "power_line_frequency", CAM_POWER_LINE_FREQUENCY)
    v4l2_set(dev_path, "white_balance_temperature", CAM_WHITE_BALANCE_TEMPERATURE)
    v4l2_set(dev_path, "sharpness", CAM_SHARPNESS)
    v4l2_set(dev_path, "backlight_compensation", CAM_BACKLIGHT_COMPENSATION)

    # 位置系（必要なら）
    # v4l2_set(dev_path, "pan_absolute", CAM_PAN_ABSOLUTE)
    # v4l2_set(dev_path, "tilt_absolute", CAM_TILT_ABSOLUTE)
    # v4l2_set(dev_path, "zoom_absolute", CAM_ZOOM_ABSOLUTE)

    # focus_absolute は inactive の可能性があるので最後に（失敗しても無視）
    # v4l2_set(dev_path, "focus_absolute", CAM_FOCUS_ABSOLUTE)

    # フォールバック（v4l2 が使えない/効かない時の最低限）
    # ※OpenCVプロパティは機種によって意味/反映が不安定です
    try:
        cap.set(cv2.CAP_PROP_BRIGHTNESS, float(CAM_BRIGHTNESS))
        cap.set(cv2.CAP_PROP_CONTRAST,   float(CAM_CONTRAST))
        cap.set(cv2.CAP_PROP_SATURATION, float(CAM_SATURATION))
        cap.set(cv2.CAP_PROP_GAIN,       float(CAM_GAIN))
        cap.set(cv2.CAP_PROP_GAMMA,      float(CAM_GAMMA))
        cap.set(cv2.CAP_PROP_SHARPNESS,  float(CAM_SHARPNESS))
        cap.set(cv2.CAP_PROP_EXPOSURE,   float(CAM_EXPOSURE_TIME_ABSOLUTE))
        # WB温度は環境によっては CAP_PROP_WB_TEMPERATURE で効く
        try:
            cap.set(cv2.CAP_PROP_WB_TEMPERATURE, float(CAM_WHITE_BALANCE_TEMPERATURE))
        except Exception:
            pass
    except Exception:
        pass

    # 反映までちょい待つ（機種によっては即時反映しない）
    time.sleep(0.05)

def setup_camera(device):
    cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        print(f"[ERROR] カメラをオープンできませんでした: {device}")
        sys.exit(1)

    # フォーマット・解像度・FPS（固定）
    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*CAM_FOURCC))
    except Exception:
        pass
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          CAM_FPS)

    dev_path = _dev_to_path(device)

    # 起動時に初期値を必ず適用
    apply_camera_init(dev_path, cap)

    return cap, dev_path

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
    do_display = not args.no_display
    use_gui = args.setting
    use_track = args.track

    # no-display が指定されていたら GUI も無効にする
    if args.no_display:
        use_gui = False

    if use_track and not _MOTPY_AVAILABLE:
        print("[ERROR] --track が指定されましたが motpy が見つかりません。`pip install motpy` を実行するか、--track を外してください。")
        # 追跡無しで継続（検出・描画は動作）

    # デバイス指定（数値 or パス）
    dev = args.device
    if isinstance(dev, str) and dev.isdigit():
        dev = int(dev)

    hsv_cfg = {
        "blue":  dict(HSV_INIT["blue"]),
        "red1":  dict(HSV_INIT["red1"]),
        "red2":  dict(HSV_INIT["red2"]),
        "redSV": dict(HSV_INIT["redSV"]),
    }
    hsv_cfg["blue"]["H_high"] = min(hsv_cfg["blue"]["H_high"], 179)
    hsv_cfg["red2"]["H_high"] = min(hsv_cfg["red2"]["H_high"], 179)

    cap, dev_path = setup_camera(dev)

    # メイン映像ウインドウ（表示 or GUI のときだけ作る）
    if do_display or use_gui:
        cv2.namedWindow(MAIN_WIN, cv2.WINDOW_NORMAL)

    # 設定GUI
    if use_gui:
        def tb_set_v4l2(ctrl_name, minv=0):
            def _f(v):
                val = v + minv
                ok = v4l2_set(dev_path, ctrl_name, val)
                if not ok:
                    pass
            return _f

        cv2.createTrackbar('ExposureAbs', MAIN_WIN, CAM_EXPOSURE_TIME_ABSOLUTE, 10000,
                           tb_set_v4l2("exposure_time_absolute", minv=0))
        cv2.createTrackbar('Gain', MAIN_WIN, CAM_GAIN, 1023,
                           tb_set_v4l2("gain", minv=0))
        cv2.createTrackbar('WhiteBalance', MAIN_WIN,
                           max(0, CAM_WHITE_BALANCE_TEMPERATURE - 2800), 6500 - 2800,
                           tb_set_v4l2("white_balance_temperature", minv=2800))
        cv2.createTrackbar('Brightness', MAIN_WIN,
                           CAM_BRIGHTNESS + 64, 128,
                           tb_set_v4l2("brightness", minv=-64))
        cv2.createTrackbar('Contrast', MAIN_WIN, CAM_CONTRAST, 95,
                           tb_set_v4l2("contrast", minv=0))
        cv2.createTrackbar('Sharpness', MAIN_WIN, CAM_SHARPNESS, 7,
                           tb_set_v4l2("sharpness", minv=0))
        cv2.createTrackbar('Saturation', MAIN_WIN, CAM_SATURATION, 255,
                           tb_set_v4l2("saturation", minv=0))
        cv2.createTrackbar('Hue', MAIN_WIN,
                           CAM_HUE + 2000, 4000,
                           tb_set_v4l2("hue", minv=-2000))
        cv2.createTrackbar('Gamma', MAIN_WIN,
                           max(0, CAM_GAMMA - 64), 300 - 64,
                           tb_set_v4l2("gamma", minv=64))

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
        cv2.createTrackbar('R1_H_low',  MAIN_WIN, hsv_cfg["red1"]["H_low"],  179, r1set("H_low"))
        cv2.createTrackbar('R1_H_high', MAIN_WIN, hsv_cfg["red1"]["H_high"], 179, r1set("H_high"))
        cv2.createTrackbar('R2_H_low',  MAIN_WIN, hsv_cfg["red2"]["H_low"],  179, r2set("H_low"))
        cv2.createTrackbar('R2_H_high', MAIN_WIN, hsv_cfg["red2"]["H_high"], 179, r2set("H_high"))
        cv2.createTrackbar('R_S_low',   MAIN_WIN, hsv_cfg["redSV"]["S_low"], 255, rsvset("S_low"))
        cv2.createTrackbar('R_S_high',  MAIN_WIN, hsv_cfg["redSV"]["S_high"],255, rsvset("S_high"))
        cv2.createTrackbar('R_V_low',   MAIN_WIN, hsv_cfg["redSV"]["V_low"], 255, rsvset("V_low"))
        cv2.createTrackbar('R_V_high',  MAIN_WIN, hsv_cfg["redSV"]["V_high"],255, rsvset("V_high"))

    # Zenoh（必要時のみ）
    publisher = None
    if do_publish:
        try:
            import zenoh
        except ImportError:
            print("[ERROR] zenoh が見つかりません。`pip install zenoh` を実施してください。")
            cap.release(); cv2.destroyAllWindows(); sys.exit(1)
        session = zenoh.open(zenoh.Config())
        key_prefix = KEY_PREFIX.strip("/")
        key_expr = f"{key_prefix}/{PUBLISH_KEY}" if key_prefix else PUBLISH_KEY
        publisher = session.declare_publisher(key_expr)

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
        tracker = MultiObjectTracker(dt=1/CAM_FPS, model_spec=model_spec)
        if (MOT_MIN_IOU is not None) and hasattr(tracker, 'min_iou'):
            tracker.min_iou = MOT_MIN_IOU
        if (MOT_MAX_STALE is not None) and hasattr(tracker, 'max_staleness'):
            tracker.max_staleness = MOT_MAX_STALE

    try:
        while True:
            ret, color_image = cap.read()
            if not ret:
                continue

            # dt / FPS更新
            now = time.time()
            dt = max(1e-6, now - last_t)
            last_t = now
            if tracker is not None:
                tracker.dt = dt
            inst_fps = 1.0 / dt
            fps = fps_alpha * inst_fps + (1 - fps_alpha) * fps

            hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

            boxes_by_color = {}
            for c in ['blue', 'red']:
                mask = get_led_mask(hsv, c, hsv_cfg)
                boxes_by_color[c] = find_boxes(mask)

            # 検出（上下ペア→unionを作り、同時にtop/bottomも保持）
            union_boxes_xywh = []
            detection_meta = []  # {xyxy, union_xywh, top, bottom, color}
            show_raw = not (use_track and _MOTPY_AVAILABLE)

            target_x, target_y = CAM_FRAME_WIDTH // 2, CAM_FRAME_HEIGHT // 2
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
                        cx0 = CAM_FRAME_WIDTH // 2
                        if abs(cx - cx0) < abs(target_to_center_distance):
                            target_to_center_distance = cx - cx0
                            target_x, target_y = int(cx), int(cy)

            # --- 追跡フェーズ ---
            chosen_from_tracks = False
            if use_track and _MOTPY_AVAILABLE and tracker is not None:
                detections = []
                for m in detection_meta:
                    ux, uy, uw, uh = m['union']
                    score = min(1.0, (uw * uh) / (CAM_FRAME_WIDTH*CAM_FRAME_HEIGHT/8.0) + 0.1)
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

            if do_display:
                cv2.imshow(MAIN_WIN, color_image)

            # Publish（必要時）
            if publisher:
                publisher.put(DamagePanelRecognition(
                    target_x=target_x,
                    target_y=target_y,
                    target_distance=int(depth_val)
                ).model_dump_json())

            #  キー入力処理も「ウインドウがあるときだけ」
            if do_display or use_gui:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                # GUI無しモードでは少し休止
                time.sleep(0.01)

    finally:
        if 'session' in locals():
            try: session.close()
            except Exception: pass
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
