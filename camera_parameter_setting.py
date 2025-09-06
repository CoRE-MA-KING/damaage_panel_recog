import pyrealsense2 as rs
import numpy as np
import cv2

# --- 色恒常性補正 (Gray-World) ---
def gray_world(img):
    b_avg, g_avg, r_avg = cv2.mean(img)[:3]
    avg = (b_avg + g_avg + r_avg) / 3.0
    kb = avg / b_avg if b_avg != 0 else 1.0
    kg = avg / g_avg if g_avg != 0 else 1.0
    kr = avg / r_avg if r_avg != 0 else 1.0
    b, g, r = cv2.split(img)
    b = cv2.multiply(b, kb)
    g = cv2.multiply(g, kg)
    r = cv2.multiply(r, kr)
    corrected = cv2.merge((b, g, r))
    return np.clip(corrected, 0, 255).astype(np.uint8)

# --- 輝度依存排除 (normalized RGB) ---
def normalize_rgb(img):
    sum_ = np.sum(img, axis=2, keepdims=True).clip(min=1)
    norm = img.astype(np.float32) / sum_
    r_norm = norm[:, :, 2]
    g_norm = norm[:, :, 1]
    b_norm = norm[:, :, 0]
    return r_norm, g_norm, b_norm

# RealSense 初期化
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
profile = pipeline.start(config)

# カメラ設定：自動をオフ
color_sensor = profile.get_device().first_color_sensor()
color_sensor.set_option(rs.option.enable_auto_exposure, 0)
color_sensor.set_option(rs.option.enable_auto_white_balance, 0)

# アライメント
align_to = rs.stream.color
align = rs.align(align_to)

# 初期値取得（安全な範囲内にある想定）
init_exposure   = int(color_sensor.get_option(rs.option.exposure))         # 1～10000
init_gain       = int(color_sensor.get_option(rs.option.gain))             # 0～128
init_wb         = int(color_sensor.get_option(rs.option.white_balance))    # 2800～6500
init_brightness = int(color_sensor.get_option(rs.option.brightness))       # -64～64
init_contrast   = int(color_sensor.get_option(rs.option.contrast))         # 0～100
init_sharpness  = int(color_sensor.get_option(rs.option.sharpness))        # 0～100
init_saturation = int(color_sensor.get_option(rs.option.saturation))       # 0～100
init_gamma      = int(color_sensor.get_option(rs.option.gamma))            # 100～500

# ウィンドウとトラックバー
cv2.namedWindow('LED Detection', cv2.WINDOW_NORMAL)

def update_exposure(val):
    actual = val + 1
    color_sensor.set_option(rs.option.exposure, float(actual))
def update_gain(val):
    color_sensor.set_option(rs.option.gain, float(val))
def update_white_balance(val):
    actual = val + 2800
    color_sensor.set_option(rs.option.white_balance, float(actual))
def update_brightness(val):
    actual = val - 64
    color_sensor.set_option(rs.option.brightness, float(actual))
def update_contrast(val):
    color_sensor.set_option(rs.option.contrast, float(val))
def update_sharpness(val):
    color_sensor.set_option(rs.option.sharpness, float(val))
def update_saturation(val):
    color_sensor.set_option(rs.option.saturation, float(val))
def update_gamma(val):
    actual = val + 100
    color_sensor.set_option(rs.option.gamma, float(actual))

cv2.createTrackbar('Exposure',     'LED Detection', init_exposure - 1, 10000 - 1, update_exposure)
cv2.createTrackbar('Gain',         'LED Detection', init_gain,        128,       update_gain)
cv2.createTrackbar('WhiteBalance', 'LED Detection', init_wb - 2800,   6500-2800, update_white_balance)
cv2.createTrackbar('Brightness',   'LED Detection', init_brightness + 64, 128,   update_brightness)
cv2.createTrackbar('Contrast',     'LED Detection', init_contrast,    100,       update_contrast)
cv2.createTrackbar('Sharpness',    'LED Detection', init_sharpness,   100,       update_sharpness)
cv2.createTrackbar('Saturation',   'LED Detection', init_saturation,  100,       update_saturation)
cv2.createTrackbar('Gamma',        'LED Detection', init_gamma - 100, 500-100,   update_gamma)

# 色範囲
COLOR_RANGES = {
    'blue': ([105, 180, 120], [125, 255, 255]),
    'red1': ([0, 180, 120], [10, 255, 255]),
    'red2': ([160, 180, 120], [180, 255, 255]),
}

def get_led_mask(hsv, color):
    if color == 'blue':
        lo, hi = COLOR_RANGES['blue']
        return cv2.inRange(hsv, np.array(lo), np.array(hi))
    else:
        lo1, hi1 = COLOR_RANGES['red1']
        lo2, hi2 = COLOR_RANGES['red2']
        m1 = cv2.inRange(hsv, np.array(lo1), np.array(hi1))
        m2 = cv2.inRange(hsv, np.array(lo2), np.array(hi2))
        return cv2.bitwise_or(m1, m2)

# 上下LEDペアのパネル検出しきい値
MIN_AREA = 200
MIN_VERTICAL_DIST = 50
MAX_HORIZONTAL_OFFSET = 50

use_soft_correction = False  # True にすれば gray_world + normalize_rgb を適用

while True:
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    color_frame = aligned_frames.get_color_frame()
    depth_frame = aligned_frames.get_depth_frame()
    if not color_frame or not depth_frame:
        continue

    color_image = np.asanyarray(color_frame.get_data())

    # 任意でソフト補正を挟む（フラグ制御）
    if use_soft_correction:
        corrected = gray_world(color_image)
        r_norm, g_norm, b_norm = normalize_rgb(corrected)
        working_image = corrected
    else:
        working_image = color_image

    hsv = cv2.cvtColor(working_image, cv2.COLOR_BGR2HSV)
    detected_panels = []

    for color in ['blue', 'red']:
        mask = get_led_mask(hsv, color)
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.dilate(mask, kernel, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        led_positions = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if w * h > MIN_AREA:
                led_positions.append((x + w//2, y + h//2))

        # 上下ペア判定
        if len(led_positions) >= 2:
            led_positions.sort(key=lambda p: p[1])  # y軸でソート
            pairs = []
            for i in range(len(led_positions)):
                for j in range(i+1, len(led_positions)):
                    dx = abs(led_positions[i][0] - led_positions[j][0])
                    dy = abs(led_positions[i][1] - led_positions[j][1])
                    if dy >= MIN_VERTICAL_DIST and dx < MAX_HORIZONTAL_OFFSET:
                        upper = led_positions[i] if led_positions[i][1] < led_positions[j][1] else led_positions[j]
                        lower = led_positions[j] if upper == led_positions[i] else led_positions[i]
                        pairs.append((upper, lower))

            for upper_led, lower_led in pairs:
                cx = (upper_led[0] + lower_led[0]) // 2
                cy = (upper_led[1] + lower_led[1]) // 2
                panel_h = int(abs(upper_led[1] - lower_led[1]) * 1.5)
                panel_w = int(abs(upper_led[1] - lower_led[1]) * 2)

                detected_panels.append({
                    'color': color,
                    'center': (cx, cy),
                    'w': panel_w,
                    'h': panel_h
                })

    # 描画
    display = working_image.copy()
    for panel in detected_panels:
        cx, cy = panel['center']
        w, h = panel['w'], panel['h']
        x1 = max(0, cx - w//2)
        y1 = max(0, cy - h//2)
        x2 = min(display.shape[1]-1, cx + w//2)
        y2 = min(display.shape[0]-1, cy + h//2)
        # パネル領域
        if panel['color'] == 'blue':
            col = (255, 0, 0)
        else:
            col = (0, 0, 255)
        cv2.rectangle(display, (x1, y1), (x2, y2), col, 2)
        cv2.circle(display, panel['center'], 5, (0,255,0), -1)
        # ラベル
        label = f"{panel['color'].upper()}"
        cv2.putText(display, label, (cx-40, cy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)

    cv2.imshow('LED Detection', display)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):  # ソフト補正の切り替え（任意で使う）
        use_soft_correction = not use_soft_correction

pipeline.stop()
cv2.destroyAllWindows()
