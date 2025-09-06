import pyrealsense2 as rs
import numpy as np
import cv2

# RealSenseカメラの初期化
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
pipeline.start(config)

# 深度フレームをカラーに合わせるための設定
align_to = rs.stream.color
align = rs.align(align_to)

# HSVの色範囲をさらに調整（特に青色を狭めて誤検知を抑制）
COLOR_RANGES = {
    'blue': ([105, 180, 120], [125, 255, 255]),
    'red1': ([0, 150, 100], [10, 255, 255]),
    'red2': ([160, 150, 100], [180, 255, 255]),
}

# 指定された色のマスクを取得する関数
def get_led_mask(hsv, color):
    if color == 'blue':
        lower, upper = COLOR_RANGES['blue']
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
    elif color == 'red':
        lower1, upper1 = COLOR_RANGES['red1']
        lower2, upper2 = COLOR_RANGES['red2']
        mask1 = cv2.inRange(hsv, np.array(lower1), np.array(upper1))
        mask2 = cv2.inRange(hsv, np.array(lower2), np.array(upper2))
        mask = cv2.bitwise_or(mask1, mask2)
    return mask

while True:
    # フレーム取得と位置合わせ
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)

    color_frame = aligned_frames.get_color_frame()
    depth_frame = aligned_frames.get_depth_frame()
    if not color_frame or not depth_frame:
        continue

    color_image = np.asanyarray(color_frame.get_data())
    hsv         = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

    for color in ['blue', 'red']:
        # blue or red のマスク画像を得る
        mask = get_led_mask(hsv, color)

        # ノイズ除去のためのモルフォロジー処理（Opening処理）
        kernel = np.ones((3,3), np.uint8)
        mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask   = cv2.dilate(mask, kernel, iterations=1)

        # 輪郭抽出
        contours, _ = cv2.findContours(
            mask, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )

        # 輪郭ごとにバウンディングボックスを計算・描画
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if w * h > 200:  # ノイズ除去
                # 描画色を BGR で指定
                if color == 'blue':
                    box_color = (255, 0, 0)
                else:  # 'red'
                    box_color = (0, 0, 255)
                cv2.rectangle(
                    color_image,
                    (x, y),
                    (x + w, y + h),
                    box_color,
                    thickness=2
                )

    # 描画結果を表示
    cv2.imshow('LED Detection', color_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

pipeline.stop()
cv2.destroyAllWindows()
