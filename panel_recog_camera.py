import pyrealsense2 as rs
import numpy as np
import cv2

# ========= 可調整パラメータ =========
AREA_MIN = 200                 # 単色領域の最小面積(px^2)
KERNEL_SZ = 3                  # モルフォロジー用カーネル
WIDTH_TOL = 0.25               # 上下の幅差許容（最大幅の何割まで許容）
MIN_H_OVERLAP = 0.50           # 上下の水平オーバーラップの下限（min(w_top,w_bottom)に対する比）
MIN_V_GAP = 2                  # 上下の最小垂直間隔（px）
# ===================================

# RealSense 初期化
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
pipeline.start(config)

align_to = rs.stream.color
align = rs.align(align_to)

# HSV 範囲
COLOR_RANGES = {
    'blue': ([105, 180, 120], [125, 255, 255]),
    'red1': ([0, 180, 120], [10, 255, 255]),
    'red2': ([160, 180, 120], [180, 255, 255]),
}

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
    else:
        return None

def find_boxes(mask):
    """マスク→輪郭→(x,y,w,h) のリストを返す（面積フィルタ込み）"""
    kernel = np.ones((KERNEL_SZ, KERNEL_SZ), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h >= AREA_MIN:
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
    """
    同色ボックス群から上下ペアを作る。
    条件: 下は上より下側 / 幅差<=WIDTH_TOL / 水平オーバーラップ>=MIN_H_OVERLAP / 垂直間隔>=MIN_V_GAP
    マッチングは「上ボックスから見て最も近い下ボックス」を優先（1対1対応）。
    """
    # y 位置でソート（上から下へ）
    boxes_sorted = sorted(boxes, key=lambda b: b[1])
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

            # 下側にあるか
            dy = (y_b) - (y_t + h_t)
            if dy < MIN_V_GAP:
                continue

            # 幅差判定
            if abs(w_t - w_b) > WIDTH_TOL * max(w_t, w_b):
                continue

            # 水平オーバーラップ
            if horiz_overlap_ratio(top, bottom) < MIN_H_OVERLAP:
                continue

            # 最も近い下ボックスを選ぶ
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

while True:
    frames = pipeline.wait_for_frames()
    aligned = align.process(frames)
    color_frame = aligned.get_color_frame()
    depth_frame = aligned.get_depth_frame()
    if not color_frame or not depth_frame:
        continue

    color_image = np.asanyarray(color_frame.get_data())
    hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

    # 色ごとに矩形候補を抽出
    boxes_by_color = {}
    for color in ['blue', 'red']:
        mask = get_led_mask(hsv, color)
        boxes_by_color[color] = find_boxes(mask)

    # 同色の上下ペアを検出
    all_pairs = []
    for color in ['blue', 'red']:
        pairs = pair_boxes_same_color(boxes_by_color[color])
        # 描画（個々の上/下は薄色、ペア矩形は緑・中心は小円）
        for (top, bottom) in pairs:
            # 個別の箱を半透明風（枠だけ）で描く
            box_col = (255, 0, 0) if color == 'blue' else (0, 0, 255)
            for (x, y, w, h) in (top, bottom):
                cv2.rectangle(color_image, (x, y), (x + w, y + h), box_col, 1)

            # ペアの最小外接矩形＋中心
            (ux, uy, uw, uh), (cx, cy) = bbox_union(top, bottom)
            cv2.rectangle(color_image, (ux, uy), (ux + uw, uy + uh), (0, 255, 0), 2)
            cv2.circle(color_image, (int(cx), int(cy)), 3, (0, 255, 0), -1)
            cv2.putText(
                color_image,
                f"{color} cx,cy=({int(cx)},{int(cy)})",
                (ux, max(0, uy - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA
            )
            all_pairs.append((color, (ux, uy, uw, uh), (cx, cy)))

    # 必要に応じて座標をログ出力（例）
    # print(all_pairs)

    cv2.imshow('Panel (paired by same-color top & bottom)', color_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

pipeline.stop()
cv2.destroyAllWindows()
