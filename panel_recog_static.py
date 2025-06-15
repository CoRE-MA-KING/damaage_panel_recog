#!/usr/bin/env python3
"""
recg_static.py

複数の静止画像中のパネルLEDをまとめて検出し、
バウンディングボックスと色ラベルを描画して
result/{元画像名}_result.png に一括保存します。

Usage:
    python recg_static.py path/to/img1.png path/to/img2.png ...
    python recg_static.py "images/*.png"
"""

import os
import sys
import glob
import cv2
import numpy as np

def detect_led_panels(image):
    """
    画像中のLEDバーを検出し、バウンディングボックスと色ラベルを描画します。
    Returns annotated image and list of detections (bbox, color_name).
    """
    # 前処理：ブラー＋HSV変換
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    hsv     = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # LED色ごとのHSV閾値
    COLOR_RANGES = {
        'BLUE':   [([85, 0,  242], [95, 77, 255])],
        'RED':    [([25, 0,  242], [35, 77, 255])],
        'YELLOW': [([20, 150, 150], [40, 255, 255])],
        'GREEN':  [([40,  50,  50], [80, 255, 255])],
    }

    detections = []

    for color_name, ranges in COLOR_RANGES.items():
        mask = None
        for lower, upper in ranges:
            lower = np.array(lower, dtype=np.uint8)
            upper = np.array(upper, dtype=np.uint8)
            m = cv2.inRange(hsv, lower, upper)
            mask = m if mask is None else cv2.bitwise_or(mask, m)

        # ノイズ除去
        kernel_open  = np.ones((3,3), np.uint8)
        kernel_close = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel_open)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

        # 輪郭抽出
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            # 横長かつ一定面積以上のものをLEDバーとみなす
            if w / float(h) > 3.0 and w * h > 500:
                # 描画
                cv2.rectangle(image, (x, y), (x+w, y+h), (0,255,0), 2)
                cv2.putText(image, color_name, (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                detections.append(((x, y, w, h), color_name))

    return image, detections

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    # ワイルドカード対応：引数ごとに glob で展開
    input_patterns = sys.argv[1:]
    img_paths = []
    for pat in input_patterns:
        # glob で一致するファイルをリストに追加
        found = glob.glob(pat)
        if not found:
            print(f"Warning: no files match pattern {pat}")
        img_paths.extend(found)

    if not img_paths:
        print("Error: no images to process.")
        sys.exit(1)

    # 出力ディレクトリ
    result_dir = "result"
    os.makedirs(result_dir, exist_ok=True)

    for img_path in sorted(set(img_paths)):
        # ファイル存在チェック
        if not os.path.isfile(img_path):
            print(f"Skipping non-file {img_path}")
            continue

        # 画像読み込み
        image = cv2.imread(img_path)
        if image is None:
            print(f"Failed to load {img_path}, skipping.")
            continue

        # 検出＆描画
        annotated, detections = detect_led_panels(image)

        # 保存パス作成
        base = os.path.basename(img_path)
        name, _ = os.path.splitext(base)
        out_path = os.path.join(result_dir, f"{name}_result.png")

        # 結果保存
        cv2.imwrite(out_path, annotated)
        print(f"[{len(detections)} detections] {img_path} -> {out_path}")

if __name__ == "__main__":
    main()
