#!/usr/bin/env python3
import cv2
import time

DURATION_SEC = 5.0  # 測定時間（秒）

def main():
    cap = cv2.VideoCapture("/dev/video4")

    if not cap.isOpened():
        print("カメラをオープンできませんでした")
        return

    # --- ここでフォーマット・解像度・FPSを指定 ---
    # フォーマット: MJPG を指定
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    cap.set(cv2.CAP_PROP_FOURCC, fourcc)

    # 解像度: 1280x720
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # FPS: 90（ドライバ側が無視する場合もあるので目標値として）
    cap.set(cv2.CAP_PROP_FPS, 90)

    # 実際に設定された値を確認（参考）
    print("Width :", cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    print("Height:", cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("FPS(prop):", cap.get(cv2.CAP_PROP_FPS))
    print(f"{DURATION_SEC}秒間フレームを読み込んで平均FPSを計測します…")

    # 計測用
    start_time = time.perf_counter()
    frames = 0

    while True:
        now = time.perf_counter()
        if now - start_time >= DURATION_SEC:
            break

        ret, frame = cap.read()
        if not ret:
            print("フレームを取得できませんでした（途中で停止）")
            break

        frames += 1
        # 画面描画や処理は一切しない

    end_time = time.perf_counter()
    elapsed = end_time - start_time

    print("=== 計測結果 ===")
    print(f"経過時間: {elapsed:.3f} 秒")
    print(f"取得フレーム数: {frames}")
    if elapsed > 0 and frames > 0:
        avg_fps = frames / elapsed
        print(f"平均FPS: {avg_fps:.2f}")
    else:
        print("有効な計測ができませんでした")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
