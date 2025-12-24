import pyrealsense2 as rs
import numpy as np
import cv2
import argparse

def main(bag_file: str, realtime_flag: bool = False):
    # パイプラインと設定の初期化
    pipeline = rs.pipeline()
    config = rs.config()
    rs.config.enable_device_from_file(config, bag_file, repeat_playback=True)

    # ストリーミングの開始
    profile = pipeline.start(config)

    # カラライザの作成
    colorizer = rs.colorizer()

    # 再生デバイスの取得とリアルタイム再生の設定
    device = profile.get_device()
    playback = device.as_playback()
    playback.set_real_time(realtime_flag)

    # バッグファイルの総再生時間の取得
    total_duration = playback.get_duration().total_seconds()

    # ストリームプロファイルから解像度とFPSを取得して表示
    streams = profile.get_streams()
    for stream in streams:
        video_stream = stream.as_video_stream_profile()
        width = video_stream.width()
        height = video_stream.height()
        fps = video_stream.fps()
        stream_type = stream.stream_type()
        format = video_stream.format()
        stream_name = stream.stream_name()
        stream_type_str = str(stream_type).split('.')[-1]
        format_str = str(format).split('.')[-1]
        print(f"Stream: {stream_name}, Type: {stream_type_str}, Format: {format_str}, Resolution: {width}x{height}, FPS: {fps}")

    try:
        while True:
            # フレームの取得
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            if not depth_frame or not color_frame:
                # 再生が終了したか確認
                if playback.current_status() == rs.playback_status.stopped:
                    print("\nReached the end of the bag file.")
                    break
                continue

            # フレームをnumpy配列に変換
            depth_image = np.asanyarray(colorizer.colorize(depth_frame).get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # 現在の再生位置を取得（秒単位）
            current_time = playback.get_position() / 1000000000.0  # ナノ秒を秒に変換

            # 進捗率を計算
            progress = min((current_time / total_duration) * 100, 100)

            # 進捗を表示（小数点以下2桁まで表示）
            print(f'\rProgress: {current_time:.2f}s / {total_duration:.2f}s ({progress:.2f}%)', end='')

            # 画像を表示
            cv2.imshow('RealSense Color', color_image)
            cv2.imshow('RealSense Depth', depth_image)

            # 'q'キーで終了
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except RuntimeError as e:
        print("\nPlayback has ended.")
    finally:
        # ストリーミングを停止
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    # コマンドライン引数の処理
    parser = argparse.ArgumentParser(description="RealSense bag file viewer.")
    parser.add_argument("bag_file", type=str, help="Path to the bag file.")
    parser.add_argument(
        "realtime_flag",
        type=str,
        nargs="?",  # オプション扱いにする
        default="False",
        help="Set real-time playback (True or False)."
    )
    args = parser.parse_args()

    # `realtime_flag` を明示的に変換
    if args.realtime_flag.lower() in ("true", "1"):
        realtime_flag = True
    elif args.realtime_flag.lower() in ("false", "0"):
        realtime_flag = False
    else:
        print("Invalid value for realtime_flag. Use 'True' or 'False'.")
        exit(1)

    # メイン関数を呼び出し
    main(args.bag_file, realtime_flag)