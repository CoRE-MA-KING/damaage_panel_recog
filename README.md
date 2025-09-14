# damage_panel_recog

画像処理のテスト実装アレコレ

## 環境

動作確認済み環境

- Ubuntu 24.04 Desktop (kernel: 6.14.0-29-generic)
- Python3.12

依存関係のインストール（なにか足りてないかも）

```bash
pip3 install numpy opencv-python pydantic eclipse-zenoh 
pip3 intall pyrealsense2  # 入らないディストリビューションもある
```

pyrealsense2が入らない場合は、[ソースビルド＆インストール](./librealsense2_source_build.md)をしてください。

## panel_recog_camera.py

上下のLEDをもとにパネルを認識するプログラム。
各種パラメータを調整します。ダメージパネルのLEDの輝度が高すぎて常に白飛びしているため、現時点では基本は露光のシャッタースピードを調整する想定。
いまの設定は基本固定で、試合時のパネルの輝度に合わせて`RS_EXPOSURE `によってシャッタースピードのみを変更すると良いと思っている。
今は赤色と青色のパネルを同時に検知しているが、試合時にはどちらか片方のみでいいはず。それを分けるとより誤検知は減ると思われる。

オプションによって`roboapp`向けに認識結果をpublishしたり、パラメータ設定モードで起動するかが変わる。パラメータは同時に複数指定可能だヨ。

```bash
# 画像処理単体で実行
python3 panel_recog_camera.py
```

オプション

```bash
-p / --publish : roboapp向けに照準対象をpublishするモード
-s / --setting : カメラパラメータと認識対象LEDの閾値HSVを設定を動的に行うモード
なし : 単体でのプログラム実行（カメラ映像+認識結果を画面に表示するだけ）
```

## panel_recog_static.py

過去の試合のカメラ画像を指定したHSVの範囲で抽出を試みたもの。
ただし、白飛びが大きくなかなかうまくいかなかった。

```bash
python3 panel_recog_static.py  --
```

## camera_parameter_setting.py

カメラパラメータを調整して見え方を可視化するツール。
青色と赤色の領域が抽出できるかも同時に試す。

```bash
python3 camera_parameter_setting.py
```