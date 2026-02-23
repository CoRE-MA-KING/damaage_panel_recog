# Calibration Tools (main_camera / panel_recog_camera)

`damaage_panel_recog` で座標変換を使うためのキャリブレーション手順です。  
命名規則は `main_camera` / `panel_recog_camera` に統一しています。

## 1) 画像取得

メインカメラと認識用カメラをケースに取り付けて同時に画像を撮影します。

```bash
uv run python3 calibration/capture_calib_images.py --main-device /dev/video4 --panel-recog-device /dev/video6 --out-main calib/main_camera --out-panel-recog calib/panel_recog_camera --width 1280 --height 720
```

`--width/--height` は両カメラ共通値として使えます。  
`fps` は引数未指定時に両カメラともデフォルトで `90` が使われます。

ペア画像は次の名前で保存されます。

- `pair_0001_main_camera.png`
- `pair_0001_panel_recog_camera.png`

## 2) 内部パラメータ推定

メインカメラ側

```bash
uv run python3 calibration/calibrate_intrinsics.py --img-dir calib/main_camera --out calib/intrinsics_main_camera.yaml --board-cols 5 --board-rows 9 --square-size 0.005
```

認識用カメラ側

```bash
uv run python3 calibration/calibrate_intrinsics.py --img-dir calib/panel_recog_camera --out calib/intrinsics_panel_recog_camera.yaml --board-cols 5 --board-rows 9 --square-size 0.05
```

## 3) 外部パラメータ推定

```bash
uv run python3 calibration/calibrate_extrinsics.py --dir-main calib/main_camera --dir-panel-recog calib/panel_recog_camera --intr-main calib/intrinsics_main_camera.yaml --intr-panel-recog calib/intrinsics_panel_recog_camera.yaml --out calib/extrinsics_panel_recog_camera_to_main_camera.yaml --board-cols 5 --board-rows 9 --square-size 0.05
```

## 4) 主なYAML命名例

- `calib/intrinsics_main_camera.yaml`
- `calib/intrinsics_panel_recog_camera.yaml`
- `calib/extrinsics_panel_recog_camera_to_main_camera.yaml`
