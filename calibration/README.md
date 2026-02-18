# Calibration Tools (main_camera / panel_recog_camera)

`damaage_panel_recog` で座標変換を使うためのキャリブレーション手順です。  
命名規則は `main_camera` / `panel_recog_camera` に統一しています。

## 1) 画像取得

### 単体（内部パラメータ用）

```bash
python3 calibration/capture_calib_images.py --mode single --device /dev/video0 --camera-role main_camera --out-dir calib/main_camera
python3 calibration/capture_calib_images.py --mode single --device /dev/video2 --camera-role panel_recog_camera --out-dir calib/panel_recog_camera
```

### 同時（外部パラメータ用のペア画像）

```bash
python3 calibration/capture_calib_images.py --mode pair --main-device /dev/video0 --panel-recog-device /dev/video2 --out-main calib/main_camera --out-panel-recog calib/panel_recog_camera
```

ペア画像は次の名前で保存されます。

- `pair_0001_main_camera.png`
- `pair_0001_panel_recog_camera.png`

## 2) 内部パラメータ推定

```bash
python3 calibration/calibrate_intrinsics.py --img-dir calib/main_camera --out calib/intrinsics_main_camera.yaml --board-cols 4 --board-rows 7 --square-size 0.09
python3 calibration/calibrate_intrinsics.py --img-dir calib/panel_recog_camera --out calib/intrinsics_panel_recog_camera.yaml --board-cols 4 --board-rows 7 --square-size 0.09
```

## 3) 外部パラメータ推定

### 理想モデル（PoC）

```bash
python3 calibration/calibrate_extrinsics.py --mode ideal --baseline 0.35 --out calib/extrinsics_panel_recog_camera_to_main_camera.yaml
```

### ペア画像から推定（stereo）

```bash
python3 calibration/calibrate_extrinsics.py --mode stereo --dir-main calib/main_camera --dir-panel-recog calib/panel_recog_camera --intr-main calib/intrinsics_main_camera.yaml --intr-panel-recog calib/intrinsics_panel_recog_camera.yaml --out calib/extrinsics_panel_recog_camera_to_main_camera.yaml --board-cols 4 --board-rows 7 --square-size 0.09
```

## 4) 主なYAML命名例

- `calib/intrinsics_main_camera.yaml`
- `calib/intrinsics_main_camera_publish.yaml`（publish先用）
- `calib/intrinsics_panel_recog_camera.yaml`
- `calib/extrinsics_panel_recog_camera_to_main_camera.yaml`
- `calib/extrinsics_panel_recog_camera_to_main_camera_publish.yaml`（publish先用）
