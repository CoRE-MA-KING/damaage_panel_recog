# damage_panel_recog_and_tracking (refactored)

元の `damage_panel_recog_and_tracking.py` を **モジュール分割**し、
「検出（bbox生成） → 追跡（motpyなど差し替え可能） → 表示/Publish」
の境界をはっきりさせた構成です。

この構成にしておくと、将来的に **C++へ移植**する場合も
`damage_panel_tracking/tracking/` 以下を別実装（SORTなど）に置き換えるだけで済むようにできます。

---

## 依存関係

### 必須

- [mise](https://mise.jdx.dev)
  - 以下のパッケージを含みます(miseを使いたくない場合は、個別にインストールしてください) )
    - [uv](https://astral.sh/uv)
    - [buf](https://buf.build/product/cli)

### 任意

- v4l2-ctl（v4l-utilsに含まれる。起動時カメラ設定をしたい場合）

    ```bash
    sudo apt install v4l-utils
    ```

## 環境構築

```bash
mise build
source ./.venv/bin/activate
```

---

## 実行方法

### 基本（画像表示あり・追跡なし）

```bash
python3 damage_panel_recog_and_tracking.py
```

### デバイス指定（番号 or デバイスパス）

```bash
python3 damage_panel_recog_and_tracking.py -d 0
python3 damage_panel_recog_and_tracking.py -d /dev/video0
```

### 画像表示なし

```bash
python3 damage_panel_recog_and_tracking.py -n
```

### 設定GUI（トラックバー）

```bash
python3 damage_panel_recog_and_tracking.py -s
```

### トラッキング（motpy backend）

```bash
pip install motpy
python3 damage_panel_recog_and_tracking.py -t
```

### Zenoh publish / subscribe

```bash
pip install zenoh
python3 damage_panel_recog_and_tracking.py -p
```

`publish` / `subscribe` は `config/default.yaml` で個別制御できます。

- `publish.enabled: false` なら、検知は行うが publish はしません。
- `subscribe.enabled: false` なら、`subscribe.default_target` (`blue` or `red`) をターゲット色として使います。
- `--default-target red` のように CLI から `subscribe.default_target` を上書きできます。

```bash
python3 damage_panel_recog_and_tracking.py --subscribe
python3 damage_panel_recog_and_tracking.py --default-target red
```

### 座標変換付きpublish（panel_recog_camera -> main_camera）

`config/default.yaml` の `coordinate_transform` を有効化するか、
CLIで `--coord-transform` を付けると、認識結果を `main_camera` 座標へ変換して publish します。

```bash
python3 damage_panel_recog_and_tracking.py -p --coord-transform
```

publish先カメラ情報は `coordinate_transform.publish_main_camera` で指定します。

- `intrinsics_path`: publish先 `main_camera` の内部パラメータ
- `extrinsics_from_panel_recog_path`: `panel_recog_camera -> main_camera` 外部パラメータ
- `frame_size`: publish先アプリ上の `main_camera` 解像度

### main_camera重畳表示デバッグ

座標変換の確認用に、`damage_panel_recog` 側でも `main_camera` へ重畳表示できます。

```bash
python3 damage_panel_recog_and_tracking.py --coord-transform --main-overlay
python3 damage_panel_recog_and_tracking.py --coord-transform --main-overlay --main-camera-device /dev/video0
```

この表示は確認用途で、publish処理とは独立です（publish先アプリの `main_camera` は別設定を保持できます）。

### (計測用) ターゲット座標ログをCSVに保存

「映像中に映る物体は1つ以下」という前提で、**フレーム間の移動量（dx,dy）**や**速度(px/s)** の当たりをつける用途向けです。
動画録画よりも負荷が軽く、90fps付近のまま統計が取りやすいです。

```bash
python3 damage_panel_recog_and_tracking.py -l
```

出力先はデフォルトで `logs/motion_log_YYYYmmdd_HHMMSS.csv` です。明示する場合:

```bash
python3 damage_panel_recog_and_tracking.py -l --log-path logs/run1.csv
```

---

## 設定ファイル

デフォルトは `config/default.yaml` を読みます。

- YAMLを使う場合: `pip install pyyaml`

例:

```bash
python3 damage_panel_recog_and_tracking.py --config config/default.yaml
```

`coordinate_transform` セクションで次を管理します。

- `panel_recog_camera` の内部パラメータ
- publish先 `main_camera` の内部/外部パラメータ
- デバッグ重畳表示に使う `main_camera` の設定（必要なら publish先とは別管理）

---

## キャリブレーション手順

`stereo_overlay` で使っていた手順は `calibration/` に統合済みです。  
詳細は `calibration/README.md` を参照してください。

主なスクリプト:

- `calibration/capture_calib_images.py`
- `calibration/calibrate_intrinsics.py`
- `calibration/calibrate_extrinsics.py`

---

## 将来的に追跡部を差し替える場所

自分用メモ

- Interface: `damage_panel_tracking/tracking/base.py`
- motpy wrapper: `damage_panel_tracking/tracking/motpy_tracker.py`

SORTを自作する場合は、例えば

- `damage_panel_tracking/tracking/sort_tracker.py` を追加し
- `damage_panel_tracking/cli.py` の `_build_tracker()` に backend名を足す

という形で置き換えられる。
