# damaage_panel_recog

`panel_recog_camera` で検出したダメージパネル座標を、
必要に応じて `main_camera` 座標へ変換して Zenoh publish するパッケージです。

この README は、インストール後に「上から順に実行すれば準備から実行まで完了する」流れで記載しています。

## 0. 前提（環境・依存パッケージ）

- OS: Ubuntu 24.04
- Python: `>=3.12`（`pyproject.toml` 準拠）
- `v4l2-ctl`（カメラ初期設定を有効
    ```bash
    sudo apt install v4l-utils
    ```

- [mise](https://mise.jdx.dev)
  - 以下のパッケージを含みます(miseを使いたくない場合は、個別にインストールしてください) 
    - [uv](https://astral.sh/uv)
    - [buf](https://buf.build/product/cli)

## 1. セットアップ

リポジトリルート（この `README.md` があるディレクトリ）で実行します。

```bash
mise install
mise build
```

`mise build` で以下が実行されます。

- Protobufコード生成（`buf generate`）
- Python依存関係の同期（`uv sync`）

uvコマンドを使わずに操作をしたい場合は、仮想環境を有効化してください。

```bash
source .venv/bin/activate
```

## 2. 最小実行（まず動作確認）

```bash
python3 damage_panel_recog_and_tracking.py
```

よく使うオプション:

```bash
# デバイス指定
python3 damage_panel_recog_and_tracking.py -d /dev/video0

# 画面表示なし
python3 damage_panel_recog_and_tracking.py -n

# トラックバー設定UI
python3 damage_panel_recog_and_tracking.py -s

# 設定ファイル指定
python3 damage_panel_recog_and_tracking.py --config config/default.yaml
```

## 3. Zenoh publish / subscribe

`publish` / `subscribe` を使う場合は、先に Zenoh ルーターと設定ファイル
`~/.config/roboapp/zenoh.json5` を準備してください
（`utils/configurator/README.md` 参照）。

```bash
# publish
python3 damage_panel_recog_and_tracking.py -p

# subscribeでターゲット色を受信
python3 damage_panel_recog_and_tracking.py --subscribe

# subscribe無効時のデフォルト色上書き
python3 damage_panel_recog_and_tracking.py --default-target red
```

`publish` / `subscribe` のキーや有効/無効は `config/default.yaml` で管理します。

## 4. 座標変換付き運用（panel_recog_camera -> main_camera）

### 4-1. キャリブレーション実施

以下を先に準備します。

- `panel_recog_camera` の内部パラメータ
- publish先 `main_camera` の内部パラメータ
- `panel_recog_camera -> main_camera` の外部パラメータ

手順は `calibration/README.md` を参照してください。

- `calibration/README.md`
- `calibration/checkerboard/README.md`（チェッカーボード表示手順）

### 4-2. 設定ファイル反映

`config/default.yaml` の `coordinate_transform` を更新します。

- `panel_recog_camera.intrinsics_path`
- `publish_main_camera.intrinsics_path`
- `publish_main_camera.extrinsics_from_panel_recog_path`
- `publish_main_camera.frame_size`

### 4-3. 座標変換有効で実行

```bash
# 変換してpublish
python3 damage_panel_recog_and_tracking.py -p --coord-transform

# main_camera重畳表示デバッグも有効
python3 damage_panel_recog_and_tracking.py -p --coord-transform --main-overlay

# デバッグ表示のmain_cameraデバイス指定
python3 damage_panel_recog_and_tracking.py -p --coord-transform --main-overlay --main-camera-device /dev/video0
```

## 5. ログ取得（任意）

```bash
python3 damage_panel_recog_and_tracking.py -l
python3 damage_panel_recog_and_tracking.py -l --log-path logs/run1.csv
```

## 6. 補助READMEへのリンク

機能別の詳細は以下を参照してください。

- キャリブレーション: `calibration/README.md`
- チェッカーボード表示: `calibration/checkerboard/README.md`
- ユーティリティ全体: `utils/README.md`
- configurator（systemd/自動起動設定）: `utils/configurator/README.md`
- Zenoh通信サンプル: `utils/examples/README.md`
- Protobuf定義: `utils/proto/README.md`

## 7. 参考情報

- エントリポイント: `damage_panel_recog_and_tracking.py`
- 実装本体: `damage_panel_tracking/`
- 変換ロジック: `damage_panel_tracking/transform/projection.py`
