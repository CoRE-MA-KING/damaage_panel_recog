# damage_panel_recog_and_tracking (refactored)

元の `damage_panel_recog_and_tracking.py` を **モジュール分割**し、
「検出（bbox生成） → 追跡（motpyなど差し替え可能） → 表示/Publish」
の境界をはっきりさせた構成です。

この構成にしておくと、将来的に **C++へ移植**する場合も
`damage_panel_tracking/tracking/` 以下を別実装（SORTなど）に置き換えるだけで済むようにできます。

---

## 依存関係

必須:
- Python 3
- numpy
- opencv-python
- pydantic

任意:
- pyyaml（YAML config使用時）
- motpy（tracking backend=motpy 使用時）
- zenoh（publish時）
- v4l2-ctl（v4l-utilsに含まれる。起動時カメラ設定をしたい場合）

```bash
pip install -r requirements.txt
sudo apt install v4l-utils
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

### Zenoh publish

```bash
pip install zenoh
python3 damage_panel_recog_and_tracking.py -p
```

---

## 設定ファイル

デフォルトは `config/default.yaml` を読みます。

- YAMLを使う場合: `pip install pyyaml`
- JSONで使う場合: `--config config/default.json`

例:

```bash
python3 damage_panel_recog_and_tracking.py --config config/default.yaml
```

---

## 将来的に追跡部を差し替える場所

自分用メモ

- Interface: `damage_panel_tracking/tracking/base.py`
- motpy wrapper: `damage_panel_tracking/tracking/motpy_tracker.py`

SORTを自作する場合は、例えば

- `damage_panel_tracking/tracking/sort_tracker.py` を追加し
- `damage_panel_tracking/cli.py` の `_build_tracker()` に backend名を足す

という形で置き換えられる。
