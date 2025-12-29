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

デフォルト設定
```python
RS_EXPOSURE     = 10
RS_GAIN         = 0
RS_WHITEBALANCE = 4600
RS_BRIGHTNESS   = 0
RS_CONTRAST     = 50
RS_SHARPNESS    = 50
RS_SATURATION   = 50
RS_GAMMA        = 100

AREA_MIN       = 100
KERNEL_SZ      = 3
WIDTH_TOL      = 0.25
MIN_H_OVERLAP  = 0.50
MIN_V_GAP      = 2
MIN_BOX_H      = 10
MIN_BOX_W      = 50

# --- 追跡表示用 ---
TRACK_MIN_STEPS = 2       # 何フレーム以上生存で可視化するか
TRACK_HISTORY   = 20      # 軌跡の履歴長
TRACK_COLOR     = (0, 255, 255)  # 黄

HSV_INIT = {
    "blue":  {"H_low":110, "H_high":135, "S_low":180, "S_high":255, "V_low":120, "V_high":255},
    "red1":  {"H_low":  0, "H_high": 15},
    "red2":  {"H_low":165, "H_high":179},
    "redSV": {"S_low":180, "S_high":255, "V_low":120, "V_high":255},
}
```


オプションによって`roboapp`向けに認識結果をpublishしたり、パラメータ設定モードで起動するかが変わる。パラメータは同時に複数指定可能だヨ。

```bash
# 画像処理単体で実行
python3 panel_recog_camera.py
```

試合用（zenohでpublish + 画像表示なし + トラッキングあり）

```bash
python3 panel_recog_camera.py -p -n -t
```

オプション

```bash
-p / --publish : roboapp向けに照準対象をpublishするモード
-n / --no-display: カメラ画像+処理結果を画面上に表示しない
-s / --setting : カメラパラメータと認識対象LEDの閾値HSVを設定を動的に行うモード
-t / --tracking : トラッキングを行うモード（照準対象の決定にも関わる）
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

# damage_panel_normal_camera

## LEDパネル検出・トラッキング（OpenCV + v4l2-ctl + Zenoh 任意）

上下2本のLED（同色）を**上下ペア**として検出し、その**union矩形の中心**をターゲット座標として出力するプログラム（内容は、`damage_panel_recog`と同じ）。
オプションで **motpy による多目標トラッキング**、**OpenCVトラックバーによる調整GUI**、**Zenoh publish** を有効化できる。

## 1. 機能概要

- **カメラ入力**: `/dev/video*`（OpenCV `cv2.VideoCapture`）
- **起動時カメラ設定**: `v4l2-ctl --set-ctrl` で各種パラメータを強制適用（失敗時は `cap.set` にフォールバック）
- **LED検出**:
  - HSVで **blue / red** を二値化（red は Hue を2レンジに分割）
  - モルフォロジー（open）＋ dilation
  - 輪郭抽出 → bbox化 → 小さすぎるbboxを除外
  - **上下ペアリング**（同色 top/bottom）し、union bbox を作成
- **ターゲット決定**:
  - トラッキングOFF: union中心が画面中心に最も近いものを採用
  - トラッキングON: motpyで追跡したトラックの中心が画面中心に最も近いものを採用
- **表示**:
  - 検出枠（top/bottom、union）
  - トラッキング枠（黄）、軌跡、連番ID、中心座標
  - ターゲットマーカー、FPS、モード表示
- **Zenoh publish（任意）**:
  - `robot/command/<field>` に `target_x`, `target_y`, `depth`, `dummy` 等を publish

## 2. 実行方法

### 2.1 基本実行（表示あり）

```bash
python3 your_script.py
```

### 2.2 デバイス指定（番号 or パス）

```bash
python3 your_script.py -d 0
python3 your_script.py -d /dev/video2
```

### 2.3 GUIなし（表示しない）

```bash
python3 your_script.py -n
```

### 2.4 設定GUI（トラックバーで調整）

```bash
python3 your_script.py -s
```

- -n を付けた場合はGUIも無効化される（no-display優先）

### 2.5 motpyトラッキング有効

```bash
python3 your_script.py -t
```

- motpy が無い場合は警告を出し、追跡無しで継続

### 2.6 Zenoh publish 有効

```bash
python3 your_script.py -p
```

zenoh が無い場合はエラー終了

## 3. 依存関係

- 必須
    - Python 3
    - opencv-python（cv2）
    - numpy
- 任意
    - v4l2-ctl（v4l-utils に含まれる）
    - motpy（--track 使用時）
    - zenoh（--publish 使用時）
    - domain.message.RobotCommand（存在すれば publish key をそれに合わせる）

```bash
pip install opencv-python numpy
sudo apt-get install v4l-utils

pip install motpy      # --track を使う場合
pip install zenoh      # --publish を使う場合
```

## 4. コマンドライン引数

| option              | 意味                            |
| ------------------- | ----------------------------- |
| `-p / --publish`    | Zenohに publish する             |
| `-n / --no-display` | 映像表示しない（GUIも無効化）              |
| `-s / --setting`    | トラックバーGUIでカメラ/HSV調整           |
| `-t / --track`      | motpyで多目標トラッキング有効             |
| `-d / --device`     | デバイス指定（例: `0`, `/dev/video0`） |

## 5. カメラ設定（起動時に強制適用）

### 5.1 固定キャプチャ仕様
- 解像度: `CAM_FRAME_WIDTH=640`, `CAM_FRAME_HEIGHT=360`
- FPS: `CAM_FPS=90`
- FourCC: `CAM_FOURCC="MJPG"`（`VideoWriter_fourcc(*"MJPG")`）

### 5.2 起動時の設定反映フロー
- `setup_camera()` で `cv2.VideoCapture()` を作成後、固定キャプチャ設定を `cap.set()` で投入
- 続いて `apply_camera_init(dev_path, cap)` を **起動時に1回だけ必ず実行**して、カメラ制御値を反映
- `dev_path` は `-d 0` のような指定も `/dev/video0` に正規化して `v4l2-ctl` に渡す（正規化できない場合は `v4l2-ctl` は無効）

### 5.3 v4l2-ctl による適用（基本）
- `v4l2-ctl --set-ctrl=<name>=<value>` を順に実行し、失敗しても例外で落とさず続行（戻り値で成功/失敗のみ判定）
- オート系は **先に止めてから** 個別値を入れる（順序依存を回避）
  - 露光: `auto_exposure` → `exposure_time_absolute`
  - WB: `white_balance_automatic` → `white_balance_temperature`
  - フォーカス: `focus_automatic_continuous` → `focus_absolute`（inactiveの可能性があるため最後）
- 主な設定項目（例）
  - 画像系: `brightness`, `contrast`, `saturation`, `hue`, `gamma`, `sharpness`, `backlight_compensation`
  - 露光/ゲイン: `auto_exposure`, `exposure_time_absolute`, `gain`
  - フリッカ対策: `power_line_frequency`（0/50Hz/60Hz）
  - 位置系: `pan_absolute`, `tilt_absolute`, `zoom_absolute`

### 5.4 OpenCV cap.set フォールバック（補助）
- `v4l2-ctl` が無い／効かない場合に備えて、同等項目を `cap.set()` でも試行（失敗は無視）
- 対象例: `CAP_PROP_BRIGHTNESS`, `CONTRAST`, `SATURATION`, `GAIN`, `GAMMA`, `SHARPNESS`, `EXPOSURE`
- WB温度は環境によって `CAP_PROP_WB_TEMPERATURE` で反映する場合があるため try で試す

### 5.5 反映待ち
- 設定投入後、機種によって即時反映しないケースがあるため `time.sleep(0.05)` を挿入


## 6. LED検出ロジック

### 6.1 HSVマスク生成（blue / red）
- フレームを `cv2.cvtColor(BGR→HSV)` 変換
- `get_led_mask()` で色ごとに `cv2.inRange()` を生成
  - blue: 1レンジ（H/S/Vすべて min/max）
  - red: Hueが循環するため2レンジ（`red1`, `red2`）＋S/V共通（`redSV`）、最後にOR合成
- GUI時はトラックバーで HSV閾値を直接更新できる

### 6.2 bbox抽出
- マスクに対して
  1) `MORPH_OPEN`（`KERNEL_SZ`）  
  2) `dilate(iterations=1)`  
  3) `findContours(RETR_EXTERNAL)`  
  4) `boundingRect()`  
- `MIN_BOX_H`, `MIN_BOX_W` 未満の小bboxは除外
- 出力: `(x, y, w, h)` のリスト


## 7. 上下ペアリング（同色 top & bottom）

### 7.1 ペア条件
- bboxを `y` 昇順で走査し、上側(top)に対して下側(bottom)候補を探索
- 条件
  - 縦方向の隙間: `dy = y_b - (y_t + h_t) >= MIN_V_GAP`
  - 幅の近さ: `abs(w_t - w_b) <= WIDTH_TOL * max(w_t, w_b)`
  - 横方向重なり率: `horiz_overlap_ratio(top,bottom) >= MIN_H_OVERLAP`
- 条件を満たす中で `dy` が最小の bottom を採用（bottomは再利用しない）

### 7.2 union bbox と中心
- top/bottom から union bbox `(ux,uy,uw,uh)` を作成
- 併せて中心 `(cx,cy)` を算出
- 検出メタとして `xyxy`, `union`, `top`, `bottom`, `color` を保持（追跡時の描画対応付けに使う）


## 8. ターゲット選択

### 8.1 トラッキングOFF
- union中心 `cx` が画面中心 `CAM_FRAME_WIDTH/2` に最も近い検出をターゲットにする

### 8.2 トラッキングON（motpy）
- union bbox を `Detection(box=xyxy, score=...)` に変換して `tracker.step()`
- `active_tracks(min_steps_alive=TRACK_MIN_STEPS)` のトラック集合から選ぶ
- トラック中心が画面中心に最も近いものをターゲットにする
- 追跡時は、各トラックと検出（union）をIoUで対応付けして、上下LED bboxも描画できるようにする


## 9. motpy トラッキング設定（要点）

- `model_spec` により状態モデルとノイズを指定
  - 例: `order_pos`, `dim_pos`, `q_var_pos`, `r_var_pos`
- `dt` は起動時 `1/CAM_FPS`、ループ内で実測 `dt` に更新
- `max_staleness` などは属性が存在する場合のみ反映
- 表示用に `t.id` を連番へマッピングし、軌跡（deque）を保持して折れ線描画


## 10. 表示（描画内容の要点）

- 検出のみ（追跡OFF時）
  - top/bottom bbox（青 or 赤）
  - union bbox（緑）＋中心点＋座標ラベル
- 追跡ON時
  - track bbox（黄）＋ID/中心座標ラベル＋軌跡
  - 対応する top/bottom bbox（青 or 赤）
- 共通
  - ターゲットマーカー（トラック由来=シアン、検出由来=白）
  - モード表示、FPS表示（EMA）
- 終了
  - ウインドウがある場合: `q` で終了
  - ない場合: `sleep(0.01)` しつつ継続


## 11. Zenoh publish（要点）

- `--publish` 指定時のみ `zenoh` を import・`session = zenoh.open(...)`
- `declare_publishers()` で publish key を決定
  - `domain.message.RobotCommand` が取れればその `model_fields.keys()`
  - 取れなければフォールバック（`target_x`, `target_y`, `depth`, `dummy`）
- publish 先: `robot/command/<key>`
- 現状の publish 値: `target_x`, `target_y`, `depth(0.0)`, `dummy(0)`
