# librealsense2とpyrealsense2をソースビルドしてインストールする

librealsense2をPCにインストールしたあと、Pythonバインディングであるpyrealsense2をインストールします。
pyrealsense2はwheelファイルを作成し、任意のワークスペースに都度インストールします。
下は Python 3.12（cp312） の例。3.11 も必要なら同じ流れで .venv-cp311 / build-cp311 みたいに作ればOK。

## Realsense-SDK本体をインストール（Viewerも入れる）

```bash
# 依存
sudo apt-get update
sudo apt-get install -y build-essential cmake git pkg-config unzip \
  libusb-1.0-0-dev libudev-dev libssl-dev \
  libgl1-mesa-dev libglu1-mesa-dev freeglut3-dev libglfw3-dev \
  qtbase5-dev qtdeclarative5-dev qml-module-qtquick-controls qml-module-qtquick-controls2

# 取得（この認識プログラムでは"2.56.5"を使用）
cd ~
git clone -b <任意のリリースタグを指定> --depth=1 https://github.com/IntelRealSense/librealsense.git
cd librealsense

# udev & 権限付与（apt版のlibrealsenseのudevがあると競合するので削除しておく）
sudo ./scripts/setup_udev_rules.sh
sudo udevadm control --reload-rules && sudo udevadm trigger
sudo usermod -aG video,plugdev $USER   # 再ログインで反映

# 本体＋viewer を RSUSB でビルド&インストール
rm -rf build && mkdir build && cd build
cmake .. -DFORCE_RSUSB_BACKEND=ON -DBUILD_TOOLS=ON -DBUILD_GRAPHICAL_EXAMPLES=ON \
         -DCHECK_FOR_UPDATES=OFF -DCMAKE_BUILD_TYPE=Release
make -j"$(nproc)"
sudo make install && sudo ldconfig
```

## Pythonバインディングをインストール

wheelファイルを作成して、任意のプロジェクトでインストールするようにします。

### リポジトリ直下に “ビルダー用 .venv（cp312）” を作る


```bash
cd ~/librealsense

# Pythonヘッダ＆pybind11（ターゲットPython用）
sudo apt-get install -y python3-dev pybind11-dev

# venv 作成＆有効化
python3 -m venv .venv-cp312
source .venv-cp312/bin/activate
python -V   # 例: Python 3.12.x
```

### venv 用の拡張 .so をビルド


```bash
# ABI（cp312）を作る
ABI=$(python -c 'import sys; print(f"cp{sys.version_info[0]}{sys.version_info[1]}")')

mkdir -p build-$ABI && cd build-$ABI
cmake .. \
  -DFORCE_RSUSB_BACKEND=ON \
  -DBUILD_PYTHON_BINDINGS=ON \
  -DBUILD_EXAMPLES=OFF -DBUILD_GRAPHICAL_EXAMPLES=OFF -DBUILD_TOOLS=OFF \
  -DCHECK_FOR_UPDATES=OFF \
  -DPYTHON_EXECUTABLE="$(python -c 'import sys; print(sys.executable)')" \
  -DPython3_EXECUTABLE="$(python -c 'import sys; print(sys.executable)')" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build . --target pyrealsense2 -j"$(nproc)"
```

### バージョンファイル生成 → .so を同梱 → wheel 作成


```bash
# 1) _version.py を生成
cd ~/librealsense
python wrappers/python/find_librs_version.py "$PWD" "wrappers/python/pyrealsense2"

# 2) ビルドした .so をパッケージ配下へコピー
SO=$(find build-$ABI -name 'pyrealsense2*.so' | head -n1)
cp -v "$SO" wrappers/python/pyrealsense2/

# 3) wheel を作成（PEP517）
python -m pip install -U pip build wheel
mkdir -p wheels
cd wrappers/python
python -m build --wheel --outdir ../../wheels

# できた wheel を確認
ls -1 ../../wheels/pyrealsense2-*-cp312-*.whl

# venv を無効化
deactivate
```

### 任意のプロジェクト .venv にインストール（wheel 使い回し）

例：別プロジェクト ~/proj の .venv（Python 3.12）に導入：

```bash
cd /path/to/your_project/
python3 -m venv .venv
source .venv/bin/activate

python -m pip install "$(ls -t ~/librealsense/wheels/pyrealsense2-*-cp312-*.whl | head -n1)"

# インストール確認
python - <<'PY'
import pyrealsense2 as rs, os
print("pyrealsense2:", rs.__version__)
print("path:", os.path.abspath(rs.__file__))
print("devices:", len(rs.context().query_devices()))
PY
```

##　アンインストール

ビルドキャッシュなどが悪さすることがあるので、わけわからんくなったらアンインストールしたほうがいい。

### udevルールの削除

```bash
# apt 版 ルール（あれば）
sudo rm -f /lib/udev/rules.d/60-librealsense2-udev-rules.rules
# ソース版の RSUSB ルール（クリーンにしたいなら削除）
sudo rm -f /etc/udev/rules.d/99-realsense-libusb.rules
sudo udevadm control --reload-rules && sudo udevadm trigger
```

### ソースからインストールしたファイルを削除（/usr/local 系）

#### install_manifest.txt が残っているビルドディレクトリがある場合

```bash
# それぞれの build ディレクトリで実行（複数あれば全部）
cd ~/librealsense/build 2>/dev/null && sudo xargs -a install_manifest.txt rm -vf || true
cd ~/librealsense/build-312 2>/dev/null && sudo xargs -a install_manifest.txt rm -vf || true
cd ~/librealsense/build-cp312 2>/dev/null && sudo xargs -a install_manifest.txt rm -vf || true
```

#### 代表的な配置先を明示的に削除（取りこぼし対策）

```bash
# 実行ファイル・ツール
sudo rm -f /usr/local/bin/realsense-viewer /usr/local/bin/rs-* 2>/dev/null
# ライブラリ・CMake設定・pkg-config
sudo rm -f /usr/local/lib/librealsense2.so* /usr/local/lib/librealsense-file.so* 2>/dev/null
sudo rm -rf /usr/local/lib/cmake/realsense2 /usr/local/lib/pkgconfig/realsense2.pc 2>/dev/null
# ヘッダ/共有データ
sudo rm -rf /usr/local/include/librealsense2 2>/dev/null
sudo rm -rf /usr/local/share/librealsense2* /usr/local/share/doc/librealsense2* 2>/dev/null
# 共有ライブラリキャッシュ更新
sudo ldconfig
```