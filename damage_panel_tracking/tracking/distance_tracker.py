from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .base import Detection, Track


@dataclass
class DistanceConfig:
    """Config for distance-based association tracker (bbox-only).

    This backend is designed for cases where:
      - appearance features are unavailable (or useless),
      - the number of objects is small (e.g., <= 4),
      - abrupt direction changes (large jerk) can happen,
      - you want to avoid "IoU=0 -> immediate split" failure modes.

    Key idea:
      - match detections to tracks by *center distance* (optionally normalized),
        plus a small penalty for bbox size change.
      - use a simple constant-velocity state as a hint, but keep matching robust
        by allowing fallback to the last observed center (helps on hard reversals).
    """

    # 対応付け / ゲーティング
    gate_px: float = 80.0                 # 1ステップで対応付けを許容する中心移動量の上限(px)
    normalize: str = "diag"               # "diag" | "sqrt_area" | "none"
    size_weight: float = 0.15             # bboxサイズ変化ペナルティの重み(log比)

    # 運動モデル
    use_prediction: bool = True           # 定速度予測を補助情報として使う
    vel_alpha: float = 0.6                # 速度更新のEMA係数(0..1、大きいほど追従優先)
    max_speed_px_s: float = 8000.0        # 速度ノルムの上限(px/s)

    # ライフサイクル
    max_age: int = 4                      # 検出欠落を許容する最大ステップ数
    min_steps_alive: int = 2              # 出力対象にするまでの最小ヒット数


class _InternalTrack:
    __slots__ = (
        "id", "box", "center", "wh", "vel", "age", "hits", "misses"
    )

    def __init__(self, tid: int, box_xyxy: np.ndarray) -> None:
        # 新規検出bboxから内部track状態を初期化する。
        self.id = tid
        self.box = box_xyxy.astype(float)
        self.center = _xyxy_center(self.box)
        self.wh = _xyxy_wh(self.box)
        self.vel = np.zeros(2, dtype=float)   # px/s
        self.age = 1
        self.hits = 1
        self.misses = 0

    def predict_center(self, dt: float) -> np.ndarray:
        # 定速度モデルで次の中心座標を予測する。
        return self.center + self.vel * float(dt)

    def predict_box(self, dt: float) -> np.ndarray:
        # 予測中心まわりに現サイズを配置して次のbboxを作る。
        c = self.predict_center(dt)
        w, h = self.wh
        x1 = c[0] - w / 2.0
        y1 = c[1] - h / 2.0
        x2 = c[0] + w / 2.0
        y2 = c[1] + h / 2.0
        return np.array([x1, y1, x2, y2], dtype=float)

    def mark_missed(self, dt: float, use_prediction: bool) -> None:
        # 未対応フレームとして年齢を進め、必要なら状態を予測更新する。
        self.age += 1
        self.misses += 1
        if use_prediction and dt > 0:
            # 短時間の欠落に耐えるため状態を前進させる
            self.center = self.predict_center(dt)
            self.box = self.predict_box(dt)


def _xyxy_center(box: np.ndarray) -> np.ndarray:
    # ベクトル計算用にbbox中心をndarrayで返す。
    return np.array([(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0], dtype=float)


def _xyxy_wh(box: np.ndarray) -> np.ndarray:
    # 0除算回避の下限付きでbbox幅/高さを返す。
    return np.array([max(1e-6, box[2] - box[0]), max(1e-6, box[3] - box[1])], dtype=float)


def _scale_for_norm(wh: np.ndarray, normalize: str) -> float:
    # 距離dをd/scaleで正規化するため、現在bboxサイズから分母scaleを決める。
    w, h = float(wh[0]), float(wh[1])
    if normalize == "none":
        # 正規化しない（px絶対距離をそのまま使う）。
        return 1.0
    if normalize == "sqrt_area":
        # 面積の平方根で正規化し、物体サイズ差の影響を緩和する。
        return max(1.0, float(np.sqrt(max(1e-6, w * h))))
    # 既定は対角長で正規化し、見かけサイズに応じた相対距離として扱う。
    return max(1.0, float(np.sqrt(w * w + h * h)))


def _size_penalty(track_wh: np.ndarray, det_wh: np.ndarray) -> float:
    # 急なbboxサイズ変化にペナルティを与えて誤対応を抑える。
    # penalty = |log(w2/w1)| + |log(h2/h1)|
    eps = 1e-6
    w1, h1 = float(track_wh[0]) + eps, float(track_wh[1]) + eps
    w2, h2 = float(det_wh[0]) + eps, float(det_wh[1]) + eps
    return abs(np.log(w2 / w1)) + abs(np.log(h2 / h1))


def _greedy_assignment(cost: np.ndarray, invalid_cost: float) -> List[Tuple[int, int]]:
    """Return list of (track_idx, det_idx) assignments."""
    # Hungarian法が使えない場合の貪欲法フォールバック。
    n_trk, n_det = cost.shape
    pairs: List[Tuple[int, int]] = []
    if n_trk == 0 or n_det == 0:
        return pairs

    # 候補一覧を作る
    cand: List[Tuple[float, int, int]] = []
    for i in range(n_trk):
        for j in range(n_det):
            c = float(cost[i, j])
            if c < invalid_cost:
                cand.append((c, i, j))
    cand.sort(key=lambda x: x[0])

    used_i = set()
    used_j = set()
    for c, i, j in cand:
        if i in used_i or j in used_j:
            continue
        used_i.add(i)
        used_j.add(j)
        pairs.append((i, j))
    return pairs


class DistanceTracker:
    """Distance-based multi-object tracker backend.

    Interface:
      step(detections, dt) -> List[Track]
    """

    def __init__(self, cfg: DistanceConfig):
        # 追跡状態と任意のHungarianソルバを準備する。
        self._cfg = cfg
        self._tracks: List[_InternalTrack] = []
        self._next_id = 1

        # SciPyがあれば最適割当（Hungarian法）を使う。
        self._use_scipy = False
        try:
            from scipy.optimize import linear_sum_assignment  # type: ignore
            self._linear_sum_assignment = linear_sum_assignment
            self._use_scipy = True
        except Exception:
            self._linear_sum_assignment = None

    def step(self, detections: List[Detection], dt: float) -> List[Track]:
        # 検出とtrackを対応付け、状態更新し、有効trackを出力する。
        # dtを浮動小数へそろえる。
        dt = float(dt)
        # 0除算回避のため最小値を設ける。
        dt = max(1e-6, dt)

        # Detectionからbbox(xyxy)配列を取り出す。
        det_boxes = [d.box_xyxy.astype(float) for d in detections]
        # 各検出bboxの中心座標を事前計算する。
        det_centers = [ _xyxy_center(b) for b in det_boxes ]
        # 各検出bboxの幅・高さを事前計算する。
        det_wh = [ _xyxy_wh(b) for b in det_boxes ]

        # 現在保持しているtrack数を得る（前stepの結果から決まる）
        n_trk = len(self._tracks)
        # 今回フレームの検出数を得る。
        n_det = len(det_boxes)

        # コスト行列を構築する
        # 無効対応に使う大きなコスト定数。
        invalid = 1e9
        # まずは全要素を無効値で初期化する（shape: track数 x 検出数）。
        cost = np.full((n_trk, n_det), invalid, dtype=float)

        # 各trackと各検出の対応コストを計算する。
        for i, tr in enumerate(self._tracks):
            # 設定に応じて予測中心または現在中心を参照する。
            # use_prediction:
            #   True: 定速度モデル
            #   False: 静止モデル
            tr_pred = tr.predict_center(dt) if self._cfg.use_prediction else tr.center
            # 直近の観測中心（予測フォールバック用）。
            tr_last = tr.center
            # 距離正規化に使うスケールを計算する。
            tr_scale = _scale_for_norm(tr.wh, self._cfg.normalize)

            # このtrackに対する全検出候補を評価する。
            for j in range(n_det):
                # 強い切り返しに備えて予測中心と直近中心の両方を候補にする
                # 検出中心と予測中心のユークリッド距離。
                d_pred = float(np.linalg.norm(det_centers[j] - tr_pred))
                # 検出中心と直近中心のユークリッド距離。
                d_last = float(np.linalg.norm(det_centers[j] - tr_last))
                # 予測利用時は小さい方の距離を採用して頑健性を上げる。
                d = min(d_pred, d_last) if self._cfg.use_prediction else d_last

                # ゲート外（動きすぎ）は対応候補から除外する。
                if d > float(self._cfg.gate_px):
                    continue

                # bboxサイズ変化に対するペナルティを計算する。
                sp = _size_penalty(tr.wh, det_wh[j])
                # 最終コスト = 正規化距離 + サイズ差ペナルティ。
                c = (d / tr_scale) + float(self._cfg.size_weight) * sp
                # 有効な対応候補としてコスト行列へ格納する。
                cost[i, j] = c

        # 対応付けを解く
        # (track_index, detection_index) の対応結果を格納する。
        matches: List[Tuple[int, int]] = []
        # track/検出がともに1件以上あるときだけ割当計算する。
        if n_trk > 0 and n_det > 0:
            # SciPy利用可ならHungarian法で最適割当を解く。
            if self._use_scipy and self._linear_sum_assignment is not None:
                row_ind, col_ind = self._linear_sum_assignment(cost)
                # 無効コストでない割当だけを採用する。
                for i, j in zip(row_ind.tolist(), col_ind.tolist()):
                    if float(cost[i, j]) < invalid:
                        matches.append((i, j))
            else:
                # SciPyがない場合は貪欲法で近似割当する。
                matches = _greedy_assignment(cost, invalid_cost=invalid)

        # 対応済みtrackインデックス集合。
        matched_trk = {i for i, _ in matches}
        # 対応済み検出インデックス集合。
        matched_det = {j for _, j in matches}

        # 対応が取れたtrackを更新する
        # 割当済みペアごとにtrack状態を最新観測へ更新する。
        for i, j in matches:
            tr = self._tracks[i]
            # 割り当てられた検出bbox。
            new_box = det_boxes[j]
            # 割り当てられた検出中心。
            new_center = det_centers[j]
            # 割り当てられた検出サイズ。
            new_wh = det_wh[j]

            # 速度更新(px/s)
            # 中心位置差分から観測速度を計算する。
            meas_v = (new_center - tr.center) / dt
            # 速度クリップ
            # 観測速度ノルムを計算する。
            spd = float(np.linalg.norm(meas_v))
            # 設定された最大速度を取得する。
            max_spd = float(self._cfg.max_speed_px_s)
            # 上限超過時は方向を保って速度ベクトルを縮小する。
            if spd > max_spd and spd > 1e-6:
                meas_v = meas_v * (max_spd / spd)

            # 速度更新のEMA係数を取得する。
            a = float(self._cfg.vel_alpha)
            # 既存速度と観測速度をEMAで合成する。
            tr.vel = (1.0 - a) * tr.vel + a * meas_v

            # 幾何状態を最新観測で更新する。
            tr.center = new_center
            tr.wh = new_wh
            tr.box = new_box
            # 生存期間カウンタを進める。
            tr.age += 1
            # 対応成功回数を増やす。
            tr.hits += 1
            # 未検出カウンタをリセットする。
            tr.misses = 0

        # 未対応trackの予測/経時更新
        # 生き残るtrackだけを詰める配列。
        survivors: List[_InternalTrack] = []
        # 全trackを見て、対応済み/未対応で処理を分ける。
        for i, tr in enumerate(self._tracks):
            # 対応済みtrackはそのまま残す。
            if i in matched_trk:
                survivors.append(tr)
            else:
                # 未対応trackはmiss処理（年齢更新・必要なら予測前進）を行う。
                tr.mark_missed(dt=dt, use_prediction=self._cfg.use_prediction)
                # miss上限以内なら一時的に保持する。
                if tr.misses <= int(self._cfg.max_age):
                    survivors.append(tr)
                # 期限超過は破棄する

        # 生存track一覧を内部状態へ反映する。
        self._tracks = survivors

        # 未対応検出から新規trackを作る
        # まだどのtrackにも割り当たっていない検出を新規track化する。
        for j in range(n_det):
            if j in matched_det:
                continue
            # 次の連番IDで内部trackを生成する。
            tr = _InternalTrack(self._next_id, det_boxes[j])
            # 次回用のIDカウンタを進める。
            self._next_id += 1
            # 内部track一覧へ追加する。
            self._tracks.append(tr)

        # 今ステップで更新され、十分成長したtrackのみ出力する
        # 呼び出し側へ返す公開Trackリストを作る。
        out: List[Track] = []
        # 内部trackを走査して出力条件を満たすものだけ返す。
        for tr in self._tracks:
            # 今フレームで未対応のtrackは返さない。
            if tr.misses != 0:
                continue
            # 生成直後などヒット数不足のtrackは返さない。
            if tr.hits < int(self._cfg.min_steps_alive):
                continue
            # 外部公開用のTrackデータへ変換して追加する。
            out.append(
                Track(
                    track_id=str(tr.id),
                    box_xyxy=tr.box.copy(),
                    age=int(tr.age),
                    hits=int(tr.hits),
                )
            )

        return out
