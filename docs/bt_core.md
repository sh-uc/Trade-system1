# bt_core.py 仕様書

## 役割
`bt_core.py` は、単一ティッカー・単一パラメータセットのバックテストを実行する中核モジュールです。価格取得、指標計算、エントリー、ポジション管理、exit 判定、損益集計を担当します。

---

## 主な責務
- `fetch_prices()` による日足 OHLCV の取得と parquet キャッシュ利用
- `fetch_intraday_prices()` による 1 時間足取得と parquet キャッシュ利用
- `compute_indicators()` による MA / RSI / MACD / ATR / 出来高系指標の計算
- `long_signal_row()` によるロングシグナル判定
- `_calc_qty()` による建玉サイズ計算
- `run_backtest()` による売買シミュレーションと結果集計
- `resolve_intraday_ambiguous_exit()` による曖昧日の補助判定

---

## 主要な設計ポイント

### 価格取得
- `fetch_prices()` は日足を取得する
- `fetch_intraday_prices()` は補助用の intraday 足を取得する
- どちらも parquet キャッシュを持つ
- index は JST に統一する

### エントリー
- 当日引けでシグナル判定し、翌営業日寄りで買いを執行
- `GAP_ENTRY_MAX` を超える寄りギャップは `SKIP`

### ポジション管理
主要変数:
- `pos`
- `entry_px`
- `entry_date`
- `entry_bar_index`
- `stop_px`
- `take_px`
- `pending_buy_for`
- `pending_sell_for`
- `pending_sell_reason`
- `pending_sell_signal_ts`
- `just_bought`
- `ambiguous_days`
- `intraday_resolved_days`

`entry_bar_index` を持つことで、保有日数は「エントリー日を 0 日目」として `i - entry_bar_index` で計算します。

---

## パラメータの考え方

### 建玉サイズとストップ距離の分離
現行版では `RISK_PCT` と `STOP_PCT` を分離しています。

- `RISK_PCT`
  - 建玉サイズ計算用
  - `_calc_qty()` 内で 1 トレードあたりの想定損失額を決める
- `STOP_PCT`
  - 初期ストップ距離用
  - `stop_px` と `take_px` の基準になる

`STOP_PCT` が未指定の場合は後方互換のため `RISK_PCT` を使います。

### intraday 補助用パラメータ
- `TICKER`
  - intraday 取得時に使う銘柄コード
- `USE_INTRADAY_RESOLUTION`
  - 曖昧日に 1 時間足補助を使うかどうか
- `INTRADAY_INTERVAL`
  - 現在は `60m` を想定
- `INTRADAY_TIE_BREAK`
  - 同じ 1 時間足の中で `SL` / `TP` の両方に触れた場合の優先ルール
  - 既定は `SL_FIRST`

### 利確・損切り価格
- `R = entry_px * STOP_PCT`
- `stop_px = entry_px - R`
- `take_px = entry_px + TAKE_PROFIT_RR * R`

---

## pending_sell 設計

### 背景
`TIME` / `REV` を当日引けで即売却すると日足バックテストで同日決済が増えるため、現行版では **当日引けで検知し、翌営業日寄りで成行執行** にしています。

### 管理変数
- `pending_sell_for`
- `pending_sell_reason`
- `pending_sell_signal_ts`

---

## exit 判定フロー

通常の保有日:
1. `SL`
2. `TP`
3. `REV`

期限到達日（`current_hold_days >= MAX_HOLD_DAYS`）:
1. `SL`
2. `TIME` を予約
3. `TP` は見ない
4. `REV` も見ない

曖昧日（`low <= stop_px` かつ `high >= take_px`）で `USE_INTRADAY_RESOLUTION=True` の場合:
1. その日だけ `60m` 足を取得
2. `resolve_intraday_ambiguous_exit()` で `SL` / `TP` の先着を判定
3. 判定できた場合はその結果を優先
4. 判定できなければ従来の日足ロジックにフォールバック

---

## 曖昧日カウント
現行版では、期限日ではない保有日について
- `low <= stop_px`
- `high >= take_px`
を同時に満たす日を `ambiguous_days` として数えます。

さらに、1 時間足補助で実際に順序判定できた日を `intraday_resolved_days` として数えます。

---

## signal_ts
- `TIME` / `REV` の検知日を `signal_ts` として保持
- `SELL` レコードに保存される
- `signal_ts -> sell_ts` の整合性確認に使える

---

## trades レコード形式

```python
{
  "date": datetime,
  "side": "BUY" | "SELL",
  "price": float,
  "qty": int,
  "reason": "SL" | "TP" | "TIME" | "REV",
  "signal_ts": Optional[datetime]
}
```

補足:
- 実装内部では `px` キーで持つが、保存時に `price` 相当に整形される
- `SKIP` は DB 保存対象ではない

---

## result に入る補助情報
`run_backtest()` の戻り値には、通常の集計に加えて次を含みます。
- `ambiguous_days`
- `intraday_resolved_days`

---

## 検証上の注意
- `just_bought` により買い当日の exit 判定は行わない
- `sold` により同日複数 exit を防ぐ
- 1 時間足補助を使っても、その 1 本の中で高値と安値のどちらが先かは分からない場合がある
- 現行版ではその場合の tie-break は `SL_FIRST` を既定にしている
- `TIME` を効かせたい場合は、優先順だけでなく `STOP_PCT` / `TAKE_PROFIT_RR` / シグナル設計もあわせて検討が必要
