# bt_core.py 仕様書

## 役割
`bt_core.py` は、単一ティッカー・単一パラメータセットのバックテストを実行する中核モジュールです。価格取得、指標計算、エントリー、ポジション管理、exit 判定、損益集計を担当します。

---

## 主な責務
- `fetch_prices()` による日足 OHLCV の取得と parquet キャッシュ利用
- `compute_indicators()` による MA / RSI / MACD / ATR / 出来高系指標の計算
- `long_signal_row()` によるロングシグナル判定
- `_calc_qty()` による建玉サイズ計算
- `run_backtest()` による売買シミュレーションと結果集計

---

## 主要な設計ポイント

### 価格取得
- `yfinance` から日足を取得
- `BT_PRICE_CACHE_DIR` / `BT_PRICE_CACHE` で parquet キャッシュを制御
- index は JST に統一

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

### 利確・損切り価格
- `R = entry_px * STOP_PCT`
- `stop_px = entry_px - R`
- `take_px = entry_px + TAKE_PROFIT_RR * R`

---

## pending_sell 設計

### 背景
`TIME` / `REV` を当日引けで即売却すると日足バックテストで同日決済が増えるため、
現行版では **当日引けで検知し、翌営業日寄りで成行執行** にしています。

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

つまり、期限到達日は「期限優先」の動きになります。

---

## TIME の扱い
- `MAX_HOLD_DAYS` はエントリー日を 0 日目とする経過営業日数
- 期限到達日は `TP` を無効化し、引けで `TIME` を検知して翌営業日寄りで売却
- ただし、同日中に `SL` に触れた場合は `SL` が優先される

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

## 検証上の注意
- `just_bought` により買い当日の exit 判定は行わない
- `sold` により同日複数 exit を防ぐ
- 日足の高値/安値判定を先に使うため、`TIME` は短期で `SL/TP` に着地しやすい戦略では出番が少ない
- `TIME` を効かせたい場合は、優先順だけでなく `STOP_PCT` / `TAKE_PROFIT_RR` / シグナル設計もあわせて検討が必要
