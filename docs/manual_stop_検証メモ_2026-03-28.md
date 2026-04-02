# manual_stop 検証メモ（2026-03-28）

## 1. 実装内容

`simulate_live.py` で `ticker_events.event_type='manual_stop'` を参照し、

- `block_entry`
- `force_close`
- `both`

を扱えるようにした。

## 2. 検証用イベント

### block_entry

- `7013.T` IHI
- `2025-08-06`
- `action='block_entry'`

### force_close

- `6363.T` 酉島製作所
- `2025-07-15`
- `action='force_close'`

## 3. 結果

比較対象:
- base: `paper_candidate9_open5_no_daily_limit_base`
- test: `paper_candidate9_manual_stop_test`

### 損益

- base: `1,594,643.86`
- test: `1,569,228.31`

### 発火確認

#### `force_close`

base では
- `6363.T` 酉島製作所
- open `2025-07-15`
- close `2025-07-16`
- `close_reason='SL'`
- `realized_pnl=2,827.96`

test では
- 同じ建玉が
- close `2025-07-15`
- `close_reason='MANUAL_STOP'`
- `realized_pnl=-2,218.10`

となり、当日引けで強制決済された。

#### `block_entry`

base では
- `7013.T` IHI
- open `2025-08-06`
- close `2025-08-07`
- `realized_pnl=18,649.85`

test ではこの建玉が消えた。

その代わり
- `7013.T` IHI
- open `2025-08-07`
- close `2025-08-08`
- `realized_pnl=-1,719.63`

が入っており、`2025-08-06` の新規建てが止まった結果として挙動が変わったことが確認できた。

## 4. 結論

- `manual_stop` の `block_entry` は機能した
- `manual_stop` の `force_close` も機能した
- 最小版としては十分に動作確認できた

## 5. 次の使い方

実運用寄りには、次の用途で使うのが自然。

- 急な材料時の銘柄別停止
- 一時的な除外銘柄の新規停止
- 既存建玉の緊急クローズ
