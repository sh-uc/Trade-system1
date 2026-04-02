# manual_stop 最小運用

## 1. 目的

`ticker_events` の `manual_stop` を使って、  
銘柄別に

- 新規停止
- 強制決済

を手動で指定できるようにする。

## 2. イベントの持ち方

`ticker_events` に次の形で入れる。

- `event_type='manual_stop'`
- `event_date`: 停止開始日
- `meta.window_end`: 停止終了日
- `meta.action`: `block_entry`, `force_close`, `both`

## 3. ルール

### `block_entry`

- 停止期間中、その銘柄の新規建てを行わない

### `force_close`

- 停止期間中、その銘柄の保有建玉を当日引けで `MANUAL_STOP` 理由で強制決済する

### `both`

- 新規停止と強制決済の両方を行う

## 4. 実装上の優先順

保有中の建玉については、

1. `manual_stop(force_close)`
2. `earnings_force_close`
3. 通常の exit 判定

の順で判定する。

## 5. 検証用 seed の例

別ファイル:
- `docs/manual_stop_test_seed.sql`

## 6. 補足

この仕組みは、将来の
- 地政学リスク時の全体停止
- 個別材料発生時の監視対象停止
- 一時的な銘柄除外

の土台として使える。
