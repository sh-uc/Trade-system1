# db_utils.py 仕様書

## 役割
バックテスト結果を Supabase に保存するユーティリティ。

---

## 対象テーブル

### backtests_runs
- run_id
- ticker
- params (jsonb)

### backtests_trades
- run_id
- ts
- side
- price
- qty
- reason
- signal_ts

---

## save_cb(run_id, trades)

- bt_core から呼ばれるコールバック
- trades 配列を DB row に変換

---

## Timestamp 処理

- pandas.Timestamp / datetime 両対応
- ts は isoformat() で保存
- 重複回避のため microseconds を付加

---

## signal_ts 対応

- trade.get("signal_ts") をそのまま保存
- JSON serialize エラー防止のため datetime → isoformat

---

## エラー耐性

- 不正 side / qty はスキップ
- insert 失敗時は warn ログのみ（run 全体は落とさない）
