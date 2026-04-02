# db_utils.py 仕様書

## 役割
Supabase 関連の共通ユーティリティ。

- バックテスト結果を Supabase に保存する
- `tickers.name` の補完を共通関数で行う

---

## 会社名補完

### `resolve_company_name(code, sb=None)`
- `tickers` から `name` を読む
- ただし `name` が空白、`None`、または銘柄コードそのものなら未設定扱いにする
- 未設定扱いの場合は `yfinance` から会社名を取得する
- 取得できたら `tickers` に `upsert` する
- 最終的に取れなければ銘柄コードを返す

### `ensure_ticker_name(sb, code, current_name=None)`
- 既存の `current_name` と DB 上の `name` を正規化して判定する
- 有効な名前がなければ `yfinance` で補完する
- `tickers` の `name` 補完処理を共通化するための内部利用向け関数

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

## `save_backtest_to_db(sb, ticker, params, result, curve, trades)`

- `backtests_runs` と `backtests_trades` に保存する
- `result` の `ambiguous_days`, `intraday_resolved_days` は `params._result_meta` として残す

---

## Timestamp 処理

- pandas.Timestamp / datetime 両対応
- ts は isoformat() で保存
- 重複回避のため microseconds を付加

---

## signal_ts 対応

- `trade.get("signal_ts")` をそのまま保存
- JSON serialize エラー防止のため datetime → isoformat

---

## エラー耐性

- 不正 side / qty はスキップ
- insert 失敗時は warn ログのみ（run 全体は落とさない）
- 会社名取得失敗時は銘柄コードへフォールバック
