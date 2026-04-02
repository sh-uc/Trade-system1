# ticker_events 最小運用

`ticker_events` は、決算日や `ex-dividend date` を
Supabase に蓄積するための最小運用です。

---

## 1. 目的

- `earnings_actual` を保持する
- 将来の `earnings_expected` へつなげる
- `ex_dividend` を保持する
- 将来の `paper simulation` 停止ロジックにつなげる

---

## 2. 適用するもの

- DDL: `supabase/ddl/083_ticker_events.sql`
- 取得スクリプト: `sync_ticker_events.py`
- 実行入口: `run_sync_ticker_events.ps1`

---

## 3. 実行例

全 active 銘柄を対象にする場合:

```powershell
.\run_sync_ticker_events.ps1
```

個別銘柄だけに絞る場合:

```powershell
.\run_sync_ticker_events.ps1 -Tickers "3103.T,7013.T,4506.T"
```

---

## 4. 初版の注意

- `earnings_actual` は `yfinance` で未来日が取れないことがある
- `ex_dividend` は比較的安定している
- 日本株の正式な権利確定日や権利付最終日は、初版では未対応

このため、初版では
「取れるイベントを蓄積し、停止ロジックの土台を作る」
ところまでを目的にする

将来は、過去の `earnings_actual` から
`earnings_expected` を生成して使う。

---

## 5. paper simulation での最小ルール

`simulate_live.py` では、最小ルールとして次を入れられる。

- `EARNINGS_BLOCK_DAYS`
  - `earnings_actual` / `earnings_expected` が近い銘柄は新規建てしない
- `EARNINGS_FORCE_CLOSE_DAYS`
  - `earnings_actual` / `earnings_expected` が近い保有銘柄は、当日引けで強制決済する

実行例:

```powershell
.\run_simulate_live.ps1 -SimulationName "paper_main_earnings" -StrategyName "swing_v1" -Start "2025-01-01" -Reset "1" -EarningsBlockDays "3" -EarningsForceCloseDays "1"
```

---

## 6. 検証用 seed

`yfinance` の `earnings_actual` は鮮度にムラがあるため、
ロジック確認だけ先に進めたい場合は検証用 seed を使う。

- seed SQL: `docs/ticker_events_earnings_test_seed.sql`

この seed は、実際に建玉が出ている期間へ少数の `earnings` を置いて、
`paper simulation` のブロック・強制決済が動くか確認するためのもの。
