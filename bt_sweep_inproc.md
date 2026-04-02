# bt_sweep_inproc.py 仕様書

## 役割
`bt_sweep_inproc.py` は、複数銘柄と複数パラメータ組み合わせに対して `bt_core.run_backtest()` を並列実行し、結果を Supabase に保存するスイープ実行モジュールです。

---

## 依存関係
- `bt_core.py`
  - `fetch_prices()`
  - `fetch_intraday_prices()`
  - `compute_indicators()`
  - `run_backtest()`
- `db_utils.py`
  - `get_supabase()`
  - `save_backtest_to_db()`

---

## 実行の流れ
1. 環境変数からティッカー・開始日・グリッド条件を読む
2. ティッカーごとに日足価格取得と指標計算を 1 回だけ行う
3. `itertools.product()` でパラメータ全組み合わせを作る
4. `ProcessPoolExecutor` で `run_backtest()` を並列実行する
5. `SAVE_BT=1` のとき Supabase に run / trades を保存する
6. 標準出力に進捗とリターンを出す

---

## make_save_cb()
`run_backtest()` の保存コールバックを作る関数です。

- ティッカーごとに `get_supabase()` でクライアント取得
- `curve`, `trades`, `params`, `summary` を受け取り `save_backtest_to_db()` へ渡す

保存時には `summary` の一部補助情報を `params["_result_meta"]` にも残します。

---

## グリッドの考え方
標準では以下のようなグリッドを持ちます。

- `CAPITAL`
- `PER_TRADE`
- `LOT_SIZE`
- `RISK_PCT`
- `STOP_PCT`
- `TAKE_PROFIT_RR`
- `MAX_HOLD_DAYS`
- `STOP_SLIPPAGE`
- `EXIT_ON_REVERSE`
- `USE_INTRADAY_RESOLUTION`
- `INTRADAY_INTERVAL`
- `INTRADAY_TIE_BREAK`
- `VOL_SPIKE_M`
- `MACD_ATR_K`
- `RSI_MIN`
- `RSI_MAX`
- `GAP_ENTRY_MAX`

必要な項目は環境変数で上書きできます。

---

## 現行版で重要な入力

### 建玉サイズとストップ距離
`bt_core.py` 側で `RISK_PCT` と `STOP_PCT` を分離しているため、スイープ側でも両方を渡せます。

- `RISK_PCT`
  - 建玉サイズ計算用
- `STOP_PCT`
  - 初期ストップ距離用
  - 未指定なら `bt_core.py` 側で `RISK_PCT` が使われる

### 1時間足補助
曖昧日だけ 1 時間足で順序判定するため、次の入力を渡せます。

- `USE_INTRADAY_RESOLUTION`
  - `True` のとき曖昧日に intraday 補助を使う
- `INTRADAY_INTERVAL`
  - 現在は `60m` を想定
- `INTRADAY_TIE_BREAK`
  - 1 時間足 1 本の中で `SL` / `TP` 両方に触れた場合の優先順
  - 既定は `SL_FIRST`

### そのほか主要パラメータ
- `TAKE_PROFIT_RR`
- `MAX_HOLD_DAYS`
- `VOL_SPIKE_M`
- `MACD_ATR_K`
- `RSI_MIN`
- `RSI_MAX`
- `GAP_ENTRY_MAX`
- `EXIT_ON_REVERSE`

---

## 環境変数 override
`bt_sweep_inproc.py` は、環境変数が設定されていればグリッドを差し替えます。

主な対応項目:
- `RISK_PCT`
- `STOP_PCT`
- `TAKE_PROFIT_RR`
- `MAX_HOLD_DAYS`
- `VOL_SPIKE_M`
- `MACD_ATR_K`
- `RSI_MIN`
- `RSI_MAX`
- `GAP_ENTRY_MAX`
- `EXIT_ON_REVERSE`
- `USE_INTRADAY_RESOLUTION`
- `INTRADAY_INTERVAL`
- `INTRADAY_TIE_BREAK`

カンマ区切りで複数値を渡すと、そのままスイープ対象になります。

例:
```powershell
$env:RISK_PCT = "0.001"
$env:STOP_PCT = "0.001"
$env:USE_INTRADAY_RESOLUTION = "1"
$env:INTRADAY_INTERVAL = "60m"
$env:INTRADAY_TIE_BREAK = "SL_FIRST"
```

---

## run_bt_sweep_inproc.ps1 との関係
通常運用では [run_bt_sweep_inproc.ps1](/C:/Users/suchida/Documents/margintr/root/run_bt_sweep_inproc.ps1) を入口にします。

主な引数:
- `-Ticker`
- `-Start`
- `-MaxHoldDays`
- `-VolSpikeM`
- `-TakeProfitRr`
- `-RiskPct`
- `-StopPct`
- `-RsiMin`
- `-RsiMax`
- `-MacdAtrK`
- `-GapEntryMax`
- `-ExitOnReverse`
- `-UseIntradayResolution`
- `-IntradayInterval`
- `-IntradayTieBreak`
- `-SweepWorkers`
- `-SaveBt`

`-StopPct` を指定しない場合は、後方互換で `STOP_PCT` は送られません。

---

## 出力
### 標準出力
- `[INFO] Using Python: ...`
- `[n/total] ticker done ret=...`
- `[SWEEP] finished: ...`

### Supabase
- `backtests_runs`
- `backtests_trades`

保存内容:
- `ticker`
- `params`
- `final_equity`
- `total_return`
- `max_drawdown`
- `sharpe`
- `n_trades`
- trades 明細

補足:
- `ambiguous_days` と `intraday_resolved_days` は、現行版では `backtests_runs.params._result_meta` に保存される

---

## 現行版の用途
- 単一銘柄の局所最適化
- 候補銘柄の横並び比較
- `RISK_PCT` と `STOP_PCT` を分離した exit 設計検証
- `MAX_HOLD_DAYS` / `TIME` が戦略上どれだけ意味を持つかの確認
- 曖昧日だけに 1 時間足補助を入れた exit 順序検証

---

## 注意点
- `STOP_PCT` を広げても `TIME` が出るとは限らない
- 日足の `SL/TP` 判定が先に着地する戦略では、出口種別の件数が変わらず損益スケールだけ変わることがある
- 1 時間足補助を入れても、その 1 本の中で順序が不明な場合は tie-break ルールに依存する
- 大きなグリッドを作ると組み合わせ数が急増するため、1 回の検証では軸を絞るほうが扱いやすい
