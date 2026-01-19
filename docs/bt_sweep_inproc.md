# bt_sweep_inproc.py 仕様書

## 役割
複数ティッカー × 複数パラメータグリッドを並列実行するドライバ。
結果を Supabase に保存する。

---

## 主な責務
- ティッカーリストの解釈（環境変数）
- グリッドサーチ生成
- 並列実行（ProcessPoolExecutor）
- run_id 単位での save_cb 呼び出し

---

## 環境変数

- SWEEP_TICKERS
- SWEEP_START
- SWEEP_WORKERS
- SAVE_BT
- MAX_HOLD_DAYS
- TAKE_PROFIT_RR
- RISK_PCT
- EXIT_ON_REVERSE

---

## グリッド定義

```python
grid = {
  "CAPITAL": [3_000_000],
  "PER_TRADE": [500_000],
  "LOT_SIZE": [100],
  "RISK_PCT": [0.003],
  "TAKE_PROFIT_RR": [1.5],
  "MAX_HOLD_DAYS": [3,5],
  ...
}
```

---

## 環境変数による上書き

grid 定義直後で env を優先する設計。
→ 実験用パラメータを即切替可能。

---

## 並列実行モデル

- 各 worker = 1 run
- fetch_prices は事前キャッシュ済み
- CPU bound 処理のみ並列化

---

## 保存単位

- backtests_runs : 1 run = 1 row
- backtests_trades : trades 配列を展開して保存
