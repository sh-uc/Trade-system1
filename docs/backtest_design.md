# 技術要約：bt_sweep_inproc / bt_core / db_utils 改修と検証

## 目的
- REV / TIME を「当日売り」ではなく「翌営業日寄り成行」で執行する
- 検知日と執行日を分離し、DB 上で検証可能にする

---

## 対象ファイル
- bt_sweep_inproc.py
- bt_core.py
- db_utils.py
- Supabase（backtests_runs / backtests_trades / best_params / indicators / tickers / prices）

---

## 1. bt_sweep_inproc.py の変更点

### 1.1 グリッド定義
```python
grid = {
    "CAPITAL":       [3_000_000.0],
    "PER_TRADE":     [500_000.0],
    "LOT_SIZE":      [100],
    "RISK_PCT":       [0.003],
    "TAKE_PROFIT_RR": [1.5],
    "MAX_HOLD_DAYS":  [3, 5],
    "STOP_SLIPPAGE":  [0.0025],
    "EXIT_ON_REVERSE":[True],
    "VOL_SPIKE_M":    [1.0, 1.1, 1.2],
    "MACD_ATR_K":     [0.05, 0.1, 0.15],
    "RSI_MIN":        [30, 35.0],
    "RSI_MAX":        [75, 80],
    "GAP_ENTRY_MAX":  [0.08, 0.12],
}
```

### 1.2 環境変数による上書き
- SWEEP_TICKERS
- MAX_HOLD_DAYS
- TAKE_PROFIT_RR
- RISK_PCT
- EXIT_ON_REVERSE

---

## 2. bt_core.py：REV / TIME を翌営業日寄りで執行

### 2.1 設計方針
- 検知：当日引け
- 執行：翌営業日寄り

### 2.2 判定ロジック
```python
if not sold and pending_sell_for is None:
    if hold_days >= MAX_HOLD_DAYS:
        pending_sell_for = next_trading_day(date.date())
        pending_sell_reason = "TIME"
        pending_signal_ts = date
        sold = True
    elif EXIT_ON_REVERSE:
        if not long_signal_row(...):
            pending_sell_for = next_trading_day(date.date())
            pending_sell_reason = "REV"
            pending_signal_ts = date
            sold = True
```

---

## 3. pending に signal_ts（検知日）を持たせる

### 3.1 目的
- REV / TIME の検知日を保存
- 執行日との差分検証を可能にする

### 3.2 実装
- pending_signal_ts を追加
- 判定時にセット
- SELL trade に signal_ts を付与

---

## 4. db_utils.py：signal_ts 保存対応

### 4.1 スキーマ
```sql
alter table backtests_trades
add column signal_ts timestamptz;
```

### 4.2 保存処理
```python
"signal_ts": signal_ts.isoformat() if signal_ts else None
```

---

## 5. 検証結果

- 当日売り：0 件
- signal_ts → ts 差：1 / 3 / 4 日のみ
- 不整合なし

---

## 結論
- ロジックは正しく動作
- DB 検証可能な設計が完成

---

## 次の作業
- 通常パラメータ sweep
