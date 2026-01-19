# bt_core.py 仕様書

## 役割
単一ティッカー・単一パラメータセットに対するバックテストの中核ロジック。
価格取得、売買判定、ポジション管理、損益計算を担当する。

---

## 主な責務
- 日足 OHLCV データの取得（fetch_prices）
- エントリー判定
- 保有中の exit 判定（SL / TP / TIME / REV）
- 翌営業日寄り執行（pending_sell）
- トレード履歴の生成

---

## fetch_prices()

- Yahoo Finance から日足データを取得
- parquet キャッシュ対応
- キャッシュ有効期限は環境変数で制御

環境変数:
- PRICE_CACHE_DIR
- PRICE_CACHE_TTL_DAYS

---

## ポジション状態管理

主要変数:
- pos
- entry_px
- entry_date
- hold_days
- stop_px
- take_px

---

## pending_sell 設計

### 背景
TIME / REV を当日引けで即売却すると「当日売り」が大量発生するため、
**翌営業日寄りでの成行売却**に変更。

### 管理変数
- pending_sell_for : date
- pending_sell_reason : str
- pending_sell_signal_ts : datetime

---

## exit 判定フロー

優先順:
1. SL（当日中に即時執行）
2. TP（当日中に即時執行）
3. TIME（引けで検知 → 翌寄り）
4. REV（引けで検知 → 翌寄り）

---

## signal_ts

- TIME / REV の「検知日」を保持
- SELL レコードに signal_ts として保存
- buy_ts → signal_ts → sell_ts の整合性検証が可能

---

## 売買レコード形式（trades）

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

---

## 検証済み不変条件

- BUY日 = SELL日 は発生しない
- signal_ts → sell_ts は常に翌営業日
- signal_ts NULL の REV / TIME は存在しない

---

## 注意点

- just_bought フラグで当日 exit を抑止
- sold フラグで同日複数 exit を防止
