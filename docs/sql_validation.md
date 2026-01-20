# Backtest SQL 検証集（Supabase / Postgres）

このドキュメントは、バックテスト開発（`bt_core.py` / `bt_sweep_inproc.py` / `db_utils.py`）で使った **確認SQL（検証SQL）** を、次スレッドでもコピペ参照できる形にまとめたものです。

## 0. 使い方

- **必ず run_id 範囲を絞って実行**してください（過去runが混ざると誤解が起きます）。
- 以降のSQLは、原則 `:run_min` / `:run_max` を差し替えて使う想定です。

```sql
-- 例：今回の run_id 範囲
-- WHERE r.run_id BETWEEN 11300 AND 11450
```

---

## 1. 最優先：params が効いているか

### 1-1. run_id 範囲内の params を確認（一覧）

> ここで意図した `MAX_HOLD_DAYS` / `TAKE_PROFIT_RR` / `RISK_PCT` / `EXIT_ON_REVERSE` 等が
> 保存されているかをまず見る。

```sql
SELECT
  r.run_id,
  r.ticker,
  r.params->>'max_hold_days'     AS max_hold_days,
  r.params->>'take_profit_rr'    AS take_profit_rr,
  r.params->>'risk_pct'          AS risk_pct,
  r.params->>'exit_on_reverse'   AS exit_on_reverse
FROM backtests_runs r
WHERE r.run_id BETWEEN :run_min AND :run_max
ORDER BY r.run_id;
```

> `params` が JSONB ではなく TEXT の場合は、アプリ側でJSON化してから保存するか、
> Supabase側の列型を合わせてください。

### 1-2. 特定の ticker だけに絞って params を確認

```sql
SELECT r.run_id, r.ticker, r.params
FROM backtests_runs r
WHERE r.run_id BETWEEN :run_min AND :run_max
  AND r.ticker = :ticker
ORDER BY r.run_id;
```

---

## 2. SELL reason の内訳を見る（全体/ run別 / ticker別）

### 2-1. run_id 範囲で SELL reason を全体集計

```sql
SELECT reason, COUNT(*) AS n
FROM backtests_trades t
WHERE t.run_id BETWEEN :run_min AND :run_max
  AND t.side = 'SELL'
GROUP BY reason
ORDER BY n DESC;
```

### 2-2. run別に TIME/SL/TP/REV を数える（グリッドの効き確認）

```sql
SELECT
  t.run_id,
  COUNT(*) FILTER (WHERE t.side='SELL' AND t.reason='TIME') AS n_time,
  COUNT(*) FILTER (WHERE t.side='SELL' AND t.reason='REV')  AS n_rev,
  COUNT(*) FILTER (WHERE t.side='SELL' AND t.reason='TP')   AS n_tp,
  COUNT(*) FILTER (WHERE t.side='SELL' AND t.reason='SL')   AS n_sl,
  COUNT(*) FILTER (WHERE t.side='SELL')                     AS n_sell
FROM backtests_trades t
WHERE t.run_id BETWEEN :run_min AND :run_max
GROUP BY t.run_id
ORDER BY t.run_id;
```

### 2-3. ticker別に reason を集計（複数tickerの傾向）

```sql
SELECT
  r.ticker,
  t.reason,
  COUNT(*) AS n
FROM backtests_trades t
JOIN backtests_runs r ON r.run_id = t.run_id
WHERE r.run_id BETWEEN :run_min AND :run_max
  AND t.side = 'SELL'
GROUP BY r.ticker, t.reason
ORDER BY r.ticker, t.reason;
```

---

## 3. 「当日売り（BUY日=SELL日）」が残っていないか

> `just_bought` 等のロジック修正後に、BUY日=SELL日が0件であることを確認する。

```sql
WITH trades AS (
  SELECT
    t.run_id,
    t.ts,
    t.side,
    ROW_NUMBER() OVER (PARTITION BY t.run_id ORDER BY t.ts) AS seq
  FROM backtests_trades t
  WHERE t.run_id BETWEEN :run_min AND :run_max
), paired AS (
  SELECT
    b.run_id,
    b.ts AS buy_ts,
    s.ts AS sell_ts
  FROM trades b
  JOIN trades s
    ON s.run_id = b.run_id
   AND s.seq = b.seq + 1
  WHERE b.side='BUY' AND s.side='SELL'
)
SELECT COUNT(*) AS n
FROM paired
WHERE DATE(buy_ts) = DATE(sell_ts);
```

---

## 4. REV/TIME を「翌寄り執行」にした検証（signal_ts）

### 4-1. signal_ts が NULL の REV/TIME が無いか

```sql
SELECT
  t.reason,
  COUNT(*) AS n
FROM backtests_trades t
WHERE t.run_id BETWEEN :run_min AND :run_max
  AND t.side = 'SELL'
  AND t.reason IN ('REV','TIME')
  AND t.signal_ts IS NULL
GROUP BY t.reason;
```

期待値：0件。

### 4-2. 「signal_ts の翌営業日 = ts（sell_ts）」になっているか

> *休日またぎ* を含めて正しく動いているかを検証。
> ここでは **“悪いものだけ抽出”** します。

```sql
-- 前提：next_trading_day のカレンダーと、DBの営業日カレンダーが無い場合は
--      “実行差分”を先に見て、異常候補だけをピックアップして目視確認します。
--      （完全自動判定は市場カレンダーが必要）

SELECT
  t.run_id,
  t.signal_ts,
  t.ts AS sell_ts,
  (DATE(t.ts) - DATE(t.signal_ts)) AS exec_days_diff
FROM backtests_trades t
WHERE t.run_id BETWEEN :run_min AND :run_max
  AND t.side='SELL'
  AND t.reason IN ('REV','TIME')
  AND t.signal_ts IS NOT NULL
  AND (DATE(t.ts) - DATE(t.signal_ts)) NOT IN (1,3,4)
ORDER BY t.run_id, t.ts;
```

期待値：0件（通常は1、金曜→月曜で3、祝日で4が稀に混入）。

### 4-3. TIME/REV だけで exec_days_diff を集計（すぐ結論が出る）

```sql
SELECT
  t.reason,
  (DATE(t.ts) - DATE(t.signal_ts)) AS exec_days_diff,
  COUNT(*) AS n
FROM backtests_trades t
WHERE t.run_id BETWEEN :run_min AND :run_max
  AND t.side='SELL'
  AND t.reason IN ('REV','TIME')
  AND t.signal_ts IS NOT NULL
GROUP BY t.reason, exec_days_diff
ORDER BY t.reason, exec_days_diff;
```

---

## 5. 「直前のBUY」とセットで見る（重要）

> *REV/TIME/SL/TP* の売りが、どの買いに対応しているかを **同一run内の直前BUY** とペアにして見る。

```sql
WITH trades AS (
  SELECT
    t.run_id,
    t.ts,
    t.side,
    t.reason,
    t.signal_ts,
    ROW_NUMBER() OVER (PARTITION BY t.run_id ORDER BY t.ts) AS seq
  FROM backtests_trades t
  WHERE t.run_id BETWEEN :run_min AND :run_max
), paired AS (
  SELECT
    b.run_id,
    b.ts AS buy_ts,
    s.ts AS sell_ts,
    s.reason,
    s.signal_ts,
    (DATE(s.ts) - DATE(b.ts)) AS days_diff,
    CASE
      WHEN s.signal_ts IS NULL THEN NULL
      ELSE (DATE(s.ts) - DATE(s.signal_ts))
    END AS exec_days_diff
  FROM trades b
  JOIN trades s
    ON s.run_id = b.run_id
   AND s.seq = b.seq + 1
  WHERE b.side='BUY' AND s.side='SELL'
)
SELECT *
FROM paired
WHERE reason IN ('REV','TIME')
ORDER BY run_id, buy_ts;
```

### 5-1. max_days_diff だけを見る（run範囲内）

```sql
WITH paired AS (
  SELECT
    b.run_id,
    b.ts AS buy_ts,
    s.ts AS sell_ts,
    s.reason,
    (DATE(s.ts) - DATE(b.ts)) AS days_diff
  FROM (
    SELECT *, ROW_NUMBER() OVER (PARTITION BY run_id ORDER BY ts) AS seq
    FROM backtests_trades
    WHERE run_id BETWEEN :run_min AND :run_max
  ) b
  JOIN (
    SELECT *, ROW_NUMBER() OVER (PARTITION BY run_id ORDER BY ts) AS seq
    FROM backtests_trades
    WHERE run_id BETWEEN :run_min AND :run_max
  ) s
    ON s.run_id=b.run_id AND s.seq=b.seq+1
  WHERE b.side='BUY' AND s.side='SELL'
)
SELECT
  MAX(days_diff) AS max_days_diff,
  MIN(days_diff) AS min_days_diff
FROM paired;
```

---

## 6. MAX_HOLD_DAYS を超える保有が存在するか？

> 「TIMEが出ていない」だけだと、SL/TP/REVに吸われているだけの可能性があるため、
> “実際のdays_diff” をペアで確認する。

### 6-1. run params をJOINして MAX_HOLD_DAYS=3 の run だけに絞る

```sql
WITH paired AS (
  SELECT
    b.run_id,
    s.reason,
    (DATE(s.ts) - DATE(b.ts)) AS days_diff
  FROM (
    SELECT *, ROW_NUMBER() OVER (PARTITION BY run_id ORDER BY ts) AS seq
    FROM backtests_trades
    WHERE run_id BETWEEN :run_min AND :run_max
  ) b
  JOIN (
    SELECT *, ROW_NUMBER() OVER (PARTITION BY run_id ORDER BY ts) AS seq
    FROM backtests_trades
    WHERE run_id BETWEEN :run_min AND :run_max
  ) s
    ON s.run_id=b.run_id AND s.seq=b.seq+1
  WHERE b.side='BUY' AND s.side='SELL'
)
SELECT
  p.reason,
  p.days_diff,
  COUNT(*) AS n
FROM paired p
JOIN backtests_runs r ON r.run_id = p.run_id
WHERE r.run_id BETWEEN :run_min AND :run_max
  AND (r.params->>'max_hold_days')::int = 3
GROUP BY p.reason, p.days_diff
ORDER BY p.reason, p.days_diff;
```

### 6-2. MAX_HOLD_DAYS=3 の run における最大 days_diff

```sql
WITH paired AS (
  SELECT
    b.run_id,
    (DATE(s.ts) - DATE(b.ts)) AS days_diff
  FROM (
    SELECT *, ROW_NUMBER() OVER (PARTITION BY run_id ORDER BY ts) AS seq
    FROM backtests_trades
    WHERE run_id BETWEEN :run_min AND :run_max
  ) b
  JOIN (
    SELECT *, ROW_NUMBER() OVER (PARTITION BY run_id ORDER BY ts) AS seq
    FROM backtests_trades
    WHERE run_id BETWEEN :run_min AND :run_max
  ) s
    ON s.run_id=b.run_id AND s.seq=b.seq+1
  WHERE b.side='BUY' AND s.side='SELL'
)
SELECT
  MAX(p.days_diff) AS max_days_diff,
  MIN(p.days_diff) AS min_days_diff
FROM paired p
JOIN backtests_runs r ON r.run_id = p.run_id
WHERE r.run_id BETWEEN :run_min AND :run_max
  AND (r.params->>'max_hold_days')::int = 3;
```

---

## 7. 異常を見つけたときの“次の一手”

### 7-1. max_days_diff だけ抜いて原因を特定する

> 「max_days_diff=8 だった」等のときは、該当取引だけ抜いて `buy_ts / signal_ts / sell_ts` を並べると一発で原因が特定できます。

```sql
WITH trades AS (
  SELECT
    t.run_id,
    t.ts,
    t.side,
    t.reason,
    t.signal_ts,
    ROW_NUMBER() OVER (PARTITION BY t.run_id ORDER BY t.ts) AS seq
  FROM backtests_trades t
  WHERE t.run_id BETWEEN :run_min AND :run_max
), paired AS (
  SELECT
    b.run_id,
    b.ts AS buy_ts,
    s.signal_ts,
    s.ts AS sell_ts,
    (DATE(s.ts) - DATE(b.ts)) AS days_diff,
    CASE WHEN s.signal_ts IS NULL THEN NULL ELSE (DATE(s.ts) - DATE(s.signal_ts)) END AS exec_days_diff,
    s.reason
  FROM trades b
  JOIN trades s
    ON s.run_id=b.run_id AND s.seq=b.seq+1
  WHERE b.side='BUY' AND s.side='SELL'
)
SELECT *
FROM paired
WHERE days_diff >= :threshold_days
ORDER BY days_diff DESC, run_id, buy_ts;
```

### 7-2. TIME を“強制的に出す”ための実験パラメータ（メモ）

- `TAKE_PROFIT_RR=999`（TPをほぼ出さない）
- `RISK_PCT` を小さく（ストップ距離が広がり、SLを遠ざけられる設計なら）
- `EXIT_ON_REVERSE=false`（REVに吸われない）
- `MAX_HOLD_DAYS=3`（TIMEを出す）

この状態で `reason` 集計（2-1）を取り、`TIME` が出ること、`exec_days_diff` が 1/3/4 に収まることを確認。

---

## 付録：よく使う最小SQLセット（コピー用）

1) params確認（1-1）
2) reason集計（2-1）
3) 当日売り0件（3）
4) REV/TIMEのsignal_ts NULL 0件（4-1）
5) exec_days_diff 集計（4-3）

