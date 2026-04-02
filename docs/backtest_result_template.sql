-- backtest_result_template.sql
-- 標準の結果抽出テンプレート
-- 使い方:
-- 1) target_ticker を差し替える
-- 2) 必要なら created_at / run_id の範囲を絞る
-- 3) 上から順に実行して、Codex に結果を渡す

-- =====================================
-- 0. 対象条件
-- =====================================
-- 例: where ticker = '3778.T'
-- 例: and created_at >= '2026-03-10 00:00:00+09'
-- 例: and id between 16972 and 17025

-- =====================================
-- 1. run の全体件数と分布
-- =====================================
select
  ticker,
  count(*) as runs,
  count(distinct final_equity) as uniq_final_equity,
  count(distinct total_return) as uniq_total_return,
  count(distinct n_trades) as uniq_n_trades,
  min(created_at) as first_created_at,
  max(created_at) as last_created_at
from backtests_runs
where ticker = '3778.T'
group by ticker;

-- =====================================
-- 2. 上位20件
-- =====================================
select
  id,
  ticker,
  final_equity,
  total_return,
  max_drawdown,
  sharpe,
  n_trades,
  created_at,
  params
from backtests_runs
where ticker = '3778.T'
order by final_equity desc
limit 20;

-- =====================================
-- 3. 下位20件
-- =====================================
select
  id,
  ticker,
  final_equity,
  total_return,
  max_drawdown,
  sharpe,
  n_trades,
  created_at,
  params
from backtests_runs
where ticker = '3778.T'
order by final_equity asc
limit 20;

-- =====================================
-- 4. パラメータ別の効き方確認
-- =====================================
select
  ticker,
  (params->>'RISK_PCT')::double precision as risk_pct,
  (params->>'TAKE_PROFIT_RR')::double precision as take_profit_rr,
  (params->>'MAX_HOLD_DAYS')::int as max_hold_days,
  (params->>'VOL_SPIKE_M')::double precision as vol_spike_m,
  (params->>'MACD_ATR_K')::double precision as macd_atr_k,
  (params->>'RSI_MIN')::double precision as rsi_min,
  (params->>'RSI_MAX')::double precision as rsi_max,
  count(*) as runs,
  avg(total_return) as avg_total_return,
  max(total_return) as max_total_return,
  avg(max_drawdown) as avg_max_drawdown,
  avg(n_trades) as avg_n_trades
from backtests_runs
where ticker = '3778.T'
group by 1,2,3,4,5,6,7,8
order by avg(total_return) desc, avg(max_drawdown) asc;

-- =====================================
-- 5. SELL reason 集計
-- =====================================
select
  t.reason,
  count(*) as n
from backtests_trades t
join backtests_runs r on r.id = t.run_id
where r.ticker = '3778.T'
  and t.side = 'SELL'
group by t.reason
order by n desc;

-- =====================================
-- 6. run ごとの SELL reason 内訳
-- =====================================
select
  t.run_id,
  t.reason,
  count(*) as n
from backtests_trades t
join backtests_runs r on r.id = t.run_id
where r.ticker = '3778.T'
  and t.side = 'SELL'
group by t.run_id, t.reason
order by t.run_id, n desc;

-- =====================================
-- 7. トップ run の trade 明細確認
-- =====================================
select *
from backtests_trades
where run_id = (
  select id
  from backtests_runs
  where ticker = '3778.T'
  order by final_equity desc
  limit 1
)
order by ts;

-- =====================================
-- 8. 当日売りが残っていないか確認
-- =====================================
with buy_rows as (
  select run_id, ts as buy_ts
  from backtests_trades
  where side = 'BUY'
),
sell_rows as (
  select run_id, ts as sell_ts
  from backtests_trades
  where side = 'SELL'
),
pairs as (
  select b.run_id, b.buy_ts, s.sell_ts
  from buy_rows b
  join sell_rows s
    on s.run_id = b.run_id
   and s.sell_ts > b.buy_ts
)
select count(*) as same_day_exits
from pairs p
join backtests_runs r on r.id = p.run_id
where r.ticker = '3778.T'
  and p.sell_ts::date = p.buy_ts::date;

-- =====================================
-- 9. 代表 run の署名比較
-- =====================================
select
  t.run_id,
  md5(string_agg(
    t.side || ':' || to_char(t.ts, 'YYYY-MM-DD"T"HH24:MI:SSOF') || ':' ||
    coalesce(t.reason, '') || ':' || t.price::text || ':' || t.qty::text,
    '|' order by t.ts, t.side
  )) as trades_sig,
  count(*) as n_rows
from backtests_trades t
join backtests_runs r on r.id = t.run_id
where r.ticker = '3778.T'
group by t.run_id
order by n_rows desc, t.run_id;
