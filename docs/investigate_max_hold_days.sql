-- investigate_max_hold_days.sql
-- MAX_HOLD_DAYS が効かない理由を調べるための標準SQL
-- 使い方:
-- 1) target_ticker または run_id 範囲を差し替える
-- 2) 上から順に実行して、TIME / REV / TP / SL の偏りを確認する

-- =====================================
-- 1. SELL reason の全体内訳
-- =====================================
select
  r.ticker,
  t.reason,
  count(*) as n
from backtests_trades t
join backtests_runs r on r.id = t.run_id
where r.ticker in ('3103.T','3964.T','4275.T','3778.T')
  and t.side = 'SELL'
group by r.ticker, t.reason
order by r.ticker, n desc;

-- =====================================
-- 2. run ごとの SELL reason 内訳
-- =====================================
select
  t.run_id,
  r.ticker,
  t.reason,
  count(*) as n
from backtests_trades t
join backtests_runs r on r.id = t.run_id
where t.run_id between 17093 and 17151
  and t.side = 'SELL'
group by t.run_id, r.ticker, t.reason
order by r.ticker, t.run_id, n desc;

-- =====================================
-- 3. BUY -> SELL の保有日数分布
-- =====================================
with buys as (
  select
    run_id,
    ts as buy_ts,
    row_number() over (partition by run_id order by ts) as seq
  from backtests_trades
  where side = 'BUY'
),
sells as (
  select
    run_id,
    ts as sell_ts,
    reason,
    row_number() over (partition by run_id order by ts) as seq
  from backtests_trades
  where side = 'SELL'
),
pairs as (
  select
    b.run_id,
    b.buy_ts,
    s.sell_ts,
    s.reason,
    (s.sell_ts::date - b.buy_ts::date) as hold_days
  from buys b
  join sells s
    on s.run_id = b.run_id
   and s.seq = b.seq
)
select
  r.ticker,
  p.reason,
  p.hold_days,
  count(*) as n
from pairs p
join backtests_runs r on r.id = p.run_id
where r.ticker in ('3103.T','3964.T','4275.T','3778.T')
group by r.ticker, p.reason, p.hold_days
order by r.ticker, p.hold_days, n desc;

-- =====================================
-- 4. TIME が本当に出ているか確認
-- =====================================
select
  r.ticker,
  count(*) as n_time
from backtests_trades t
join backtests_runs r on r.id = t.run_id
where t.side = 'SELL'
  and t.reason = 'TIME'
  and r.ticker in ('3103.T','3964.T','4275.T','3778.T')
group by r.ticker
order by r.ticker;

-- =====================================
-- 5. REV が本当に出ているか確認
-- =====================================
select
  r.ticker,
  count(*) as n_rev
from backtests_trades t
join backtests_runs r on r.id = t.run_id
where t.side = 'SELL'
  and t.reason = 'REV'
  and r.ticker in ('3103.T','3964.T','4275.T','3778.T')
group by r.ticker
order by r.ticker;

-- =====================================
-- 6. signal_ts -> ts の日数差
-- REV / TIME が出ている場合だけ意味を持つ
-- =====================================
select
  r.ticker,
  t.reason,
  (t.ts::date - t.signal_ts::date) as signal_to_sell_days,
  count(*) as n
from backtests_trades t
join backtests_runs r on r.id = t.run_id
where t.side = 'SELL'
  and t.signal_ts is not null
  and r.ticker in ('3103.T','3964.T','4275.T','3778.T')
group by r.ticker, t.reason, (t.ts::date - t.signal_ts::date)
order by r.ticker, signal_to_sell_days, n desc;

-- =====================================
-- 7. TP より前に SL で落ちていないかのざっくり確認
-- =====================================
select
  r.ticker,
  sum(case when t.reason = 'TP' then 1 else 0 end) as n_tp,
  sum(case when t.reason = 'SL' then 1 else 0 end) as n_sl,
  count(*) as n_sell,
  round(sum(case when t.reason = 'TP' then 1 else 0 end)::numeric / nullif(count(*),0), 4) as tp_ratio,
  round(sum(case when t.reason = 'SL' then 1 else 0 end)::numeric / nullif(count(*),0), 4) as sl_ratio
from backtests_trades t
join backtests_runs r on r.id = t.run_id
where t.side = 'SELL'
  and r.ticker in ('3103.T','3964.T','4275.T','3778.T')
group by r.ticker
order by r.ticker;
