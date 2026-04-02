① ほぼ同じ結果だらけか？
（銘柄ごとに、final_equity が何種類あるかを見る）
select
  ticker,
  count(*) as runs,
  count(distinct final_equity) as uniq_final,
  count(distinct n_trades) as uniq_trades
from backtests_runs
where ticker in ('4506.T','9600.T')
group by ticker;

“上位が弱い”の正体を見る（勝てないパターンを特定）
② まずトップ20をDBから出す
select
  id, ticker, final_equity, total_return, max_drawdown, sharpe, n_trades,
  params
from backtests_runs
where ticker in ('4506.T','9600.T')
order by final_equity desc
limit 20;

見てほしいのは この3つだけです：

n_trades が極端に少ない（例：7〜15）
→ 期待値以前に検証になりづらい

total_return がほぼゼロ近辺に集まる
→ 仕掛け自体に優位性が薄い可能性

max_drawdown が小さいのに増えない
→ “入ってない”か“利確が伸びない”かのどちらか




A) まず「効いてる/効いてない」をSQLで可視化する（最短で改善に直結）
1) パラメータごとに “結果が変わっているか” を確認

例：MACD_ATR_K を変えているのに結果が同じか？
select
  ticker,
  (params->>'MACD_ATR_K') as macd_k,
  count(*) as runs,
  count(distinct final_equity) as uniq_final,
  count(distinct total_return) as uniq_ret
from backtests_runs
where ticker='9600.T'
group by 1,2
order by 2;

C) まず “実際にどの日に売買してるのか” を見よう（SQL 2本でOK）
1) その run_id（例：9600.T のトップの id=1907）のトレード一覧を見る
select *
from backtests_trades
where run_id = 1907
order by ts;

「その銘柄は観察できるだけ取引したか？」
select
  ticker,
  count(*) as runs,
  avg(n_trades) as avg_trades,
  max(n_trades) as max_trades,
  count(distinct n_trades) as uniq_trades
from backtests_runs
group by ticker
order by avg_trades desc;

A) 8306.T の “ベスト1本” を DB から抜いて「どの条件が効いてるか」を確認
select
  id, ticker, total_return, final_equity, max_drawdown, sharpe, n_trades, params
from backtests_runs
where ticker='8306.T'
order by final_equity desc
limit 10;

1) runごとの SELL reason 内訳（TIMEがあるか確認）
SELECT
  run_id,
  reason,
  COUNT(*) AS n
FROM public.backtests_trades
WHERE side = 'SELL' AND run_id >= 8741

GROUP BY run_id, reason
ORDER BY run_id, n DESC;

2) runごとの SELL件数（= n_tradesの実態）
SELECT
  run_id,
  COUNT(*) AS n_sells
FROM public.backtests_trades
WHERE side = 'SELL'
GROUP BY run_id
ORDER BY n_sells DESC;

3) “同じトレード列か？”の判定（runごとの署名）
SELECT
  run_id,
  md5(string_agg(
        side || ':' || to_char(ts, 'YYYY-MM-DD"T"HH24:MI:SSOF') || ':' ||
        COALESCE(reason,'') || ':' || price::text || ':' || qty::text,
        '|' ORDER BY ts, side
      )) AS trades_sig,
  COUNT(*) AS n_rows
FROM public.backtests_trades
GROUP BY run_id
ORDER BY n_rows DESC;

当日売り（BUY日=SELL日）が残ってないか
with b as (
  select run_id, ts as buy_ts
  from backtests_trades
  where side='BUY' and run_id >= 8885
),
s as (
  select run_id, ts as sell_ts
  from backtests_trades
  where side='SELL' and run_id >= 8885
),
pairs as (
  select b.run_id, b.buy_ts, s.sell_ts
  from b join s on s.run_id=b.run_id and s.sell_ts > b.buy_ts
)
select count(*) as same_day_exits
from pairs
where sell_ts::date = buy_ts::date;

「REV/TIME が出た run」を探すSQL
select run_id, reason, count(*) as n
from backtests_trades
where run_id >= 9029 and run_id <= 9172
  and side = 'SELL'
  and reason in ('REV','TIME')
group by run_id, reason
order by run_id, reason;

run_id 9750〜 の範囲で、SELL reason を全部集計してみてください。
select reason, count(*) as n
from backtests_trades
where run_id >= 9750
  and side='SELL'
group by reason
order by n desc;

「runごとに TIME/SL の比率がどう変わるか」を見ると、グリッドの効きが分かります。
select run_id,
       sum(case when reason='TIME' then 1 else 0 end) as n_time,
       sum(case when reason='SL' then 1 else 0 end) as n_sl,
       count(*) as n_sell
from backtests_trades
where run_id >= 9750
  and side='SELL'
group by run_id
order by run_id;




