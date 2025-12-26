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


