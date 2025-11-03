create or replace view best_params as
select
  ticker,
  jsonb_extract_path_text(params, 'TAKE_PROFIT_RR')::float as take_rr,
  jsonb_extract_path_text(params, 'VOL_SPIKE_M')::float as vol_m,
  jsonb_extract_path_text(params, 'MACD_ATR_K')::float as macd_k,
  total_return, max_drawdown, sharpe, n_trades
from backtests_runs
order by sharpe desc
limit 30;
