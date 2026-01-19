create view public.best_params as
select
  ticker,
  jsonb_extract_path_text(params, variadic array['TAKE_PROFIT_RR'::text])::double precision as take_rr,
  jsonb_extract_path_text(params, variadic array['VOL_SPIKE_M'::text])::double precision as vol_m,
  jsonb_extract_path_text(params, variadic array['MACD_ATR_K'::text])::double precision as macd_k,
  total_return,
  max_drawdown,
  sharpe,
  n_trades
from
  backtests_runs
order by
  sharpe desc
limit
  30;