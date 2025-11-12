create index if not exists idx_signals_code_date on signals(code, date);
create index if not exists idx_prices_code_date on prices(code, date);
create index if not exists idx_runs_created_at on backtests_runs(created_at desc);
-- #