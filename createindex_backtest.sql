create index if not exists backtests_runs_ticker_started_at_idx
on backtests_runs (ticker, started_at desc);
-- #