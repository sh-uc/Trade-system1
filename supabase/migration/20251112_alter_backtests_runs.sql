alter table backtests_runs
  add column if not exists created_at timestamptz not null default now();
-- #