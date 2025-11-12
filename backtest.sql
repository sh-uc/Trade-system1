create table if not exists backtests_runs (
  id bigserial primary key,
  ticker text not null,
  started_at timestamptz not null default now(),
  params jsonb not null,
  final_equity numeric not null,
  total_return numeric not null,
  max_drawdown numeric not null,
  sharpe numeric,
  n_trades int not null
);

create table if not exists backtests_trades (
  run_id bigint references backtests_runs(id) on delete cascade,
  ts timestamptz not null,
  side text not null check (side in ('BUY','SELL')),
  price numeric not null,
  qty int not null,
  reason text,
  primary key (run_id, ts, side)
);
-- #