create table public.backtests_runs (
  id bigserial not null,
  ticker text not null,
  started_at timestamp with time zone not null default now(),
  params jsonb not null,
  final_equity numeric not null,
  total_return numeric not null,
  max_drawdown numeric not null,
  sharpe numeric null,
  n_trades integer not null,
  created_at timestamp with time zone not null default now(),
  constraint backtests_runs_pkey primary key (id)
) TABLESPACE pg_default;

create index IF not exists idx_runs_created_at on public.backtests_runs using btree (created_at desc) TABLESPACE pg_default;

create index IF not exists backtests_runs_ticker_started_at_idx on public.backtests_runs using btree (ticker, started_at desc) TABLESPACE pg_default;