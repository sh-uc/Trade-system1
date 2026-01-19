create table public.backtests_trades (
  run_id bigint not null,
  ts timestamp with time zone not null,
  side text not null,
  price numeric not null,
  qty integer not null,
  reason text null,
  signal_ts timestamp with time zone null,
  constraint backtests_trades_pkey primary key (run_id, ts, side),
  constraint backtests_trades_run_id_fkey foreign KEY (run_id) references backtests_runs (id) on delete CASCADE,
  constraint backtests_trades_side_check check ((side = any (array['BUY'::text, 'SELL'::text])))
) TABLESPACE pg_default;