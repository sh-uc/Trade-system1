-- Draft: 約定履歴
create table if not exists public.live_trade_fills (
  id bigserial primary key,
  position_id bigint not null references public.live_positions(id) on delete cascade,
  simulation_name text not null,
  ticker text not null references public.tickers(code),
  strategy_name text not null,
  param_set_id bigint null references public.strategy_param_sets(id),
  executed_at timestamp with time zone not null,
  side text not null check (side in ('BUY', 'SELL')),
  price numeric not null,
  qty integer not null check (qty > 0),
  notional numeric not null,
  fee numeric not null default 0,
  reason text null,
  signal_ts timestamp with time zone null,
  meta jsonb not null default '{}'::jsonb,
  created_at timestamp with time zone not null default now()
);

create index if not exists idx_live_trade_fills_position
  on public.live_trade_fills (position_id, executed_at);

create index if not exists idx_live_trade_fills_simulation
  on public.live_trade_fills (simulation_name, ticker, executed_at desc);
