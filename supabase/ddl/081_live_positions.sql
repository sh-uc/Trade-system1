-- Draft: 建玉管理
create table if not exists public.live_positions (
  id bigserial primary key,
  simulation_name text not null,
  ticker text not null references public.tickers(code),
  strategy_name text not null,
  param_set_id bigint null references public.strategy_param_sets(id),
  status text not null check (status in ('OPEN', 'CLOSED')),
  opened_at timestamp with time zone not null,
  closed_at timestamp with time zone null,
  entry_signal_ts timestamp with time zone null,
  entry_price numeric not null,
  exit_price numeric null,
  qty integer not null check (qty > 0),
  lot_size integer not null check (lot_size > 0),
  current_stop_price numeric null,
  take_profit_price numeric null,
  break_even_armed boolean not null default false,
  trailing_armed boolean not null default false,
  highest_price_since_entry numeric null,
  close_reason text null,
  realized_pnl numeric not null default 0,
  fees numeric not null default 0,
  meta jsonb not null default '{}'::jsonb,
  created_at timestamp with time zone not null default now(),
  updated_at timestamp with time zone not null default now()
);

create index if not exists idx_live_positions_open
  on public.live_positions (simulation_name, status, ticker, opened_at desc);

create index if not exists idx_live_positions_param_set
  on public.live_positions (param_set_id);
