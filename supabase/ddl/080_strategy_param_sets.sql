-- Draft: 推奨パラメータ管理
create table if not exists public.strategy_param_sets (
  id bigserial primary key,
  strategy_name text not null,
  ticker text not null references public.tickers(code),
  param_set_name text not null,
  params jsonb not null,
  source_run_id bigint null references public.backtests_runs(id),
  is_active boolean not null default true,
  effective_from date not null default current_date,
  effective_to date null,
  notes text null,
  created_at timestamp with time zone not null default now(),
  updated_at timestamp with time zone not null default now(),
  constraint strategy_param_sets_unique unique (strategy_name, ticker, param_set_name, effective_from)
);

create index if not exists idx_strategy_param_sets_active
  on public.strategy_param_sets (strategy_name, ticker, is_active, effective_from desc);
