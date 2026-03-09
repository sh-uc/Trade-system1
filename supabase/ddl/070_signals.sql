create table public.signals (
  date date not null,
  code text not null,
  action text not null,
  summary text null,
  reasons jsonb null,
  close numeric null,
  qty integer null,
  sl numeric null,
  tp numeric null,
  risk_jpy numeric null,
  created_at timestamp with time zone null default now(),
  diff_pct double precision null,
  prev_close double precision null,
  constraint signals_pkey primary key (date, code)
) TABLESPACE pg_default;

create index IF not exists idx_signals_code_date on public.signals using btree (code, date) TABLESPACE pg_default;