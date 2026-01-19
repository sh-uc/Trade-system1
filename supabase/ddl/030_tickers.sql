create table public.tickers (
  code text not null,
  name text not null,
  updated_at timestamp with time zone not null default now(),
  constraint tickers_pkey primary key (code)
) TABLESPACE pg_default;