create table public.tickers (
  code text not null,
  name text not null,
  lot_size integer not null default 100,
  updated_at timestamp with time zone not null default now(),
  constraint tickers_pkey primary key (code),
  constraint tickers_lot_size_check check (lot_size > 0)
) TABLESPACE pg_default;
