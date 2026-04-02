create table if not exists public.jpx_ticker_master (
  code text not null,
  ticker_code text not null,
  name text not null,
  market_product_category text not null,
  source_date date null,
  source_file text null,
  updated_at timestamp with time zone not null default now(),
  constraint jpx_ticker_master_pkey primary key (code),
  constraint jpx_ticker_master_ticker_code_key unique (ticker_code)
);

create index if not exists idx_jpx_ticker_master_ticker_code
  on public.jpx_ticker_master (ticker_code);
