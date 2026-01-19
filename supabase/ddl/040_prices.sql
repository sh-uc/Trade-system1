create table public.prices (
  date date not null,
  code text not null,
  open numeric null,
  high numeric null,
  low numeric null,
  close numeric null,
  volume bigint null,
  constraint prices_pkey primary key (date, code)
) TABLESPACE pg_default;

create index IF not exists idx_prices_code_date on public.prices using btree (code, date) TABLESPACE pg_default;
