create table if not exists tickers (
  code text primary key,         -- 例: '3778.T'
  name text not null,            -- 会社名
  updated_at timestamptz not null default now()
);
