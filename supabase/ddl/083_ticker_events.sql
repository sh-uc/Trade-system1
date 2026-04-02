-- Draft: 銘柄イベント管理
create table if not exists public.ticker_events (
  id bigserial primary key,
  ticker text not null references public.tickers(code),
  event_type text not null,
  event_date date not null,
  event_ts timestamp with time zone null,
  source text not null default 'yfinance',
  source_key text not null default '',
  event_label text null,
  event_value numeric null,
  currency text null,
  confidence text not null default 'medium',
  is_active boolean not null default true,
  meta jsonb not null default '{}'::jsonb,
  created_at timestamp with time zone not null default now(),
  updated_at timestamp with time zone not null default now(),
  constraint ticker_events_type_check check (
    event_type in (
      'earnings_actual',
      'earnings_expected',
      'ex_dividend',
      'record_date',
      'rights',
      'manual_stop',
      'manual_note'
    )
  ),
  constraint ticker_events_confidence_check check (
    confidence in ('low', 'medium', 'high')
  ),
  constraint ticker_events_unique unique (ticker, event_type, event_date, source, source_key)
);

create index if not exists idx_ticker_events_lookup
  on public.ticker_events (ticker, event_date, event_type, is_active);

create index if not exists idx_ticker_events_upcoming
  on public.ticker_events (event_date, event_type, is_active);
