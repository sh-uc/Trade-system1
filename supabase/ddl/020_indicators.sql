create table public.indicators (
  date date not null,
  code text not null,
  rsi14 numeric null,
  macd numeric null,
  macd_signal numeric null,
  atr14 numeric null,
  stdev20 numeric null,
  ma25 numeric null,
  ma75 numeric null,
  vol_ma20 numeric null,
  vol_spike boolean null,
  swing_low20 numeric null,
  swing_high20 numeric null,
  constraint indicators_pkey primary key (date, code)
) TABLESPACE pg_default;