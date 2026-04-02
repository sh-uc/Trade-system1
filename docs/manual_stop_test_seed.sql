-- block_entry の検証例
insert into public.ticker_events (
  ticker, event_type, event_date, source, source_key, event_label, confidence, is_active, meta
) values (
  '7013.T',
  'manual_stop',
  '2025-08-06',
  'manual_test',
  'manual_stop_7013_block_20250806',
  'manual_stop',
  'high',
  true,
  jsonb_build_object(
    'action', 'block_entry',
    'window_end', '2025-08-06',
    'note', 'test block entry'
  )
)
on conflict (ticker, event_type, event_date, source, source_key) do update
set is_active = excluded.is_active,
    meta = excluded.meta,
    updated_at = now();

-- force_close の検証例
insert into public.ticker_events (
  ticker, event_type, event_date, source, source_key, event_label, confidence, is_active, meta
) values (
  '6363.T',
  'manual_stop',
  '2025-07-15',
  'manual_test',
  'manual_stop_6363_force_20250715',
  'manual_stop',
  'high',
  true,
  jsonb_build_object(
    'action', 'force_close',
    'window_end', '2025-07-15',
    'note', 'test force close'
  )
)
on conflict (ticker, event_type, event_date, source, source_key) do update
set is_active = excluded.is_active,
    meta = excluded.meta,
    updated_at = now();
