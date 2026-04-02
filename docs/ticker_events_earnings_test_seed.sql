-- 検証用 earnings seed
-- 目的:
-- - paper simulation の EARNINGS_BLOCK_DAYS を確認
-- - EARNINGS_FORCE_CLOSE_DAYS を確認

insert into public.ticker_events
  (ticker, event_type, event_date, source, source_key, event_label, confidence, is_active, meta)
values
  ('3103.T', 'earnings_actual', '2025-06-27', 'manual_test', '3103_20250627', 'earnings_test', 'high', true, '{"note":"paper simulation test seed"}'::jsonb),
  ('4506.T', 'earnings_actual', '2025-08-05', 'manual_test', '4506_20250805', 'earnings_test', 'high', true, '{"note":"paper simulation test seed"}'::jsonb),
  ('7013.T', 'earnings_actual', '2025-10-29', 'manual_test', '7013_20251029', 'earnings_test', 'high', true, '{"note":"paper simulation test seed"}'::jsonb)
on conflict (ticker, event_type, event_date, source, source_key)
do update set
  event_label = excluded.event_label,
  confidence = excluded.confidence,
  is_active = excluded.is_active,
  meta = excluded.meta,
  updated_at = now();
