import calendar
import datetime as dt
import os
from collections import Counter, defaultdict

import pandas as pd

from db_utils import get_supabase
from bt_core import is_trading_day


def prev_trading_day(d: dt.date) -> dt.date:
    t = d - dt.timedelta(days=1)
    while not is_trading_day(t):
        t -= dt.timedelta(days=1)
    return t


def next_trading_day_local(d: dt.date) -> dt.date:
    t = d + dt.timedelta(days=1)
    while not is_trading_day(t):
        t += dt.timedelta(days=1)
    return t


def add_trading_days(d: dt.date, offset: int) -> dt.date:
    cur = d
    if offset == 0:
        return cur
    if offset > 0:
        for _ in range(offset):
            cur = next_trading_day_local(cur)
    else:
        for _ in range(-offset):
            cur = prev_trading_day(cur)
    return cur


def clamp_day(year: int, month: int, day: int) -> dt.date:
    last_day = calendar.monthrange(year, month)[1]
    return dt.date(year, month, min(day, last_day))


def main():
    sb = get_supabase()
    target_year = int(os.getenv("EARNINGS_EXPECTED_YEAR", str(dt.date.today().year)))
    min_samples = int(os.getenv("EARNINGS_MIN_SAMPLES", "2"))
    pre_days = int(os.getenv("EARNINGS_WINDOW_PRE_DAYS", "3"))
    post_days = int(os.getenv("EARNINGS_WINDOW_POST_DAYS", "1"))
    tickers_env = os.getenv("EARNINGS_EXPECTED_TICKERS", "")

    q = (
        sb.table("ticker_events")
        .select("ticker,event_date,meta")
        .eq("event_type", "earnings_actual")
        .eq("is_active", True)
    )
    rows = q.execute().data or []
    if tickers_env.strip():
        tickers_filter = {t.strip() for t in tickers_env.split(",") if t.strip()}
        rows = [r for r in rows if r.get("ticker") in tickers_filter]

    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        ticker = row["ticker"]
        event_date = pd.to_datetime(row["event_date"]).date()
        meta = row.get("meta") or {}
        grouped[ticker][event_date.month].append(
            {
                "event_date": event_date,
                "fiscal_label": str(meta.get("fiscal_label") or "").strip(),
                "fiscal_term": str(meta.get("fiscal_term") or "").strip(),
            }
        )

    upsert_count = 0
    for ticker, month_map in grouped.items():
        for month, samples in sorted(month_map.items()):
            if len(samples) < min_samples:
                continue
            dates = [s["event_date"] for s in samples]
            median_day = int(round(pd.Series([d.day for d in dates]).median()))
            center = clamp_day(target_year, month, median_day)
            window_start = add_trading_days(center, -pre_days)
            window_end = add_trading_days(center, post_days)
            label_counter = Counter(
                s["fiscal_label"] for s in samples if s.get("fiscal_label")
            )
            term_counter = Counter(
                s["fiscal_term"] for s in samples if s.get("fiscal_term")
            )
            fiscal_label = label_counter.most_common(1)[0][0] if label_counter else ""
            fiscal_term = term_counter.most_common(1)[0][0] if term_counter else ""
            payload = {
                "ticker": ticker,
                "event_type": "earnings_expected",
                "event_date": center.isoformat(),
                "source": "derived_actual",
                "source_key": f"{ticker}_{target_year}_{month:02d}",
                "event_label": "earnings_expected",
                "confidence": "medium" if len(dates) == 2 else "high",
                "is_active": True,
                "meta": {
                    "kind": "expected",
                    "window_start": window_start.isoformat(),
                    "window_end": window_end.isoformat(),
                    "sample_years": sorted({d.year for d in dates}),
                    "sample_dates": [d.isoformat() for d in sorted(dates)],
                    "sample_count": len(dates),
                    "estimation_method": "median_day_of_month",
                    "pre_days": pre_days,
                    "post_days": post_days,
                    "fiscal_label": fiscal_label,
                    "fiscal_labels": sorted(label_counter.keys()),
                    "fiscal_term": fiscal_term,
                },
            }
            sb.table("ticker_events").upsert(
                payload,
                on_conflict="ticker,event_type,event_date,source,source_key",
            ).execute()
            upsert_count += 1
            print(
                f"[EARNINGS_EXPECTED] {ticker} center={center.isoformat()} "
                f"window={window_start.isoformat()}..{window_end.isoformat()} "
                f"samples={len(dates)} fiscal_label={fiscal_label or '-'}"
            )

    print(f"[EARNINGS_EXPECTED] target_year={target_year} rows={upsert_count}")


if __name__ == "__main__":
    main()
