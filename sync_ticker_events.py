import os
import datetime as dt
from typing import Iterable

import pandas as pd
import yfinance as yf

from db_utils import get_supabase, ensure_ticker_name


def _coerce_date(v) -> dt.date | None:
    if v is None:
        return None
    try:
        ts = pd.to_datetime(v)
    except Exception:
        return None
    if pd.isna(ts):
        return None
    if getattr(ts, "tzinfo", None) is not None:
        try:
            ts = ts.tz_convert("Asia/Tokyo")
        except Exception:
            pass
    return ts.date()


def _iter_earnings_events(ticker: str):
    tk = yf.Ticker(ticker)
    seen: set[tuple[str, str]] = set()

    try:
        cal = tk.calendar
        if isinstance(cal, pd.DataFrame) and not cal.empty:
            for idx in cal.index:
                label = str(idx)
                if "earn" not in label.lower():
                    continue
                raw = cal.loc[idx]
                if isinstance(raw, pd.Series):
                    for value in raw.tolist():
                        event_date = _coerce_date(value)
                        if not event_date:
                            continue
                        key = ("earnings", event_date.isoformat())
                        if key in seen:
                            continue
                        seen.add(key)
                        yield {
                            "event_type": "earnings_actual",
                            "event_date": event_date.isoformat(),
                            "event_label": label,
                            "source": "yfinance",
                            "confidence": "medium",
                            "meta": {"provider": "calendar"},
                        }
    except Exception:
        pass

    try:
        ed = tk.get_earnings_dates(limit=8)
        if isinstance(ed, pd.DataFrame) and not ed.empty:
            for idx, row in ed.iterrows():
                event_date = _coerce_date(idx)
                if not event_date:
                    continue
                key = ("earnings", event_date.isoformat())
                if key in seen:
                    continue
                seen.add(key)
                yield {
                    "event_type": "earnings_actual",
                    "event_date": event_date.isoformat(),
                    "event_label": "earnings_date",
                    "source": "yfinance",
                    "confidence": "medium",
                    "meta": {
                        "provider": "get_earnings_dates",
                        "eps_estimate": None if "EPS Estimate" not in row else row.get("EPS Estimate"),
                        "reported_eps": None if "Reported EPS" not in row else row.get("Reported EPS"),
                        "surprise_percent": None if "Surprise(%)" not in row else row.get("Surprise(%)"),
                    },
                }
    except Exception:
        pass


def _iter_dividend_events(ticker: str):
    tk = yf.Ticker(ticker)
    try:
        dividends = tk.dividends
    except Exception:
        dividends = None
    if dividends is None or len(dividends) == 0:
        return
    for idx, value in dividends.items():
        event_date = _coerce_date(idx)
        if not event_date:
            continue
        yield {
            "event_type": "ex_dividend",
            "event_date": event_date.isoformat(),
            "event_label": "ex_dividend",
            "event_value": None if pd.isna(value) else float(value),
            "source": "yfinance",
            "confidence": "high",
            "meta": {"provider": "dividends"},
        }


def upsert_events(sb, ticker: str, events: Iterable[dict]) -> int:
    count = 0
    for ev in events:
        row = {
            "ticker": ticker,
            "event_type": ev["event_type"],
            "event_date": ev["event_date"],
            "event_ts": ev.get("event_ts"),
            "source": ev.get("source", "yfinance"),
            "source_key": ev.get("source_key") or "",
            "event_label": ev.get("event_label"),
            "event_value": ev.get("event_value"),
            "currency": ev.get("currency"),
            "confidence": ev.get("confidence", "medium"),
            "is_active": True,
            "meta": ev.get("meta", {}),
        }
        sb.table("ticker_events").upsert(
            row,
            on_conflict="ticker,event_type,event_date,source,source_key",
        ).execute()
        count += 1
    return count


def main():
    sb = get_supabase()
    tickers_env = os.getenv("EVENT_TICKERS", "")
    if tickers_env.strip():
        tickers = [t.strip() for t in tickers_env.split(",") if t.strip()]
    else:
        rows = sb.table("strategy_param_sets").select("ticker").eq("is_active", True).execute().data or []
        tickers = sorted({r["ticker"] for r in rows})

    if not tickers:
        raise RuntimeError("No tickers found for event sync")

    total = 0
    for ticker in tickers:
        ensure_ticker_name(sb, ticker)
        events = list(_iter_earnings_events(ticker))
        events.extend(list(_iter_dividend_events(ticker)))
        total += upsert_events(sb, ticker, events)
        print(f"[EVENTS] {ticker} synced={len(events)}")

    print(f"[EVENTS] done tickers={len(tickers)} rows={total}")


if __name__ == "__main__":
    main()
