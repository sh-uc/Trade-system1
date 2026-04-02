import datetime as dt
import os
from pathlib import Path
from typing import Any

import pandas as pd

from bt_core import compute_indicators, fetch_prices, long_signal_row, next_trading_day
from db_utils import ensure_ticker_name, get_supabase
from simulate_live import (
    fetch_active_param_sets,
    fetch_event_date_map,
    fetch_manual_stop_rules,
    has_event_within_days,
    has_manual_stop,
    merge_event_dates,
    parse_label_set,
)


def to_date(value: str | None, default: dt.date) -> dt.date:
    if not value:
        return default
    return pd.to_datetime(value).date()


def fetch_name_map(sb, tickers: list[str]) -> dict[str, str]:
    out = {}
    if not tickers:
        return out
    try:
        rows = sb.table("tickers").select("code,name").in_("code", tickers).execute().data or []
    except Exception:
        rows = []
    for row in rows:
        code = row.get("code")
        if code:
            out[code] = str(row.get("name") or code)
    for ticker in tickers:
        out.setdefault(ticker, ticker)
    return out


def fetch_open_positions(sb, simulation_name: str) -> list[dict[str, Any]]:
    if not simulation_name:
        return []
    try:
        return (
            sb.table("live_positions")
            .select(
                "id,ticker,opened_at,entry_price,qty,current_stop_price,"
                "take_profit_price,break_even_armed,trailing_armed,highest_price_since_entry"
            )
            .eq("simulation_name", simulation_name)
            .eq("status", "OPEN")
            .order("opened_at")
            .execute()
            .data
            or []
        )
    except Exception:
        return []


def fmt_yen(value: float) -> str:
    return f"{value:,.0f}"


def fmt_pct(value: float) -> str:
    return f"{value:.2f}%"


def build_monitor_report(
    *,
    as_of: dt.date,
    strategy_name: str,
    position_simulation_name: str = "",
    report_dir: str = "docs/daily_reports",
    earnings_block_days: int = 2,
    earnings_block_labels: set[str] | None = None,
    ex_dividend_block_days: int = 1,
    upcoming_event_days: int = 7,
) -> dict[str, Any]:
    report_dir_path = Path(report_dir)
    report_dir_path.mkdir(parents=True, exist_ok=True)
    sb = get_supabase()
    param_sets = fetch_active_param_sets(sb, strategy_name)
    if not param_sets:
        raise RuntimeError(f"No active strategy_param_sets found for {strategy_name}")

    tickers = sorted({row["ticker"] for row in param_sets})
    for ticker in tickers:
        ensure_ticker_name(sb, ticker)
    name_map = fetch_name_map(sb, tickers)

    earnings_actual = fetch_event_date_map(
        sb, tickers, "earnings_actual", allowed_labels=earnings_block_labels or None
    )
    earnings_expected = fetch_event_date_map(
        sb, tickers, "earnings_expected", allowed_labels=earnings_block_labels or None
    )
    earnings_events = merge_event_dates(earnings_actual, earnings_expected)
    ex_dividend_events = fetch_event_date_map(sb, tickers, "ex_dividend")
    manual_stop_rules = fetch_manual_stop_rules(sb, tickers)
    open_positions = fetch_open_positions(sb, position_simulation_name)
    open_by_ticker = {row["ticker"]: row for row in open_positions}

    candidates: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []
    watchlist: list[dict[str, Any]] = []
    latest_dates: list[dt.date] = []

    for row in param_sets:
        ticker = row["ticker"]
        params = dict(row.get("params") or {})
        prices = fetch_prices(ticker, (as_of - dt.timedelta(days=420)).isoformat(), as_of.isoformat())
        ind = compute_indicators(prices)
        if ind.empty:
            continue
        as_of_ts = pd.Timestamp(as_of)
        if getattr(ind.index, "tz", None) is not None:
            as_of_ts = as_of_ts.tz_localize(ind.index.tz)
        current_ts = ind.index[ind.index <= as_of_ts]
        if len(current_ts) == 0:
            continue
        current_date = current_ts[-1]
        latest_dates.append(current_date.date())
        current = ind.loc[current_date]
        buy_for = next_trading_day(current_date.date())
        signal = long_signal_row(
            current,
            MACD_ATR_K=float(params.get("MACD_ATR_K", 0.13)),
            RSI_MIN=float(params.get("RSI_MIN", 30.0)),
            RSI_MAX=float(params.get("RSI_MAX", 80.0)),
            VOL_SPIKE_M=float(params.get("VOL_SPIKE_M", 1.0)),
        )
        earnings_block = has_event_within_days(
            earnings_events.get(ticker, set()), buy_for, earnings_block_days
        )
        ex_dividend_block = has_event_within_days(
            ex_dividend_events.get(ticker, set()), buy_for, ex_dividend_block_days
        )
        manual_block = has_manual_stop(manual_stop_rules.get(ticker, []), buy_for, "block_entry")

        event_notes = []
        upcoming_earnings = sorted(
            d for d in earnings_events.get(ticker, set()) if 0 <= (d - as_of).days <= upcoming_event_days
        )
        upcoming_exdiv = sorted(
            d for d in ex_dividend_events.get(ticker, set()) if 0 <= (d - as_of).days <= upcoming_event_days
        )
        if upcoming_earnings:
            event_notes.append(f"決算:{','.join(d.isoformat() for d in upcoming_earnings)}")
        if upcoming_exdiv:
            event_notes.append(f"権利落ち:{','.join(d.isoformat() for d in upcoming_exdiv)}")

        item = {
            "ticker": ticker,
            "name": name_map.get(ticker, ticker),
            "latest_date": current_date.date().isoformat(),
            "buy_for": buy_for.isoformat(),
            "close": float(current["close"]),
            "rsi14": float(current.get("rsi14", 0.0)),
            "macd_atr": float(current.get("macd_atr", 0.0)),
            "signal": bool(signal),
            "blocked_reasons": [
                label
                for label, enabled in (
                    ("FY決算回避", earnings_block),
                    ("権利落ち回避", ex_dividend_block),
                    ("手動停止", manual_block),
                )
                if enabled
            ],
            "events": " / ".join(event_notes),
        }

        if ticker in open_by_ticker:
            position = open_by_ticker[ticker]
            entry_price = float(position.get("entry_price") or 0)
            qty = int(position.get("qty") or 0)
            current_notional = float(current["close"]) * qty
            unrealized = (float(current["close"]) - entry_price) * qty
            watchlist.append(
                {
                    **item,
                    "opened_at": str(position.get("opened_at") or "")[:10],
                    "entry_price": entry_price,
                    "qty": qty,
                    "current_stop_price": float(position.get("current_stop_price") or 0),
                    "take_profit_price": float(position.get("take_profit_price") or 0),
                    "unrealized_pnl": unrealized,
                    "unrealized_pct": ((float(current["close"]) / entry_price - 1) * 100.0) if entry_price > 0 else 0.0,
                    "current_notional": current_notional,
                    "break_even_armed": bool(position.get("break_even_armed")),
                    "trailing_armed": bool(position.get("trailing_armed")),
                }
            )
        elif signal and item["blocked_reasons"]:
            blocked.append(item)
        elif signal:
            candidates.append(item)

    latest_market_date = max(latest_dates) if latest_dates else as_of

    candidates.sort(key=lambda x: (x["macd_atr"], x["rsi14"]), reverse=True)
    blocked.sort(key=lambda x: (x["macd_atr"], x["rsi14"]), reverse=True)
    watchlist.sort(key=lambda x: x["ticker"])

    report_lines: list[str] = []
    report_lines.append(f"# Daily Paper Monitor {as_of.isoformat()}")
    report_lines.append("")
    report_lines.append(f"- strategy: `{strategy_name}`")
    report_lines.append(f"- as_of: `{as_of.isoformat()}`")
    report_lines.append(f"- latest_market_date: `{latest_market_date.isoformat()}`")
    report_lines.append(f"- monitored_tickers: `{len(tickers)}`")
    report_lines.append(f"- open_positions: `{len(watchlist)}`")
    report_lines.append(f"- entry_candidates: `{len(candidates)}`")
    report_lines.append(f"- blocked_signals: `{len(blocked)}`")
    report_lines.append("")

    report_lines.append("## Open Positions")
    report_lines.append("")
    if not watchlist:
        report_lines.append("なし")
    else:
        report_lines.append("| ticker | name | opened_at | qty | close | stop | take | unrealized | flags | events |")
        report_lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |")
        for item in watchlist:
            flags = []
            if item["break_even_armed"]:
                flags.append("BE")
            if item["trailing_armed"]:
                flags.append("TRAIL")
            report_lines.append(
                f"| {item['ticker']} | {item['name']} | {item['opened_at']} | {item['qty']} | "
                f"{item['close']:.2f} | {item['current_stop_price']:.2f} | {item['take_profit_price']:.2f} | "
                f"{fmt_yen(item['unrealized_pnl'])} ({fmt_pct(item['unrealized_pct'])}) | "
                f"{','.join(flags) or '-'} | {item['events'] or '-'} |"
            )
    report_lines.append("")

    report_lines.append("## New Entry Candidates")
    report_lines.append("")
    if not candidates:
        report_lines.append("なし")
    else:
        report_lines.append("| ticker | name | signal_date | buy_for | close | RSI14 | MACD_ATR | events |")
        report_lines.append("| --- | --- | --- | --- | ---: | ---: | ---: | --- |")
        for item in candidates:
            report_lines.append(
                f"| {item['ticker']} | {item['name']} | {item['latest_date']} | {item['buy_for']} | "
                f"{item['close']:.2f} | {item['rsi14']:.2f} | {item['macd_atr']:.4f} | {item['events'] or '-'} |"
            )
    report_lines.append("")

    report_lines.append("## Blocked Signals")
    report_lines.append("")
    if not blocked:
        report_lines.append("なし")
    else:
        report_lines.append("| ticker | name | signal_date | buy_for | blocked_reasons | close | RSI14 | MACD_ATR | events |")
        report_lines.append("| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- |")
        for item in blocked:
            report_lines.append(
                f"| {item['ticker']} | {item['name']} | {item['latest_date']} | {item['buy_for']} | "
                f"{','.join(item['blocked_reasons'])} | {item['close']:.2f} | {item['rsi14']:.2f} | "
                f"{item['macd_atr']:.4f} | {item['events'] or '-'} |"
            )
    report_lines.append("")

    report_lines.append("## Upcoming Events")
    report_lines.append("")
    upcoming_rows = []
    for ticker in tickers:
        notes = []
        for d in sorted(
            d for d in earnings_events.get(ticker, set()) if 0 <= (d - as_of).days <= upcoming_event_days
        ):
            notes.append(f"FY決算:{d.isoformat()}")
        for d in sorted(
            d for d in ex_dividend_events.get(ticker, set()) if 0 <= (d - as_of).days <= upcoming_event_days
        ):
            notes.append(f"権利落ち:{d.isoformat()}")
        if notes:
            upcoming_rows.append((ticker, name_map.get(ticker, ticker), " / ".join(notes)))
    if not upcoming_rows:
        report_lines.append("なし")
    else:
        report_lines.append("| ticker | name | events |")
        report_lines.append("| --- | --- | --- |")
        for ticker, name, notes in upcoming_rows:
            report_lines.append(f"| {ticker} | {name} | {notes} |")

    report_text = "\n".join(report_lines) + "\n"
    report_path = report_dir_path / f"daily_paper_monitor_{strategy_name}_{as_of.isoformat()}.md"
    report_path.write_text(report_text, encoding="utf-8")

    return {
        "strategy_name": strategy_name,
        "as_of": as_of,
        "latest_market_date": latest_market_date,
        "tickers": tickers,
        "open_positions": watchlist,
        "candidates": candidates,
        "blocked": blocked,
        "report_text": report_text,
        "report_path": report_path,
        "position_simulation_name": position_simulation_name,
    }


def build_line_summary(report: dict[str, Any]) -> str:
    lines = [
        f"【日次紙上モニター】{report['as_of'].isoformat()}",
        f"戦略: {report['strategy_name']}",
        f"市場データ: {report['latest_market_date'].isoformat()}",
        (
            f"保有 {len(report['open_positions'])}件 / "
            f"新規候補 {len(report['candidates'])}件 / "
            f"見送り {len(report['blocked'])}件"
        ),
    ]

    if report["open_positions"]:
        lines.append("")
        lines.append("【保有中】")
        for item in report["open_positions"][:5]:
            lines.append(
                f"・{item['ticker']}（{item['name']}） "
                f"含み {fmt_yen(item['unrealized_pnl'])} "
                f"({fmt_pct(item['unrealized_pct'])}) / stop {item['current_stop_price']:.2f}"
            )

    if report["candidates"]:
        lines.append("")
        lines.append("【新規候補】")
        for item in report["candidates"][:5]:
            lines.append(
                f"・{item['ticker']}（{item['name']}） "
                f"買付候補 {item['buy_for']} / 終値 {item['close']:.2f}"
            )

    if report["blocked"]:
        lines.append("")
        lines.append("【見送り】")
        for item in report["blocked"][:5]:
            lines.append(
                f"・{item['ticker']}（{item['name']}） "
                f"{','.join(item['blocked_reasons'])}"
            )

    return "\n".join(lines)[:4900]


def main() -> None:
    today = dt.date.today()
    as_of = to_date(os.getenv("MONITOR_DATE"), today)
    strategy_name = os.getenv("STRATEGY_NAME", "swing_v1_candidate9")
    position_simulation_name = os.getenv("POSITION_SIMULATION_NAME", "")
    report_dir = os.getenv("REPORT_DIR", "docs/daily_reports")

    earnings_block_days = int(os.getenv("EARNINGS_BLOCK_DAYS", "2"))
    earnings_block_labels = parse_label_set(os.getenv("EARNINGS_BLOCK_LABELS", "FY"))
    ex_dividend_block_days = int(os.getenv("EX_DIVIDEND_BLOCK_DAYS", "1"))
    upcoming_event_days = int(os.getenv("UPCOMING_EVENT_DAYS", "7"))

    report = build_monitor_report(
        as_of=as_of,
        strategy_name=strategy_name,
        position_simulation_name=position_simulation_name,
        report_dir=report_dir,
        earnings_block_days=earnings_block_days,
        earnings_block_labels=earnings_block_labels,
        ex_dividend_block_days=ex_dividend_block_days,
        upcoming_event_days=upcoming_event_days,
    )

    print(
        f"[MONITOR] strategy_name={strategy_name} as_of={as_of.isoformat()} "
        f"latest_market_date={report['latest_market_date'].isoformat()}"
    )
    print(
        f"[MONITOR] open_positions={len(report['open_positions'])} "
        f"entry_candidates={len(report['candidates'])} blocked_signals={len(report['blocked'])}"
    )
    print(f"[MONITOR] report={report['report_path']}")


if __name__ == "__main__":
    main()
