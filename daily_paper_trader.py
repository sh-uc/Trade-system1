import datetime as dt
import math
import os
from typing import Any

import pandas as pd
import requests

from bt_core import (
    compute_indicators,
    fetch_prices,
    long_signal_row,
    next_trading_day,
    resolve_intraday_ambiguous_exit,
)
from daily_paper_monitor import build_line_summary, build_monitor_report
from db_utils import ensure_ticker_name, get_supabase
from simulate_live import (
    PositionState,
    as_bool,
    calc_qty,
    fetch_active_param_sets,
    fetch_event_date_map,
    fetch_lot_size_map,
    fetch_manual_stop_rules,
    has_event_within_days,
    has_manual_stop,
    insert_fill,
    is_force_close_day,
    merge_event_dates,
    parse_label_set,
)


def send_line_message_broadcast(text: str) -> None:
    token = os.environ.get("LINE_CHANNEL_TOKEN")
    if not token:
        print("[WARN] LINE_CHANNEL_TOKEN not set; skip LINE broadcast")
        return
    url = "https://api.line.me/v2/bot/message/broadcast"
    headers = {"Authorization": f"Bearer {token}"}
    payload = {"messages": [{"type": "text", "text": text[:4900]}]}
    response = requests.post(url, json=payload, headers=headers, timeout=10)
    print("[LINE broadcast]", response.status_code, response.text)


def send_line_message_push(user_id: str, text: str) -> None:
    token = os.environ.get("LINE_CHANNEL_TOKEN")
    if not token:
        print("[WARN] LINE_CHANNEL_TOKEN not set; skip LINE push")
        return
    url = "https://api.line.me/v2/bot/message/push"
    headers = {"Authorization": f"Bearer {token}"}
    payload = {"to": user_id, "messages": [{"type": "text", "text": text[:4900]}]}
    response = requests.post(url, json=payload, headers=headers, timeout=10)
    print("[LINE push]", response.status_code, response.text)


def to_date(value: str | None, default: dt.date) -> dt.date:
    if not value:
        return default
    return pd.to_datetime(value).date()


def fetch_pending_orders(sb, simulation_name: str, scheduled_for: dt.date) -> list[dict[str, Any]]:
    return (
        sb.table("daily_trader_orders")
        .select("*")
        .eq("simulation_name", simulation_name)
        .eq("status", "PENDING")
        .eq("scheduled_for", scheduled_for.isoformat())
        .order("id")
        .execute()
        .data
        or []
    )


def fetch_open_positions_state(sb, simulation_name: str) -> dict[str, PositionState]:
    rows = (
        sb.table("live_positions")
        .select("*")
        .eq("simulation_name", simulation_name)
        .eq("status", "OPEN")
        .execute()
        .data
        or []
    )
    out: dict[str, PositionState] = {}
    for row in rows:
        meta = row.get("meta") or {}
        params = dict(meta.get("params_snapshot") or {})
        opened_at = pd.to_datetime(row["opened_at"])
        out[row["ticker"]] = PositionState(
            position_id=int(row["id"]),
            ticker=row["ticker"],
            strategy_name=row["strategy_name"],
            param_set_id=int(row["param_set_id"]) if row.get("param_set_id") is not None else 0,
            params=params,
            qty=int(row["qty"]),
            lot_size=int(row["lot_size"]),
            entry_px=float(row["entry_price"]),
            entry_date=opened_at,
            entry_bar_index=int(meta.get("entry_bar_index", -1)),
            stop_px=float(row.get("current_stop_price") or 0.0),
            take_px=float(row.get("take_profit_price") or 0.0),
            break_even_armed=bool(row.get("break_even_armed")),
            trailing_armed=bool(row.get("trailing_armed")),
            highest_high_since_entry=float(row.get("highest_price_since_entry") or math.nan),
            pending_sell_for=(
                pd.to_datetime(meta["pending_sell_for"]).date() if meta.get("pending_sell_for") else None
            ),
            pending_sell_reason=meta.get("pending_sell_reason"),
            pending_sell_signal_ts=meta.get("pending_sell_signal_ts"),
            just_bought=bool(meta.get("just_bought", False)),
        )
    return out


def compute_cash(sb, simulation_name: str, initial_capital: float) -> float:
    fills = (
        sb.table("live_trade_fills")
        .select("side,notional,fee")
        .eq("simulation_name", simulation_name)
        .execute()
        .data
        or []
    )
    cash = initial_capital
    for row in fills:
        side = row.get("side")
        notional = float(row.get("notional") or 0)
        fee = float(row.get("fee") or 0)
        if side == "BUY":
            cash -= notional + fee
        elif side == "SELL":
            cash += notional - fee
    return cash


def upsert_order(
    sb,
    *,
    simulation_name: str,
    ticker: str,
    strategy_name: str,
    param_set_id: int,
    order_type: str,
    scheduled_for: dt.date,
    reason: str,
    signal_ts: str | None,
    position_id: int | None = None,
    meta: dict[str, Any] | None = None,
) -> None:
    existing = (
        sb.table("daily_trader_orders")
        .select("id")
        .eq("simulation_name", simulation_name)
        .eq("ticker", ticker)
        .eq("status", "PENDING")
        .eq("order_type", order_type)
        .eq("scheduled_for", scheduled_for.isoformat())
        .execute()
        .data
        or []
    )
    payload = {
        "simulation_name": simulation_name,
        "ticker": ticker,
        "strategy_name": strategy_name,
        "param_set_id": param_set_id or None,
        "position_id": position_id,
        "order_type": order_type,
        "scheduled_for": scheduled_for.isoformat(),
        "status": "PENDING",
        "reason": reason,
        "signal_ts": signal_ts,
        "meta": meta or {},
        "updated_at": pd.Timestamp.utcnow().isoformat(),
    }
    if existing:
        sb.table("daily_trader_orders").update(payload).eq("id", existing[0]["id"]).execute()
    else:
        sb.table("daily_trader_orders").insert(payload).execute()


def update_order_status(sb, order_id: int, status: str, meta: dict[str, Any] | None = None) -> None:
    payload: dict[str, Any] = {"status": status, "updated_at": pd.Timestamp.utcnow().isoformat()}
    if meta is not None:
        payload["meta"] = meta
    sb.table("daily_trader_orders").update(payload).eq("id", order_id).execute()


def save_monitor_report(report: dict[str, Any]) -> None:
    sb = get_supabase()
    row = {
        "report_date": report["as_of"].isoformat(),
        "strategy_name": report["strategy_name"],
        "simulation_name": report["position_simulation_name"] or "",
        "latest_market_date": report["latest_market_date"].isoformat(),
        "open_positions_count": len(report["open_positions"]),
        "entry_candidates_count": len(report["candidates"]),
        "blocked_signals_count": len(report["blocked"]),
        "report_markdown": report["report_text"],
        "line_summary": build_line_summary(report),
        "meta": {"report_path": str(report["report_path"]), "tickers": report["tickers"]},
    }
    sb.table("daily_monitor_reports").upsert(
        row, on_conflict="report_date,strategy_name,simulation_name"
    ).execute()


def main() -> None:
    process_date = to_date(os.getenv("TRADER_DATE"), dt.date.today())
    simulation_name = os.getenv("TRADER_SIMULATION_NAME", "paper_candidate9_daily")
    strategy_name = os.getenv("STRATEGY_NAME", "swing_v1_candidate9")
    report_dir = os.getenv("REPORT_DIR", "docs/daily_reports")
    initial_capital = float(os.getenv("SIM_CAPITAL", "3000000"))
    max_open_positions = int(os.getenv("MAX_OPEN_POSITIONS", "5"))
    max_new_positions_per_day = int(os.getenv("MAX_NEW_POSITIONS_PER_DAY", "2"))
    max_total_exposure = float(os.getenv("MAX_TOTAL_EXPOSURE", "0"))
    earnings_block_days = int(os.getenv("EARNINGS_BLOCK_DAYS", "2"))
    earnings_force_close_days = int(os.getenv("EARNINGS_FORCE_CLOSE_DAYS", "0"))
    earnings_block_labels = parse_label_set(os.getenv("EARNINGS_BLOCK_LABELS", "FY"))
    earnings_force_close_labels = parse_label_set(os.getenv("EARNINGS_FORCE_CLOSE_LABELS", ""))
    ex_dividend_block_days = int(os.getenv("EX_DIVIDEND_BLOCK_DAYS", "1"))
    ex_dividend_force_close_days = int(os.getenv("EX_DIVIDEND_FORCE_CLOSE_DAYS", "0"))

    sb = get_supabase()
    param_sets = fetch_active_param_sets(sb, strategy_name)
    if not param_sets:
        raise RuntimeError(f"No active strategy_param_sets found for {strategy_name}")

    tickers = sorted({row["ticker"] for row in param_sets})
    lot_sizes = fetch_lot_size_map(sb, tickers)
    for ticker in tickers:
        ensure_ticker_name(sb, ticker)

    earnings_actual_events = fetch_event_date_map(
        sb, tickers, "earnings_actual", allowed_labels=earnings_block_labels or None
    )
    earnings_expected_events = fetch_event_date_map(
        sb, tickers, "earnings_expected", allowed_labels=earnings_block_labels or None
    )
    earnings_force_actual_events = fetch_event_date_map(
        sb, tickers, "earnings_actual", allowed_labels=earnings_force_close_labels or None
    )
    earnings_force_expected_events = fetch_event_date_map(
        sb, tickers, "earnings_expected", allowed_labels=earnings_force_close_labels or None
    )
    earnings_events = merge_event_dates(earnings_actual_events, earnings_expected_events)
    earnings_force_events = merge_event_dates(earnings_force_actual_events, earnings_force_expected_events)
    ex_dividend_events = fetch_event_date_map(sb, tickers, "ex_dividend")
    manual_stop_rules = fetch_manual_stop_rules(sb, tickers)

    per_ticker_data: dict[str, pd.DataFrame] = {}
    active_by_ticker: dict[str, dict[str, Any]] = {}
    market_dates: list[dt.date] = []
    for row in param_sets:
        ticker = row["ticker"]
        params = dict(row.get("params") or {})
        params.setdefault("TICKER", ticker)
        params.setdefault("LOT_SIZE", lot_sizes.get(ticker, 100))
        prices = fetch_prices(ticker, (process_date - dt.timedelta(days=420)).isoformat(), process_date.isoformat())
        ind = compute_indicators(prices)
        current_ts = ind.index[ind.index.date <= process_date]
        if len(current_ts) == 0:
            continue
        last_ts = current_ts[-1]
        market_dates.append(last_ts.date())
        per_ticker_data[ticker] = ind
        active_by_ticker[ticker] = {
            "param_set_id": row["id"],
            "strategy_name": row["strategy_name"],
            "params": params,
            "current_ts": last_ts,
        }

    if not market_dates:
        raise RuntimeError(f"No market data found on or before {process_date.isoformat()}")
    market_date = max(market_dates)

    open_positions = fetch_open_positions_state(sb, simulation_name)
    cash = compute_cash(sb, simulation_name, initial_capital)
    pending_orders = fetch_pending_orders(sb, simulation_name, market_date)
    new_entries_today = 0

    for order in pending_orders:
        ticker = order["ticker"]
        if ticker not in per_ticker_data or ticker not in active_by_ticker:
            update_order_status(sb, int(order["id"]), "CANCELLED", meta={"note": "missing_market_data"})
            continue
        current_ts = active_by_ticker[ticker]["current_ts"]
        if current_ts.date() != market_date:
            continue
        row = per_ticker_data[ticker].loc[current_ts]
        params = active_by_ticker[ticker]["params"]
        slippage = float(params.get("SLIPPAGE", 0.0005))
        fee_pct = float(params.get("FEE_PCT", 0.0))
        stop_slippage = float(params.get("STOP_SLIPPAGE", 0.0015))
        o = float(row["open"])

        if order["order_type"] == "ENTRY":
            if has_manual_stop(manual_stop_rules.get(ticker, []), market_date, "block_entry"):
                update_order_status(sb, int(order["id"]), "CANCELLED", meta={"note": "manual_stop_block"})
                continue
            if has_event_within_days(earnings_events.get(ticker, set()), market_date, earnings_block_days):
                update_order_status(sb, int(order["id"]), "CANCELLED", meta={"note": "earnings_block"})
                continue
            if has_event_within_days(ex_dividend_events.get(ticker, set()), market_date, ex_dividend_block_days):
                update_order_status(sb, int(order["id"]), "CANCELLED", meta={"note": "ex_dividend_block"})
                continue
            if max_open_positions > 0 and len(open_positions) >= max_open_positions:
                update_order_status(sb, int(order["id"]), "CANCELLED", meta={"note": "max_open_positions"})
                continue
            if max_new_positions_per_day > 0 and new_entries_today >= max_new_positions_per_day:
                update_order_status(sb, int(order["id"]), "CANCELLED", meta={"note": "max_new_positions_per_day"})
                continue

            lot_size = int(params.get("LOT_SIZE", lot_sizes.get(ticker, 100)))
            per_trade = float(params.get("PER_TRADE", 500000))
            risk_pct = float(params.get("RISK_PCT", 0.001))
            stop_pct = float(params.get("STOP_PCT", risk_pct))
            qty = calc_qty(o, per_trade, risk_pct, cash, slippage, fee_pct, lot_size)
            if qty <= 0:
                update_order_status(sb, int(order["id"]), "CANCELLED", meta={"note": "qty_zero"})
                continue
            fill = o * (1 + slippage)
            current_total_exposure = sum(p.entry_px * p.qty for p in open_positions.values())
            new_exposure = fill * qty
            if max_total_exposure > 0 and (current_total_exposure + new_exposure) > max_total_exposure:
                update_order_status(sb, int(order["id"]), "CANCELLED", meta={"note": "max_total_exposure"})
                continue
            fee = fill * qty * fee_pct
            cash -= fill * qty + fee

            ind = per_ticker_data[ticker]
            entry_bar_index = int(ind.index.get_loc(current_ts))
            initial_r = fill * stop_pct
            stop_px = fill - initial_r
            take_profit_rr = float(params.get("TAKE_PROFIT_RR", 2.0))
            take_px = fill + take_profit_rr * initial_r
            position_row = {
                "simulation_name": simulation_name,
                "ticker": ticker,
                "strategy_name": order["strategy_name"],
                "param_set_id": order.get("param_set_id"),
                "status": "OPEN",
                "opened_at": current_ts.isoformat(),
                "entry_signal_ts": order.get("signal_ts"),
                "entry_price": float(fill),
                "qty": int(qty),
                "lot_size": int(lot_size),
                "current_stop_price": float(stop_px),
                "take_profit_price": float(take_px),
                "break_even_armed": False,
                "trailing_armed": False,
                "highest_price_since_entry": float(row["high"]),
                "meta": {
                    "params_snapshot": params,
                    "entry_bar_index": entry_bar_index,
                    "just_bought": True,
                },
                "updated_at": current_ts.isoformat(),
            }
            res = sb.table("live_positions").insert(position_row).execute()
            position_id = int(res.data[0]["id"])
            insert_fill(
                sb,
                position_id=position_id,
                simulation_name=simulation_name,
                ticker=ticker,
                strategy_name=order["strategy_name"],
                param_set_id=order.get("param_set_id"),
                executed_at=current_ts.isoformat(),
                side="BUY",
                price=fill,
                qty=qty,
                fee=fee,
                reason=order.get("reason") or "ENTRY",
                signal_ts=order.get("signal_ts"),
            )
            open_positions[ticker] = PositionState(
                position_id=position_id,
                ticker=ticker,
                strategy_name=order["strategy_name"],
                param_set_id=int(order.get("param_set_id") or 0),
                params=params,
                qty=qty,
                lot_size=lot_size,
                entry_px=fill,
                entry_date=current_ts,
                entry_bar_index=entry_bar_index,
                stop_px=stop_px,
                take_px=take_px,
                highest_high_since_entry=float(row["high"]),
                just_bought=True,
            )
            new_entries_today += 1
            update_order_status(sb, int(order["id"]), "EXECUTED", meta={"executed_at": current_ts.isoformat()})
        elif order["order_type"] == "EXIT":
            pos = open_positions.get(ticker)
            if not pos:
                update_order_status(sb, int(order["id"]), "CANCELLED", meta={"note": "position_not_found"})
                continue
            fill = o * (1 - slippage if order.get("reason") != "SL" else 1 - stop_slippage)
            fee = fill * pos.qty * fee_pct
            cash += fill * pos.qty - fee
            realized = (fill - pos.entry_px) * pos.qty - fee
            sb.table("live_positions").update({
                "status": "CLOSED",
                "closed_at": current_ts.isoformat(),
                "exit_price": float(fill),
                "close_reason": order.get("reason"),
                "realized_pnl": float(realized),
                "fees": float(fee),
                "updated_at": current_ts.isoformat(),
                "meta": {"params_snapshot": pos.params, "executed_exit_order": int(order["id"])},
            }).eq("id", pos.position_id).execute()
            insert_fill(
                sb,
                position_id=pos.position_id,
                simulation_name=simulation_name,
                ticker=ticker,
                strategy_name=pos.strategy_name,
                param_set_id=pos.param_set_id,
                executed_at=current_ts.isoformat(),
                side="SELL",
                price=fill,
                qty=pos.qty,
                fee=fee,
                reason=order.get("reason") or "EXIT",
                signal_ts=order.get("signal_ts"),
            )
            del open_positions[ticker]
            update_order_status(sb, int(order["id"]), "EXECUTED", meta={"executed_at": current_ts.isoformat()})

    for ticker, pos in list(open_positions.items()):
        if ticker not in per_ticker_data or ticker not in active_by_ticker:
            continue
        current_ts = active_by_ticker[ticker]["current_ts"]
        if current_ts.date() != market_date:
            continue
        row = per_ticker_data[ticker].loc[current_ts]
        params = pos.params
        slippage = float(params.get("SLIPPAGE", 0.0005))
        fee_pct = float(params.get("FEE_PCT", 0.0))
        stop_slippage = float(params.get("STOP_SLIPPAGE", 0.0015))
        break_even_r = float(params.get("BREAK_EVEN_R", 0.0))
        trailing_start_r = float(params.get("TRAILING_START_R", 0.0))
        trailing_stop_r = float(params.get("TRAILING_STOP_R", 0.0))
        risk_pct = float(params.get("RISK_PCT", 0.001))
        stop_pct = float(params.get("STOP_PCT", risk_pct))
        max_hold_days = int(params.get("MAX_HOLD_DAYS", 5))
        use_intraday = as_bool(params.get("USE_INTRADAY_RESOLUTION", False), default=False)
        intraday_interval = str(params.get("INTRADAY_INTERVAL", "60m"))
        intraday_tie_break = str(params.get("INTRADAY_TIE_BREAK", "SL_FIRST")).upper()

        o = float(row["open"])
        h = float(row["high"])
        l = float(row["low"])
        c = float(row["close"])

        if has_manual_stop(manual_stop_rules.get(ticker, []), market_date, "force_close"):
            fill = c * (1 - slippage)
            fee = fill * pos.qty * fee_pct
            cash += fill * pos.qty - fee
            realized = (fill - pos.entry_px) * pos.qty - fee
            sb.table("live_positions").update({
                "status": "CLOSED",
                "closed_at": current_ts.isoformat(),
                "exit_price": float(fill),
                "close_reason": "MANUAL_STOP",
                "realized_pnl": float(realized),
                "fees": float(fee),
                "updated_at": current_ts.isoformat(),
                "meta": {"params_snapshot": pos.params},
            }).eq("id", pos.position_id).execute()
            insert_fill(sb, position_id=pos.position_id, simulation_name=simulation_name, ticker=ticker, strategy_name=pos.strategy_name, param_set_id=pos.param_set_id, executed_at=current_ts.isoformat(), side="SELL", price=fill, qty=pos.qty, fee=fee, reason="MANUAL_STOP", signal_ts=current_ts.isoformat())
            del open_positions[ticker]
            continue

        if is_force_close_day(earnings_force_events.get(ticker, set()), market_date, earnings_force_close_days):
            fill = c * (1 - slippage)
            fee = fill * pos.qty * fee_pct
            cash += fill * pos.qty - fee
            realized = (fill - pos.entry_px) * pos.qty - fee
            sb.table("live_positions").update({
                "status": "CLOSED",
                "closed_at": current_ts.isoformat(),
                "exit_price": float(fill),
                "close_reason": "EARNINGS",
                "realized_pnl": float(realized),
                "fees": float(fee),
                "updated_at": current_ts.isoformat(),
                "meta": {"params_snapshot": pos.params},
            }).eq("id", pos.position_id).execute()
            insert_fill(sb, position_id=pos.position_id, simulation_name=simulation_name, ticker=ticker, strategy_name=pos.strategy_name, param_set_id=pos.param_set_id, executed_at=current_ts.isoformat(), side="SELL", price=fill, qty=pos.qty, fee=fee, reason="EARNINGS", signal_ts=current_ts.isoformat())
            del open_positions[ticker]
            continue

        if is_force_close_day(ex_dividend_events.get(ticker, set()), market_date, ex_dividend_force_close_days):
            fill = c * (1 - slippage)
            fee = fill * pos.qty * fee_pct
            cash += fill * pos.qty - fee
            realized = (fill - pos.entry_px) * pos.qty - fee
            sb.table("live_positions").update({
                "status": "CLOSED",
                "closed_at": current_ts.isoformat(),
                "exit_price": float(fill),
                "close_reason": "EX_DIVIDEND",
                "realized_pnl": float(realized),
                "fees": float(fee),
                "updated_at": current_ts.isoformat(),
                "meta": {"params_snapshot": pos.params},
            }).eq("id", pos.position_id).execute()
            insert_fill(sb, position_id=pos.position_id, simulation_name=simulation_name, ticker=ticker, strategy_name=pos.strategy_name, param_set_id=pos.param_set_id, executed_at=current_ts.isoformat(), side="SELL", price=fill, qty=pos.qty, fee=fee, reason="EX_DIVIDEND", signal_ts=current_ts.isoformat())
            del open_positions[ticker]
            continue

        if pos.just_bought:
            pos.just_bought = False
        else:
            ind = per_ticker_data[ticker]
            entry_idx = pos.entry_bar_index
            if entry_idx < 0:
                entry_idx = int(ind.index.get_loc(pd.Timestamp(pos.entry_date)))
            current_hold_days = int(ind.index.get_loc(current_ts)) - entry_idx
            expiry_day = current_hold_days >= max_hold_days
            ambiguous = (not expiry_day) and (l <= pos.stop_px) and (h >= pos.take_px)
            ambiguous_reason = None
            if ambiguous and use_intraday:
                ambiguous_reason = resolve_intraday_ambiguous_exit(
                    ticker, current_ts, pos.stop_px, pos.take_px, interval=intraday_interval, tie_break=intraday_tie_break
                )
            if ambiguous_reason == "SL" or l <= pos.stop_px:
                fill = max(o, pos.stop_px) * (1 - stop_slippage)
                fee = fill * pos.qty * fee_pct
                cash += fill * pos.qty - fee
                realized = (fill - pos.entry_px) * pos.qty - fee
                sb.table("live_positions").update({
                    "status": "CLOSED",
                    "closed_at": current_ts.isoformat(),
                    "exit_price": float(fill),
                    "close_reason": "SL",
                    "realized_pnl": float(realized),
                    "fees": float(fee),
                    "updated_at": current_ts.isoformat(),
                    "meta": {"params_snapshot": pos.params},
                }).eq("id", pos.position_id).execute()
                insert_fill(sb, position_id=pos.position_id, simulation_name=simulation_name, ticker=ticker, strategy_name=pos.strategy_name, param_set_id=pos.param_set_id, executed_at=current_ts.isoformat(), side="SELL", price=fill, qty=pos.qty, fee=fee, reason="SL", signal_ts=current_ts.isoformat())
                del open_positions[ticker]
                continue
            if expiry_day:
                upsert_order(sb, simulation_name=simulation_name, ticker=ticker, strategy_name=pos.strategy_name, param_set_id=pos.param_set_id, order_type="EXIT", scheduled_for=next_trading_day(market_date), reason="TIME", signal_ts=current_ts.isoformat(), position_id=pos.position_id, meta={"source": "expiry"})
            elif ambiguous_reason == "TP" or h >= pos.take_px:
                fill = max(pos.take_px, o) * (1 - slippage)
                fee = fill * pos.qty * fee_pct
                cash += fill * pos.qty - fee
                realized = (fill - pos.entry_px) * pos.qty - fee
                sb.table("live_positions").update({
                    "status": "CLOSED",
                    "closed_at": current_ts.isoformat(),
                    "exit_price": float(fill),
                    "close_reason": "TP",
                    "realized_pnl": float(realized),
                    "fees": float(fee),
                    "updated_at": current_ts.isoformat(),
                    "meta": {"params_snapshot": pos.params},
                }).eq("id", pos.position_id).execute()
                insert_fill(sb, position_id=pos.position_id, simulation_name=simulation_name, ticker=ticker, strategy_name=pos.strategy_name, param_set_id=pos.param_set_id, executed_at=current_ts.isoformat(), side="SELL", price=fill, qty=pos.qty, fee=fee, reason="TP", signal_ts=current_ts.isoformat())
                del open_positions[ticker]
                continue
            else:
                still_long = long_signal_row(
                    row,
                    MACD_ATR_K=float(params.get("MACD_ATR_K", 0.13)),
                    RSI_MIN=float(params.get("RSI_MIN", 30.0)),
                    RSI_MAX=float(params.get("RSI_MAX", 80.0)),
                    VOL_SPIKE_M=float(params.get("VOL_SPIKE_M", 1.0)),
                )
                if not still_long and as_bool(params.get("EXIT_ON_REVERSE", True), default=True):
                    upsert_order(sb, simulation_name=simulation_name, ticker=ticker, strategy_name=pos.strategy_name, param_set_id=pos.param_set_id, order_type="EXIT", scheduled_for=next_trading_day(market_date), reason="REV", signal_ts=current_ts.isoformat(), position_id=pos.position_id, meta={"source": "reverse"})

        initial_r = pos.entry_px * stop_pct
        pos.highest_high_since_entry = max(pos.highest_high_since_entry, h) if not math.isnan(pos.highest_high_since_entry) else h
        if (not pos.break_even_armed) and break_even_r > 0 and h >= (pos.entry_px + break_even_r * initial_r):
            pos.stop_px = max(pos.stop_px, pos.entry_px)
            pos.break_even_armed = True
        if trailing_stop_r > 0 and pos.highest_high_since_entry >= (pos.entry_px + trailing_start_r * initial_r):
            trailing_stop_px = pos.highest_high_since_entry - (trailing_stop_r * initial_r)
            pos.stop_px = max(pos.stop_px, trailing_stop_px)
            pos.trailing_armed = True

        sb.table("live_positions").update({
            "current_stop_price": float(pos.stop_px),
            "take_profit_price": float(pos.take_px),
            "break_even_armed": bool(pos.break_even_armed),
            "trailing_armed": bool(pos.trailing_armed),
            "highest_price_since_entry": float(pos.highest_high_since_entry),
            "updated_at": current_ts.isoformat(),
            "meta": {
                "params_snapshot": pos.params,
                "entry_bar_index": pos.entry_bar_index,
                "just_bought": pos.just_bought,
            },
        }).eq("id", pos.position_id).execute()

    for ticker, cfg in active_by_ticker.items():
        if ticker in open_positions:
            continue
        current_ts = cfg["current_ts"]
        if current_ts.date() != market_date:
            continue
        row = per_ticker_data[ticker].loc[current_ts]
        params = cfg["params"]
        buy_for = next_trading_day(market_date)
        if not long_signal_row(row, MACD_ATR_K=float(params.get("MACD_ATR_K", 0.13)), RSI_MIN=float(params.get("RSI_MIN", 30.0)), RSI_MAX=float(params.get("RSI_MAX", 80.0)), VOL_SPIKE_M=float(params.get("VOL_SPIKE_M", 1.0))):
            continue
        if has_manual_stop(manual_stop_rules.get(ticker, []), buy_for, "block_entry"):
            continue
        if has_event_within_days(earnings_events.get(ticker, set()), buy_for, earnings_block_days):
            continue
        if has_event_within_days(ex_dividend_events.get(ticker, set()), buy_for, ex_dividend_block_days):
            continue
        upsert_order(sb, simulation_name=simulation_name, ticker=ticker, strategy_name=cfg["strategy_name"], param_set_id=cfg["param_set_id"], order_type="ENTRY", scheduled_for=buy_for, reason="ENTRY", signal_ts=current_ts.isoformat(), meta={"source": "daily_signal"})

    report = build_monitor_report(
        as_of=process_date,
        strategy_name=strategy_name,
        position_simulation_name=simulation_name,
        report_dir=report_dir,
        earnings_block_days=earnings_block_days,
        earnings_block_labels=earnings_block_labels,
        ex_dividend_block_days=ex_dividend_block_days,
        upcoming_event_days=int(os.getenv("UPCOMING_EVENT_DAYS", "7")),
    )
    save_monitor_report(report)
    summary = build_line_summary(report)
    user_id = os.getenv("LINE_USER_ID", "").strip()
    if user_id:
        send_line_message_push(user_id, summary)
    else:
        send_line_message_broadcast(summary)

    print(f"[TRADER] simulation_name={simulation_name} strategy_name={strategy_name} process_date={process_date.isoformat()} market_date={market_date.isoformat()}")
    print(f"[TRADER] open_positions={len(report['open_positions'])} entry_candidates={len(report['candidates'])} blocked={len(report['blocked'])}")


if __name__ == "__main__":
    main()
