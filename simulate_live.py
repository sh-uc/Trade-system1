import os
import math
import datetime as dt
from dataclasses import dataclass
from typing import Any, Dict, Optional

import pandas as pd

from db_utils import get_supabase, ensure_ticker_name
from bt_core import (
    fetch_prices,
    compute_indicators,
    long_signal_row,
    next_trading_day,
    resolve_intraday_ambiguous_exit,
)


@dataclass
class PositionState:
    position_id: int
    ticker: str
    strategy_name: str
    param_set_id: int
    params: Dict[str, Any]
    qty: int
    lot_size: int
    entry_px: float
    entry_date: Any
    entry_bar_index: int
    stop_px: float
    take_px: float
    break_even_armed: bool = False
    trailing_armed: bool = False
    highest_high_since_entry: float = math.nan
    pending_sell_for: Optional[dt.date] = None
    pending_sell_reason: Optional[str] = None
    pending_sell_signal_ts: Optional[str] = None
    just_bought: bool = True


def as_bool(v: Any, default: bool = False) -> bool:
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "y", "on"):
        return True
    if s in ("0", "false", "no", "n", "off", ""):
        return False
    return default


def calc_qty(open_px: float, per_trade_cap: float, risk_pct: float, cash: float, slippage: float, fee_pct: float, lot_size: int) -> int:
    if open_px <= 0 or lot_size <= 0:
        return 0
    max_lots_by_cap = int(per_trade_cap // (open_px * lot_size))
    if max_lots_by_cap <= 0:
        return 0
    risk_jpy = per_trade_cap * risk_pct
    stop_px = open_px * (1 - risk_pct)
    risk_per_share = open_px - stop_px
    if risk_per_share <= 0:
        return 0
    max_shares_by_risk = int(risk_jpy // risk_per_share)
    max_lots_by_risk = max_shares_by_risk // lot_size
    max_shares_by_cash = int(cash // (open_px * (1 + slippage + fee_pct)))
    max_lots_by_cash = max_shares_by_cash // lot_size
    lots = min(max_lots_by_cap, max_lots_by_risk, max_lots_by_cash)
    return max(0, lots) * lot_size


def fetch_active_param_sets(sb, strategy_name: str, tickers: list[str] | None = None):
    q = sb.table("strategy_param_sets").select("id,strategy_name,ticker,params").eq("is_active", True)
    if strategy_name:
        q = q.eq("strategy_name", strategy_name)
    rows = q.execute().data or []
    if tickers:
        tickers_set = {t.strip() for t in tickers}
        rows = [r for r in rows if r.get("ticker") in tickers_set]
    return rows


def fetch_lot_size_map(sb, tickers: list[str]) -> dict[str, int]:
    if not tickers:
        return {}
    try:
        rows = sb.table("tickers").select("code,lot_size").in_("code", tickers).execute().data or []
    except Exception:
        return {t: 100 for t in tickers}
    out = {t: 100 for t in tickers}
    for row in rows:
        try:
            out[row["code"]] = int(row.get("lot_size") or 100)
        except Exception:
            out[row["code"]] = 100
    return out


def parse_label_set(raw: str) -> set[str]:
    return {s.strip().upper() for s in (raw or "").split(",") if s.strip()}


def fetch_event_date_map(
    sb,
    tickers: list[str],
    event_type: str,
    allowed_labels: set[str] | None = None,
) -> dict[str, set[dt.date]]:
    out = {t: set() for t in tickers}
    if not tickers:
        return out
    try:
        rows = (
            sb.table("ticker_events")
            .select("ticker,event_date,meta")
            .eq("event_type", event_type)
            .eq("is_active", True)
            .in_("ticker", tickers)
            .execute()
            .data
            or []
        )
    except Exception:
        return out
    for row in rows:
        ticker = row.get("ticker")
        if ticker not in out:
            continue
        meta = row.get("meta") or {}
        fiscal_label = str(meta.get("fiscal_label") or "").strip().upper()
        if allowed_labels and fiscal_label not in allowed_labels:
            continue
        try:
            out[ticker].add(pd.to_datetime(row["event_date"]).date())
        except Exception:
            pass
    return out


def fetch_manual_stop_rules(sb, tickers: list[str]) -> dict[str, list[dict[str, Any]]]:
    out = {t: [] for t in tickers}
    if not tickers:
        return out
    try:
        rows = (
            sb.table("ticker_events")
            .select("ticker,event_date,meta")
            .eq("event_type", "manual_stop")
            .eq("is_active", True)
            .in_("ticker", tickers)
            .execute()
            .data
            or []
        )
    except Exception:
        return out
    for row in rows:
        ticker = row.get("ticker")
        if ticker not in out:
            continue
        meta = row.get("meta") or {}
        try:
            start_date = pd.to_datetime(row["event_date"]).date()
        except Exception:
            continue
        try:
            end_date = pd.to_datetime(meta.get("window_end") or row["event_date"]).date()
        except Exception:
            end_date = start_date
        action = str(meta.get("action") or "block_entry").strip().lower()
        out[ticker].append(
            {
                "start_date": start_date,
                "end_date": end_date,
                "action": action,
            }
        )
    return out


def has_manual_stop(rules: list[dict[str, Any]], current_date: dt.date, action: str) -> bool:
    if not rules:
        return False
    action = action.strip().lower()
    for rule in rules:
        start_date = rule.get("start_date")
        end_date = rule.get("end_date")
        rule_action = str(rule.get("action") or "").strip().lower()
        if start_date is None or end_date is None:
            continue
        if start_date <= current_date <= end_date:
            if rule_action in (action, "both"):
                return True
    return False


def merge_event_dates(*maps: dict[str, set[dt.date]]) -> dict[str, set[dt.date]]:
    merged: dict[str, set[dt.date]] = {}
    tickers = set()
    for m in maps:
        tickers.update(m.keys())
    for t in tickers:
        merged[t] = set()
        for m in maps:
            merged[t].update(m.get(t, set()))
    return merged


def has_event_within_days(event_dates: set[dt.date], current_date: dt.date, window_days: int) -> bool:
    if window_days <= 0 or not event_dates:
        return False
    end_date = current_date + dt.timedelta(days=window_days)
    return any(current_date <= d <= end_date for d in event_dates)


def is_force_close_day(event_dates: set[dt.date], current_date: dt.date, window_days: int) -> bool:
    if window_days <= 0 or not event_dates:
        return False
    return any(0 <= (d - current_date).days <= window_days for d in event_dates)


def insert_fill(sb, *, position_id: int, simulation_name: str, ticker: str, strategy_name: str, param_set_id: int, executed_at: str, side: str, price: float, qty: int, fee: float, reason: str, signal_ts: Optional[str]):
    sb.table("live_trade_fills").insert({
        "position_id": position_id,
        "simulation_name": simulation_name,
        "ticker": ticker,
        "strategy_name": strategy_name,
        "param_set_id": param_set_id,
        "executed_at": executed_at,
        "side": side,
        "price": float(price),
        "qty": int(qty),
        "notional": float(price) * int(qty),
        "fee": float(fee),
        "reason": reason,
        "signal_ts": signal_ts,
    }).execute()


def main():
    simulation_name = os.getenv("SIMULATION_NAME", "paper_main")
    strategy_name = os.getenv("STRATEGY_NAME", "swing_v1")
    start = os.getenv("SIM_START", "2026-01-01")
    end = os.getenv("SIM_END")
    reset = as_bool(os.getenv("SIM_RESET", "0"), default=False)
    tickers_env = os.getenv("SIM_TICKERS", "")
    requested_tickers = [t.strip() for t in tickers_env.split(",") if t.strip()]
    max_open_positions = int(os.getenv("MAX_OPEN_POSITIONS", "0"))
    max_new_positions_per_day = int(os.getenv("MAX_NEW_POSITIONS_PER_DAY", "0"))
    max_total_exposure = float(os.getenv("MAX_TOTAL_EXPOSURE", "0"))

    sb = get_supabase()
    param_sets = fetch_active_param_sets(sb, strategy_name, requested_tickers or None)
    if not param_sets:
        raise RuntimeError("No active strategy_param_sets found")

    tickers = sorted({row["ticker"] for row in param_sets})
    lot_sizes = fetch_lot_size_map(sb, tickers)
    earnings_block_labels = parse_label_set(os.getenv("EARNINGS_BLOCK_LABELS", ""))
    earnings_force_close_labels = parse_label_set(os.getenv("EARNINGS_FORCE_CLOSE_LABELS", ""))

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
    earnings_force_events = merge_event_dates(
        earnings_force_actual_events, earnings_force_expected_events
    )
    ex_dividend_events = fetch_event_date_map(sb, tickers, "ex_dividend")
    manual_stop_rules = fetch_manual_stop_rules(sb, tickers)

    if reset:
        sb.table("live_trade_fills").delete().eq("simulation_name", simulation_name).execute()
        sb.table("live_positions").delete().eq("simulation_name", simulation_name).execute()

    per_ticker_data = {}
    active_by_ticker = {}
    for row in param_sets:
        ticker = row["ticker"]
        params = dict(row.get("params") or {})
        params.setdefault("TICKER", ticker)
        params.setdefault("LOT_SIZE", lot_sizes.get(ticker, 100))
        ensure_ticker_name(sb, ticker)
        prices = fetch_prices(ticker, start, end)
        ind = compute_indicators(prices)
        per_ticker_data[ticker] = ind
        active_by_ticker[ticker] = {"param_set_id": row["id"], "strategy_name": row["strategy_name"], "params": params}

    all_dates = sorted({idx for df in per_ticker_data.values() for idx in df.index})
    cash = float(os.getenv("SIM_CAPITAL", "3000000"))
    pending_buys: dict[str, dict[str, Any]] = {}
    open_positions: dict[str, PositionState] = {}

    for date in all_dates:
        new_entries_today = 0
        for ticker in tickers:
            ind = per_ticker_data.get(ticker)
            if ind is None or date not in ind.index:
                continue
            row = ind.loc[date]
            cfg = active_by_ticker[ticker]
            params = cfg["params"]
            lot_size = int(params.get("LOT_SIZE", lot_sizes.get(ticker, 100)))
            slippage = float(params.get("SLIPPAGE", 0.0005))
            fee_pct = float(params.get("FEE_PCT", 0.0))
            stop_slippage = float(params.get("STOP_SLIPPAGE", 0.0015))
            break_even_r = float(params.get("BREAK_EVEN_R", 0.0))
            trailing_start_r = float(params.get("TRAILING_START_R", 0.0))
            trailing_stop_r = float(params.get("TRAILING_STOP_R", 0.0))
            risk_pct = float(params.get("RISK_PCT", 0.001))
            stop_pct = float(params.get("STOP_PCT", risk_pct))
            take_profit_rr = float(params.get("TAKE_PROFIT_RR", 2.0))
            max_hold_days = int(params.get("MAX_HOLD_DAYS", 5))
            earnings_block_days = int(params.get("EARNINGS_BLOCK_DAYS", os.getenv("EARNINGS_BLOCK_DAYS", "0")))
            earnings_force_close_days = int(params.get("EARNINGS_FORCE_CLOSE_DAYS", os.getenv("EARNINGS_FORCE_CLOSE_DAYS", "0")))
            ex_dividend_block_days = int(
                params.get("EX_DIVIDEND_BLOCK_DAYS", os.getenv("EX_DIVIDEND_BLOCK_DAYS", "0"))
            )
            ex_dividend_force_close_days = int(
                params.get("EX_DIVIDEND_FORCE_CLOSE_DAYS", os.getenv("EX_DIVIDEND_FORCE_CLOSE_DAYS", "0"))
            )
            use_intraday = as_bool(params.get("USE_INTRADAY_RESOLUTION", False), default=False)
            intraday_interval = str(params.get("INTRADAY_INTERVAL", "60m"))
            intraday_tie_break = str(params.get("INTRADAY_TIE_BREAK", "SL_FIRST")).upper()

            o = float(row["open"])
            h = float(row["high"])
            l = float(row["low"])

            pos = open_positions.get(ticker)
            if pos and pos.pending_sell_for and date.date() == pos.pending_sell_for:
                fill = o * (1 - slippage)
                fee = fill * pos.qty * fee_pct
                cash += fill * pos.qty - fee
                realized = (fill - pos.entry_px) * pos.qty - fee
                sb.table("live_positions").update({
                    "status": "CLOSED",
                    "closed_at": date.isoformat(),
                    "exit_price": float(fill),
                    "close_reason": pos.pending_sell_reason,
                    "realized_pnl": float(realized),
                    "fees": float(fee),
                    "updated_at": date.isoformat(),
                }).eq("id", pos.position_id).execute()
                insert_fill(sb, position_id=pos.position_id, simulation_name=simulation_name, ticker=ticker, strategy_name=pos.strategy_name, param_set_id=pos.param_set_id, executed_at=date.isoformat(), side="SELL", price=fill, qty=pos.qty, fee=fee, reason=pos.pending_sell_reason or "REV", signal_ts=pos.pending_sell_signal_ts)
                del open_positions[ticker]
                pos = None

            pending_buy = pending_buys.get(ticker)
            if pos is None and pending_buy and date.date() == pending_buy["buy_for"]:
                if has_manual_stop(manual_stop_rules.get(ticker, []), date.date(), "block_entry"):
                    pending_buys.pop(ticker, None)
                    continue
                if max_open_positions > 0 and len(open_positions) >= max_open_positions:
                    pending_buys.pop(ticker, None)
                    continue
                if max_new_positions_per_day > 0 and new_entries_today >= max_new_positions_per_day:
                    pending_buys.pop(ticker, None)
                    continue
                if has_event_within_days(earnings_events.get(ticker, set()), date.date(), earnings_block_days):
                    pending_buys.pop(ticker, None)
                    continue
                if has_event_within_days(
                    ex_dividend_events.get(ticker, set()), date.date(), ex_dividend_block_days
                ):
                    pending_buys.pop(ticker, None)
                    continue
                per_trade = float(params.get("PER_TRADE", 500000))
                qty = calc_qty(o, per_trade, risk_pct, cash, slippage, fee_pct, lot_size)
                if qty > 0:
                    fill = o * (1 + slippage)
                    current_total_exposure = sum(p.entry_px * p.qty for p in open_positions.values())
                    new_exposure = fill * qty
                    if max_total_exposure > 0 and (current_total_exposure + new_exposure) > max_total_exposure:
                        pending_buys.pop(ticker, None)
                        continue
                    fee = fill * qty * fee_pct
                    cost = fill * qty + fee
                    cash -= cost
                    entry_bar_index = ind.index.get_loc(date)
                    initial_r = fill * stop_pct
                    stop_px = fill - initial_r
                    take_px = fill + take_profit_rr * initial_r
                    position_row = {
                        "simulation_name": simulation_name,
                        "ticker": ticker,
                        "strategy_name": cfg["strategy_name"],
                        "param_set_id": cfg["param_set_id"],
                        "status": "OPEN",
                        "opened_at": date.isoformat(),
                        "entry_signal_ts": pending_buy["signal_ts"],
                        "entry_price": float(fill),
                        "qty": int(qty),
                        "lot_size": int(lot_size),
                        "current_stop_price": float(stop_px),
                        "take_profit_price": float(take_px),
                        "break_even_armed": False,
                        "trailing_armed": False,
                        "highest_price_since_entry": float(h),
                        "meta": {"params_snapshot": params},
                        "updated_at": date.isoformat(),
                    }
                    res = sb.table("live_positions").insert(position_row).execute()
                    position_id = int(res.data[0]["id"])
                    insert_fill(sb, position_id=position_id, simulation_name=simulation_name, ticker=ticker, strategy_name=cfg["strategy_name"], param_set_id=cfg["param_set_id"], executed_at=date.isoformat(), side="BUY", price=fill, qty=qty, fee=fee, reason="ENTRY", signal_ts=pending_buy["signal_ts"])
                    open_positions[ticker] = PositionState(
                        position_id=position_id,
                        ticker=ticker,
                        strategy_name=cfg["strategy_name"],
                        param_set_id=cfg["param_set_id"],
                        params=params,
                        qty=qty,
                        lot_size=lot_size,
                        entry_px=fill,
                        entry_date=date,
                        entry_bar_index=entry_bar_index,
                        stop_px=stop_px,
                        take_px=take_px,
                        highest_high_since_entry=h,
                        just_bought=True,
                    )
                    pos = open_positions[ticker]
                    new_entries_today += 1
                pending_buys.pop(ticker, None)

            if pos is None:
                buy_for = next_trading_day(date.date())
                if long_signal_row(row, MACD_ATR_K=float(params.get("MACD_ATR_K", 0.13)), RSI_MIN=float(params.get("RSI_MIN", 30.0)), RSI_MAX=float(params.get("RSI_MAX", 80.0)), VOL_SPIKE_M=float(params.get("VOL_SPIKE_M", 1.0))):
                    if (
                        not has_event_within_days(earnings_events.get(ticker, set()), buy_for, earnings_block_days)
                        and not has_event_within_days(
                            ex_dividend_events.get(ticker, set()), buy_for, ex_dividend_block_days
                        )
                        and not has_manual_stop(manual_stop_rules.get(ticker, []), buy_for, "block_entry")
                    ):
                        pending_buys[ticker] = {"buy_for": buy_for, "signal_ts": date.isoformat()}
                continue

            if has_manual_stop(manual_stop_rules.get(ticker, []), date.date(), "force_close"):
                fill = float(row["close"]) * (1 - slippage)
                fee = fill * pos.qty * fee_pct
                cash += fill * pos.qty - fee
                realized = (fill - pos.entry_px) * pos.qty - fee
                sb.table("live_positions").update({
                    "status": "CLOSED",
                    "closed_at": date.isoformat(),
                    "exit_price": float(fill),
                    "close_reason": "MANUAL_STOP",
                    "realized_pnl": float(realized),
                    "fees": float(fee),
                    "updated_at": date.isoformat(),
                }).eq("id", pos.position_id).execute()
                insert_fill(sb, position_id=pos.position_id, simulation_name=simulation_name, ticker=ticker, strategy_name=pos.strategy_name, param_set_id=pos.param_set_id, executed_at=date.isoformat(), side="SELL", price=fill, qty=pos.qty, fee=fee, reason="MANUAL_STOP", signal_ts=date.isoformat())
                del open_positions[ticker]
                continue

            if is_force_close_day(earnings_force_events.get(ticker, set()), date.date(), earnings_force_close_days):
                fill = float(row["close"]) * (1 - slippage)
                fee = fill * pos.qty * fee_pct
                cash += fill * pos.qty - fee
                realized = (fill - pos.entry_px) * pos.qty - fee
                sb.table("live_positions").update({
                    "status": "CLOSED",
                    "closed_at": date.isoformat(),
                    "exit_price": float(fill),
                    "close_reason": "EARNINGS",
                    "realized_pnl": float(realized),
                    "fees": float(fee),
                    "updated_at": date.isoformat(),
                }).eq("id", pos.position_id).execute()
                insert_fill(sb, position_id=pos.position_id, simulation_name=simulation_name, ticker=ticker, strategy_name=pos.strategy_name, param_set_id=pos.param_set_id, executed_at=date.isoformat(), side="SELL", price=fill, qty=pos.qty, fee=fee, reason="EARNINGS", signal_ts=date.isoformat())
                del open_positions[ticker]
                continue

            if is_force_close_day(
                ex_dividend_events.get(ticker, set()), date.date(), ex_dividend_force_close_days
            ):
                fill = float(row["close"]) * (1 - slippage)
                fee = fill * pos.qty * fee_pct
                cash += fill * pos.qty - fee
                realized = (fill - pos.entry_px) * pos.qty - fee
                sb.table("live_positions").update({
                    "status": "CLOSED",
                    "closed_at": date.isoformat(),
                    "exit_price": float(fill),
                    "close_reason": "EX_DIVIDEND",
                    "realized_pnl": float(realized),
                    "fees": float(fee),
                    "updated_at": date.isoformat(),
                }).eq("id", pos.position_id).execute()
                insert_fill(
                    sb,
                    position_id=pos.position_id,
                    simulation_name=simulation_name,
                    ticker=ticker,
                    strategy_name=pos.strategy_name,
                    param_set_id=pos.param_set_id,
                    executed_at=date.isoformat(),
                    side="SELL",
                    price=fill,
                    qty=pos.qty,
                    fee=fee,
                    reason="EX_DIVIDEND",
                    signal_ts=date.isoformat(),
                )
                del open_positions[ticker]
                continue

            if pos.just_bought:
                pos.just_bought = False
            else:
                current_hold_days = ind.index.get_loc(date) - pos.entry_bar_index
                expiry_day = current_hold_days >= max_hold_days
                ambiguous = (not expiry_day) and (l <= pos.stop_px) and (h >= pos.take_px)
                ambiguous_reason = None
                if ambiguous and use_intraday:
                    ambiguous_reason = resolve_intraday_ambiguous_exit(ticker, date, pos.stop_px, pos.take_px, interval=intraday_interval, tie_break=intraday_tie_break)
                sold = False
                if ambiguous_reason == "SL" or l <= pos.stop_px:
                    fill = max(o, pos.stop_px) * (1 - stop_slippage)
                    fee = fill * pos.qty * fee_pct
                    cash += fill * pos.qty - fee
                    realized = (fill - pos.entry_px) * pos.qty - fee
                    sb.table("live_positions").update({
                        "status": "CLOSED",
                        "closed_at": date.isoformat(),
                        "exit_price": float(fill),
                        "close_reason": "SL",
                        "realized_pnl": float(realized),
                        "fees": float(fee),
                        "updated_at": date.isoformat(),
                    }).eq("id", pos.position_id).execute()
                    insert_fill(sb, position_id=pos.position_id, simulation_name=simulation_name, ticker=ticker, strategy_name=pos.strategy_name, param_set_id=pos.param_set_id, executed_at=date.isoformat(), side="SELL", price=fill, qty=pos.qty, fee=fee, reason="SL", signal_ts=date.isoformat())
                    del open_positions[ticker]
                    sold = True
                elif expiry_day:
                    pos.pending_sell_for = next_trading_day(date.date())
                    pos.pending_sell_reason = "TIME"
                    pos.pending_sell_signal_ts = date.isoformat()
                    sold = True
                elif ambiguous_reason == "TP" or h >= pos.take_px:
                    fill = max(pos.take_px, o) * (1 - slippage)
                    fee = fill * pos.qty * fee_pct
                    cash += fill * pos.qty - fee
                    realized = (fill - pos.entry_px) * pos.qty - fee
                    sb.table("live_positions").update({
                        "status": "CLOSED",
                        "closed_at": date.isoformat(),
                        "exit_price": float(fill),
                        "close_reason": "TP",
                        "realized_pnl": float(realized),
                        "fees": float(fee),
                        "updated_at": date.isoformat(),
                    }).eq("id", pos.position_id).execute()
                    insert_fill(sb, position_id=pos.position_id, simulation_name=simulation_name, ticker=ticker, strategy_name=pos.strategy_name, param_set_id=pos.param_set_id, executed_at=date.isoformat(), side="SELL", price=fill, qty=pos.qty, fee=fee, reason="TP", signal_ts=date.isoformat())
                    del open_positions[ticker]
                    sold = True
                else:
                    still_long = long_signal_row(row, MACD_ATR_K=float(params.get("MACD_ATR_K", 0.13)), RSI_MIN=float(params.get("RSI_MIN", 30.0)), RSI_MAX=float(params.get("RSI_MAX", 80.0)), VOL_SPIKE_M=float(params.get("VOL_SPIKE_M", 1.0)))
                    if not still_long and as_bool(params.get("EXIT_ON_REVERSE", True), default=True):
                        pos.pending_sell_for = next_trading_day(date.date())
                        pos.pending_sell_reason = "REV"
                        pos.pending_sell_signal_ts = date.isoformat()
                        sold = True
                if sold:
                    continue

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
                "updated_at": date.isoformat(),
            }).eq("id", pos.position_id).execute()

    open_count = len(open_positions)
    print(f"[SIM] simulation_name={simulation_name} tickers={len(tickers)} open_positions={open_count} cash={cash:,.0f}")


if __name__ == "__main__":
    main()
