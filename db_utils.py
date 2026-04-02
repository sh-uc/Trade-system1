from supabase import create_client, Client
import os
import csv
import re
import unicodedata
from datetime import timedelta
from functools import lru_cache
from pathlib import Path
import pandas as pd
import yfinance as yf


def get_supabase() -> Client:
    url = os.environ["SUPABASE_URL"]
    key = os.environ["SUPABASE_KEY"]
    return create_client(url, key)


def normalize_display_company_name(name: str | None) -> str | None:
    if name is None:
        return None
    text = unicodedata.normalize("NFKC", str(name))
    text = re.sub(r"\bHLDGS?\b", "HD", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


def _normalize_company_name(name: str | None, code: str) -> str | None:
    if name is None:
        return None
    cleaned = normalize_display_company_name(name)
    if not cleaned:
        return None
    if cleaned.upper() == str(code).strip().upper():
        return None
    return cleaned


def fetch_company_name_from_yfinance(code: str) -> str | None:
    try:
        tkr = yf.Ticker(code)
        fast_info = getattr(tkr, "fast_info", None)
        name = getattr(fast_info, "shortName", None)
        name = _normalize_company_name(name, code)
        if name:
            return name
    except Exception:
        pass

    try:
        name = yf.Ticker(code).info.get("shortName")
        return _normalize_company_name(name, code)
    except Exception:
        return None


def _bare_code(code: str) -> str:
    text = str(code).strip()
    return text[:-2] if text.upper().endswith(".T") else text


def fetch_company_name_from_jpx_master(sb: Client | None, code: str) -> str | None:
    if sb is None:
        return None
    bare = _bare_code(code)
    try:
        r = (
            sb.table("jpx_ticker_master")
            .select("name")
            .or_(f"ticker_code.eq.{code},code.eq.{bare}")
            .limit(1)
            .execute()
        )
        if r.data:
            return _normalize_company_name(r.data[0].get("name"), code)
    except Exception:
        return None
    return None


@lru_cache(maxsize=8)
def _load_screening_csv_name_map(paths_key: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw in [p.strip() for p in paths_key.split("|") if p.strip()]:
        path = Path(raw)
        if not path.exists():
            continue
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    code = str(row.get("コード") or row.get("code") or row.get("ticker") or "").strip()
                    name = str(row.get("銘柄名") or row.get("name") or "").strip()
                    if not code or not name:
                        continue
                    if not code.endswith(".T") and code and code[-1].isalnum():
                        ticker_code = f"{code}.T"
                    else:
                        ticker_code = code
                    out[ticker_code] = name
        except Exception:
            continue
    return out


def fetch_company_name_from_screening_csv(code: str) -> str | None:
    raw = os.getenv("TICKER_NAME_SOURCE_CSVS", "").strip()
    if not raw:
        return None
    paths_key = "|".join([p.strip() for p in raw.split(",") if p.strip()])
    if not paths_key:
        return None
    name = _load_screening_csv_name_map(paths_key).get(code)
    return _normalize_company_name(name, code)


def ensure_ticker_name(sb: Client | None, code: str, current_name: str | None = None) -> str:
    resolved_name = _normalize_company_name(current_name, code)
    db_name = None

    if sb is not None:
        try:
            r = sb.table("tickers").select("name").eq("code", code).limit(1).execute()
            if r.data:
                db_name = r.data[0].get("name")
                db_name = _normalize_company_name(db_name, code)
        except Exception:
            db_name = None

    master_name = fetch_company_name_from_jpx_master(sb, code)
    if master_name:
        if sb is not None:
            try:
                if db_name != master_name:
                    sb.table("tickers").upsert({"code": code, "name": master_name}).execute()
            except Exception:
                pass
        return master_name

    csv_name = fetch_company_name_from_screening_csv(code)
    if csv_name:
        if sb is not None:
            try:
                if db_name != csv_name:
                    sb.table("tickers").upsert({"code": code, "name": csv_name}).execute()
            except Exception:
                pass
        return csv_name

    if db_name:
        return db_name
    if resolved_name:
        if sb is not None:
            try:
                sb.table("tickers").upsert({"code": code, "name": resolved_name}).execute()
            except Exception:
                pass
        return resolved_name

    fetched_name = fetch_company_name_from_yfinance(code)
    final_name = fetched_name or code

    if sb is not None:
        try:
            sb.table("tickers").upsert({"code": code, "name": final_name}).execute()
        except Exception:
            pass

    return final_name


def resolve_company_name(code: str, sb: Client | None = None) -> str:
    return ensure_ticker_name(sb, code)



def save_backtest_to_db(sb: Client, ticker: str, params: dict, result: dict, curve: pd.DataFrame, trades: list):
    params_to_save = dict(params)
    result_meta = {}
    for key in ("ambiguous_days", "intraday_resolved_days"):
        if result.get(key) is not None:
            result_meta[key] = int(result[key])
    if result_meta:
        params_to_save["_result_meta"] = result_meta
    run_row = {
        "ticker": ticker,
        "params": params_to_save,
        "final_equity": float(result["final_equity"]),
        "total_return": float(result["total_return"]),
        "max_drawdown": float(result["max_drawdown"]),
        "sharpe": float(result["sharpe"]) if result.get("sharpe") is not None else None,
        "n_trades": int(result["n_trades"]),
    }

    res = sb.table("backtests_runs").insert(run_row).execute()
    run_id = int(res.data[0]["id"])

    rows = []
    for i, t in enumerate(trades):
        ts = t.get("ts") or t.get("date")
        if ts is None:
            continue

        if hasattr(ts, "to_pydatetime"):
            ts = ts.to_pydatetime()
        ts = ts + timedelta(microseconds=i)

        side = (t.get("side") or "").upper()
        if side not in ("BUY", "SELL"):
            continue
        qty = int(t.get("qty", 0))
        if qty <= 0:
            continue

        signal_ts = t.get("signal_ts")
        if signal_ts is not None:
            if hasattr(signal_ts, "to_pydatetime"):
                signal_ts = signal_ts.to_pydatetime()
            if hasattr(signal_ts, "isoformat"):
                signal_ts = signal_ts.isoformat()
            else:
                signal_ts = str(signal_ts)

        rows.append({
            "run_id": run_id,
            "ts": ts.isoformat(),
            "side": side,
            "price": float(t.get("price", t.get("px", 0.0))),
            "qty": int(t.get("qty", 0)),
            "reason": t.get("reason"),
            "signal_ts": signal_ts,
        })

    if rows:
        sb.table("backtests_trades").insert(rows).execute()

    print(f"[DB] saved run_id={run_id} ({ticker}) {len(rows)} trades.")
