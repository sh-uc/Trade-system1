# db_utils.py
from supabase import create_client, Client
import os, uuid, pandas as pd

def get_supabase() -> Client:
    url = os.environ["SUPABASE_URL"]
    key = os.environ["SUPABASE_KEY"]
    return create_client(url, key)

def save_backtest_to_db(sb: Client, ticker: str, params: dict, result: dict, curve: pd.DataFrame, trades: list):
    # --- backtests_runs に保存（id は DB に任せる）---
    run_row = {
        "ticker": ticker,
        "params": params,
        "final_equity": float(result["final_equity"]),
        "total_return": float(result["total_return"]),
        "max_drawdown": float(result["max_drawdown"]),
        "sharpe": float(result["sharpe"]) if result.get("sharpe") is not None else None,
        "n_trades": int(result["n_trades"]),
    }

    # insert して採番された id(bigint) を受け取る
    res = sb.table("backtests_runs").insert(run_row).execute()
    run_id = res.data[0]["id"]  # ← bigint

    # --- backtests_trades に保存（run_id は bigint）---
    rows = []
    for i, t in enumerate(trades):
        dt = t.get("date")
        rows.append({
            "run_id": int(run_id),
            "seq": i + 1,
            "date": dt.isoformat() if hasattr(dt, "isoformat") else None,
            "side": t.get("side"),
            "price": float(t.get("px", 0)),
            "qty": int(t.get("qty", 0)),
            "reason": t.get("reason", ""),
        })

    if rows:
        sb.table("backtests_trades").insert(rows).execute()

    print(f"[DB] saved run_id={run_id} ({ticker}) {len(rows)} trades.")

#