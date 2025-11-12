# db_utils.py
from supabase import create_client, Client
import os, uuid, pandas as pd

def get_supabase() -> Client:
    url = os.environ["SUPABASE_URL"]
    key = os.environ["SUPABASE_KEY"]
    return create_client(url, key)

def save_backtest_to_db(sb: Client, ticker: str, params: dict, result: dict, curve: pd.DataFrame, trades: list):
    run_id = str(uuid.uuid4())
    start = curve.index[0].date().isoformat() if not curve.empty else None
    end   = curve.index[-1].date().isoformat() if not curve.empty else None

    # backtests_runs に保存
    sb.table("backtests_runs").upsert({
        "run_id": run_id,
        "ticker": ticker,
        "start": start,
        "end": end,
        "params": params,
        "metrics": {
            "final_equity": result["final_equity"],
            "total_return": result["total_return"],
            "max_drawdown": result["max_drawdown"],
            "sharpe": result["sharpe"],
            "win_rate": result["win_rate"],
            "n_trades": result["n_trades"],
        }
    }).execute()

    # backtests_trades に保存
    rows = []
    for i, t in enumerate(trades):
        rows.append({
            "run_id": run_id,
            "seq": i + 1,
            "date": t.get("date").isoformat() if hasattr(t.get("date"), "isoformat") else None,
            "side": t.get("side"),
            "price": float(t.get("px", 0)),
            "qty": int(t.get("qty", 0)),
            "reason": t.get("reason", ""),
        })
    if rows:
        sb.table("backtests_trades").insert(rows).execute()

    print(f"[DB] saved run_id={run_id} ({ticker}) {len(rows)} trades.")

#