# db_utils.py
from supabase import create_client, Client
import os, pandas as pd
from datetime import timedelta

def get_supabase() -> Client:
    url = os.environ["SUPABASE_URL"]
    key = os.environ["SUPABASE_KEY"]
    return create_client(url, key)



def save_backtest_to_db(sb: Client, ticker: str, params: dict, result: dict, curve: pd.DataFrame, trades: list):
    # --- backtests_runs に保存（id は bigserial：DBに任せる）---
    run_row = {
        "ticker": ticker,
        "params": params,
        "final_equity": float(result["final_equity"]),
        "total_return": float(result["total_return"]),
        "max_drawdown": float(result["max_drawdown"]),
        "sharpe": float(result["sharpe"]) if result.get("sharpe") is not None else None,
        "n_trades": int(result["n_trades"]),
    }

    res = sb.table("backtests_runs").insert(run_row).execute()
    run_id = int(res.data[0]["id"])  # bigint

    # --- backtests_trades に保存（DDL: run_id, ts, side, price, qty, reason,signal_ts）---
    rows = []
    for i, t in enumerate(trades):
        ts = t.get("ts") or t.get("date")
        if ts is None:
            continue

        # pandas.Timestamp / datetime 対応
        if hasattr(ts, "to_pydatetime"):
            ts = ts.to_pydatetime()
        # もし date だけで時刻が 00:00:00 になりがちなら重複回避で少しずらす
        ts = ts + timedelta(microseconds=i)

        side = (t.get("side") or "").upper()
        # DDL: BUY / SELL 制約に合わせる（必要ならここで変換）
        # 例：'LONG'/'SHORT' などが来るなら適宜マッピングしてください
        # ✅ backtests_trades は BUY/SELL だけ保存（DDLの制約に合わせる）
        if side not in ("BUY", "SELL"):
            continue
        qty = int(t.get("qty", 0))
        if qty <= 0:
            continue
        # --- signal_ts を JSON で送れる形（ISO文字列）に変換 ---
        signal_ts = t.get("signal_ts")
        if signal_ts is not None:
            if hasattr(signal_ts, "to_pydatetime"):
                signal_ts = signal_ts.to_pydatetime()
            # datetime なら ISO 文字列へ
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
            "signal_ts": signal_ts,   # ★追加（BUYやSL/TPは None のままでOK）
        })

    if rows:
        sb.table("backtests_trades").insert(rows).execute()

    print(f"[DB] saved run_id={run_id} ({ticker}) {len(rows)} trades.")

#