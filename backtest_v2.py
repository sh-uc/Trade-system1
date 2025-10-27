# backtest_v2.py
# 取引日ベースの現実的約定モデル（翌寄り約定、逆指値滑り、利確/時間切れ）

import os, math, datetime as dt
import numpy as np
import pandas as pd
import yfinance as yf
import jpholiday
from supabase import create_client, Client


# ====== 設定（ENVで上書き）======
TICKER         = os.environ.get("BT_TICKER", "3778.T")
START          = os.environ.get("BT_START",  "2023-01-01")
END            = os.environ.get("BT_END",    dt.date.today().isoformat())
CAPITAL_JPY    = float(os.environ.get("BT_CAPITAL",   "1000000"))
PER_TRADE_CAP  = float(os.environ.get("BT_PER_TRADE", "200000"))
RISK_PCT       = float(os.environ.get("BT_RISK_PCT",  "0.005"))   # 0.5%（初期ストップ距離の目安）
SLIPPAGE       = float(os.environ.get("BT_SLIPPAGE",  "0.0005"))  # 成行スリッページ率
FEE_PCT        = float(os.environ.get("BT_FEE_PCT",   "0.000"))   # 手数料率（片道）
STOP_SLIPPAGE  = float(os.environ.get("BT_STOP_SLIP","0.0015"))   # ストップ時追加滑り（ギャップ想定）
TAKE_PROFIT_RR = float(os.environ.get("BT_TP_RR",     "2.0"))     # 2Rで利確
MAX_HOLD_DAYS  = int(os.environ.get("BT_MAX_HOLD",    "15"))      # 経過日数で時間切れ
EXIT_ON_REVERSE= (os.environ.get("BT_EXIT_REV","1")=="1")         # 逆シグナルで手仕舞い

RSI_MIN, RSI_MAX = 40.0, 75.0
RSI_EXIT         = 80.0

# ====== supabase client作成　=======
def create_supabase_from_env() -> Client | None:
    import os
    url = os.environ.get("SUPABASE_URL", "").strip()
    key = (os.environ.get("SUPABASE_SERVICE_ROLE") or os.environ.get("SUPABASE_KEY") or "").strip()
    if not url or not key:
        print("[INFO] Supabase env not set; skip saving.")
        return None
    try:
        return create_client(url, key)
    except Exception as e:
        print("[WARN] create_client failed:", e)
        return None

# ====== ユーティリティ：日本の次の営業日 ======
def is_trading_day(d: dt.date) -> bool:
    if d.weekday() >= 5: return False
    if jpholiday.is_holiday(d): return False
    if (d.month, d.day) in {(12,31),(1,1),(1,2),(1,3)}: return False
    return True

def next_trading_day(d: dt.date) -> dt.date:
    t = d + dt.timedelta(days=1)
    while not is_trading_day(t):
        t += dt.timedelta(days=1)
    return t

# ====== 指標 ======
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    o = df.copy()
    # MultiIndex列をフラット化
    if isinstance(o.columns, pd.MultiIndex):
        PRICE_KEYS = {"Open","High","Low","Close","Adj Close","Volume"}
        def pick(col_tuple):
            for part in col_tuple:
                if part in PRICE_KEYS: return part
            return col_tuple[0]
        o.columns = [pick(c) if isinstance(c, tuple) else c for c in o.columns]

    # 列名マップ
    cols = {c.lower(): c for c in o.columns}
    for need in ["open","high","low","close","volume"]:
        uc = need.capitalize()
        if need not in cols and uc in o.columns:
            cols[need] = uc

    H = o[cols.get("high","High")]
    L = o[cols.get("low","Low")]
    C = o[cols.get("close","Close")]
    V = o[cols.get("volume","Volume")]

    # MA
    o["ma25"] = C.rolling(25, min_periods=25).mean()
    o["ma75"] = C.rolling(75, min_periods=75).mean()
    # RSI
    dlt = C.diff()
    up  = np.where(dlt>0, dlt, 0.0)
    dn  = np.where(dlt<0, -dlt, 0.0)
    ru  = pd.Series(up, index=o.index).ewm(span=14, adjust=False).mean()
    rd  = pd.Series(dn, index=o.index).ewm(span=14, adjust=False).mean()
    rs  = ru / rd.replace(0, np.nan)
    o["rsi14"] = 100 - (100/(1+rs))
    # MACD
    ema12 = C.ewm(span=12, adjust=False).mean()
    ema26 = C.ewm(span=26, adjust=False).mean()
    o["macd"] = ema12 - ema26
    o["macd_signal"] = o["macd"].ewm(span=9, adjust=False).mean()
    # ATR（単純）
    tr = pd.concat([(H-L), (H-C.shift()).abs(), (L-C.shift()).abs()], axis=1).max(axis=1)
    o["atr14"] = tr.rolling(14, min_periods=14).mean()
    # 出来高スパイク
    o["vol_ma20"] = V.rolling(20, min_periods=20).mean()
    o["vol_spike"] = (V >= o["vol_ma20"]).fillna(False)
    # Open列保証
    o["open"] = o[cols.get("open","Open")]
    o["high"] = H; o["low"] = L; o["close"] = C
    return o

# ====== シグナル ======
def long_signal_row(r: pd.Series) -> bool:
    try:
        close = float(r["close"]); ma25 = float(r["ma25"]); ma75 = float(r["ma75"])
        macd = float(r["macd"]); macd_sig = float(r["macd_signal"])
        rsi = float(r["rsi14"]); atr = float(r["atr14"])
        vol_spike = bool(r["vol_spike"])
    except Exception:
        return False
    cond_trend    = (close > ma25) and (ma25 >= ma75)
    cond_momentum = (macd > macd_sig)
    cond_rsi      = (RSI_MIN <= rsi <= RSI_MAX)
    cond_vol      = vol_spike
    cond_vola     = atr > 0
    return all([cond_trend, cond_momentum, cond_rsi, cond_vol, cond_vola])

# ====== バックテスト（翌寄り約定 + ストップ滑り + TP/時間切れ） ======
def backtest():
    df = yf.download(
        TICKER, start=START, end=END, auto_adjust=False, progress=False, group_by="column"
    )
    if df.empty:
        print(f"[BT] No data: {TICKER}")
        return
    ind = compute_indicators(df)

    cash = CAPITAL_JPY
    pos = 0
    entry_px = np.nan
    entry_date = None
    stop_px = np.nan
    take_px = np.nan
    hold_days = 0
    pending_buy_for = None  # 日付: 翌営業日に執行

    equity_curve, trades = [], []

    dates = list(ind.index)

    for i, date in enumerate(dates):
        row = ind.loc[date]
        o = float(row["open"]); h = float(row["high"]); l = float(row["low"]); c = float(row["close"])

        # 1) ペンディング（買い）→ 翌営業日の寄りで執行
        if pending_buy_for and date.date() == pending_buy_for:
            # 数量決定（前日の終値ベースのRでも良いが、今回は当日寄りで再計算）
            risk_jpy = PER_TRADE_CAP * RISK_PCT
            # 初期ストップ：寄りから -RISK_PCT
            stop_px = o * (1 - RISK_PCT)
            risk_per_share = max(o - stop_px, o * 0.005)
            qty_by_risk = int(risk_jpy / risk_per_share) if risk_per_share>0 else 0
            qty_by_cap  = int(PER_TRADE_CAP // o)
            est_qty = max(0, min(qty_by_risk, qty_by_cap))
            est_qty = min(est_qty, int(cash // (o * (1 + SLIPPAGE + FEE_PCT))))
            if est_qty > 0:
                fill = o * (1 + SLIPPAGE)
                cost = fill * est_qty * (1 + FEE_PCT)
                cash -= cost
                pos = est_qty
                entry_px = fill
                entry_date = date
                hold_days = 0
                # TP価格（R倍）
                R = entry_px - stop_px
                take_px = entry_px + TAKE_PROFIT_RR * R if R>0 else np.nan
                trades.append({"date": date, "side": "BUY", "px": fill, "qty": est_qty})
            pending_buy_for = None

        # 2) 当日引けで判定 → 翌営業日に買いセット
        if pos == 0:
            if long_signal_row(row):
                nd = next_trading_day(date.date())
                pending_buy_for = nd

        # 3) ポジション保有時の手仕舞い判定（順序：ストップ→利確→時間切れ→逆シグナル）
        else:
            # ストップ：日中で割れていたら、次の約定は「その日の寄り価格相当」に滑りを上乗せ
            stop_hit = (l <= stop_px)
            if stop_hit:
                # ギャップも含め「寄り相当」に STOP_SLIPPAGE を上乗せして約定
                fill = max(o, stop_px) * (1 - STOP_SLIPPAGE)
                proceeds = fill * pos * (1 - FEE_PCT)
                cash += proceeds
                trades.append({"date": date, "side": "SELL", "px": fill, "qty": pos, "reason":"SL"})
                pos = 0; entry_px = np.nan; entry_date=None; take_px=np.nan; hold_days=0
            else:
                # 利確
                tp_hit = (h >= take_px) if not math.isnan(take_px) else False
                if tp_hit:
                    fill = max(take_px, o) * (1 - SLIPPAGE)  # TP到達は少し有利執行想定
                    cash += fill * pos * (1 - FEE_PCT)
                    trades.append({"date": date, "side": "SELL", "px": fill, "qty": pos, "reason":"TP"})
                    pos = 0; entry_px = np.nan; entry_date=None; take_px=np.nan; hold_days=0
                else:
                    # 時間切れ
                    if hold_days >= MAX_HOLD_DAYS:
                        fill = c * (1 - SLIPPAGE)
                        cash += fill * pos * (1 - FEE_PCT)
                        trades.append({"date": date, "side": "SELL", "px": fill, "qty": pos, "reason":"TIME"})
                        pos = 0; entry_px = np.nan; entry_date=None; take_px=np.nan; hold_days=0
                    else:
                        # 逆シグナル（任意）
                        if EXIT_ON_REVERSE and (not long_signal_row(row)):
                            fill = c * (1 - SLIPPAGE)
                            cash += fill * pos * (1 - FEE_PCT)
                            trades.append({"date": date, "side": "SELL", "px": fill, "qty": pos, "reason":"REV"})
                            pos = 0; entry_px = np.nan; entry_date=None; take_px=np.nan; hold_days=0

        # 4) 評価額
        hold_days = (hold_days + 1) if (pos>0 and entry_date is not None and date>=entry_date) else hold_days
        position_val = pos * c
        equity = cash + position_val
        equity_curve.append({"date": date, "equity": equity})

    curve = pd.DataFrame(equity_curve).set_index("date")
    if curve.empty:
        print("[BT] Curve empty."); return
    total_return = curve["equity"].iloc[-1] / CAPITAL_JPY - 1.0
    roll_max = curve["equity"].cummax()
    drawdown = (roll_max - curve["equity"]) / roll_max
    max_dd = drawdown.max()

    # 約定ペア損益
    pnl=[]; last_buy=None
    for t in trades:
        if t["side"]=="BUY": last_buy=t
        elif t["side"]=="SELL" and last_buy:
            pnl.append((t["px"]-last_buy["px"])*min(t["qty"], last_buy["qty"]))
            last_buy=None
    pnl_s = pd.Series(pnl)
    win_rate = float((pnl_s>0).mean()) if len(pnl_s) else np.nan

    # 日次リターンで簡易シャープ
    daily_ret = curve["equity"].pct_change().dropna()
    sharpe = float(np.sqrt(245) * (daily_ret.mean() / (daily_ret.std()+1e-12))) if len(daily_ret) else np.nan
    n_trades = int((np.array([t["side"] for t in trades])=="SELL").sum())

    print(f"[BT2] {TICKER} {START}..{END}")
    print(f"  初期資金  : JPY{CAPITAL_JPY:,.0f}")
    print(f"  最終資産  : JPY{curve['equity'].iloc[-1]:,.0f}  (累計損益 {total_return*100:.2f}%)")
    print(f"  取引数(決済): {n_trades}  勝率 {win_rate*100:.1f}%  最大DD {max_dd*100:.2f}%  Sharpe {sharpe:.2f}")

    out_dir = os.environ.get("BT_OUT","./backtest_out")
    os.makedirs(out_dir, exist_ok=True)
    curve.to_csv(os.path.join(out_dir, f"curve_{TICKER}_v2.csv"))
    pd.DataFrame(trades).to_csv(os.path.join(out_dir, f"trades_{TICKER}_v2.csv"), index=False)
    print(f"[BT2] 出力: {out_dir}/curve_{TICKER}_v2.csv, trades_{TICKER}_v2.csv")

def save_backtest_run(sb: Client, ticker: str, params: dict, curve_final: float,
                      total_return: float, max_dd: float, sharpe: float, n_trades: int) -> int | None:
    try:
        res = sb.table("backtests_runs").insert({
            "ticker": ticker,
            "params": params,                 # jsonb
            "final_equity": round(curve_final, 4),
            "total_return": round(total_return, 6),
            "max_drawdown": round(max_dd, 6),
            "sharpe": round(sharpe, 6),
            "n_trades": n_trades,
        }).execute()
        run_id = res.data[0]["id"]
        print(f"[SAVE] runs id={run_id}")
        return int(run_id)
    except Exception as e:
        print("[ERROR] save_backtest_run:", e)
        return None

def save_backtest_trades(sb: Client, run_id: int, trades: list[dict]):
    # trades: {"date": pd.Timestamp, "side": "BUY"/"SELL", "px": float, "qty": int, "reason": Optional[str]}
    rows = []
    for t in trades:
        ts = t["date"]
        # pandas.Timestamp → ISO8601（tzなしならUTC想定でそのまま）
        if hasattr(ts, "isoformat"):
            ts_iso = ts.isoformat()
        else:
            ts_iso = str(ts)
        rows.append({
            "run_id": run_id,
            "ts": ts_iso,
            "side": t["side"],
            "price": float(t["px"]),
            "qty": int(t["qty"]),
            "reason": t.get("reason"),
        })
    try:
        # バルク insert（必要に応じて分割）
        for i in range(0, len(rows), 500):
            sb.table("backtests_trades").insert(rows[i:i+500]).execute()
        print(f"[SAVE] trades rows={len(rows)}")
    except Exception as e:
        print("[ERROR] save_backtest_trades:", e)

if __name__ == "__main__":
    import os
    os.makedirs("./backtest_out", exist_ok=True)
    backtest()
    # ---- ここから追記: Supabase保存（SAVE_BT=1 の時だけ） ----
if os.environ.get("SAVE_BT", "0") == "1":
    sb = create_supabase_from_env()
    if sb is not None:
        params = {
            "CAPITAL_JPY": CAPITAL_JPY,
            "PER_TRADE_CAP": PER_TRADE_CAP,
            "RISK_PCT": RISK_PCT,
            "SLIPPAGE": SLIPPAGE,
            "FEE_PCT": FEE_PCT,
            "STOP_SLIPPAGE": STOP_SLIPPAGE,
            "TAKE_PROFIT_RR": TAKE_PROFIT_RR,
            "MAX_HOLD_DAYS": MAX_HOLD_DAYS,
            "EXIT_ON_REVERSE": EXIT_ON_REVERSE,
            "START": START,
            "END": END,
        }
        run_id = save_backtest_run(
            sb, TICKER, params,
            float(curve["equity"].iloc[-1]),
            float(total_return),
            float(max_dd),
            float(sharpe),
            int(n_trades),
        )
        if run_id is not None and len(trades):
            save_backtest_trades(sb, run_id, trades)
# ---- 追記ここまで ----

