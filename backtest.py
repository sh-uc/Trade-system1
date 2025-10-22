# backtest.py
# 現行ロジック準拠の最小実用バックテスト（引け判定・日足）
import os, math, datetime as dt
import numpy as np
import pandas as pd
import yfinance as yf

# ===== 設定（ENVで上書き可）=====
TICKER         = os.environ.get("BT_TICKER", "3778.T")
START          = os.environ.get("BT_START",  "2023-01-01")
END            = os.environ.get("BT_END",    dt.date.today().isoformat())
CAPITAL_JPY    = float(os.environ.get("BT_CAPITAL",    "1000000"))  # 100万円
PER_TRADE_CAP  = float(os.environ.get("BT_PER_TRADE",  "200000"))   # 1回20万円
RISK_PCT       = float(os.environ.get("BT_RISK_PCT",   "0.005"))    # 0.5%（許容損失率）
SLIPPAGE       = float(os.environ.get("BT_SLIPPAGE",   "0.0005"))   # 0.05%想定
FEE_PCT        = float(os.environ.get("BT_FEE_PCT",    "0.000"))    # 手数料率（必要なら設定）
RSI_MIN, RSI_MAX = 40.0, 75.0                                       # 買いの快適帯
RSI_EXIT        = 80.0                                              # 利確的手仕舞い閾値

# ===== 指標計算（app/daily_taskに合わせる）=====
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    o = df.copy()
    # 必要列を揃える
    cols = {c.lower(): c for c in o.columns}
    for need in ["open","high","low","close","volume"]:
        if need not in cols:
            # yfinanceは列頭大文字が来ることが多い
            uc = need.capitalize()
            if uc in o.columns: cols[need] = uc
    H = o[cols.get("high","High")]
    L = o[cols.get("low","Low")]
    C = o[cols.get("close","Close")]
    V = o[cols.get("volume","Volume")]

    # MA
    o["ma25"] = C.rolling(25, min_periods=25).mean()
    o["ma75"] = C.rolling(75, min_periods=75).mean()

    # RSI(14)
    delta = C.diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up, index=o.index).ewm(span=14, adjust=False).mean()
    roll_dn = pd.Series(down, index=o.index).ewm(span=14, adjust=False).mean()
    rs = roll_up / roll_dn.replace(0, np.nan)
    o["rsi14"] = 100 - (100 / (1 + rs))

    # MACD(12,26,9)
    ema12 = C.ewm(span=12, adjust=False).mean()
    ema26 = C.ewm(span=26, adjust=False).mean()
    o["macd"] = ema12 - ema26
    o["macd_signal"] = o["macd"].ewm(span=9, adjust=False).mean()

    # ATR(14)（単純版）
    tr = pd.concat([(H - L), (H - C.shift()).abs(), (L - C.shift()).abs()], axis=1).max(axis=1)
    o["atr14"] = tr.rolling(14, min_periods=14).mean()

    # 出来高スパイク（20日平均以上）
    o["vol_ma20"] = V.rolling(20, min_periods=20).mean()
    o["vol_spike"] = (V >= o["vol_ma20"]).fillna(False)

    # クローズ列を保証
    o["close"] = C
    return o

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

# ===== バックテスト =====
def backtest():
    df = yf.download(TICKER, start=START, end=END, auto_adjust=False, progress=False)
    if df.empty:
        print(f"[BT] No data: {TICKER} {START}..{END}")
        return
    ind = compute_indicators(df)

    cash = CAPITAL_JPY
    qty  = 0
    entry_px = np.nan
    entry_date = None

    equity_curve = []
    trades = []  # {date, side, px, qty}

    for i in range(len(ind)):
        row = ind.iloc[i]
        date = row.name
        close = float(row["close"])

        # 引けで判定 → 引け成行で執行（シンプル版）
        if qty == 0:
            if long_signal_row(row):
                # 許容損失から理論数量
                risk_jpy = PER_TRADE_CAP * RISK_PCT
                stop_px  = close * (1 - RISK_PCT)
                risk_per_share = max(close - stop_px, close * 0.005)  # 下限0.5%
                qty_by_risk = int(risk_jpy / risk_per_share) if risk_per_share > 0 else 0
                qty_by_cap  = int(PER_TRADE_CAP // close)
                est_qty = max(0, min(qty_by_risk, qty_by_cap))
                # 所持現金チェック
                est_qty = min(est_qty, int(cash // (close * (1 + SLIPPAGE + FEE_PCT))))
                if est_qty > 0:
                    fill = close * (1 + SLIPPAGE)
                    cost = fill * est_qty * (1 + FEE_PCT)
                    cash -= cost
                    qty = est_qty
                    entry_px = fill
                    entry_date = date
                    trades.append({"date": date, "side": "BUY", "px": fill, "qty": est_qty})
        else:
            # 損切り / 逆シグナル / RSI過熱のどれかで手仕舞い
            stop_trigger = (close <= entry_px * (1 - RISK_PCT))
            exit_signal  = (not long_signal_row(row)) or float(row["rsi14"]) >= RSI_EXIT
            if stop_trigger or exit_signal:
                fill = close * (1 - SLIPPAGE)
                proceeds = fill * qty * (1 - FEE_PCT)
                cash += proceeds
                trades.append({"date": date, "side": "SELL", "px": fill, "qty": qty})
                qty = 0
                entry_px = np.nan
                entry_date = None

        # 評価額
        position_val = qty * close
        equity = cash + position_val
        equity_curve.append({"date": date, "equity": equity})

    curve = pd.DataFrame(equity_curve).set_index("date")
    if curve.empty:
        print("[BT] Curve empty.")
        return
    total_return = curve["equity"].iloc[-1] / CAPITAL_JPY - 1.0
    # 最大DD
    roll_max = curve["equity"].cummax()
    drawdown = (roll_max - curve["equity"]) / roll_max
    max_dd = drawdown.max()

    # 約定ペアから単純損益
    pnl = []
    last_buy = None
    for t in trades:
        if t["side"] == "BUY":
            last_buy = t
        elif t["side"] == "SELL" and last_buy:
            pnl.append((t["px"] - last_buy["px"]) * min(t["qty"], last_buy["qty"]))
            last_buy = None
    pnl_s = pd.Series(pnl)
    win_rate = float((pnl_s > 0).mean()) if len(pnl_s) else np.nan
    avg_win  = float(pnl_s[pnl_s>0].mean()) if (pnl_s>0).any() else np.nan
    avg_loss = float(pnl_s[pnl_s<0].mean()) if (pnl_s<0).any() else np.nan

    # 年率換算（営業日 ~ 245）
    daily_ret = curve["equity"].pct_change().dropna()
    sharpe   = float(np.sqrt(245) * (daily_ret.mean() / (daily_ret.std() + 1e-12))) if len(daily_ret) else np.nan
    n_trades = int((np.array([t["side"] for t in trades]) == "SELL").sum())

    print(f"[BT] {TICKER} {START}..{END}")
    print(f"  初期資金  : ¥{CAPITAL_JPY:,.0f}")
    print(f"  最終資産  : ¥{curve['equity'].iloc[-1]:,.0f}  (累計損益 {total_return*100:.2f}%)")
    print(f"  トレード数: {n_trades}  勝率 {win_rate*100:.1f}%  平均勝ち ¥{0 if math.isnan(avg_win) else round(avg_win):,}  平均負け ¥{0 if math.isnan(avg_loss) else round(avg_loss):,}")
    print(f"  最大DD    : {max_dd*100:.2f}%  シャープレシオ(概算): {sharpe:.2f}")

    out_dir = os.environ.get("BT_OUT", "./backtest_out")
    os.makedirs(out_dir, exist_ok=True)
    curve.to_csv(os.path.join(out_dir, f"curve_{TICKER}.csv"))
    pd.DataFrame(trades).to_csv(os.path.join(out_dir, f"trades_{TICKER}.csv"), index=False)
    print(f"[BT] 出力: {out_dir}/curve_{TICKER}.csv, {out_dir}/trades_{TICKER}.csv")

if __name__ == "__main__":
    backtest()
