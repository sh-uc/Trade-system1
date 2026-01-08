# bt_core.py
# backtest_v2 のロジックを関数化：インプロセスで高速に複数パラメータを回すための中核モジュール

from __future__ import annotations
import math
from typing import Callable, Dict, Any, Optional

import os
import numpy as np
import pandas as pd
import yfinance as yf
import jpholiday
import datetime as dt
from datetime import timedelta, timezone

JST = timezone(timedelta(hours=9))

# =========================
# 取引カレンダー（日本）
# =========================
def is_trading_day(d: dt.date) -> bool:
    if d.weekday() >= 5:
        return False
    if jpholiday.is_holiday(d):
        return False
    if (d.month, d.day) in {(12, 31), (1, 1), (1, 2), (1, 3)}:
        return False
    return True

def next_trading_day(d: dt.date) -> dt.date:
    t = d + dt.timedelta(days=1)
    while not is_trading_day(t):
        t += dt.timedelta(days=1)
    return t

# =========================
# 価格取得（ティッカー×期間は1回だけ）
# =========================
def fetch_prices(ticker: str, start: str, end: Optional[str] = None) -> pd.DataFrame:
    """
    Yahooから日足を取得して整形（JST index、主要列のみ、MultiIndex→フラット）
    """
    df = yf.download(
        ticker, start=start, end=end, progress=False, auto_adjust=False, group_by="column"
    )
    if df is None or df.empty:
        raise RuntimeError(f"No data for {ticker}")

    # 列フラット化
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        PRICE_KEYS = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}

        def pick(col_tuple):
            for part in col_tuple:
                if part in PRICE_KEYS:
                    return part
            return col_tuple[0]

        out.columns = [pick(c) if isinstance(c, tuple) else c for c in out.columns]

    # 欲しい列だけ
    cols_lower = {c.lower(): c for c in out.columns}
    need = ["open", "high", "low", "close", "volume"]
    miss = [k for k in need if k not in cols_lower and k.capitalize() not in out.columns]
    if miss:
        # 必須列が見つからないケース（銘柄停止など）
        raise RuntimeError(f"Missing columns for {ticker}: {miss}")

    # 統一名にそろえる
    o = pd.DataFrame(index=out.index)
    for k in need:
        src = cols_lower.get(k, k.capitalize())
        o[k] = pd.to_numeric(out[src], errors="coerce")

    # JST化（yfinanceはtzなしIndex→UTC扱いでJSTへ）
    if o.index.tz is None:
        o.index = pd.to_datetime(o.index).tz_localize("UTC").tz_convert(JST)
    else:
        o.index = o.index.tz_convert(JST)

    return o

# =========================
# 指標計算（backtest_v2相当）
# =========================
def compute_indicators(prices: pd.DataFrame, VOL_SPIKE_M: float = 1.4) -> pd.DataFrame:
    """
    MA25/75, RSI(14), MACD(12,26,9), ATR(14: rolling mean), vol_ma20, vol_spike(倍率) を追加
    """
    o = prices.copy()
    for c in ["open", "high", "low", "close", "volume"]:
        if c in o:
            o[c] = pd.to_numeric(o[c], errors="coerce")

    C = o["close"]; H = o["high"]; L = o["low"]; V = o["volume"]

    # MA
    o["ma25"] = C.rolling(25, min_periods=25).mean()
    o["ma75"] = C.rolling(75, min_periods=75).mean()

    # RSI(14)（backtest_v2相当）
    dlt = C.diff()
    up = np.where(dlt > 0, dlt, 0.0)
    dn = np.where(dlt < 0, -dlt, 0.0)
    ru = pd.Series(up, index=o.index).ewm(span=14, adjust=False).mean()
    rd = pd.Series(dn, index=o.index).ewm(span=14, adjust=False).mean()
    rs = ru / rd.replace(0, np.nan)
    o["rsi14"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = C.ewm(span=12, adjust=False).mean()
    ema26 = C.ewm(span=26, adjust=False).mean()
    o["macd"] = ema12 - ema26
    o["macd_signal"] = o["macd"].ewm(span=9, adjust=False).mean()

    # ATR（rolling mean 14）
    tr = pd.concat([(H - L), (H - C.shift()).abs(), (L - C.shift()).abs()], axis=1).max(axis=1)
    o["atr14"] = tr.rolling(14, min_periods=14).mean()

    # 出来高
    o["vol_ma20"] = V.rolling(20, min_periods=20).mean()
    # vol_spikeは作らない　o["vol_spike"] = (V >= (o["vol_ma20"] * float(VOL_SPIKE_M))).fillna(False)

    return o

# =========================
# シグナル（1行）
# =========================
def long_signal_row(r: pd.Series, *, MACD_ATR_K: float, RSI_MIN: float, RSI_MAX: float, VOL_SPIKE_M: float) -> bool:
    try:
        close = float(r["close"]); ma25 = float(r["ma25"]); ma75 = float(r["ma75"])
        macd = float(r["macd"]); macd_sig = float(r["macd_signal"])
        rsi = float(r["rsi14"]); atr = float(r["atr14"])
        vol = float(r["volume"]); vol_ma20 = float(r["vol_ma20"])
    except Exception:
        return False

    cond_trend = (close > ma25) and (ma25 >= ma75)
    # MACD差が「ATR/Close に係数をかけた相対しきい値」超え　→　削除2026.1.7
    # MACD差が「ATR に対する割合」以上（単位を揃える：どちらも"円"）
    # 例) MACD_ATR_K=0.1 なら「MACD差 > 0.1 * ATR」
    cond_momentum = (macd - macd_sig) > (MACD_ATR_K * atr)
    cond_rsi = (RSI_MIN <= rsi <= RSI_MAX)
    cond_vol = (vol_ma20 > 0) and (vol >= vol_ma20 * VOL_SPIKE_M)
    cond_vola = atr > 0
    return all([cond_trend, cond_momentum, cond_rsi, cond_vol, cond_vola])

# =========================
# 約定数量の計算
# =========================
def _calc_qty(open_px: float, per_trade_cap: float, risk_pct: float,
              cash: float, slippage: float, fee_pct: float,
              lot_size: int = 100) -> int:
    """
    LOT単位（lot_size株）で数量を決定する。
    """
    if open_px <= 0:
        return 0

    # 1ロットも買えない株価は即スキップ
    max_lots_by_cap = per_trade_cap // (open_px * lot_size)
    max_lots_by_cap = int(max_lots_by_cap)
    if max_lots_by_cap <= 0:
        return 0

    # リスク制約
    risk_jpy = per_trade_cap * risk_pct
    stop_px = open_px * (1 - risk_pct)
    # ここで勝手に下限(例:0.5%)を入れると、RISK_PCTの探索が死にやすいので素直に使う 2025.12.26
    # risk_per_share = max(open_px - stop_px, open_px * 0.005)
    risk_per_share = open_px - stop_px

    if risk_per_share <= 0:
        return 0
    max_shares_by_risk = int(risk_jpy // risk_per_share)
    max_lots_by_risk = max_shares_by_risk // lot_size

    # 現金制約
    max_shares_by_cash = int(cash // (open_px * (1 + slippage + fee_pct)))
    max_lots_by_cash = max_shares_by_cash // lot_size

    lots = min(max_lots_by_cap, max_lots_by_risk, max_lots_by_cash)
    if lots <= 0:
        return 0

    return lots * lot_size

# =========================
# バックテスト（翌寄り約定／ギャップ制限／TP・時間切れ・逆シグナル）
# =========================
def run_backtest(
    ind: pd.DataFrame,
    params: Dict[str, Any],
    save_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """
    ind    : compute_indicators() 済みの日足DF（JST index）
    params : 以下キーを期待（存在しない場合はデフォルト）
        CAPITAL        : 初期資金 JPY
        PER_TRADE      : 1トレード上限 JPY
        RISK_PCT       : 許容損失率（初期ストップ距離の目安）
        LOT_SIZE       : 売買単位　2025.11.23　追加
        SLIPPAGE       : 成行スリッページ
        FEE_PCT        : 片道手数料（必要なら）
        STOP_SLIPPAGE  : ストップ時追加滑り
        TAKE_PROFIT_RR : R倍で利確
        MAX_HOLD_DAYS  : 時間切れ日数
        EXIT_ON_REVERSE: 逆シグナルで手仕舞い（bool）
        VOL_SPIKE_M    : 出来高スパイク倍率（compute側と合わせる）
        MACD_ATR_K     : MACD強度の相対下限
        RSI_MIN        : エントリ許容RSI下限
        RSI_MAX        : エントリ許容RSI上限
        GAP_ENTRY_MAX  : 前日終値比の寄りギャップ上限（超えたら新規見送り）
    save_cb: 取引確定時や最後に結果を保存したい場合のコールバック（任意）

    戻り値 : {"final_equity": float, "total_return": float, "sharpe": float, ...}
    """
    CAPITAL = float(params.get("CAPITAL", 3_000_000)) # 1000000 => 3000000 2025.11.23
    PER_TRADE = float(params.get("PER_TRADE", 500_000)) # 200000 => 500000 2025.11.23
    RISK_PCT = float(params.get("RISK_PCT", 0.003)) # 0.005 => 0.003 2025.11.23
    LOT_SIZE = int(params.get("LOT_SIZE", 100)) # added 2025.11.23
    SLIPPAGE = float(params.get("SLIPPAGE", 0.0005))
    FEE_PCT = float(params.get("FEE_PCT", 0.0))
    STOP_SLIPPAGE = float(params.get("STOP_SLIPPAGE", 0.0015))
    TAKE_PROFIT_RR = float(params.get("TAKE_PROFIT_RR", 2.0))
    MAX_HOLD_DAYS = int(params.get("MAX_HOLD_DAYS", 10))
    EXIT_ON_REVERSE = bool(params.get("EXIT_ON_REVERSE", True))
    VOL_SPIKE_M = float(params.get("VOL_SPIKE_M", 1.4))
    MACD_ATR_K = float(params.get("MACD_ATR_K", 0.15))
    RSI_MIN = float(params.get("RSI_MIN", 45.0))
    RSI_MAX = float(params.get("RSI_MAX", 70.0))
    GAP_ENTRY_MAX = float(params.get("GAP_ENTRY_MAX", 0.05))

    cash = CAPITAL
    pos = 0
    entry_px = math.nan
    entry_date = None
    stop_px = math.nan
    take_px = math.nan
    hold_days = 0
    pending_buy_for: Optional[dt.date] = None
    # TIME / REV を「翌営業日寄り」で成行決済するためのペンディング
    pending_sell_for: Optional[dt.date] = None
    pending_sell_reason: Optional[str] = None    
    just_bought = False  # 当日エントリーしたら当日のexit判定を禁止する

    equity_curve = []
    trades = []

    dates = list(ind.index)

    for i, date in enumerate(dates):
        row = ind.loc[date]
        o = float(row["open"]); h = float(row["high"]); l = float(row["low"]); c = float(row["close"])

        # 0) ペンディングSELL（翌営業日寄りで執行）
        #    TIME/REV は「当日引けで判断」→「翌営業日寄りで成行」に統一する
        if pending_sell_for and date.date() == pending_sell_for and pos > 0:
            fill = o * (1 - SLIPPAGE)
            cash += fill * pos * (1 - FEE_PCT)
            trades.append({
                "date": date,
                "side": "SELL",
                "px": fill,
                "qty": pos,
                "reason": pending_sell_reason or "MKT"
            })
            pos = 0; entry_px = math.nan; entry_date = None; stop_px = math.nan; take_px = math.nan; hold_days = 0
            pending_sell_for = None
            pending_sell_reason = None
        
        # 前日終値（ギャップ判定用）
        c_prev = float(ind["close"].shift(1).loc[date]) if date in ind.index else math.nan

        # 1) ペンディング（翌寄りで執行）
        if pending_buy_for and date.date() == pending_buy_for:
            if not math.isnan(c_prev) and GAP_ENTRY_MAX > 0:
                gap = (o - c_prev) / c_prev
                if gap > GAP_ENTRY_MAX:
                    trades.append({"date": date, "side": "SKIP", "px": o, "qty": 0, "reason": f"GAP>{GAP_ENTRY_MAX:.2%}"})
                    pending_buy_for = None
                else:
                    qty = _calc_qty(o, PER_TRADE, RISK_PCT, cash, SLIPPAGE, FEE_PCT, lot_size=LOT_SIZE)
                    if qty > 0:
                        fill = o * (1 + SLIPPAGE)
                        cost = fill * qty * (1 + FEE_PCT)
                        cash -= cost
                        pos = qty
                        entry_px = fill
                        entry_date = date
                        hold_days = 0
                        just_bought = True
                        # stop/take は「リスク幅(RISK_PCT)」で決める
                        # STOP_SLIPPAGE は約定滑り(実行)であり、ストップ距離(リスク幅)には混ぜない 2025.12.26
                        # R = max(entry_px * RISK_PCT, entry_px * STOP_SLIPPAGE)
                        R = entry_px * RISK_PCT
                        stop_px = entry_px - R
                        take_px = entry_px + TAKE_PROFIT_RR * R
                        trades.append({"date": date, "side": "BUY", "px": fill, "qty": qty})
                    else:
                        trades.append({"date": date, "side": "SKIP", "px": o, "qty": 0, "reason": "NOFUNDS/SMALL_RISK"})
                    pending_buy_for = None
            else:
                qty = _calc_qty(o, PER_TRADE, RISK_PCT, cash, SLIPPAGE, FEE_PCT, lot_size=LOT_SIZE)
                if qty > 0:
                    fill = o * (1 + SLIPPAGE)
                    cost = fill * qty * (1 + FEE_PCT)
                    cash -= cost
                    pos = qty
                    entry_px = fill
                    entry_date = date
                    hold_days = 0
                    just_bought = True
                    # stop/take は「リスク幅(RISK_PCT)」で決める
                    # STOP_SLIPPAGE は約定滑り(実行)であり、ストップ距離(リスク幅)には混ぜない 2025.12.26
                    # R = max(entry_px * RISK_PCT, entry_px * STOP_SLIPPAGE)
                    R = entry_px * RISK_PCT
                    stop_px = entry_px - R
                    take_px = entry_px + TAKE_PROFIT_RR * R
                    trades.append({"date": date, "side": "BUY", "px": fill, "qty": qty})
                else:
                    trades.append({"date": date, "side": "SKIP", "px": o, "qty": 0, "reason": "NOFUNDS/SMALL_RISK"})
                pending_buy_for = None

        # 2) 当日引けでシグナル判定 → 翌営業日の寄りで成行
        if pos == 0:
            if long_signal_row(row, MACD_ATR_K=MACD_ATR_K, RSI_MIN=RSI_MIN, RSI_MAX=RSI_MAX, VOL_SPIKE_M=VOL_SPIKE_M):
                # 過度なGUは翌朝のギャップチェックで最終遮断（ここではpendingだけセット）
                nd = next_trading_day(date.date())
                pending_buy_for = nd

        # 3) 保有中の手仕舞い（順序：ストップ→利確→時間切れ→逆シグナル）
        else:
            # 日足前提：寄りで買った当日に、その日の高値/安値/引けで決済すると
            # 「同日決済」が大量に起きるので、当日のexit判定を禁止（翌日から判定）
            if just_bought:
                just_bought = False
                # exit判定は行わず、評価額記録へ（この日の売り判定を完全スキップ）
                sold = False  # ← これを入れるだけでも UnboundLocalError は防げる
            else:
                sold = False
                # ストップ（割れたら寄り相当-滑り）
                if l <= stop_px:
                    fill = max(o, stop_px) * (1 - STOP_SLIPPAGE)
                    cash += fill * pos * (1 - FEE_PCT)
                    trades.append({"date": date, "side": "SELL", "px": fill, "qty": pos, "reason": "SL"})
                    pos = 0; entry_px = math.nan; entry_date = None; take_px = math.nan; hold_days = 0
                    sold = True
                # 利確
                if not sold and (not math.isnan(take_px)) and (h >= take_px):
                    fill = max(take_px, o) * (1 - SLIPPAGE)
                    cash += fill * pos * (1 - FEE_PCT)
                    trades.append({"date": date, "side": "SELL", "px": fill, "qty": pos, "reason": "TP"})
                    pos = 0; entry_px = math.nan; entry_date = None; take_px = math.nan; hold_days = 0
                    sold = True
                # 時間切れ / 逆シグナル は「当日引けで判定」→「翌営業日寄りで成行」に変更
                if not sold and pending_sell_for is None:
                    # 時間切れ（引けで判断 → 翌寄りで売る）
                    if hold_days >= MAX_HOLD_DAYS:
                        pending_sell_for = next_trading_day(date.date())
                        pending_sell_reason = "TIME"
                    # 逆シグナル（引けで判断 → 翌寄りで売る）
                    elif EXIT_ON_REVERSE:
                        if not long_signal_row(
                            row,
                            MACD_ATR_K=MACD_ATR_K,
                            RSI_MIN=RSI_MIN,
                            RSI_MAX=RSI_MAX,
                            VOL_SPIKE_M=VOL_SPIKE_M,
                        ):
                            pending_sell_for = next_trading_day(date.date())
                            pending_sell_reason = "REV"
            # end else(if just_bought)

        # 4) 評価額を記録
        hold_days = (hold_days + 1) if (pos > 0 and entry_date is not None and date >= entry_date) else hold_days
        equity_curve_val = cash + (pos * c if pos > 0 else 0.0)
        equity_curve.append({"date": date, "equity": equity_curve_val})

    # 集計
    curve = pd.DataFrame(equity_curve).set_index("date")
    if curve.empty:
        return {"final_equity": float(CAPITAL), "total_return": 0.0, "sharpe": 0.0, "n_points": 0, "n_trades": 0}

    final_eq = float(curve["equity"].iloc[-1])
    total_return = final_eq / float(CAPITAL) - 1.0

    roll_max = curve["equity"].cummax()
    drawdown = (roll_max - curve["equity"]) / roll_max
    max_dd = float(drawdown.max()) if not drawdown.empty else 0.0

    # 約定ペア損益から勝率
    pnl = []
    last_buy = None
    for t in trades:
        if t["side"] == "BUY":
            last_buy = t
        elif t["side"] == "SELL" and last_buy:
            pnl.append((t["px"] - last_buy["px"]) * min(t["qty"], last_buy["qty"]))
            last_buy = None
    pnl_s = pd.Series(pnl, dtype="float64")
    win_rate = float((pnl_s > 0).mean()) if len(pnl_s) else float("nan")

    daily_ret = curve["equity"].pct_change().dropna()
    sharpe = float(np.sqrt(245) * (daily_ret.mean() / (daily_ret.std() + 1e-12))) if len(daily_ret) else 0.0

    result = {
        "final_equity": final_eq,
        "total_return": total_return,
        "max_drawdown": max_dd,
        "sharpe": sharpe,
        "win_rate": win_rate,
        "n_points": int(len(curve)),
        "n_trades": int(sum(1 for t in trades if t["side"] == "SELL")),
        # 任意で返す詳細
        # "curve": curve, "trades": trades
    }

    # オプションで保存コールバック
    if save_cb is not None:
        try:
            save_cb({"curve": curve, "trades": trades, "params": params, "summary": result})
        except Exception as e:
            print("[WARN] save_cb failed:", e)

    return result
#
