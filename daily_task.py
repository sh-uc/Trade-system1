# daily_task.py — 毎日実行用：直近データ取得→指標→判定→Supabaseへ保存
import os, sys, time, json
from datetime import timezone, timedelta
import numpy as np
import pandas as pd
import yfinance as yf
from supabase import create_client

JST = timezone(timedelta(hours=9))

# ---------- Utils ----------
def chunks(iterable, size=500):
    buf=[]; 
    for x in iterable:
        buf.append(x)
        if len(buf)>=size:
            yield buf; buf=[]
    if buf: yield buf

def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    d = close.diff()
    up = pd.Series(np.where(d>0, d, 0.0), index=close.index)
    dn = pd.Series(np.where(d<0, -d, 0.0), index=close.index)
    ru = up.ewm(span=period, adjust=False).mean()
    rd = dn.ewm(span=period, adjust=False).mean()
    rs = ru/(rd+1e-12)
    return 100 - 100/(1+rs)

def macd(close: pd.Series, fast=12, slow=26, sig=9):
    m = ema(close, fast) - ema(close, slow)
    return m, ema(m, sig), m - ema(m, sig)

def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        new_cols=[]
        for c in df.columns:
            if isinstance(c, tuple):
                parts=[str(x) for x in c if str(x)!=""]
                new_cols.append(parts[0] if parts else "")
            else:
                new_cols.append(c)
        df = df.copy()
        df.columns = new_cols
    return df.loc[:, ~pd.Index(df.columns).duplicated()].copy()

def true_range(df_ohlc: pd.DataFrame) -> pd.Series:
    H = df_ohlc["High"];  H = H.iloc[:,0] if isinstance(H, pd.DataFrame) else H
    L = df_ohlc["Low"];   L = L.iloc[:,0] if isinstance(L, pd.DataFrame) else L
    C = df_ohlc["Close"]; C = C.iloc[:,0] if isinstance(C, pd.DataFrame) else C
    pc = C.shift(1)
    a=(H-L).abs(); b=(H-pc).abs(); c=(L-pc).abs()
    return pd.concat([a,b,c],axis=1).max(axis=1)

def fetch_yf(ticker: str, period="90d", interval="1d", tries=4, sleep_sec=3) -> pd.DataFrame:
    last_err=None
    for _ in range(tries):
        try:
            df = yf.download(ticker, period=period, interval=interval,
                             progress=False, auto_adjust=False, threads=False)
            if isinstance(df, pd.DataFrame) and not df.empty:
                return df
            t = yf.Ticker(ticker)
            df = t.history(period=period, interval=interval, auto_adjust=False)
            if isinstance(df, pd.DataFrame) and not df.empty:
                if "Open" not in df.columns and "open" in df.columns:
                    df = df.rename(columns={"open":"Open","high":"High","low":"Low","close":"Close","volume":"Volume"})
                return df
        except Exception as e:
            last_err=e
        time.sleep(sleep_sec)
    raise RuntimeError(f"Failed to download {ticker}. last_err={last_err!r}")

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = flatten_columns(df)
    raw = (df.rename(columns={
        "Open":"open","High":"high","Low":"low","Close":"close","Adj Close":"adj_close","Volume":"volume"
    }).dropna().copy())

    ind = raw.copy()
    ind["ma25"]    = ind["close"].rolling(25, min_periods=25).mean()
    ind["ma75"]    = ind["close"].rolling(75, min_periods=75).mean()
    ind["rsi14"]   = rsi(ind["close"], 14)
    macd_line, macd_sig, _ = macd(ind["close"])
    ind["macd"], ind["macd_signal"] = macd_line, macd_sig
    ind["atr14"]   = true_range(df).rolling(14, min_periods=14).mean()
    ind["stdev20"] = ind["close"].rolling(20, min_periods=20).std()
    vol_series = pd.Series(np.ravel(np.asarray(ind["volume"])), index=ind.index)
    ind["vol_ma20"]  = vol_series.rolling(20, min_periods=20).mean()
    ind["vol_spike"] = (vol_series >= ind["vol_ma20"]).fillna(False).astype(bool)
    ind["swing_low20"]  = ind["low"].rolling(20, min_periods=20).min()
    ind["swing_high20"] = ind["high"].rolling(20, min_periods=20).max()
    ind = ind.dropna()
    return raw, ind

def judge_action(last_row: pd.Series) -> dict:
    # スカラ化
    getf = lambda s: float(s.iloc[0]) if isinstance(s, pd.Series) else float(s)
    getb = lambda s: bool(s.iloc[0]) if isinstance(s, pd.Series) else bool(s)

    close     = getf(last_row["close"])
    ma25      = getf(last_row["ma25"])
    ma75      = getf(last_row["ma75"])
    macd_val  = getf(last_row["macd"])
    macd_sig  = getf(last_row["macd_signal"])
    rsi14     = getf(last_row["rsi14"])
    atr14     = getf(last_row["atr14"])
    vol_spike = getb(last_row["vol_spike"])

    cond_trend    = (close > ma25) and (ma25 >= ma75)
    cond_momentum = (macd_val > macd_sig)
    cond_rsi      = (40.0 <= rsi14 <= 68.0)
    cond_volume   = vol_spike

    passed = all([cond_trend, cond_momentum, cond_rsi, cond_volume])

    reasons = [
        f"トレンド: close({close:.1f}) > MA25({ma25:.1f}) & MA25≥MA75 → {'OK' if cond_trend else 'NG'}",
        f"モメンタム: MACD({macd_val:.3f}) > Signal({macd_sig:.3f}) → {'OK' if cond_momentum else 'NG'}",
        f"過熱回避: RSI14={rsi14:.1f} ∈ [40,68] → {'OK' if cond_rsi else 'NG'}",
        f"出来高: 20日平均以上 → {'OK' if cond_volume else 'NG'}",
    ]
    action = "買い" if passed else "見送り"
    return {"action": action, "passed": passed, "reasons": reasons, "close": close, "atr14": atr14}

def upsert_prices(sb, code: str, raw: pd.DataFrame, index_align):
    rows=[]
    for ts, r in raw.loc[index_align].iterrows():
        rows.append({"date": ts.date().isoformat(), "code": code,
                     "open": float(r.open), "high": float(r.high), "low": float(r.low),
                     "close": float(r.close), "volume": int(r.volume)})
    for c in chunks(rows, 500):
        sb.table("prices").upsert(c, on_conflict="date,code").execute()

def upsert_indicators(sb, code: str, ind: pd.DataFrame):
    rows=[]
    for ts, r in ind.iterrows():
        rows.append({"date": ts.date().isoformat(), "code": code,
                     "rsi14": float(r.rsi14), "macd": float(r.macd), "macd_signal": float(r.macd_signal),
                     "atr14": float(r.atr14), "stdev20": float(r.stdev20),
                     "ma25": float(r.ma25), "ma75": float(r.ma75),
                     "vol_ma20": float(r.vol_ma20), "vol_spike": bool(r.vol_spike),
                     "swing_low20": float(r.swing_low20), "swing_high20": float(r.swing_high20)})
    for c in chunks(rows, 500):
        sb.table("indicators").upsert(c, on_conflict="date,code").execute()

def upsert_signal(sb, code: str, ts: pd.Timestamp, decision: dict):
    date_iso = (ts.tz_convert(JST) if ts.tzinfo else ts.tz_localize(JST)).date().isoformat()
    payload = {
        "date": date_iso,
        "code": code,
        "action": decision["action"],
        "summary": " & ".join([x.split("→")[0]+("OK" if "OK" in x else "NG") for x in decision["reasons"]]),
        "reasons": decision["reasons"],
        "close": decision["close"],
        "qty": None,
        "sl": None,
        "tp": None,
        "risk_jpy": None,
    }
    sb.table("signals").upsert(payload, on_conflict="date,code").execute()

def run_one(ticker: str, period="90d"):
    df = fetch_yf(ticker, period=period, interval="1d")
    raw, ind = compute_indicators(df)
    last_ts = ind.index[-1]
    decision = judge_action(ind.iloc[-1])

    url = os.environ["SUPABASE_URL"]; key = os.environ["SUPABASE_KEY"]
    sb = create_client(url, key)
    upsert_prices(sb, ticker, raw, ind.index)
    upsert_indicators(sb, ticker, ind)
    upsert_signal(sb, ticker, last_ts, decision)
    print(f"OK {ticker}: last={last_ts.date().isoformat()} action={decision['action']}")

if __name__ == "__main__":
    try:
        tickers = os.environ.get("TICKERS") or os.environ.get("TICKER", "3778.T")
        period  = os.environ.get("PERIOD", "90d")
        for t in [x.strip() for x in tickers.split(",") if x.strip()]:
            run_one(t, period=period)
    except Exception as e:
        print("ERROR:", repr(e), file=sys.stderr); sys.exit(1)
import os, requests

def send_line_message_broadcast(text: str):
    token = os.environ.get("LINE_CHANNEL_TOKEN")
    if not token:
        print("[WARN] LINE_CHANNEL_TOKEN not set; skip LINE broadcast")
        return
    url = "https://api.line.me/v2/bot/message/broadcast"
    headers = {"Authorization": f"Bearer {token}"}
    payload = {"messages": [{"type": "text", "text": text[:4900]}]}  # 5000字制限対策
    r = requests.post(url, json=payload, headers=headers, timeout=10)
    print("[LINE broadcast]", r.status_code, r.text)

