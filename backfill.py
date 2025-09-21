# backfill.py —— Supabaseへ過去データを一括投入（リトライ＋フォールバック＋MultiIndex対策）

import os, sys, time
from datetime import timezone, timedelta
import numpy as np
import pandas as pd
import yfinance as yf
from supabase import create_client

JST = timezone(timedelta(hours=9))

# -------- ユーティリティ --------
def chunks(iterable, size=500):
    buf=[]
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

# ★ MultiIndex → 単一レベルに平坦化
def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        new_cols = []
        for c in df.columns:
            if isinstance(c, tuple):
                # 先頭レベルを優先。空文字は無視
                parts = [str(x) for x in c if str(x) != ""]
                new_cols.append(parts[0] if parts else "")
            else:
                new_cols.append(c)
        df = df.copy()
        df.columns = new_cols
    # 重複名を除去
    df = df.loc[:, ~pd.Index(df.columns).duplicated()].copy()
    return df

# ★ TR計算をDataFrame/Series両対応に
def true_range(df_ohlc: pd.DataFrame) -> pd.Series:
    H = df_ohlc["High"];  H = H.iloc[:,0] if isinstance(H, pd.DataFrame) else H
    L = df_ohlc["Low"];   L = L.iloc[:,0] if isinstance(L, pd.DataFrame) else L
    C = df_ohlc["Close"]; C = C.iloc[:,0] if isinstance(C, pd.DataFrame) else C
    pc = C.shift(1)
    a = (H - L).abs()
    b = (H - pc).abs()
    c = (L - pc).abs()
    return pd.concat([a, b, c], axis=1).max(axis=1)

# -------- Yahoo取得（リトライ＋フォールバック）--------
def fetch_yf(ticker: str, period="2y", interval="1d", tries=4, sleep_sec=3) -> pd.DataFrame:
    last_err = None
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
            last_err = e
        time.sleep(sleep_sec)
    raise RuntimeError(f"Failed to download {ticker}. last_err={last_err!r}")

# -------- メイン --------
def main():
    TICKER  = os.environ.get("TICKER", "3778.T")
    PERIOD  = os.environ.get("PERIOD", "2y")
    SUPA_URL = os.environ.get("SUPABASE_URL") or os.environ.get("supabase_url")
    SUPA_KEY = os.environ.get("SUPABASE_KEY") or os.environ.get("supabase_key")
    if not SUPA_URL or not SUPA_KEY:
        print("ERROR: Set SUPABASE_URL and SUPABASE_KEY", file=sys.stderr); sys.exit(1)

    sb = create_client(SUPA_URL, SUPA_KEY)

    # ① 取得 → ★ 平坦化
    df = fetch_yf(TICKER, period=PERIOD, interval="1d")
    df = flatten_columns(df)
    if df.empty:
        print(f"No data for {TICKER}"); sys.exit(1)

    # ② 整形
    raw = (df.rename(columns={"Open":"open","High":"high","Low":"low","Close":"close","Adj Close":"adj_close","Volume":"volume"})
             .dropna().copy())

    # ③ 指標
    ind = raw.copy()
    ind["ma25"]    = ind["close"].rolling(25, min_periods=25).mean()
    ind["ma75"]    = ind["close"].rolling(75, min_periods=75).mean()
    ind["rsi14"]   = rsi(ind["close"], 14)
    macd_line, macd_sig, _ = macd(ind["close"])
    ind["macd"], ind["macd_signal"] = macd_line, macd_sig
    ind["atr14"]   = true_range(df).rolling(14, min_periods=14).mean()
    ind["stdev20"] = ind["close"].rolling(20, min_periods=20).std()

    # 出来高（必ず1次元へ）
    vol_series = pd.Series(np.ravel(np.asarray(ind["volume"])), index=ind.index)
    ind["vol_ma20"]  = vol_series.rolling(20, min_periods=20).mean()
    ind["vol_spike"] = (vol_series >= ind["vol_ma20"]).fillna(False).astype(bool)

    ind["swing_low20"]  = ind["low"].rolling(20, min_periods=20).min()
    ind["swing_high20"] = ind["high"].rolling(20, min_periods=20).max()

    ind = ind.dropna()

    # ④ Supabase upsert（バルク）
    # prices（指標が出た日付に合わせる）
    prices_rows = []
    for ts, r in raw.loc[ind.index].iterrows():
        prices_rows.append({
            "date": ts.date().isoformat(), "code": TICKER,
            "open": float(r.open), "high": float(r.high), "low": float(r.low),
            "close": float(r.close), "volume": int(r.volume),
        })
    for c in chunks(prices_rows, 500):
        sb.table("prices").upsert(c, on_conflict="date,code").execute()

    # indicators
    ind_rows = []
    for ts, r in ind.iterrows():
        ind_rows.append({
            "date": ts.date().isoformat(), "code": TICKER,
            "rsi14": float(r.rsi14), "macd": float(r.macd), "macd_signal": float(r.macd_signal),
            "atr14": float(r.atr14), "stdev20": float(r.stdev20),
            "ma25": float(r.ma25), "ma75": float(r.ma75),
            "vol_ma20": float(r.vol_ma20), "vol_spike": bool(r.vol_spike),
            "swing_low20": float(r.swing_low20), "swing_high20": float(r.swing_high20),
        })
    for c in chunks(ind_rows, 500):
        sb.table("indicators").upsert(c, on_conflict="date,code").execute()

    print(f"Backfilled {TICKER}: prices={len(prices_rows)}, indicators={len(ind_rows)}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR:", repr(e), file=sys.stderr); sys.exit(1)
