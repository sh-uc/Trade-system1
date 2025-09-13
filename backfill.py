# backfill.py  —— さくら(3778.T)の過去データをSupabaseへ一括投入
import os, json, math, numpy as np, pandas as pd, yfinance as yf
from datetime import timezone, timedelta
JST = timezone(timedelta(hours=9))

# ===== Supabase接続 =====
from supabase import create_client
SUPA_URL = os.environ.get("SUPABASE_URL") or os.environ.get("SUPABASE_URL".lower())
SUPA_KEY = os.environ.get("SUPABASE_KEY") or os.environ.get("SUPABASE_KEY".lower())
assert SUPA_URL and SUPA_KEY, "Set SUPABASE_URL / SUPABASE_KEY as env vars"
sb = create_client(SUPA_URL, SUPA_KEY)

# ===== 指標 =====
def ema(s, span): return s.ewm(span=span, adjust=False).mean()
def rsi(close, period=14):
    d = close.diff()
    up = pd.Series(np.where(d>0, d, 0.0), index=close.index)
    dn = pd.Series(np.where(d<0, -d, 0.0), index=close.index)
    ru = up.ewm(span=period, adjust=False).mean()
    rd = dn.ewm(span=period, adjust=False).mean()
    rs = ru/(rd+1e-12)
    return 100 - 100/(1+rs)
def macd(close, f=12, s=26, sig=9):
    m = ema(close,f)-ema(close,s)
    return m, ema(m,sig), m-ema(m,sig)
def tr(df):
    pc = df["Close"].shift(1)
    return pd.concat([(df["High"]-df["Low"]).abs(), (df["High"]-pc).abs(), (df["Low"]-pc).abs()],axis=1).max(axis=1)

# ===== 取得 & 計算 =====
ticker = os.environ.get("TICKER","3778.T")
df = yf.download(ticker, period="2y", interval="1d", auto_adjust=False, progress=False)
if df.empty: raise SystemExit("No data")

raw = df.rename(columns={"Open":"open","High":"high","Low":"low","Close":"close","Adj Close":"adj_close","Volume":"volume"}).dropna().copy()
ind = raw.copy()
ind["ma25"] = ind["close"].rolling(25, min_periods=25).mean()
ind["ma75"] = ind["close"].rolling(75, min_periods=75).mean()
ind["rsi14"] = rsi(ind["close"],14)
macd_line, macd_sig, _ = macd(ind["close"])
ind["macd"], ind["macd_signal"] = macd_line, macd_sig
ind["atr14"] = tr(df).rolling(14, min_periods=14).mean()
ind["stdev20"] = ind["close"].rolling(20, min_periods=20).std()
vol = pd.Series(np.ravel(np.asarray(ind["volume"])), index=ind.index)
ind["vol_ma20"] = vol.rolling(20, min_periods=20).mean()
ind["vol_spike"] = (vol >= ind["vol_ma20"]).fillna(False).astype(bool)
ind["swing_low20"]  = ind["low"].rolling(20, min_periods=20).min()
ind["swing_high20"] = ind["high"].rolling(20, min_periods=20).max()
ind = ind.dropna()

# ===== Supabaseへ upsert（バルク）=====
def chunks(iterable, size=500):
    buf=[]; 
    for x in iterable:
        buf.append(x)
        if len(buf)>=size:
            yield buf; buf=[]
    if buf: yield buf

# prices
rows = []
for ts, r in raw.loc[ind.index].iterrows():  # 指標が出た日以降に合わせる
    rows.append({"date": ts.date().isoformat(), "code": ticker,
                 "open": float(r.open), "high": float(r.high), "low": float(r.low),
                 "close": float(r.close), "volume": int(r.volume)})
for c in chunks(rows): sb.table("prices").upsert(c, on_conflict="date,code").execute()

# indicators
rows=[]
for ts, r in ind.iterrows():
    rows.append({"date": ts.date().isoformat(), "code": ticker,
      "rsi14": float(r.rsi14), "macd": float(r.macd), "macd_signal": float(r.macd_signal),
      "atr14": float(r.atr14), "stdev20": float(r.stdev20), "ma25": float(r.ma25), "ma75": float(r.ma75),
      "vol_ma20": float(r.vol_ma20), "vol_spike": bool(r.vol_spike),
      "swing_low20": float(r.swing_low20), "swing_high20": float(r.swing_high20)})
for c in chunks(rows): sb.table("indicators").upsert(c, on_conflict="date,code").execute()

print(f"Backfilled {ticker}: prices={len(raw.loc[ind.index])}, indicators={len(ind)}")
