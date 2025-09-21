# backfill.py  —— Supabaseへ過去データを一括投入（リトライ付き）
# 使い方（Windows PowerShell 例）:
#   .\.venv\Scripts\Activate.ps1
#   pip install -r requirements.txt
#   $env:SUPABASE_URL="https://xxxxxx.supabase.co"
#   $env:SUPABASE_KEY="xxxxxx"         # anon でも可（RLSにinsert/update許可が必要）
#   $env:TICKER="3778.T"               # 省略時は 3778.T
#   $env:PERIOD="2y"                   # 省略時は 2y
#   python backfill.py

import os
import sys
import time
from datetime import timezone, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
from supabase import create_client

JST = timezone(timedelta(hours=9))

# ====== 安全ユーティリティ ======
def chunks(iterable, size=500):
    buf = []
    for x in iterable:
        buf.append(x)
        if len(buf) >= size:
            yield buf
            buf = []
    if buf:
        yield buf

def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    d = close.diff()
    up = pd.Series(np.where(d > 0, d, 0.0), index=close.index)
    dn = pd.Series(np.where(d < 0, -d, 0.0), index=close.index)
    ru = up.ewm(span=period, adjust=False).mean()
    rd = dn.ewm(span=period, adjust=False).mean()
    rs = ru / (rd + 1e-12)
    return 100.0 - 100.0 / (1.0 + rs)

def macd(close: pd.Series, fast=12, slow=26, sig=9):
    m = ema(close, fast) - ema(close, slow)
    return m, ema(m, sig), m - ema(m, sig)

def true_range(df_ohlc: pd.DataFrame) -> pd.Series:
    pc = df_ohlc["Close"].shift(1)
    a = (df_ohlc["High"] - df_ohlc["Low"]).abs()
    b = (df_ohlc["High"] - pc).abs()
    c = (df_ohlc["Low"] - pc).abs()
    return pd.concat([a, b, c], axis=1).max(axis=1)

# ====== Yahooから堅牢に取得（リトライ + フォールバック）======
def fetch_yf(ticker: str, period="2y", interval="1d", tries=4, sleep_sec=3) -> pd.DataFrame:
    last_err = None
    for i in range(tries):
        try:
            # ①標準API
            df = yf.download(
                ticker,
                period=period,
                interval=interval,
                progress=False,
                auto_adjust=False,
                threads=False,
            )
            if isinstance(df, pd.DataFrame) and not df.empty:
                return df

            # ②Ticker.history フォールバック
            t = yf.Ticker(ticker)
            df = t.history(period=period, interval=interval, auto_adjust=False)
            if isinstance(df, pd.DataFrame) and not df.empty:
                # history側は列名が小文字の場合に備えて揃える
                if "Open" not in df.columns and "open" in df.columns:
                    df = df.rename(
                        columns={
                            "open": "Open",
                            "high": "High",
                            "low": "Low",
                            "close": "Close",
                            "volume": "Volume",
                        }
                    )
                return df
        except Exception as e:
            last_err = e
        time.sleep(sleep_sec)
    raise RuntimeError(f"Failed to download {ticker}. last_err={last_err!r}")

# ====== メイン処理 ======
def main():
    # ---- 環境変数から設定 ----
    TICKER = os.environ.get("TICKER", "3778.T")
    PERIOD = os.environ.get("PERIOD", "2y")
    SUPA_URL = os.environ.get("SUPABASE_URL") or os.environ.get("supabase_url")
    SUPA_KEY = os.environ.get("SUPABASE_KEY") or os.environ.get("supabase_key")

    if not SUPA_URL or not SUPA_KEY:
        print("ERROR: Set SUPABASE_URL / SUPABASE_KEY environment variables.", file=sys.stderr)
        sys.exit(1)

    # ---- Supabase 接続 ----
    sb = create_client(SUPA_URL, SUPA_KEY)

    # ---- 価格取得（リトライ付き）----
    df = fetch_yf(TICKER, period=PERIOD, interval="1d")
    if df.empty:
        print(f"No data for {TICKER}")
        sys.exit(1)

    # ---- 整形 ----
    raw = (
        df.rename(
            columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Adj Close": "adj_close",
                "Volume": "volume",
            }
        )
        .dropna()
        .copy()
    )

    # ---- 指標計算 ----
    ind = raw.copy()
    ind["ma25"] = ind["close"].rolling(25, min_periods=25).mean()
    ind["ma75"] = ind["close"].rolling(75, min_periods=75).mean()
    ind["rsi14"] = rsi(ind["close"], 14)
    macd_line, macd_sig, _ = macd(ind["close"])
    ind["macd"], ind["macd_signal"] = macd_line, macd_sig
    ind["atr14"] = true_range(df).rolling(14, min_periods=14).mean()
    ind["stdev20"] = ind["close"].rolling(20, min_periods=20).std()

    # 出来高系（必ず1次元に正規化）
    vol_series = pd.Series(np.ravel(np.asarray(ind["volume"])), index=ind.index)
    ind["vol_ma20"] = vol_series.rolling(20, min_periods=20).mean()
    ind["vol_spike"] = (vol_series >= ind["vol_ma20"]).fillna(False).astype(bool)

    ind["swing_low20"] = ind["low"].rolling(20, min_periods=20).min()
    ind["swing_high20"] = ind["high"].rolling(20, min_periods=20).max()

    # 指標が出そろった日以降に合わせる
    ind = ind.dropna()

    # ---- Supabase upsert（バルク）----
    # prices：指標のある日付に揃えて保存
    prices_rows = []
    for ts, r in raw.loc[ind.index].iterrows():
        prices_rows.append(
            {
                "date": ts.date().isoformat(),
                "code": TICKER,
                "open": float(r.open),
                "high": float(r.high),
                "low": float(r.low),
                "close": float(r.close),
                "volume": int(r.volume),
            }
        )
    for c in chunks(prices_rows, size=500):
        sb.table("prices").upsert(c, on_conflict="date,code").execute()

    # indicators
    ind_rows = []
    for ts, r in ind.iterrows():
        ind_rows.append(
            {
                "date": ts.date().isoformat(),
                "code": TICKER,
                "rsi14": float(r.rsi14),
                "macd": float(r.macd),
                "macd_signal": float(r.macd_signal),
                "atr14": float(r.atr14),
                "stdev20": float(r.stdev20),
                "ma25": float(r.ma25),
                "ma75": float(r.ma75),
                "vol_ma20": float(r.vol_ma20),
                "vol_spike": bool(r.vol_spike),
                "swing_low20": float(r.swing_low20),
                "swing_high20": float(r.swing_high20),
            }
        )
    for c in chunks(ind_rows, size=500):
        sb.table("indicators").upsert(c, on_conflict="date,code").execute()

    print(f"Backfilled {TICKER}: prices={len(prices_rows)}, indicators={len(ind_rows)}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR:", repr(e), file=sys.stderr)
        sys.exit(1)
