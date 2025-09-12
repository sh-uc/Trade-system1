# -*- coding: utf-8 -*-
"""
MVP: 引け判定・損失率0.5% さくらインターネット（3778）売買支援アプリ（**Supabase/Postgres対応版**）

・Streamlit 1ファイルで動作（ローカル/Streamlit Cloud）
・DBは任意：SupabaseのSecretsが設定されていれば保存・参照が有効化されます
・後から UI で「判定タイミング」「許容損失率」など変更可能

依存:
  streamlit, pandas, numpy, yfinance, plotly, supabase, python-dotenv（任意）
"""

import os
import math
import json
import datetime as dt
from datetime import timezone, timedelta
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import streamlit as st

try:
    from supabase import create_client, Client  # supabase-py v2
except Exception:
    create_client = None
    Client = None

JST = timezone(timedelta(hours=9))

# ------------------------------
# 設定（UIで変更可能）
# ------------------------------
DEFAULTS = {
    "ticker": "3778.T",             # さくらインターネット（東証）
    "capital": 1_000_000,            # 総資金
    "per_trade_cap": 200_000,        # 1回の最大発注金額
    "risk_pct": 0.005,               # 許容損失率（=0.5%）
    "decision_mode": "close",       # "close"（引け判定） or "open"（朝判定/翌寄付）
    "lookback_days": 500,            # 取得日数目安
    "atr_mult_sl": 1.5,              # SLのATR倍率
    "r_multiple_take": 2.0,          # 利確 2R
}

# ------------------------------
# Supabase クライアント
# ------------------------------

def get_supabase() -> Optional[Client]:
    try:
        url = st.secrets.get("supabase", {}).get("url")
        key = st.secrets.get("supabase", {}).get("key")
        if not url or not key or create_client is None:
            return None
        return create_client(url, key)
    except Exception:
        return None

# ------------------------------
# データ取得・指標
# ------------------------------

def jst_now():
    return dt.datetime.now(JST)


def fetch_ohlc(ticker: str, lookback_days: int = 500) -> pd.DataFrame:
    period = f"{lookback_days}d"
    df = yf.download(ticker, period=period, interval="1d", auto_adjust=False, progress=False)
    if df.empty:
        return df
    df = df.rename(columns={"Open":"open","High":"high","Low":"low","Close":"close","Adj Close":"adj_close","Volume":"volume"})
    df.index = pd.to_datetime(df.index)
    df = df.dropna()
    return df


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)

    # numpy配列を安全に1次元にしてから Series 化
    up_series = pd.Series(up.ravel(), index=series.index)
    down_series = pd.Series(down.ravel(), index=series.index)

    roll_up = up_series.ewm(span=period, adjust=False).mean()
    roll_down = down_series.ewm(span=period, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-12)
    return 100.0 - (100.0 / (1.0 + rs))



def macd(series: pd.Series, fast=12, slow=26, signal=9):
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df['close'].shift(1)
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - prev_close).abs(),
        (df['low'] - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tr = true_range(df)
    return tr.rolling(period).mean()


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out['ma25'] = out['close'].rolling(25, min_periods=25).mean()
    out['ma75'] = out['close'].rolling(75, min_periods=75).mean()
    out['rsi14'] = rsi(out['close'], 14)
    macd_line, signal_line, hist = macd(out['close'])
    out['macd'] = macd_line
    out['macd_signal'] = signal_line
    out['atr14'] = atr(out, 14)
    out['stdev20'] = out['close'].rolling(20, min_periods=20).std()

# volume を必ず 1次元 Series 化（(n,1) DataFrame や ndarray でもOKにする）
    if 'volume' in out.columns:
        vol_values = np.ravel(np.asarray(out['volume']))  # どんな形でも1次元に
    else:
    # 念のため: volume列が無い場合は欠損で埋める
        vol_values = np.full(len(out), np.nan, dtype=float)

    # 🔽 volumeを安全に数値化してから平均と比較を行う
    vol_series = pd.Series(vol_values, index=out.index)
    vol_series = pd.to_numeric(vol_series, errors='coerce')  # 数値化（文字列/NaN吸収）

    out['vol_ma20']  = vol_series.rolling(20, min_periods=20).mean()
    out['vol_spike'] = (vol_series >= out['vol_ma20']).fillna(False).astype(bool)   # ← 追加

    out['swing_low20'] = out['low'].rolling(20, min_periods=20).min()
    out['swing_high20'] = out['high'].rolling(20, min_periods=20).max()
    return out.dropna()

# ------------------------------
# ルール評価 & サイジング
# ------------------------------

def long_signal_row(row: pd.Series) -> Dict[str, Any]:
    close = float(row['close'])
    ma25 = float(row['ma25'])
    ma75 = float(row['ma75'])
    macd_val = float(row['macd'])
    macd_sig = float(row['macd_signal'])
    rsi14 = float(row['rsi14'])
    atr14 = float(row['atr14'])
    # vol_spike を確実にスカラ bool へ
    _vs = row['vol_spike']
    if isinstance(_vs, (pd.Series, np.ndarray, list)):
        arr = np.ravel(np.asarray(_vs))
        vol_spike = bool(arr[0]) if arr.size > 0 else False
    else:
        vol_spike = bool(_vs)


    cond_trend = (close > ma25) and (ma25 >= ma75)
    cond_momentum = (macd_val > macd_sig)
    cond_rsi = (40.0 <= rsi14 <= 68.0)
    cond_volume = vol_spike

    passed = all([cond_trend, cond_momentum, cond_rsi, cond_volume])

    reasons = []
    reasons.append(f"トレンド: close({close:.1f}) > MA25({ma25:.1f}) & MA25≥MA75 → {'OK' if cond_trend else 'NG'}")
    reasons.append(f"モメンタム: MACD({macd_val:.3f}) > Signal({macd_sig:.3f}) → {'OK' if cond_momentum else 'NG'}")
    reasons.append(f"過熱回避: RSI14={rsi14:.1f} ∈ [40,68] → {'OK' if cond_rsi else 'NG'}")
    reasons.append(f"出来高: 20日平均以上 → {'OK' if cond_volume else 'NG'}")

    return {
        "passed": passed,
        "reasons": reasons,
        "close": close,
        "atr14": atr14,
        "swing_low20": float(row['swing_low20'])
    }


def position_sizing(close: float, atr14: float, swing_low20: float, settings: dict) -> Dict[str, Any]:
    atr_mult = settings.get("atr_mult_sl", 1.5)
    per_trade_cap = settings.get("per_trade_cap", 200_000)
    capital = settings.get("capital", 1_000_000)
    risk_pct = settings.get("risk_pct", 0.005)

    sl_by_atr = close - atr14 * atr_mult
    sl_by_swing = swing_low20
    sl = min(sl_by_atr, sl_by_swing)
    sl = max(1.0, sl)
    sl_dist = max(1.0, close - sl)

    allow_risk_jpy = max(1.0, capital * risk_pct)

    qty_by_risk = math.floor(allow_risk_jpy / sl_dist)
    qty_by_cash = math.floor(per_trade_cap / close)
    qty = max(0, min(qty_by_risk, qty_by_cash))

    r_multiple = settings.get("r_multiple_take", 2.0)
    tp = close + sl_dist * r_multiple

    return {"sl": sl, "sl_dist": sl_dist, "qty": qty, "risk_jpy": sl_dist * qty, "tp": tp}


def judge_action(df: pd.DataFrame, settings: dict) -> Dict[str, Any]:
    last = df.iloc[-1]
    sig = long_signal_row(last)

    if not sig["passed"]:
        return {"action": "見送り", "summary": "ルール合致せず（買い条件不成立）", "reasons": sig["reasons"], "close": sig["close"]}

    ps = position_sizing(sig['close'], sig['atr14'], sig['swing_low20'], settings)
    if ps['qty'] <= 0:
        return {"action": "見送り", "summary": "許容損失率・資金制約内で建て玉が立たない", "reasons": sig["reasons"], "close": sig["close"]}

    return {"action": "買い", "summary": "買い条件成立。ATRベースで数量・SL/TPを設定", "reasons": sig["reasons"], "close": sig["close"], "qty": ps['qty'], "sl": ps['sl'], "tp": ps['tp'], "risk_jpy": ps['risk_jpy']}

# ------------------------------
# Supabase 保存/読取
# ------------------------------

def upsert_prices(sb: Client, code: str, df: pd.DataFrame):
    if sb is None or df.empty:
        return
    rows = []
    for ts, r in df.tail(1).iterrows():  # 直近1行のみアップサート
        rows.append({
            "date": ts.date().isoformat(),
            "code": code,
            "open": float(r['open']),
            "high": float(r['high']),
            "low": float(r['low']),
            "close": float(r['close']),
            "volume": int(r['volume'])
        })
    if rows:
        sb.table("prices").upsert(rows, on_conflict="date,code").execute()


def upsert_indicators(sb: Client, code: str, ind: pd.DataFrame):
    if sb is None or ind.empty:
        return
    r = ind.iloc[-1]
    row = {
        "date": ind.index[-1].date().isoformat(),
        "code": code,
        "rsi14": float(r['rsi14']),
        "macd": float(r['macd']),
        "macd_signal": float(r['macd_signal']),
        "atr14": float(r['atr14']),
        "stdev20": float(r['stdev20']),
        "ma25": float(r['ma25']),
        "ma75": float(r['ma75']),
        "vol_ma20": float(r['vol_ma20']),
        "vol_spike": bool(r['vol_spike']),
        "swing_low20": float(r['swing_low20']),
        "swing_high20": float(r['swing_high20']),
    }
    sb.table("indicators").upsert(row, on_conflict="date,code").execute()


def upsert_signal(sb: Client, code: str, date_iso: str, dec: Dict[str, Any]):
    if sb is None:
        return
    row = {
        "date": date_iso,
        "code": code,
        "action": dec.get("action"),
        "summary": dec.get("summary"),
        "reasons": json.dumps(dec.get("reasons", [])),
        "close": float(dec.get("close", 0.0)),
        "qty": int(dec.get("qty", 0)) if dec.get("qty") is not None else None,
        "sl": float(dec.get("sl", 0.0)) if dec.get("sl") is not None else None,
        "tp": float(dec.get("tp", 0.0)) if dec.get("tp") is not None else None,
        "risk_jpy": float(dec.get("risk_jpy", 0.0)) if dec.get("risk_jpy") is not None else None,
    }
    sb.table("signals").upsert(row, on_conflict="date,code").execute()


def fetch_recent_signals(sb: Client, code: str, limit: int = 20) -> pd.DataFrame:
    if sb is None:
        return pd.DataFrame()
    res = sb.table("signals").select("*").eq("code", code).order("date", desc=True).limit(limit).execute()
    data = res.data or []
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    if 'reasons' in df.columns:
        df['reasons'] = df['reasons'].apply(lambda x: x if isinstance(x, list) else json.loads(x) if x else [])
    return df

# ------------------------------
# Streamlit UI
# ------------------------------

st.set_page_config(page_title="3778 売買支援（引け判定MVP/DB対応）", layout="wide")

st.title("さくらインターネット（3778） 売買支援MVP / Supabase対応")

with st.sidebar:
    st.header("設定")
    ticker = st.text_input("ティッカー（Yahoo形式）", value=DEFAULTS['ticker'])
    capital = st.number_input("総資金(¥)", value=DEFAULTS['capital'], step=10000)
    per_trade_cap = st.number_input("1回の最大発注(¥)", value=DEFAULTS['per_trade_cap'], step=10000)
    risk_pct = st.slider("許容損失率(%)", min_value=0.1, max_value=2.0, value=DEFAULTS['risk_pct']*100, step=0.1)
    decision_mode = st.selectbox("判定タイミング", options=["close","open"], index=0)

    atr_mult_sl = st.slider("SLのATR倍率", min_value=1.0, max_value=3.0, value=DEFAULTS['atr_mult_sl'], step=0.1)
    r_multiple_take = st.slider("利確R（倍）", min_value=1.0, max_value=4.0, value=DEFAULTS['r_multiple_take'], step=0.5)

    enable_db = st.checkbox("DB保存を有効化（Supabase secrets 必須）", value=True)

    settings = {
        "capital": int(capital),
        "per_trade_cap": int(per_trade_cap),
        "risk_pct": float(risk_pct)/100.0,
        "decision_mode": decision_mode,
        "atr_mult_sl": float(atr_mult_sl),
        "r_multiple_take": float(r_multiple_take),
    }

# データ取得 & 指標
with st.spinner("データ取得中..."):
    raw = fetch_ohlc(ticker, DEFAULTS['lookback_days'])

if raw.empty:
    st.error("価格データの取得に失敗しました。ティッカーやネットワークをご確認ください。")
    st.stop()

ind = compute_indicators(raw)
latest_ts = ind.index[-1]
latest_day = latest_ts.astimezone(JST) if latest_ts.tzinfo else latest_ts.tz_localize(JST)

# 判定
decision = judge_action(ind, settings)

# Supabase 連携
sb = get_supabase() if enable_db else None
col_sb1, col_sb2 = st.columns(2)
with col_sb1:
    st.metric("DB接続", "OK" if sb else "未設定/無効")
with col_sb2:
    st.metric("判定タイミング", "引け" if settings['decision_mode']=="close" else "朝(翌寄付)")

if sb is None and enable_db:
    st.warning("SupabaseのURL/KEYが未設定です。.streamlit/secrets.toml を設定してください。")

# 保存実行（最新のみ）
if sb is not None:
    code = ticker
    upsert_prices(sb, code, raw)
    upsert_indicators(sb, code, ind)
    upsert_signal(sb, code, latest_day.date().isoformat(), decision)

# 1. 今日の結論カード
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("今日の結論", decision.get("action", "-"))
with col2:
    st.metric("終値(最新)", f"{decision.get('close', float(ind.iloc[-1]['close'])):,.0f} 円")
with col3:
    st.metric("本日(判定日)", latest_day.strftime("%Y-%m-%d"))
with col4:
    st.metric("許容損失率", f"{settings['risk_pct']*100:.1f}%")

st.subheader("根拠（ルールチェック）")
for r in decision.get("reasons", []):
    st.write("- ", r)

if decision.get("action") == "買い":
    # f-stringや全角スペース混入によるSyntaxError回避
    qty = int(decision.get("qty", 0))
    risk_jpy = float(decision.get("risk_jpy", 0.0))
    sl = float(decision.get("sl", 0.0))
    tp = float(decision.get("tp", 0.0))
    msg = (
        "数量: {qty} 株 / 予想リスク: ¥{risk:,.0f}\n"
        "\n"
        "SL: ¥{sl:,.1f} / TP(目安): ¥{tp:,.1f}"
    ).format(qty=qty, risk=risk_jpy, sl=sl, tp=tp)
    st.success(msg)
else:
    st.info(decision.get("summary", "-"))
    
# 2. チャート（ローソク＋MA）
st.subheader("価格チャート")
plot_df = ind.tail(250)
fig = go.Figure()
fig.add_trace(go.Candlestick(x=plot_df.index, open=plot_df['open'], high=plot_df['high'], low=plot_df['low'], close=plot_df['close'], name='OHLC'))
fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['ma25'], name='MA25'))
fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['ma75'], name='MA75'))
st.plotly_chart(fig, use_container_width=True)

# 3. サブ指標（RSI, MACD, ATR）
st.subheader("サブ指標")
col_a, col_b, col_c = st.columns(3)
with col_a:
    st.line_chart(plot_df['rsi14'])
with col_b:
    st.line_chart(plot_df[['macd','macd_signal']])
with col_c:
    st.line_chart(plot_df['atr14'])

# 4. 直近シグナル履歴（DB）
st.subheader("直近シグナル履歴（DB保存）")
if sb is not None:
    sig_df = fetch_recent_signals(sb, ticker, limit=20)
    if not sig_df.empty:
        show = sig_df[["date","action","summary","close","qty","sl","tp","risk_jpy"]].copy()
        st.dataframe(show)
    else:
        st.caption("DBにシグナル履歴がまだありません。初回保存後に表示されます。")
else:
    st.caption("DB未接続のため履歴は表示しません。サイドバーからDB保存を有効化し、secretsを設定してください。")

st.caption("※ 本ツールは投資助言ではなく参考情報の提供です。投資判断はご自身の責任でお願いします。")
