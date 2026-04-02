# daily_task.py — 毎日実行用：直近データ取得→指標→判定→Supabaseへ保存
import os, sys, time, json
from datetime import timezone, timedelta
import numpy as np
import re
import pandas as pd
import yfinance as yf
from supabase import create_client
from db_utils import resolve_company_name
import datetime as dt
import pytz
import jpholiday

JST = pytz.timezone("Asia/Tokyo")

def is_trading_day_jst(d: dt.date) -> bool:
    if d.weekday() >= 5:  # 土日
        return False
    if jpholiday.is_holiday(d):  # 祝日＋振替休日
        return False
    if (d.month, d.day) in {(12,31),(1,1),(1,2),(1,3)}:
        return False
    return True

today_jst = dt.datetime.now(JST).date()
if not is_trading_day_jst(today_jst):
    print(f"[SKIP] 非取引日: {today_jst.isoformat()}")
    raise SystemExit(0)

# --- add: safe Supabase client creator ---
import re
def fmt_line(value):
    # 3桁区切り & Noneガード
    return f"{value:,.0f}" if isinstance(value, (int, float)) else "-"

def decide_emoji(action: str, pct_change: float | None):
    # 行動と騰落でざっくり絵文字
    if action == "買い":
        return "🟢🛒"
    if pct_change is None:
        return "ℹ️"
    if pct_change >= 0.0:
        return "📈"
    return "📉"

def build_summary_message(date_str: str, results: list[dict]) -> str:
    """
    results: [{ticker, display, close, diff, pct, action, reasons[]}]
    """
    header = f"【日次まとめ】{date_str}\n"
    lines = [header]

    # 1行サマリー（銘柄ごと）
    for r in results:
        em = decide_emoji(r["action"], r["pct"])
        name = r.get("display") or r["ticker"]  # ← 会社名（キャッシュ）を優先
        if r["pct"] is None:
            diff_part = "-"
        else:
            diff_part = f"{r['diff']:+.0f} / {r['pct']:+.2f}%"
        lines.append(
            f"{em} {name}: 終値¥{fmt_line(r['close'])}（前日比 {diff_part}）｜結論: {r['action']}"
        )

    # 詳細（必要なら）
    lines.append("\n— 詳細 —")
    for r in results:
        name = r.get("display") or r["ticker"]
        lines.append(f"〔{name}〕")
        for reason in r["reasons"]:
            lines.append(f"・{reason[:120]}")

    msg = "\n".join(lines)
    return msg[:4900]


def create_supabase_from_env():
    raw_url = (os.environ.get("SUPABASE_URL") or os.environ.get("supabase_url") or "").strip().strip("\"'").strip()
    raw_key = (os.environ.get("SUPABASE_KEY") or os.environ.get("supabase_key") or "").strip().strip("\"'").strip()

    if not raw_url or not raw_key:
        raise ValueError("Missing SUPABASE_URL or SUPABASE_KEY")

    # 参照IDだけが入っていた場合の補完
    if re.fullmatch(r"[a-z0-9]{15,32}", raw_url) and ".supabase.co" not in raw_url:
        raw_url = f"https://{raw_url}.supabase.co"

    if not raw_url.startswith("https://") or ".supabase.co" not in raw_url:
        raise ValueError(f"SUPABASE_URL looks invalid: {raw_url}")

    return create_client(raw_url, raw_key)


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

def upsert_signal(sb, code: str, ts: pd.Timestamp, decision: dict, prev_close, diff_pct):
    date_iso = (ts.tz_convert(JST) if ts.tzinfo else ts.tz_localize(JST)).date().isoformat()
    payload = {
        "date": date_iso,
        "code": code,
        "action": decision["action"],
        "summary": " & ".join([x.split("→")[0]+("OK" if "OK" in x else "NG") for x in decision["reasons"]]),
        "reasons": decision["reasons"],
        "close": decision["close"],
        "prev_close": float(prev_close) if isinstance(prev_close, (int, float)) else None,
        "diff_pct": float(diff_pct) if isinstance(diff_pct, (int, float)) else None,
        "qty": None,
        "sl": None,
        "tp": None,
        "risk_jpy": None,
    }
    sb.table("signals").upsert(payload, on_conflict="date,code").execute()

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

def send_line_message_push(user_id: str, text: str):
    token = os.environ.get("LINE_CHANNEL_TOKEN")
    if not token:
        print("[WARN] LINE_CHANNEL_TOKEN not set; skip LINE push")
        return
    url = "https://api.line.me/v2/bot/message/push"
    headers = {"Authorization": f"Bearer {token}"}
    payload = {"to": user_id, "messages": [{"type": "text", "text": text[:4900]}]}
    r = requests.post(url, json=payload, headers=headers, timeout=10)
    print("[LINE push]", r.status_code, r.text)

def run_one(ticker: str, period="90d"):
    df = fetch_yf(ticker, period=period, interval="1d")
    raw, ind = compute_indicators(df)
    last_ts = ind.index[-1]
    decision = judge_action(ind.iloc[-1])

    # 前日比
    aligned = raw.loc[ind.index]
    close = float(aligned.iloc[-1]["close"])
    prev  = float(aligned.iloc[-2]["close"]) if len(aligned) >= 2 else None
    diff  = (close - prev) if prev is not None else None
    pct   = (diff / prev * 100.0) if prev else None

    # DB 反映
    sb = create_supabase_from_env()
    upsert_prices(sb, ticker, raw, ind.index)
    upsert_indicators(sb, ticker, ind)
    upsert_signal(sb, ticker, last_ts, decision, prev, pct)

    # LINE 本文
 
    date_str = (last_ts.tz_convert(JST) if last_ts.tzinfo else last_ts.tz_localize(JST)).strftime("%Y-%m-%d")
    sb = create_supabase_from_env()
    company = resolve_company_name(ticker, sb)
    title = f"{company}（{ticker}）" if company and company != ticker else ticker

    lines = [
       f"【日次判定】{title} / {date_str}",
        f"終値: ¥{close:,.0f}" + (f"（前日比 {diff:+.0f} / {pct:+.2f}%）" if prev else ""),
        f"結論: {decision['action']}",
       "理由:",
       *decision["reasons"]
    ]
    send_line_message_broadcast("\n".join(lines))

    print(f"OK {ticker}: last={date_str} action={decision['action']}")

if __name__ == "__main__":
    import traceback, re
    ok, ng, results = [], [], []
    try:
        tickers_raw = os.environ.get("TICKERS") or os.environ.get("TICKER", "3778.T")
        period  = os.environ.get("PERIOD", "90d")

        cleaned = tickers_raw.replace("；", ";").replace("，", ",").replace("；", ";")
        cleaned = cleaned.replace(";", ",")
        parts = re.split(r"[,\s]+", cleaned.strip())
        tickers = [p.strip().strip("\"'") for p in parts if p.strip()]

        print(f"[INFO] TICKERS={tickers} PERIOD={period}")

        last_date_str = None

        for t in tickers:
            try:
                print(f"[RUN] {t} ...")
                # === 各銘柄処理 ===
                df = fetch_yf(t, period=period, interval="1d")
                raw, ind = compute_indicators(df)
                last_ts = ind.index[-1]
                last_date_str = (last_ts.tz_convert(JST) if last_ts.tzinfo else last_ts.tz_localize(JST)).strftime("%Y-%m-%d")
                decision = judge_action(ind.iloc[-1])

                aligned = raw.loc[ind.index]
                close = float(aligned.iloc[-1]["close"])
                prev  = float(aligned.iloc[-2]["close"]) if len(aligned) >= 2 else None
                diff  = (close - prev) if prev is not None else None
                pct   = (diff / prev * 100.0) if prev else None

                sb = create_supabase_from_env()
                upsert_prices(sb, t, raw, ind.index)
                upsert_indicators(sb, t, ind)
                upsert_signal(sb, t, last_ts, decision, prev, pct)
                company = resolve_company_name(t, sb)
                display = f"{company}（{t}）" if company and company != t else t

                results.append({
                    "ticker": t,
                    "display": display,
                    "close": close,
                    "diff": diff,
                    "pct": pct,
                    "action": decision["action"],
                    "reasons": decision["reasons"]
                })

                ok.append(t)
                print(f"[OK] {t} {last_date_str} {decision['action']}")
            except Exception as e:
                ng.append((t, repr(e)))
                print(f"[ERROR] {t} failed: {e!r}")
                print(traceback.format_exc())

        # === ここで1通だけ送る ===
        if results:
            text = build_summary_message(last_date_str or "", results)
            uid = os.environ.get("LINE_USER_ID", "").strip()
            if uid:
                send_line_message_push(uid, text)         # push が設定済みなら優先
            else:
                send_line_message_broadcast(text)         # 未設定なら broadcast

        print(f"[SUMMARY] success={ok} failed={ng}")
        sys.exit(0 if ok else 1)
    except Exception as e:
        print("[FATAL]", repr(e))
        print(traceback.format_exc())
        sys.exit(1)
#








