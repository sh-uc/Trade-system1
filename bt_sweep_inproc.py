# bt_sweep_inproc.py
import os, json, time, itertools
from concurrent.futures import ProcessPoolExecutor, as_completed
from bt_core import fetch_prices, compute_indicators, run_backtest
from db_utils import get_supabase, save_backtest_to_db

SAVE_BT = os.environ.get("SAVE_BT", "0") == "1"


def make_save_cb(ticker: str):
    """ Supabase 保存コールバックを生成 """
    sb = get_supabase()
    def _cb(obj):
        curve  = obj["curve"]
        trades = obj["trades"]
        params = obj["params"]
        result = obj["summary"]
        save_backtest_to_db(sb, ticker, params, result, curve, trades)
    return _cb

def _worker_run(ticker, ind, params, enable_save=False):
    # 各プロセスで DataFrame を再送すると重いので、
    # ここでは “各プロセス起動時に一度だけ” 読み込ませる方法もあるが、
    # シンプルに picklable な DataFrame を渡しても十分速い規模ならOK。
    if enable_save:
        cb = make_save_cb(ticker)
    else:
        cb = None
    m = run_backtest(ind, params, save_cb=cb)
    return {"ticker": ticker, "params": params, "metrics": m}


if __name__ == "__main__":
    # 価格キャッシュ（bt_core.fetch_prices がこの env を参照する想定）
    os.environ.setdefault("PRICE_CACHE_DIR", ".cache/prices")
    os.environ.setdefault("PRICE_CACHE_TTL_DAYS", "14")  # 例：14日以内ならキャッシュ優先

    t0 = time.time()

    tickers_env = os.environ.get("SWEEP_TICKERS") or os.environ.get("SWEEP_TICKER") or "3778.T"
    tickers = [t.strip() for t in tickers_env.split(",") if t.strip()]
    start   = os.environ.get("SWEEP_START", "2023-01-01")
    max_workers = int(os.environ.get("SWEEP_WORKERS", "4"))
    enable_save = os.environ.get("SAVE_BT", "0") == "1"

    # グリッド（必要なら “環境変数レンジ版”に差し替え可能）
    grid = {
        # ★ 追加：資金・1トレード上限・ロットサイズ
        "CAPITAL":       [3_000_000.0],
        "PER_TRADE":     [500_000.0],
        "LOT_SIZE":      [100],

        # ★ RISK_PCT は新レンジに変更
        "RISK_PCT":       [0.003],

        # ここから下は今まで通り（必要なら後でいじる）
        "TAKE_PROFIT_RR": [1.5],
        "MAX_HOLD_DAYS":  [3, 5],
        "STOP_SLIPPAGE":  [0.0025],
        "EXIT_ON_REVERSE":[True],
        "VOL_SPIKE_M":    [1.0, 1.1, 1.2],
        "MACD_ATR_K":     [0.05, 0.1, 0.15],
        "RSI_MIN":        [30, 35.0],
        "RSI_MAX":        [75, 80],
        "GAP_ENTRY_MAX":  [0.08, 0.12],
    }
    # --- env override (single value) ---
    def _env_float(name: str):
        v = os.getenv(name)
        return None if v is None or v == "" else float(v)

    def _env_int(name: str):
        v = os.getenv(name)
        return None if v is None or v == "" else int(v)

    def _env_bool(name: str):
        v = os.getenv(name)
        if v is None or v == "":
            return None
        s = v.strip().lower()
        if s in ("1", "true", "yes", "y", "on"):
            return True
        if s in ("0", "false", "no", "n", "off"):
            return False
        raise ValueError(f"Invalid bool env {name}={v}")

    v = _env_float("RISK_PCT")
    if v is not None:
        grid["RISK_PCT"] = [v]

    v = _env_float("TAKE_PROFIT_RR")
    if v is not None:
        grid["TAKE_PROFIT_RR"] = [v]

    v = _env_int("MAX_HOLD_DAYS")
    if v is not None:
        grid["MAX_HOLD_DAYS"] = [v]

    v = _env_bool("EXIT_ON_REVERSE")
    if v is not None:
        grid["EXIT_ON_REVERSE"] = [v]
    # --- end env override ---
    keys   = list(grid.keys())
    combos = list(itertools.product(*grid.values()))

    # 価格→指標を “ティッカー毎に1回” だけ作る
    
    per_ticker_ind = {}
    for tkr in tickers:
        pr  = fetch_prices(tkr, start)
        ind = compute_indicators(pr)
        per_ticker_ind[tkr] = ind
        time.sleep(float(os.environ.get("YF_SLEEP", "0.7")))
    print(f"[SWEEP] tickers={tickers} combos={len(combos)} workers={max_workers} save={enable_save}")

    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        fut2meta = {}
        for tkr in tickers:
            ind = per_ticker_ind[tkr]
            for combo in combos:
                params = dict(zip(keys, combo))
                fut = ex.submit(_worker_run, tkr, ind, params, enable_save)
                fut2meta[fut] = (tkr, params)

        for i, fut in enumerate(as_completed(fut2meta), start=1):
            row = fut.result()
            tkr     = row["ticker"]
            params  = row["params"]
            metrics = row["metrics"]   # ← ここが正しい
            results.append(row)
            # print(f"[{i}/{len(fut2meta)}] {row['ticker']} done")
            print(f"[{i}/{len(fut2meta)}] {tkr} done  ret={metrics['total_return']:.4%}")

    # 結果ダンプ（必要ならここで Supabase 保存 or 上位抽出）
    # with open("sweep_inproc.json", "w", encoding="utf-8") as f:
    #     json.dump(results, f, ensure_ascii=False, indent=2)

    elapsed = time.time() - t0
    m, s = divmod(elapsed, 60)
    print(f"[SWEEP] finished: {len(results)} runs  time={int(m)}m{s:04.1f}s")
#