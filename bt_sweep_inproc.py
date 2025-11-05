# bt_sweep_inproc.py
import os, json, time, itertools
from concurrent.futures import ProcessPoolExecutor, as_completed
from bt_core import fetch_prices, compute_indicators, run_backtest

def _worker_run(ticker, ind, params):
    # 各プロセスで DataFrame を再送すると重いので、
    # ここでは “各プロセス起動時に一度だけ” 読み込ませる方法もあるが、
    # シンプルに picklable な DataFrame を渡しても十分速い規模ならOK。
    m = run_backtest(ind, params)
    return {"ticker": ticker, "params": params, "metrics": m}

if __name__ == "__main__":
    t0 = time.time()

    tickers_env = os.environ.get("SWEEP_TICKERS") or os.environ.get("SWEEP_TICKER") or "3778.T"
    tickers = [t.strip() for t in tickers_env.split(",") if t.strip()]
    start   = os.environ.get("SWEEP_START", "2023-01-01")
    max_workers = int(os.environ.get("SWEEP_WORKERS", "4"))

    # グリッド（必要なら “環境変数レンジ版”に差し替え可能）
    grid = {
        "RISK_PCT":       [0.007, 0.010],
        "TAKE_PROFIT_RR": [1.2, 1.6, 2.0],
        "MAX_HOLD_DAYS":  [6, 10],
        "STOP_SLIPPAGE":  [0.0025, 0.0040],
        "EXIT_ON_REVERSE":[True],
        "VOL_SPIKE_M":    [1.4, 1.6, 1.8],
        "MACD_ATR_K":     [0.10, 0.15, 0.20],
        "RSI_MIN":        [45.0],
        "RSI_MAX":        [70.0],
        "GAP_ENTRY_MAX":  [0.03, 0.05],
    }
    keys   = list(grid.keys())
    combos = list(itertools.product(*grid.values()))

    # 価格→指標を “ティッカー毎に1回” だけ作る
    per_ticker_ind = {}
    for tkr in tickers:
        pr  = fetch_prices(tkr, start)
        ind = compute_indicators(pr)
        per_ticker_ind[tkr] = ind

    print(f"[SWEEP] tickers={tickers} combos={len(combos)} workers={max_workers}")

    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        fut2meta = {}
        for tkr in tickers:
            ind = per_ticker_ind[tkr]
            for combo in combos:
                params = dict(zip(keys, combo))
                fut = ex.submit(_worker_run, tkr, ind, params)
                fut2meta[fut] = (tkr, params)

        for i, fut in enumerate(as_completed(fut2meta), start=1):
            row = fut.result()
            results.append(row)
            print(f"[{i}/{len(fut2meta)}] {row['ticker']} done")

    # 結果ダンプ（必要ならここで Supabase 保存 or 上位抽出）
    # with open("sweep_inproc.json", "w", encoding="utf-8") as f:
    #     json.dump(results, f, ensure_ascii=False, indent=2)

    elapsed = time.time() - t0
    m, s = divmod(elapsed, 60)
    print(f"[SWEEP] finished: {len(results)} runs  time={int(m)}m{s:04.1f}s")
