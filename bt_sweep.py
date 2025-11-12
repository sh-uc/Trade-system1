# bt_sweep.py
import os, itertools, subprocess, json, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed

def run_once(ticker, start, params):
    env = os.environ.copy()
    env["BT_TICKER"]   = ticker
    env["BT_START"]    = start
    env["BT_CAPITAL"]  = str(params.get("CAPITAL", 1_000_000))
    env["BT_PER_TRADE"]= str(params.get("PER_TRADE", 200_000))
    env["BT_RISK_PCT"] = str(params.get("RISK_PCT", 0.005))
    env["BT_SLIPPAGE"] = str(params.get("SLIPPAGE", 0.0005))
    env["BT_FEE_PCT"]  = str(params.get("FEE_PCT", 0.0))
    env["BT_STOP_SLIP"]= str(params.get("STOP_SLIPPAGE", 0.0015))
    env["BT_TP_RR"]    = str(params.get("TAKE_PROFIT_RR", 2.0))
    env["BT_MAX_HOLD"] = str(params.get("MAX_HOLD_DAYS", 15))
    env["BT_EXIT_REV"] = "1" if params.get("EXIT_ON_REVERSE", True) else "0"
    # 追加
    env["BT_VOL_SPIKE_M"] = str(params.get("VOL_SPIKE_M", 1.4))
    env["BT_MACD_ATR_K"]  = str(params.get("MACD_ATR_K",  0.15))
    env["BT_RSI_MIN"]     = str(params.get("RSI_MIN",     45.0))
    env["BT_RSI_MAX"]     = str(params.get("RSI_MAX",     70.0))
    env["BT_GAP_MAX"]     = str(params.get("GAP_ENTRY_MAX", 0.05))

    # backtest_v2.py をサブプロセス実行して標準出力をパース
    env["PYTHONIOENCODING"] = "utf-8"  # 子のstdout/stderrをUTF-8で出す
    p = subprocess.run(
        [sys.executable, "backtest_v2.py"],
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",     # ← 親のデコード指定
        errors="replace",     # ← もし不正バイトが来ても置換して継続
    )
    out = (p.stdout or "").strip()
    err = (p.stderr or "").strip()
    # 末尾に「[BT2] …」の行がある前提で簡易抽出
    metrics = {"ok": p.returncode == 0, "stdout": out, "stderr": p.stderr.strip()}
    for line in out.splitlines():
        if line.startswith("[BT2] "):
            metrics["line"] = line
    return metrics

if __name__ == "__main__":
    start_time = time.time()  # ← 実行開始時刻を記録

    # 1) ティッカー（単数でも複数でもOK）
    tickers_env = os.environ.get("SWEEP_TICKERS") or os.environ.get("SWEEP_TICKER") or "3778.T"
    tickers = [t.strip() for t in tickers_env.split(",") if t.strip()]
    start   = os.environ.get("SWEEP_START", "2023-01-01")

    # 2) グリッド（必要に応じて環境変数で上書きしてもOK）
    grid = {
        "RISK_PCT":       [0.007, 0.010],
        "TAKE_PROFIT_RR": [1.2, 1.6, 2.0],
        "MAX_HOLD_DAYS":  [6, 10],
        "STOP_SLIPPAGE":  [0.0025, 0.0040],
        "EXIT_ON_REVERSE":[True],          # 固定

        "VOL_SPIKE_M":    [1.4, 1.6, 1.8],
        "MACD_ATR_K":     [0.10, 0.15, 0.20],
        "RSI_MIN":        [45.0],
        "RSI_MAX":        [70.0],
        "GAP_ENTRY_MAX":  [0.03, 0.05],
    }

    keys    = list(grid.keys())
    combos  = list(itertools.product(*grid.values()))
    results = []

    # 3) 並列実行（コア数/回線に合わせて調整。Yahoo対策で 2〜4 推奨）
    max_workers = int(os.environ.get("SWEEP_WORKERS", "4"))
    futures = {}
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        for tkr in tickers:
            for combo in combos:
                params = dict(zip(keys, combo))
                fut = ex.submit(run_once, tkr, start, params)   # ← 既存の run_once をそのまま使用
                futures[fut] = (tkr, params)

        for fut in as_completed(futures):
            tkr, params = futures[fut]
            metrics = fut.result()
            row = {"ticker": tkr, "params": params, "metrics": metrics}
            results.append(row)
            print(json.dumps(row, ensure_ascii=False))  # ストリーム出力

    # 4) 所要時間を出力
    elapsed = time.time() - start_time
    m, s = divmod(elapsed, 60)
    print("\n[SWEEP] All runs completed.")
    print(f"[SWEEP] Total elapsed time: {int(m)} min {s:4.1f} sec")
    print(f"[SWEEP] Results: {len(results)} combinations processed.\n")

    # （必要なら）最後にまとめて再出力
    # for r in results:
    #     print(json.dumps(r, ensure_ascii=False))
#