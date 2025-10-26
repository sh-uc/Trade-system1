# bt_sweep.py
import os, itertools, subprocess, json, sys

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

    # backtest_v2.py をサブプロセス実行して標準出力をパース
    p = subprocess.run([sys.executable, "backtest_v2.py"], env=env, capture_output=True, text=True)
    out = p.stdout.strip()
    # 末尾に「[BT2] …」の行がある前提で簡易抽出
    metrics = {"ok": p.returncode == 0, "stdout": out, "stderr": p.stderr.strip()}
    for line in out.splitlines():
        if line.startswith("[BT2] "):
            metrics["line"] = line
    return metrics

if __name__ == "__main__":
    # ティッカー別の網羅（まずは 3778.T を重点確認）
    ticker = os.environ.get("SWEEP_TICKER", "3778.T")
    start  = os.environ.get("SWEEP_START", "2023-01-01")

    grid = {
        "RISK_PCT":       [0.005, 0.007, 0.010],
        "TAKE_PROFIT_RR": [1.2, 1.6, 2.0],
        "MAX_HOLD_DAYS":  [6, 10, 15],
        "STOP_SLIPPAGE":  [0.0015, 0.0025, 0.0040],
        "EXIT_ON_REVERSE":[True],  # 固定
    }

    combos = list(itertools.product(*grid.values()))
    keys = list(grid.keys())
    results = []

    for combo in combos:
        params = dict(zip(keys, combo))
        m = run_once(ticker, start, params)
        # 一行サマリに主要指標を乗せる（簡易正規表現でもOKだが今回はそのまま保持）
        results.append({"params": params, "metrics": m})

    # 画面に見やすくダンプ
    for r in results:
        print(json.dumps(r, ensure_ascii=False))
