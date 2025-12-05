# 実行方法　root directoryにいる状態で、python sweep_inproc_json_select_sort.py
import json

# TARGET_TICKERS = ["6702.T", "3778.T"]
TARGET_TICKERS = ["4506.T", "9600.T"]

with open("sweep_inproc.json", "r", encoding="utf-8") as f:
    data = json.load(f)

for ticker in TARGET_TICKERS:
    # ティッカーごとにフィルタ
    filtered = [run for run in data if run.get("ticker") == ticker]

    if not filtered:
        print(f"=== {ticker} はデータなし ===")
        continue

    # 上位20件
    top20 = sorted(filtered, key=lambda x: x["metrics"]["final_equity"], reverse=True)[:20]

    # 下位20件
    # bottom20 = sorted(filtered, key=lambda x: x["metrics"]["final_equity"])[:20]

    print(f"\n=== TICKER: {ticker} / TOP 20 ===")
    print(json.dumps(top20, ensure_ascii=False, indent=2))

    # print(f"\n=== TICKER: {ticker} / BOTTOM 20 ===")
    # print(json.dumps(bottom20, ensure_ascii=False, indent=2))
