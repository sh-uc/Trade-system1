import json

with open("runs.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 上位20件
top20 = sorted(data, key=lambda x: x["metrics"]["final_equity"], reverse=True)[:20]

# 下位20件
bottom20 = sorted(data, key=lambda x: x["metrics"]["final_equity"])[:20]

print("=== TOP 20 ===")
print(json.dumps(top20, ensure_ascii=False, indent=2))

print("\n=== BOTTOM 20 ===")
print(json.dumps(bottom20, ensure_ascii=False, indent=2))
