# earnings_expected 最小運用

この文書は、少数銘柄について

1. `earnings_actual` を CSV で入れる
2. そこから `earnings_expected` を作る

ための最小運用をまとめたものです。

---

## 1. 用意したもの

- CSV ひな形: `docs/earnings_actual_import_sample.csv`
- actual 取り込み: `import_earnings_actual_csv.py`
- 実行入口: `run_import_earnings_actual_csv.ps1`
- expected 生成: `build_earnings_expected.py`
- 実行入口: `run_build_earnings_expected.ps1`

---

## 2. actual の入れ方

CSV は最低限これだけあればよい。

- `ticker`
- `event_date`

その他:

- `source`
- `source_key`
- `event_label`
- `confidence`
- `note`

実行例:

```powershell
.\run_import_earnings_actual_csv.ps1 -CsvPath "docs/earnings_actual_import_sample.csv"
```

---

## 3. expected の作り方

`build_earnings_expected.py` は、銘柄ごとに
- 同じ月の `earnings_actual`
- 2件以上
を対象に、
- 日付の中央値
- 前 `3` 営業日
- 後 `1` 営業日

で危険ウィンドウを作る。

出力先は `ticker_events.event_type='earnings_expected'`。

実行例:

```powershell
.\run_build_earnings_expected.ps1 -TargetYear "2026" -Tickers "3103.T,7013.T"
```

---

## 4. 初版の割り切り

- 同じ月の実績を1クラスタとして扱う
- 四半期識別はまだしていない
- まずは少数銘柄を手動で育てる

このため、最初は
「危険ウィンドウの土台を作る」
ことを優先する。
