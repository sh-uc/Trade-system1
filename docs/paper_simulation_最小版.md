# paper simulation 最小版

この文書は、`simulate_live.py` の最小版の使い方をまとめたものです。
ここでいう `paper simulation` は、今日以降の毎日運用そのものではなく、
過去データを使って建玉管理や資金管理も含めた実運用寄りの検証を行うためのものです。

---

## 1. 目的

- `strategy_param_sets` の active row を読み込む
- 日足ベースで疑似運用を回す
- `live_positions` と `live_trade_fills` に結果を保存する

---

## 1.1 位置づけ

- `backtest`
  - 1銘柄ごとの条件比較やパラメータ探索に向く
- `paper simulation`
  - 複数銘柄を同時に扱い、資金拘束や建玉管理を含めて過去期間で検証する
- 将来の日次運用モード
  - 当日以降を対象に、前日までの建玉を引き継ぎながら毎日更新する

`paper simulation` は、この3つの中では
「過去期間を、実運用に近いルールで再生する」役割です。

---

## 2. 前提

以下が先に必要です。

1. `tickers` に `lot_size` が入っている
2. `strategy_param_sets`, `live_positions`, `live_trade_fills` が作成済み
3. `strategy_param_sets` に active な初期パラメータが入っている
4. `SUPABASE_URL`, `SUPABASE_KEY` が設定されている

---

## 3. 実行入口

PowerShell からは次で実行する。

```powershell
.\run_simulate_live.ps1 -SimulationName "paper_main" -StrategyName "swing_v1" -Start "2026-01-01" -Reset "1"
```

個別銘柄だけに絞る例:

```powershell
.\run_simulate_live.ps1 -SimulationName "paper_main" -StrategyName "swing_v1" -Start "2026-01-01" -Tickers "3103.T,7013.T"
```

---

## 4. 最小版でやっていること

- active な推奨パラメータセットを読む
- 日足データと指標を事前計算する
- シグナルが出たら翌営業日寄りで買う
- `SL`, `TP`, `TIME`, `REV` を日次で判定する
- `BREAK_EVEN` と `trailing stop` を更新する
- 建玉と約定履歴を Supabase へ保存する

---

## 5. まだやっていないこと

- 決算前停止
- 権利確定日回避
- 複数戦略の資金配分最適化
- 異常時の全停止
- 日次サマリーテーブルへの保存

最小版では、まず「有力銘柄セットで疑似運用を回せること」を優先している。
