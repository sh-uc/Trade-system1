# ticker_events 設計メモ

この文書は、決算日や配当権利落ち日などのイベントを保持する
`ticker_events` テーブルの初版設計をまとめたものです。

目的は、将来の
- 決算前後の新規停止
- 決算前の強制決済
- 配当権利落ち日前後の回避
- 異常時の手動停止

を、銘柄ごとの時系列イベントとして扱えるようにすることです。

---

## 1. なぜ `tickers` ではなく別テーブルか

決算日や権利関連の日付は、銘柄に対して1つだけある固定属性ではありません。

- 決算日は年に複数回ある
- 配当権利落ち日も複数回ある
- 将来は `manual_stop` のような手動イベントも入れたい

このため、`tickers` に列を足していくより、
`ticker_events` として時系列で保持する方が自然です。

---

## 2. 初版で扱うイベント

初版では次の3つを主対象にする。

- `earnings_actual`
  - 過去実績または正式予定日として扱う決算イベント
- `earnings_expected`
  - 過去実績から推定した危険ウィンドウの中心日
- `ex_dividend`
  - 権利落ち日ベースの配当イベント

将来拡張候補:

- `record_date`
- `rights`
- `manual_stop`
- `manual_note`

---

## 3. 主なカラム

- `ticker`
  - 銘柄コード。`tickers.code` を参照
- `event_type`
  - `earnings_actual`, `earnings_expected`, `ex_dividend` など
- `event_date`
  - イベント日付。運用ロジックで最も使いやすい主キー的日付
- `event_ts`
  - 時刻まで取れる場合だけ保持
- `source`
  - 取得元。初版では `yfinance`
- `source_key`
  - 取得元の一意キーがある場合に保持
- `event_label`
  - 表示用ラベル
- `event_value`
  - 配当額など、数値情報がある場合に保持
- `currency`
  - 通貨
- `confidence`
  - `low`, `medium`, `high`
- `is_active`
  - 無効化用フラグ
- `meta`
  - 取得時の補助情報を JSON で保持

---

## 4. `yfinance` との対応

### 4.1 決算実績 / 決算予定

`yfinance` では、銘柄によって
- `Ticker.calendar`
- `Ticker.get_earnings_dates()`

などから決算関連情報が取れることがある。

ただし、future earnings dates は不安定なことがあるため、
初版では次の扱いにする。

- 取れたら `earnings_actual` として入れる
- 取れない場合は無理に補完しない
- `confidence='medium'`

### 4.1.1 推定危険ウィンドウ

将来は、過去の `earnings_actual` から
`earnings_expected` を生成する。

この `earnings_expected` は、
- 次回危険ウィンドウの中心日
- `meta.window_start`
- `meta.window_end`
を持つ想定。

### 4.2 配当権利落ち日

`Ticker.dividends` は、基本的に `ex-dividend date` ベースで扱える。
このため初版では、

- `dividends` の index を `event_date`
- 値を `event_value`

として `ex_dividend` を入れる。

---

## 5. 運用ロジックでの使い方

将来の `paper simulation` / 日次運用モードでは、例えば次のように使う。

- `earnings_actual`
  - `event_date` の N 営業日前から新規停止
  - `event_date` 前日引けで強制決済
- `earnings_expected`
  - 正式予定日がない場合の代替危険ウィンドウとして使う
- `ex_dividend`
  - 権利落ち日をまたぎたくない場合は新規停止
  - 将来、配当狙い保有との切替にも使える
- `manual_stop`
  - 急変時に特定銘柄を停止

---

## 6. 初版の割り切り

初版では次を割り切る。

- 日本株の正式な権利確定日や権利付最終日はまだ入れない
- `yfinance` で取れない決算予定日は空でもよしとする
- まずは `earnings_actual` と `ex_dividend` を Supabase に蓄積できることを優先する
- `earnings_expected` は次段階で生成する

---

## 7. 次の実装

次に必要なのはこれです。

1. `083_ticker_events.sql` を適用する
2. `sync_ticker_events.py` で `yfinance` から `earnings_actual` / `ex_dividend` を取り込む
3. 過去実績から `earnings_expected` を生成する
4. `paper simulation` 側で `ticker_events` を参照して停止ルールへつなげる
