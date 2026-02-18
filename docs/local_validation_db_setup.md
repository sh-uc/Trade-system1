# 本番を触らない検証用DBの作成手順（Supabase/Postgres）

この手順は、**本番 Supabase に接続せず**に、ローカルで DDL を適用して検証を進めるための標準フローです。
`service_role` キーは不要です。

---

## 1. 目的と前提

- 目的
  - DDL の適用可否確認
  - テーブル構造 / 制約 / インデックスの検証
  - SQL 検証や簡易テストを安全に実施
- 前提
  - このリポジトリの DDL を使用する
  - 本番データは使わない（必要ならダミーデータのみ投入）

---

## 2. DDLの適用順（このリポジトリ）

依存関係の観点で、次の順番で適用します。

1. `supabase/ddl/010_backtests_runs.sql`
2. `supabase/ddl/011_backtests_trades.sql`
3. `supabase/ddl/012_view_best_params.sql`
4. `supabase/ddl/020_indicators.sql`
5. `supabase/ddl/030_tickers.sql`
6. `supabase/ddl/040_prices.sql`
7. `supabase/ddl/050_repo_chunks.sql`
8. `supabase/ddl/051_match_repo_chunks.sql`
9. `supabase/ddl/060_repo_docs.sql`

※ `011_backtests_trades.sql` は `backtests_runs` への FK があるため、`010` より後に適用が必須です。

---

## 3. 推奨手順A: Docker + PostgreSQL（最短）

### 3-1. PostgreSQL コンテナ起動

```bash
docker run --name trade-verify-db \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=trade_verify \
  -p 55432:5432 \
  -d postgres:16
```

### 3-2. 接続確認

```bash
psql "postgresql://postgres:postgres@localhost:55432/trade_verify" -c "select version();"
```

### 3-3. DDL一括適用

```bash
for f in \
  supabase/ddl/010_backtests_runs.sql \
  supabase/ddl/011_backtests_trades.sql \
  supabase/ddl/012_view_best_params.sql \
  supabase/ddl/020_indicators.sql \
  supabase/ddl/030_tickers.sql \
  supabase/ddl/040_prices.sql \
  supabase/ddl/050_repo_chunks.sql \
  supabase/ddl/051_match_repo_chunks.sql \
  supabase/ddl/060_repo_docs.sql; do
  echo "Applying: $f"
  psql "postgresql://postgres:postgres@localhost:55432/trade_verify" -v ON_ERROR_STOP=1 -f "$f"
done
```

### 3-4. 適用結果チェック

```bash
psql "postgresql://postgres:postgres@localhost:55432/trade_verify" -c "\dt public.*"
psql "postgresql://postgres:postgres@localhost:55432/trade_verify" -c "\dv public.*"
```

### 3-5. 後片付け

```bash
docker rm -f trade-verify-db
```

---

## 4. 代替手順B: 既存PostgreSQLに作る

Docker が使えない場合は既存 PostgreSQL で空DBを作成し、同じ順に `psql -f` で適用します。

```bash
createdb trade_verify
for f in \
  supabase/ddl/010_backtests_runs.sql \
  supabase/ddl/011_backtests_trades.sql \
  supabase/ddl/012_view_best_params.sql \
  supabase/ddl/020_indicators.sql \
  supabase/ddl/030_tickers.sql \
  supabase/ddl/040_prices.sql \
  supabase/ddl/050_repo_chunks.sql \
  supabase/ddl/051_match_repo_chunks.sql \
  supabase/ddl/060_repo_docs.sql; do
  psql -d trade_verify -v ON_ERROR_STOP=1 -f "$f"
done
```

---

## 5. 最低限の検証チェックリスト

1. テーブル/ビューが作成されている
2. FK が有効（`backtests_trades.run_id -> backtests_runs.id`）
3. 主なインデックスが作成されている
4. ダミーデータ INSERT/SELECT が成功

サンプル:

```sql
insert into backtests_runs (ticker, params, final_equity, total_return, max_drawdown, sharpe, n_trades)
values ('TEST', '{}'::jsonb, 100000, 0.01, 0.02, 1.23, 1)
returning id;

-- 返ってきた id を :run_id に差し替え
insert into backtests_trades (run_id, ts, side, price, qty, reason)
values (:run_id, now(), 'BUY', 1000, 100, null);

select * from backtests_runs order by id desc limit 5;
select * from backtests_trades order by ts desc limit 5;
```

---

## 6. 運用ルール（本番保護）

- 本番 Supabase の資格情報は渡さない
- ローカル検証が通ってから、本番反映は人間が実行
- 本番適用時は別途 migration SQL をレビューしてから実行

---

## 7. 依頼テンプレート（この手順を使う場合）

次回以降は、以下の形式で依頼するとスムーズです。

1. 「このDDLを検証用DBに適用してエラーを潰して」
2. 「最終的な migration SQL を1本にまとめて」
3. 「本番反映手順（ロールバック付き）を出して」

これで `service_role` を共有せずに、DDL 品質を先に固められます。
