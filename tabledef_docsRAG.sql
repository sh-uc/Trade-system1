-- 1) ドキュメントテーブル（本文はテキストで保持）
create table if not exists repo_docs (
  id uuid primary key default gen_random_uuid(),
  repo text not null,
  branch text not null,
  path text not null,          -- 例: docs/spec.md
  title text,                  -- ファイル名から抽出
  filetype text,               -- md, py, yml 等
  content text not null,       -- 本文
  sha text not null,           -- 内容ハッシュ(SHA-256)
  size_bytes int not null,
  committed_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  unique (repo, branch, path)  -- ← 最新状態に上書き(upsert)用
);

create index if not exists idx_repo_docs_repo_branch on repo_docs(repo, branch);
create index if not exists idx_repo_docs_filetype on repo_docs(filetype);

-- 2) RLS（匿名読み取りOK、書き込みはサービスキー専用推奨）
alter table repo_docs enable row level security;

-- 匿名selectは許可（アプリ/将来のRAGから読めるように）
create policy repo_docs_select_public
on repo_docs for select
to anon
using (true);

-- insert/update/delete はサービスロールのみ（Actionsで使用）
-- （サービスロールキーはRLSをバイパスするため、追加policyは不要）
