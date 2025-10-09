-- ベクトル拡張
create extension if not exists vector;

-- チャンク格納（RAG用）
create table if not exists repo_chunks (
  id uuid primary key default gen_random_uuid(),
  repo text not null,
  branch text not null,
  path text not null,          -- 例: docs/spec.md
  chunk_idx int not null,      -- 0,1,2...
  content text not null,
  embedding vector(1536),      -- text-embedding-3-small
  updated_at timestamptz not null default now(),
  unique (repo, branch, path, chunk_idx)
);

-- 検索高速化（cosine）
create index if not exists idx_repo_chunks_vec
on repo_chunks using ivfflat (embedding vector_cosine_ops);
