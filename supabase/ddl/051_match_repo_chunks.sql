-- supabase function for chunks SQL Editorで実行して作成する
-- 既存（prefix無し）
create or replace function match_repo_chunks(
  query_embedding vector(1536),
  match_threshold float,
  match_count int
)
returns table (
  path text,
  chunk_idx int,
  content text,
  similarity float
)
language sql stable
as $$
  select
    path,
    chunk_idx,
    content,
    1 - (embedding <=> query_embedding) as similarity
  from repo_chunks
  where 1 - (embedding <=> query_embedding) > match_threshold
  order by embedding <=> query_embedding
  limit match_count;
$$;

-- 追加（prefixで絞り込みたい場合）
create or replace function match_repo_chunks_with_prefix(
  query_embedding vector(1536),
  match_threshold float,
  match_count int,
  path_prefix text
)
returns table (
  path text,
  chunk_idx int,
  content text,
  similarity float
)
language sql stable
as $$
  select
    path,
    chunk_idx,
    content,
    1 - (embedding <=> query_embedding) as similarity
  from repo_chunks
  where path like (path_prefix || '%')
    and 1 - (embedding <=> query_embedding) > match_threshold
  order by embedding <=> query_embedding
  limit match_count;
$$;
