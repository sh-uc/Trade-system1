create table public.repo_chunks (
  id uuid not null default gen_random_uuid (),
  repo text not null,
  branch text not null,
  path text not null,
  chunk_idx integer not null,
  content text not null,
  embedding public.vector null,
  updated_at timestamp with time zone not null default now(),
  constraint repo_chunks_pkey primary key (id),
  constraint repo_chunks_repo_branch_path_chunk_idx_key unique (repo, branch, path, chunk_idx)
) TABLESPACE pg_default;

create index IF not exists idx_repo_chunks_vec on public.repo_chunks using ivfflat (embedding vector_cosine_ops) TABLESPACE pg_default;