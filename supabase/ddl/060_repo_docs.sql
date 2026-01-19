create table public.repo_docs (
  id uuid not null default gen_random_uuid (),
  repo text not null,
  branch text not null,
  path text not null,
  title text null,
  filetype text null,
  content text not null,
  sha text not null,
  size_bytes integer not null,
  committed_at timestamp with time zone not null default now(),
  updated_at timestamp with time zone not null default now(),
  constraint repo_docs_pkey primary key (id),
  constraint repo_docs_repo_branch_path_key unique (repo, branch, path)
) TABLESPACE pg_default;

create index IF not exists idx_repo_docs_repo_branch on public.repo_docs using btree (repo, branch) TABLESPACE pg_default;

create index IF not exists idx_repo_docs_filetype on public.repo_docs using btree (filetype) TABLESPACE pg_default;