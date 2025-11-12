# sync_repo_to_supabase.py (Embeddings 付き完全版)
import os, sys, hashlib, re, time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict

from supabase import create_client
from openai import OpenAI

ALLOW_EXT = {
    ".md", ".markdown", ".py", ".txt", ".yml", ".yaml", ".toml",
    ".sql", ".json", ".ini", ".cfg", ".sh"
}
EXCLUDE_DIRS = {".git", "__pycache__", ".venv", "venv", "node_modules"}

MAX_SIZE = 512 * 1024   # まずはテキストのみ対象
CHUNK_TOKENS = 800      # だいたいの文字数基準で分割（日本語はざっくり2~3文字=1token目安）
CHUNK_OVERLAP = 120
EMBED_MODEL = "text-embedding-3-small"

def sha256_text(s: str) -> str:
    import hashlib
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def list_files(root: Path):
    for p in root.rglob("*"):
        if p.is_dir():
            if set(p.parts) & EXCLUDE_DIRS:
                continue
            continue
        if p.suffix.lower() in ALLOW_EXT and not (set(p.parts) & EXCLUDE_DIRS):
            yield p

def create_supabase():
    url = (os.environ.get("SUPABASE_URL") or "").strip().strip("\"'")
    key = (os.environ.get("SUPABASE_SERVICE_ROLE") or os.environ.get("SUPABASE_KEY") or "").strip().strip("\"'")
    if not url or not key:
        raise RuntimeError("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE/KEY")
    return create_client(url, key)

def get_repo_branch_root():
    repo = os.environ.get("GITHUB_REPOSITORY", "local/repo")
    branch = os.environ.get("GITHUB_REF_NAME", os.environ.get("BRANCH", "main"))
    root = Path(os.environ.get("GITHUB_WORKSPACE", "."))
    return repo, branch, root

def simple_sentence_split(text: str) -> List[str]:
    # ざっくり文区切り（句点・改行で）
    text = text.replace("\r\n", "\n")
    parts = re.split(r"(?<=[。．！!？?])\s+|\n{2,}", text)
    # 行だけのコードブロックが崩れないよう最低限の結合
    return [p.strip() for p in parts if p.strip()]

def chunk_text(text: str, chunk_chars=CHUNK_TOKENS, overlap=CHUNK_OVERLAP) -> List[str]:
    # 厳密トークナイザなしの簡易分割（十分に実用）
    sents = simple_sentence_split(text)
    chunks, cur = [], ""
    for s in sents:
        if len(cur) + len(s) + 1 <= chunk_chars:
            cur = (cur + "\n" + s).strip()
        else:
            if cur:
                chunks.append(cur)
            # overlap のため最後の一部を引き継ぐ
            tail = cur[-overlap:] if overlap > 0 else ""
            cur = (tail + "\n" + s).strip()
    if cur:
        chunks.append(cur)
    # 極端に長い行がある場合の保険：ハードラップ
    fixed = []
    for c in chunks:
        if len(c) <= chunk_chars * 2:
            fixed.append(c)
        else:
            for i in range(0, len(c), chunk_chars):
                fixed.append(c[i:i+chunk_chars])
    return fixed

def embed_texts(client: OpenAI, texts: List[str]) -> List[List[float]]:
    # バッチで一気に埋め込み
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]

def upsert_repo_docs(sb, docs: List[Dict]):
    CHUNK = 200
    for i in range(0, len(docs), CHUNK):
        sb.table("repo_docs").upsert(docs[i:i+CHUNK], on_conflict="repo,branch,path").execute()
        print(f"[UPSERT repo_docs] {i+1}-{min(i+CHUNK,len(docs))}")

def upsert_repo_chunks(sb, repo: str, branch: str, path: str, chunks: List[str], embeds: List[List[float]]):
    rows = []
    for idx, (content, emb) in enumerate(zip(chunks, embeds)):
        rows.append({
            "repo": repo, "branch": branch, "path": path,
            "chunk_idx": idx, "content": content, "embedding": emb
        })
    CHUNK = 200
    for i in range(0, len(rows), CHUNK):
        sb.table("repo_chunks").upsert(rows[i:i+CHUNK], on_conflict="repo,branch,path,chunk_idx").execute()
        print(f"[UPSERT repo_chunks] {path} {i+1}-{min(i+CHUNK,len(rows))}")

def main():
    repo, branch, root = get_repo_branch_root()
    sb = create_supabase()

    # 1) repo_docs の更新
    docs = []
    for p in list_files(root):
        try:
            text = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        size = len(text.encode("utf-8"))
        if size > MAX_SIZE:
            print(f"[SKIP large] {p} ({size} bytes)")
            continue
        rel = p.relative_to(root).as_posix()
        docs.append({
            "repo": repo, "branch": branch, "path": rel,
            "title": p.name, "filetype": p.suffix.lstrip(".").lower(),
            "content": text, "sha": sha256_text(text),
            "size_bytes": size, "committed_at": datetime.now(timezone.utc).isoformat()
        })
    print(f"[INFO] files to upsert (repo_docs): {len(docs)}")
    upsert_repo_docs(sb, docs)

    # 2) repo_chunks の更新（md/py/…を対象）
    openai_key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if not openai_key:
        print("[WARN] OPENAI_API_KEY not set; skip embeddings")
        return
    client = OpenAI(api_key=openai_key)

    target_paths = [d for d in docs if d["filetype"] in {"md","markdown","py","txt","yml","yaml","sql"}]
    print(f"[INFO] files to embed (repo_chunks): {len(target_paths)}")

    for d in target_paths:
        text = d["content"]
        path = d["path"]
        chunks = chunk_text(text, CHUNK_TOKENS, CHUNK_OVERLAP)
        if not chunks:
            continue
        # OpenAI Embeddings（大きすぎる場合は分割バッチ）
        BATCH = 128
        embeddings = []
        for i in range(0, len(chunks), BATCH):
            batch = chunks[i:i+BATCH]
            emb = embed_texts(client, batch)
            embeddings.extend(emb)
            time.sleep(0.2)  # レート保守
        upsert_repo_chunks(sb, repo, branch, path, chunks, embeddings)

    print("[DONE] sync repo → supabase (docs + embeddings)")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR:", repr(e))
        sys.exit(1)
#