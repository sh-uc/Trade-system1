# sync_repo_to_supabase.py
import os, sys, hashlib, json
from datetime import datetime, timezone
from pathlib import Path

from supabase import create_client

ALLOW_EXT = {
    ".md", ".markdown", ".py", ".txt", ".yml", ".yaml", ".toml",
    ".sql", ".json", ".ini", ".cfg", ".sh"
}
# 除外パス（ビルド産物や巨大ディレクトリ）
EXCLUDE_DIRS = {".git", ".github/workflows/__pycache__", "__pycache__", ".venv", "venv", "node_modules", ".streamlit/config.toml"}

MAX_SIZE = 512 * 1024  # 512KB超は今回はスキップ（後でStorage併用に拡張）

def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def list_files(root: Path):
    for p in root.rglob("*"):
        if p.is_dir():
            # 除外ディレクトリ
            parts = set(p.parts)
            if parts & EXCLUDE_DIRS:
                continue
            continue
        if p.suffix.lower() in ALLOW_EXT:
            # 除外ディレクトリ判定（ファイル側）
            if set(p.parts) & EXCLUDE_DIRS:
                continue
            yield p

def create_supabase():
    url = (os.environ.get("SUPABASE_URL") or "").strip().strip("\"'")
    key = (os.environ.get("SUPABASE_SERVICE_ROLE") or os.environ.get("SUPABASE_KEY") or "").strip().strip("\"'")
    if not url or not key:
        raise RuntimeError("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE/KEY")
    return create_client(url, key)

def main():
    repo = os.environ.get("GITHUB_REPOSITORY", "local/repo")
    branch = os.environ.get("GITHUB_REF_NAME", os.environ.get("BRANCH", "main"))
    root = Path(os.environ.get("GITHUB_WORKSPACE", "."))

    sb = create_supabase()
    docs = []
    for p in list_files(root):
        try:
            text = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            # バイナリ/非UTF-8はスキップ
            continue
        size = len(text.encode("utf-8"))
        if size > MAX_SIZE:
            print(f"[SKIP large] {p} ({size} bytes)")
            continue
        rel = p.relative_to(root).as_posix()
        doc = {
            "repo": repo,
            "branch": branch,
            "path": rel,
            "title": p.name,
            "filetype": p.suffix.lstrip(".").lower(),
            "content": text,
            "sha": sha256_text(text),
            "size_bytes": size,
            "committed_at": datetime.now(timezone.utc).isoformat()
        }
        docs.append(doc)

    print(f"[INFO] files to upsert: {len(docs)}")

    # 分割upsert（大きすぎるペイロード対策）
    CHUNK = 200
    for i in range(0, len(docs), CHUNK):
        batch = docs[i:i+CHUNK]
        # on_conflict は unique (repo,branch,path)
        sb.table("repo_docs").upsert(batch, on_conflict="repo,branch,path").execute()
        print(f"[UPSERT] {i+1}-{min(i+CHUNK,len(docs))}")

    print("[DONE] sync repo → supabase")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR:", repr(e))
        sys.exit(1)
