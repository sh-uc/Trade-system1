
```mermaid
graph LR
A[spec.md, app.py] --> B[Embedding化 (OpenAI API)]
B --> C[Supabase repo_chunks<br/>vector列に保存]
D[質問/検索リクエスト] --> E[Edge Function]
E -->|類似検索 SQL| C
C --> E
E --> D
D -->|関連箇所の本文| ChatGPT
```

