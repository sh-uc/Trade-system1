// supabase/functions/rag-search/index.ts
import { createClient } from "https://esm.sh/@supabase/supabase-js@2.44.3";
import OpenAI from "https://esm.sh/openai@4.24.7";

// --- 環境変数 ---
const openai_key = Deno.env.get("OPENAI_API_KEY")!;
const supabase_url = Deno.env.get("SUPABASE_URL")!;
const supabase_service = Deno.env.get("SERVICE_ROLE")!;

// --- 初期化 ---
const sb = createClient(supabase_url, supabase_service);
const openai = new OpenAI({ apiKey: openai_key });

// --- 定数 ---
const MODEL = "text-embedding-3-small";
const LIMIT = 5;

// --- エントリーポイント ---
Deno.serve(async (req) => {
  try {
    const { query } = await req.json();
    if (!query) {
      return new Response(JSON.stringify({ error: "Missing query" }), { status: 400 });
    }

    // 1️⃣ クエリをEmbedding化
    const emb = await openai.embeddings.create({
      model: MODEL,
      input: query,
    });
    const vector = emb.data[0].embedding;

    // 2️⃣ repo_chunksをコサイン距離で検索
    const { data, error } = await sb.rpc("match_repo_chunks", {
      query_embedding: vector,
      match_threshold: 0.35, // 小さいほど類似度が高い
      match_count: LIMIT,
    });

    if (error) throw error;

    return new Response(JSON.stringify({ query, results: data }), {
      headers: { "Content-Type": "application/json" },
    });
  } catch (e) {
    console.error("ERROR", e);
    return new Response(JSON.stringify({ error: e.message }), { status: 500 });
  }
});
