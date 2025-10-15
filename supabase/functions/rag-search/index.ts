// supabase/functions/rag-search/index.ts
import { createClient } from "https://esm.sh/@supabase/supabase-js@2.44.3";

const OPENAI_API_KEY = Deno.env.get("OPENAI_API_KEY");
const SUPABASE_URL   = Deno.env.get("SUPABASE_URL");
const SERVICE_ROLE   = Deno.env.get("SERVICE_ROLE");

const MODEL = "text-embedding-3-small";

async function embed(text: string, signal?: AbortSignal): Promise<number[]> {
  const res = await fetch("https://api.openai.com/v1/embeddings", {
    method: "POST",
    headers: {
      "Authorization": `Bearer ${OPENAI_API_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ model: MODEL, input: text }),
    signal,
  });
  if (!res.ok) {
    const detail = await res.text().catch(() => "");
    throw new Error(`openai ${res.status} ${res.statusText}: ${detail}`);
  }
  const json = await res.json();
  return json.data[0].embedding as number[];
}

Deno.serve(async (req) => {
  try {
    if (!OPENAI_API_KEY || !SUPABASE_URL || !SERVICE_ROLE) {
      return new Response(JSON.stringify({
        error: "missing_env",
        OPENAI_API_KEY: !!OPENAI_API_KEY,
        SUPABASE_URL: !!SUPABASE_URL,
        SERVICE_ROLE: !!SERVICE_ROLE,
      }), { status: 500, headers: { "Content-Type": "application/json" } });
    }

    const { query, threshold = 0.35, limit = 5, prefix } = await req.json().catch(() => ({}));
    if (!query) {
      return new Response(JSON.stringify({ error: "missing_query" }), {
        status: 400, headers: { "Content-Type": "application/json" },
      });
    }

    // タイムアウト（15秒）
    const ctrl = new AbortController();
    const t = setTimeout(() => ctrl.abort("timeout"), 15_000);

    // 1) Embedding
    let vector: number[];
    try {
      vector = await embed(query, ctrl.signal);
    } catch (e) {
      clearTimeout(t);
      return new Response(JSON.stringify({ error: "openai_error", detail: String(e) }),
        { status: 502, headers: { "Content-Type": "application/json" } });
    }

    // 2) RPC（必要に応じて prefix で絞り込み）
    const sb = createClient(SUPABASE_URL, SERVICE_ROLE);
    const rpcName = prefix ? "match_repo_chunks_with_prefix" : "match_repo_chunks";
    const params: Record<string, unknown> = {
      query_embedding: vector,
      match_threshold: threshold,
      match_count: limit,
    };
    if (prefix) params.path_prefix = prefix;

    const { data, error } = await sb.rpc(rpcName, params);
    clearTimeout(t);

    if (error) {
      return new Response(JSON.stringify({ error: "rpc_error", detail: error }),
        { status: 502, headers: { "Content-Type": "application/json" } });
    }

    return new Response(JSON.stringify({ query, results: data }),
      { headers: { "Content-Type": "application/json" } });
  } catch (e) {
    return new Response(JSON.stringify({ error: "fatal", detail: String(e) }),
      { status: 500, headers: { "Content-Type": "application/json" } });
  }
});
