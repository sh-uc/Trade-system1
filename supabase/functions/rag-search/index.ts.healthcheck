// health check only
Deno.serve((_req) => {
  const hasUrl = !!Deno.env.get("SUPABASE_URL");
  const hasRole = !!Deno.env.get("SERVICE_ROLE");
  const hasOpenAI = !!Deno.env.get("OPENAI_API_KEY");
  return new Response(
    JSON.stringify({ ok: true, env: { SUPABASE_URL: hasUrl, SERVICE_ROLE: hasRole, OPENAI_API_KEY: hasOpenAI } }),
    { headers: { "Content-Type": "application/json" } }
  );
});
