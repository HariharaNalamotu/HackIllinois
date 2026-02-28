/**
 * Hackillinois Fine-Tuning API — Cloudflare Worker
 *
 * Acts as the edge API layer between the frontend (CF Pages) and the
 * Modal compute backend. Handles auth, routing, R2 model storage,
 * KV job tracking, and SSE log streaming.
 *
 * Deploy:  cd worker && npm run deploy
 * Dev:     cd worker && npm run dev
 */

import { Hono } from "hono";
import { cors } from "hono/cors";

// ── Cloudflare bindings ───────────────────────────────────────────────────────
export interface Env {
  // Cloudflare bindings (declared in wrangler.toml)
  R2_BUCKET: R2Bucket;
  JOB_KV: KVNamespace;

  // Vars / secrets
  MODAL_API_BASE: string;
  MODAL_SECRET: string;

  // Forwarded to Modal for R2 uploads
  R2_ENDPOINT_URL: string;
  R2_ACCESS_KEY_ID: string;
  R2_SECRET_ACCESS_KEY: string;
}

// ── Typed shapes ──────────────────────────────────────────────────────────────
interface JobRecord {
  status: "running" | "complete" | "error";
  created_at: string;
  base_model: string;
  output_model: string;
  chunks_collected?: number;
  modal_call_id?: string;
  result?: unknown;
  error?: string;
}

interface ModalStatusResponse {
  status: "running" | "complete" | "error";
  result?: unknown;
  error?: string;
}

// ── Helpers ───────────────────────────────────────────────────────────────────
function modalHeaders(secret: string, extra: Record<string, string> = {}) {
  return { "X-Modal-Secret": secret, ...extra };
}

async function proxyToModal(
  env: Env,
  path: string,
  init: RequestInit
): Promise<Response> {
  const url = `${env.MODAL_API_BASE}${path}`;
  const res = await fetch(url, {
    ...init,
    headers: {
      ...((init.headers as Record<string, string>) ?? {}),
      "X-Modal-Secret": env.MODAL_SECRET,
    },
  });
  return res;
}

// ── App ───────────────────────────────────────────────────────────────────────
const app = new Hono<{ Bindings: Env }>();

app.use(
  "*",
  cors({
    origin: "*", // tighten to your CF Pages domain in production
    allowMethods: ["GET", "POST", "DELETE", "OPTIONS"],
    allowHeaders: ["Content-Type", "Authorization", "X-Request-ID"],
    exposeHeaders: ["X-Job-ID", "X-Total-Count"],
  })
);

// ── Health ────────────────────────────────────────────────────────────────────
app.get("/api/health", (c) =>
  c.json({ status: "ok", platform: "cloudflare-workers", ts: Date.now() })
);

// ── Models ────────────────────────────────────────────────────────────────────
app.get("/api/models", async (c) => {
  const listed = await c.env.R2_BUCKET.list({ prefix: "models/", delimiter: "/" });

  const models = await Promise.all(
    (listed.delimitedPrefixes ?? []).map(async (prefix) => {
      const modelId = prefix.replace("models/", "").replace(/\/$/, "");
      const metaObj = await c.env.R2_BUCKET.get(`models/${modelId}/meta.json`);
      const meta = metaObj ? await metaObj.json<Record<string, unknown>>() : {};
      return { id: modelId, ...meta };
    })
  );

  c.header("X-Total-Count", String(models.length));
  return c.json({ models, total: models.length });
});

app.delete("/api/models/:id", async (c) => {
  const id = c.req.param("id");
  const meta = await c.env.R2_BUCKET.get(`models/${id}/meta.json`);
  if (!meta) return c.json({ error: `Model '${id}' not found.` }, 404);

  const metaData = await meta.json<{ is_finetuned?: boolean }>();
  if (!metaData.is_finetuned) {
    return c.json({ error: "Only fine-tuned models can be deleted via the API." }, 403);
  }

  // Delete all objects under models/{id}/
  const objects = await c.env.R2_BUCKET.list({ prefix: `models/${id}/` });
  await Promise.all(objects.objects.map((o) => c.env.R2_BUCKET.delete(o.key)));

  return c.json({ deleted: id });
});

// ── Chunking methods (static — mirrors chunk.py TOOLS definitions) ────────────
app.get("/api/chunking/methods", (c) => {
  const methods: Record<string, { description: string; parameters: Record<string, { type: string; description: string; default?: unknown }> }> = {
    sentence:         { description: "Split on sentence boundaries. Best for Q&A, transcripts, conversational text.", parameters: { sentences_per_chunk: { type: "integer", description: "Sentences per chunk.", default: 2 } } },
    paragraph:        { description: "Split on blank lines. Best for articles, blogs, documentation.", parameters: { min_words: { type: "integer", description: "Min words to keep a paragraph.", default: 10 } } },
    sliding_window:   { description: "Overlapping fixed-size word windows. Best for dense documents where context bleeds across boundaries.", parameters: { chunk_size: { type: "integer", description: "Words per chunk.", default: 100 }, overlap: { type: "integer", description: "Overlapping words.", default: 20 } } },
    fixed_size:       { description: "Non-overlapping fixed word-count chunks. Best for homogeneous or unstructured text.", parameters: { chunk_size: { type: "integer", description: "Words per chunk.", default: 200 } } },
    markdown_headers: { description: "Split on # / ## / ### headers. Best for .md files, wikis, GitHub READMEs.", parameters: {} },
    html_sections:    { description: "Split HTML-derived text on heading-like breaks. Best for scraped web content.", parameters: {} },
    recursive:        { description: "LangChain-style recursive splitting. Best for mixed-format or inconsistent structure.", parameters: { chunk_size: { type: "integer", description: "Max chars per chunk.", default: 500 }, overlap: { type: "integer", description: "Char overlap.", default: 50 } } },
    code_blocks:      { description: "Split on function/class definitions or code fences. Best for source code files.", parameters: {} },
    csv_rows:         { description: "Group CSV rows into N-row chunks. Best for tabular data.", parameters: { rows_per_chunk: { type: "integer", description: "Rows per chunk.", default: 10 } } },
    json_objects:     { description: "Each top-level JSON object/array element becomes a chunk. Best for .json/.jsonl.", parameters: {} },
    page:             { description: "Split on PDF page breaks. Best for slide decks and scanned documents.", parameters: {} },
  };
  return c.json({ methods, auto_available: true, total: Object.keys(methods).length });
});

// ── Chunking preview (proxied to Modal) ───────────────────────────────────────
app.post("/api/chunking/preview", async (c) => {
  const body = await c.req.formData();
  const res = await proxyToModal(c.env, "/chunk", { method: "POST", body });
  if (!res.ok) {
    const err = await res.text();
    return c.json({ error: err }, res.status as 400 | 422 | 500);
  }
  return c.json(await res.json());
});

// ── Generate training script without running it ───────────────────────────────
app.post("/api/generate-script", async (c) => {
  const body = await c.req.json();
  const res = await proxyToModal(c.env, "/script", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    return c.json({ error: await res.text() }, res.status as 400 | 422 | 500);
  }
  return c.json(await res.json());
});

// ── Start fine-tuning job ─────────────────────────────────────────────────────
app.post("/api/finetune", async (c) => {
  const body = await c.req.formData();

  const baseModel = body.get("base_model")?.toString();
  const outputModel = body.get("output_model_name")?.toString();
  if (!baseModel || !outputModel) {
    return c.json({ error: "base_model and output_model_name are required." }, 400);
  }

  // Inject R2 credentials so Modal can upload the finished weights
  const jobId = crypto.randomUUID();
  body.set("job_id", jobId);
  body.set("r2_endpoint", c.env.R2_ENDPOINT_URL);
  body.set("r2_key_id", c.env.R2_ACCESS_KEY_ID);
  body.set("r2_secret", c.env.R2_SECRET_ACCESS_KEY);
  body.set("r2_bucket", "hackillinois-models");
  body.set("r2_model_prefix", `models/${outputModel}`);

  const res = await proxyToModal(c.env, "/train", { method: "POST", body });
  if (!res.ok) {
    const err = await res.text();
    return c.json({ error: err }, res.status as 400 | 422 | 500);
  }

  const modalResult = await res.json<{ modal_call_id?: string }>();

  // Persist job metadata in KV (7-day TTL)
  const jobRecord: JobRecord = {
    status: "running",
    created_at: new Date().toISOString(),
    base_model: baseModel,
    output_model: outputModel,
    modal_call_id: modalResult.modal_call_id,
  };
  await c.env.JOB_KV.put(`job:${jobId}`, JSON.stringify(jobRecord), {
    expirationTtl: 60 * 60 * 24 * 7,
  });

  c.header("X-Job-ID", jobId);
  return c.json(
    { job_id: jobId, status: "running", poll_url: `/api/finetune/${jobId}`, ...modalResult },
    202
  );
});

// ── Poll job status ───────────────────────────────────────────────────────────
app.get("/api/finetune/:jobId", async (c) => {
  const jobId = c.req.param("jobId");
  const stored = await c.env.JOB_KV.get<JobRecord>(`job:${jobId}`, "json");
  if (!stored) return c.json({ error: "Job not found." }, 404);

  // If still running, poll Modal for latest status
  if (stored.status === "running") {
    const res = await proxyToModal(c.env, `/status/${jobId}`, { method: "GET" });
    if (res.ok) {
      const modalStatus = await res.json<ModalStatusResponse>();
      if (modalStatus.status !== "running") {
        const updated: JobRecord = { ...stored, ...modalStatus };
        await c.env.JOB_KV.put(`job:${jobId}`, JSON.stringify(updated), {
          expirationTtl: 60 * 60 * 24 * 7,
        });
        return c.json({ job_id: jobId, ...updated });
      }
    }
  }

  return c.json({ job_id: jobId, ...stored });
});

// ── Stream training logs via SSE ──────────────────────────────────────────────
app.get("/api/finetune/:jobId/logs", async (c) => {
  const jobId = c.req.param("jobId");

  const res = await proxyToModal(c.env, `/logs/${jobId}`, { method: "GET" });
  if (!res.ok) {
    return c.json({ error: "Could not connect to log stream." }, 502);
  }

  // Pass the Modal SSE stream straight through to the client
  return new Response(res.body, {
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      "X-Accel-Buffering": "no",
      "Access-Control-Allow-Origin": "*",
    },
  });
});

// ── Semantic search via Actian VectorAI DB ────────────────────────────────────
app.post("/api/search", async (c) => {
  const body = await c.req.json<{
    query: string;
    collection?: string;
    top_k?: number;
    with_payload?: boolean;
  }>();

  if (!body.query?.trim()) {
    return c.json({ error: "query is required." }, 400);
  }

  const res = await proxyToModal(c.env, "/search", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      query:        body.query,
      collection:   body.collection ?? "corpus_embeddings",
      top_k:        body.top_k ?? 10,
      with_payload: body.with_payload ?? true,
    }),
  });

  if (!res.ok) {
    return c.json({ error: await res.text() }, res.status as 400 | 500);
  }
  return c.json(await res.json());
});

// ── 404 catch-all ─────────────────────────────────────────────────────────────
app.notFound((c) => c.json({ error: "Not found." }, 404));

export default app;
