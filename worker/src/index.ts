/**
 * HackIllinois — Cloudflare Worker (backend API)
 *
 * This Worker IS the backend. It handles all routing, storage, and
 * business logic. Modal is called only when Python compute is required
 * (chunking, training, inference, embedding-based search).
 *
 * Modal endpoints (individual on-demand functions, not a persistent server):
 *   MODAL_BASE--chunk.modal.run       POST  chunk a file (needs Python NLP)
 *   MODAL_BASE--train.modal.run       POST  spawn GPU training job
 *   MODAL_BASE--infer.modal.run       POST  run inference pipeline (CPU)
 *   MODAL_BASE--job-status.modal.run  GET   ?job_id=<id>
 *   MODAL_BASE--job-logs.modal.run    GET   ?job_id=<id>  (SSE)
 *   MODAL_BASE--search.modal.run      POST  embed query + Actian search
 *
 * After `modal deploy modal/app.py`, set MODAL_BASE in wrangler.toml to
 * the printed URL base, e.g.:
 *   MODAL_BASE = "https://harihara--hackillinois-pipeline"
 *
 * Worker routes:
 *   GET  /api/health
 *   GET  /api/nodes/types               — static node catalogue
 *   GET  /api/chunking/methods          — static method list
 *
 *   Workflows (stored in KV):
 *   GET    /api/workflows
 *   POST   /api/workflows
 *   GET    /api/workflows/:id
 *   PUT    /api/workflows/:id
 *   DELETE /api/workflows/:id
 *
 *   Models (stored in R2):
 *   GET    /api/models
 *   DELETE /api/models/:id
 *
 *   Compute (proxied to Modal on demand):
 *   POST /api/chunking/preview          → Modal /chunk
 *   POST /api/pipeline/train            → Modal /train (spawns GPU job, 202)
 *   POST /api/pipeline/infer            → Modal /infer (sync, returns results)
 *   GET  /api/pipeline/:id/status       → Modal /job-status?job_id=<id>
 *   GET  /api/pipeline/:id/logs         → Modal /job-logs?job_id=<id> (SSE)
 *   POST /api/search                    → Modal /search (embed + Actian)
 *
 *   Actian health (direct):
 *   GET  /api/actian/health             → Actian HTTP gateway
 *
 *   Agentic AI (merged from backend/):
 *   POST /api/chat                      → SSE streaming chat (OpenAI + tools + sub-agents)
 *   GET  /api/feedback                  → retrieve RLHF feedback entries
 */

import { Hono } from "hono";
import { chatHandler } from "./services/agent_service";
import { storeFeedback, getRecentFeedback } from "./services/rlhf_service";

export interface Env {
  // KV namespace — workflows (workflow:<id>) and job metadata (job:<id>)
  JOB_KV: KVNamespace;
  // R2 bucket — model weights (models/<name>/...) and uploads (uploads/<wfId>/...)
  R2_MODELS: R2Bucket;
  /**
   * Modal endpoint base URL.
   * After `modal deploy modal/app.py`, set this to:
   *   https://<modal-workspace>--hackillinois-pipeline
   * Each on-demand endpoint is then:
   *   ${MODAL_BASE}--chunk.modal.run
   *   ${MODAL_BASE}--train.modal.run  etc.
   */
  MODAL_BASE: string;
  // Shared secret between Worker and Modal (set via `wrangler secret put`)
  MODAL_SECRET: string;
  // R2 credentials — injected into Modal for model weight uploads
  R2_ACCESS_KEY_ID: string;
  R2_SECRET_ACCESS_KEY: string;
  R2_ENDPOINT_URL: string;
  // Actian VectorAI DB — Cloudflare Container HTTP gateway
  ACTIAN_HTTP_URL: string;
  // CORS origins (comma-separated, or *)
  ALLOWED_ORIGINS: string;
  // OpenAI API key for LLM chat proxy
  OPENAI_API_KEY: string;
}

const app = new Hono<{ Bindings: Env }>();

// ── CORS ──────────────────────────────────────────────────────────────────────
app.use("*", async (c, next) => {
  const origins = (c.env.ALLOWED_ORIGINS || "*").split(",").map((s) => s.trim());
  const origin  = c.req.header("Origin") || "";
  const allowed = origins.includes("*") || origins.includes(origin) ? origin : origins[0];

  if (c.req.method === "OPTIONS") {
    return new Response(null, {
      status: 204,
      headers: {
        "Access-Control-Allow-Origin":  allowed,
        "Access-Control-Allow-Methods": "GET,POST,PUT,DELETE,OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type,X-Modal-Secret,Authorization,X-API-Key",
      },
    });
  }
  await next();
  c.res.headers.set("Access-Control-Allow-Origin", allowed);
});

// ── Helpers ───────────────────────────────────────────────────────────────────

/** Headers to authenticate Worker → Modal requests. */
function modalHeaders(env: Env): Record<string, string> {
  return env.MODAL_SECRET ? { "X-Modal-Secret": env.MODAL_SECRET } : {};
}

/** Build a Modal on-demand endpoint URL from the base + label. */
function modalUrl(env: Env, label: string): string {
  // MODAL_BASE = https://<workspace>--hackillinois-pipeline
  // endpoint   = https://<workspace>--hackillinois-pipeline--<label>.modal.run
  return `${env.MODAL_BASE}--${label}.modal.run`;
}

/** Return the first input-node id from a backend pipeline JSON string. */
function firstInputNodeIdFromPipeline(pipelineJson: string): string | null {
  try {
    const parsed = JSON.parse(pipelineJson) as { nodes?: Array<{ id?: string; type?: string }> };
    const inputNode = (parsed.nodes || []).find((n) => typeof n.type === "string" && n.type.endsWith("_input"));
    return inputNode?.id || null;
  } catch {
    return null;
  }
}

// ═════════════════════════════════════════════════════════════════════════════
// Static routes — handled entirely by the Worker
// ═════════════════════════════════════════════════════════════════════════════

app.get("/api/health", (c) =>
  c.json({
    status:    "ok",
    timestamp: new Date().toISOString(),
    actian:    c.env.ACTIAN_HTTP_URL,
    modal_base: c.env.MODAL_BASE,
  })
);

// Node catalogue — static (no Modal cold-start on page load)
app.get("/api/nodes/types", (c) => c.json(NODE_CATALOGUE));

// Chunking methods — static
app.get("/api/chunking/methods", (c) =>
  c.json({
    methods: [
      { id: "auto",            label: "Auto (AI agent)",     description: "OpenAI picks the best method" },
      { id: "sentence",        label: "Sentence",            description: "Split by sentence boundaries" },
      { id: "paragraph",       label: "Paragraph",           description: "Split on blank lines" },
      { id: "sliding_window",  label: "Sliding Window",      description: "Overlapping fixed-size windows" },
      { id: "fixed_size",      label: "Fixed Size",          description: "Non-overlapping fixed char count" },
      { id: "markdown_headers",label: "Markdown Headers",    description: "Split on # headings" },
      { id: "html_sections",   label: "HTML Sections",       description: "Split on semantic HTML elements" },
      { id: "recursive",       label: "Recursive",           description: "LangChain-style recursive splitting" },
      { id: "code_blocks",     label: "Code Blocks",         description: "Split on ``` fences" },
      { id: "csv_rows",        label: "CSV Rows",            description: "One chunk per CSV row" },
      { id: "json_objects",    label: "JSON Objects",        description: "One chunk per JSON array element" },
    ],
  })
);

// Actian health — direct proxy (no Modal involvement)
app.get("/api/actian/health", async (c) => {
  const resp = await fetch(`${c.env.ACTIAN_HTTP_URL}/health`);
  return new Response(resp.body, {
    status:  resp.status,
    headers: { "Content-Type": "application/json" },
  });
});

// ═════════════════════════════════════════════════════════════════════════════
// Workflow CRUD — stored in KV (no Modal involvement)
// ═════════════════════════════════════════════════════════════════════════════

app.get("/api/workflows", async (c) => {
  const list = await c.env.JOB_KV.list({ prefix: "workflow:" });
  const items = await Promise.all(list.keys.map(({ name }) => c.env.JOB_KV.get(name, "json")));
  return c.json({ workflows: items.filter(Boolean) });
});

app.post("/api/workflows", async (c) => {
  const body = await c.req.json<{
    id?: string; name?: string; nodes?: unknown[]; edges?: unknown[];
  }>();
  const id = body.id || crypto.randomUUID();
  const wf = {
    id, name: body.name || "Untitled",
    nodes: body.nodes || [], edges: body.edges || [],
    createdAt: new Date().toISOString(), updatedAt: new Date().toISOString(),
  };
  await c.env.JOB_KV.put(`workflow:${id}`, JSON.stringify(wf), { expirationTtl: 60 * 60 * 24 * 90 });
  return c.json(wf, 201);
});

app.get("/api/workflows/:id", async (c) => {
  const wf = await c.env.JOB_KV.get(`workflow:${c.req.param("id")}`, "json");
  return wf ? c.json(wf) : c.json({ error: "Not found" }, 404);
});

app.put("/api/workflows/:id", async (c) => {
  const id  = c.req.param("id");
  const key = `workflow:${id}`;
  const existing = (await c.env.JOB_KV.get(key, "json")) as Record<string, unknown> | null;
  if (!existing) return c.json({ error: "Not found" }, 404);
  const updated = { ...existing, ...(await c.req.json()), id, updatedAt: new Date().toISOString() };
  await c.env.JOB_KV.put(key, JSON.stringify(updated), { expirationTtl: 60 * 60 * 24 * 90 });
  return c.json(updated);
});

app.delete("/api/workflows/:id", async (c) => {
  await c.env.JOB_KV.delete(`workflow:${c.req.param("id")}`);
  return c.json({ deleted: true });
});

// ═════════════════════════════════════════════════════════════════════════════
// Model management — R2 bucket (no Modal involvement)
// ═════════════════════════════════════════════════════════════════════════════

app.get("/api/models", async (c) => {
  const listed = await c.env.R2_MODELS.list({ prefix: "models/" });
  const names  = new Set<string>();
  for (const obj of listed.objects) {
    const part = obj.key.replace("models/", "").split("/")[0];
    if (part) names.add(part);
  }
  return c.json({ models: [...names].map((n) => ({ id: n, name: n })) });
});

app.delete("/api/models/:id", async (c) => {
  const listed = await c.env.R2_MODELS.list({ prefix: `models/${c.req.param("id")}/` });
  await Promise.all(listed.objects.map((o) => c.env.R2_MODELS.delete(o.key)));
  return c.json({ deleted: true, files: listed.objects.length });
});

// ═════════════════════════════════════════════════════════════════════════════
// Compute routes — proxied to Modal on-demand functions
// ═════════════════════════════════════════════════════════════════════════════

// Chunk preview (needs Python PDF/DOCX parsers + NLP)
app.post("/api/chunking/preview", async (c) => {
  const fd   = await c.req.formData();
  const resp = await fetch(modalUrl(c.env, "chunk"), {
    method: "POST", headers: modalHeaders(c.env), body: fd,
  });
  return new Response(resp.body, { status: resp.status, headers: { "Content-Type": "application/json" } });
});

// Spawn GPU training job (generates Python script + executes on Modal GPU)
app.post("/api/pipeline/train", async (c) => {
  const jobId = crypto.randomUUID();
  const fd    = await c.req.formData();

  // Inject credentials for Modal to upload weights to R2
  fd.set("job_id",      jobId);
  fd.set("r2_endpoint", c.env.R2_ENDPOINT_URL        || "");
  fd.set("r2_key_id",   c.env.R2_ACCESS_KEY_ID       || "");
  fd.set("r2_secret",   c.env.R2_SECRET_ACCESS_KEY   || "");
  fd.set("r2_bucket",   "hackillinois-models");
  fd.set("actian_url",  c.env.ACTIAN_HTTP_URL         || "");

  const resp = await fetch(modalUrl(c.env, "train"), {
    method: "POST", headers: modalHeaders(c.env), body: fd,
  });

  let data: Record<string, unknown> = {};
  try { data = (await resp.json()) as Record<string, unknown>; } catch (_) {}
  const returnedId = (data.job_id as string) || jobId;

  // Record job in KV for frontend polling
  await c.env.JOB_KV.put(
    `job:${returnedId}`,
    JSON.stringify({ status: "running", startedAt: new Date().toISOString() }),
    { expirationTtl: 60 * 60 * 24 * 7 }
  );

  return c.json({ job_id: returnedId, status: "running" }, 202);
});

// Run inference pipeline (sync, CPU, returns results directly)
app.post("/api/pipeline/infer", async (c) => {
  const fd = await c.req.formData();
  fd.set("actian_url", c.env.ACTIAN_HTTP_URL || "");

  const resp = await fetch(modalUrl(c.env, "infer"), {
    method: "POST", headers: modalHeaders(c.env), body: fd,
  });
  return new Response(resp.body, { status: resp.status, headers: { "Content-Type": "application/json" } });
});

// Poll training job status
app.get("/api/pipeline/:id/status", async (c) => {
  const id = c.req.param("id");

  // Ask Modal (the authoritative source for running jobs)
  try {
    const resp = await fetch(
      `${modalUrl(c.env, "job-status")}?job_id=${encodeURIComponent(id)}`,
      { headers: modalHeaders(c.env) }
    );
    if (resp.ok) {
      const data = (await resp.json()) as Record<string, unknown>;
      // Persist terminal state to KV
      if (data.status === "complete" || data.status === "error") {
        await c.env.JOB_KV.put(
          `job:${id}`,
          JSON.stringify({ ...data, finishedAt: new Date().toISOString() }),
          { expirationTtl: 60 * 60 * 24 * 7 }
        );
      }
      return c.json(data);
    }
  } catch (_) {}

  // Fall back to KV (terminal states persisted above)
  const kv = await c.env.JOB_KV.get(`job:${id}`, "json");
  return kv ? c.json(kv) : c.json({ status: "not_found" }, 404);
});

// Stream training logs (SSE)
app.get("/api/pipeline/:id/logs", async (c) => {
  const id = c.req.param("id");
  const upstream = await fetch(
    `${modalUrl(c.env, "job-logs")}?job_id=${encodeURIComponent(id)}`,
    { headers: { ...modalHeaders(c.env), Accept: "text/event-stream" } }
  );
  return new Response(upstream.body, {
    status: upstream.status,
    headers: { "Content-Type": "text/event-stream", "Cache-Control": "no-cache" },
  });
});

// Semantic search (embed query via Modal → query Actian directly from Modal)
app.post("/api/search", async (c) => {
  const body = await c.req.json();
  const resp = await fetch(modalUrl(c.env, "search"), {
    method: "POST",
    headers: { ...modalHeaders(c.env), "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return new Response(resp.body, { status: resp.status, headers: { "Content-Type": "application/json" } });
});

// ═════════════════════════════════════════════════════════════════════════════
// Per-workflow training & upload endpoints
// ═════════════════════════════════════════════════════════════════════════════

/** Build a minimal ustar tar archive from in-memory file entries. */
function buildTar(entries: Array<{ name: string; data: Uint8Array }>): Uint8Array {
  const parts: Uint8Array[] = [];
  for (const { name, data } of entries) {
    const hdr = new Uint8Array(512);
    const enc = (s: string, off: number, len: number) => {
      for (let i = 0; i < Math.min(s.length, len); i++) hdr[off + i] = s.charCodeAt(i);
    };
    enc(name.slice(0, 100), 0, 100);
    enc("0000644\0", 100, 8);
    enc("0000000\0", 108, 8);
    enc("0000000\0", 116, 8);
    enc(data.length.toString(8).padStart(11, "0") + "\0", 124, 12);
    enc(Math.floor(Date.now() / 1000).toString(8).padStart(11, "0") + "\0", 136, 12);
    for (let i = 148; i < 156; i++) hdr[i] = 0x20;
    hdr[156] = 0x30;
    let cs = 0;
    for (let i = 0; i < 512; i++) cs += hdr[i];
    enc(cs.toString(8).padStart(6, "0") + "\0 ", 148, 8);
    parts.push(hdr);
    const padded = new Uint8Array(Math.ceil(data.length / 512) * 512);
    padded.set(data);
    parts.push(padded);
  }
  parts.push(new Uint8Array(1024)); // EOF blocks
  const total = parts.reduce((s, p) => s + p.length, 0);
  const out = new Uint8Array(total);
  let off = 0;
  for (const p of parts) { out.set(p, off); off += p.length; }
  return out;
}

// POST /api/workflow/:id/train
//
// For each node with uploaded files, routes by node type:
//   text_input  → checks KV for pre-embedded Actian collection, OR chunks+embeds fresh file
//   image/audio/tabular → checks KV for pre-uploaded R2 key, OR stores fresh file in R2
//
// Modal receives:
//   actian_collections[nodeId] — collection name to scan at train time (text)
//   r2_keys[nodeId]            — R2 object key to download at train time (binary)
app.post("/api/workflow/:id/train", async (c) => {
  const workflowId = c.req.param("id");
  const jobId      = crypto.randomUUID();
  const fd         = await c.req.formData();

  // Debug: log all form keys received from frontend
  const allKeys: string[] = [];
  for (const [key, value] of fd.entries()) {
    const isFile = typeof value !== "string";
    allKeys.push(`${key}(${isFile ? `file:${(value as File).name || "?"},${(value as File).size || 0}b` : "string"})`);
  }
  console.log(`[train] form keys from frontend: [${allKeys.join(", ")}]`);

  const pipelineVal = fd.get("pipeline");
  if (!pipelineVal || typeof pipelineVal !== "string") {
    return c.json({ error: "Missing 'pipeline' field." }, 400);
  }

  // Build node-type map from pipeline spec
  const nodeTypeMap: Record<string, string> = {};
  try {
    const spec = JSON.parse(pipelineVal) as { nodes?: Array<{ id: string; type: string }> };
    for (const node of spec.nodes || []) nodeTypeMap[node.id] = node.type;
  } catch (_) {}

  // Assemble fresh FormData for Modal
  const modalFd = new FormData();
  modalFd.set("pipeline",    pipelineVal);
  modalFd.set("job_id",      jobId);
  modalFd.set("workflow_id", workflowId);
  modalFd.set("r2_endpoint", c.env.R2_ENDPOINT_URL      || "");
  modalFd.set("r2_key_id",   c.env.R2_ACCESS_KEY_ID     || "");
  modalFd.set("r2_secret",   c.env.R2_SECRET_ACCESS_KEY || "");
  modalFd.set("r2_bucket",   "hackillinois-models");
  modalFd.set("actian_url",  c.env.ACTIAN_HTTP_URL       || "");

  // Forward any scalar fields (e.g. gpu)
  for (const [key, value] of fd.entries()) {
    if (key === "pipeline" || (key.startsWith("files[") && key.endsWith("]"))) continue;
    if (typeof value === "string") modalFd.set(key, value);
  }

  // Track which nodes we've handled via fresh files (to avoid double-processing KV)
  const freshNodes = new Set<string>();

  // ── Process fresh file uploads ────────────────────────────────────────────
  for (const [key, rawValue] of fd.entries()) {
    if (!key.startsWith("files[") || !key.endsWith("]")) continue;
    const file = rawValue as unknown as File;
    if (!(file instanceof File)) continue;

    const nodeId   = key.slice(6, -1);
    const nodeType = nodeTypeMap[nodeId] || "";
    const buf      = await file.arrayBuffer();
    freshNodes.add(nodeId);

    if (nodeType === "text_input") {
      // Chunk the file, pass raw chunks directly to Modal.
      // The encoder is fine-tuned on these chunks first; THEN Modal embeds with the
      // fine-tuned model and stores in Actian. No pre-training embed-store needed.
      try {
        const chunkFd = new FormData();
        chunkFd.set("file", new Blob([buf], { type: file.type || "text/plain" }), file.name);
        chunkFd.set("method", "auto");
        chunkFd.set("preview_limit", "100000");

        const chunkResp = await fetch(modalUrl(c.env, "chunk"), {
          method: "POST", headers: modalHeaders(c.env), body: chunkFd,
        });

        if (chunkResp.ok) {
          const chunks = ((await chunkResp.json() as { preview?: string[] }).preview) || [];
          if (chunks.length > 0) {
            // Pass raw chunks to Modal — fine-tune first, THEN Actian embedding
            modalFd.set(`text_chunks[${nodeId}]`, JSON.stringify(chunks));
            console.log(`[train] text node ${nodeId}: ${chunks.length} chunks → Modal (raw)`);
          } else {
            // Chunking returned 0 chunks — send raw file so TextInputNode can read it
            console.warn(`[train] chunk returned 0 chunks for ${nodeId}, sending raw file`);
            modalFd.set(`files[${nodeId}]`, new Blob([buf], { type: file.type || "text/plain" }), file.name);
          }
        } else {
          // Fallback: send raw file bytes
          console.error(`[train] chunk failed (${chunkResp.status}) for ${nodeId}, sending raw`);
          modalFd.set(`files[${nodeId}]`, new Blob([buf], { type: file.type || "text/plain" }), file.name);
        }
      } catch (e) {
        console.error(`[train] chunk error for ${nodeId}:`, e);
        modalFd.set(`files[${nodeId}]`, new Blob([buf], { type: file.type || "text/plain" }), file.name);
      }

    } else {
      // Binary node: store in R2, pass key to Modal
      const r2Key = `uploads/${workflowId}/${nodeId}/${file.name}`;
      await c.env.R2_MODELS.put(r2Key, buf, {
        httpMetadata: { contentType: file.type || "application/octet-stream" },
      });
      modalFd.append(`r2_keys[${nodeId}]`, r2Key);
      console.log(`[train] binary node ${nodeId}: stored R2 ${r2Key}`);
    }
  }

  // ── Check KV for nodes pre-uploaded via /upload endpoint ──────────────────
  const uploadKeys = await c.env.JOB_KV.list({ prefix: `upload:${workflowId}:` });
  for (const { name } of uploadKeys.keys) {
    const nodeId = name.slice(`upload:${workflowId}:`.length);
    if (freshNodes.has(nodeId)) continue;   // already handled via fresh file

    const entry = (await c.env.JOB_KV.get(name, "json")) as { type: string; collection?: string; keys?: string[]; chunks?: string[] } | null;
    if (!entry) continue;

    if (entry.type === "text" && entry.chunks?.length) {
      // Raw chunks from pre-upload — fine-tune first, THEN Actian embed
      modalFd.set(`text_chunks[${nodeId}]`, JSON.stringify(entry.chunks));
      console.log(`[train] node ${nodeId}: ${entry.chunks.length} raw chunks from KV`);
    } else if (entry.type === "actian" && entry.collection) {
      // Legacy: collection was pre-embedded (old upload path) — scan Actian at train time
      modalFd.set(`actian_collections[${nodeId}]`, entry.collection);
      console.log(`[train] node ${nodeId}: Actian collection from KV '${entry.collection}'`);
    } else if (entry.type === "r2" && entry.keys?.length) {
      for (const r2Key of entry.keys) {
        modalFd.append(`r2_keys[${nodeId}]`, r2Key);
      }
      console.log(`[train] node ${nodeId}: R2 keys from KV (${entry.keys.length})`);
    }
  }

  const resp = await fetch(modalUrl(c.env, "train"), {
    method: "POST", headers: modalHeaders(c.env), body: modalFd,
  });
  if (!resp.ok) {
    const errText = await resp.text();
    console.error(`[train] Modal returned ${resp.status}:`, errText);
    return c.json({ error: `Modal /train returned ${resp.status}`, detail: errText.slice(0, 800) }, 502);
  }
  let data: Record<string, unknown> = {};
  try { data = (await resp.json()) as Record<string, unknown>; } catch (_) {}
  const retId = (data.job_id as string) || jobId;

  await c.env.JOB_KV.put(
    `job:${retId}`,
    JSON.stringify({ status: "running", workflowId, startedAt: new Date().toISOString() }),
    { expirationTtl: 60 * 60 * 24 * 7 }
  );
  return c.json({ job_id: retId, status: "running" }, 202);
});

// POST /api/workflow/:id/upload — route files by node type:
//   text_input  → chunk via Modal /chunk → embed via Modal /embed-store → stored in Actian
//   image/audio/tabular → stored in R2
// Saves a KV entry (upload:<workflowId>:<nodeId>) so the train endpoint can look it up.
app.post("/api/workflow/:id/upload", async (c) => {
  const workflowId = c.req.param("id");
  const fd         = await c.req.formData();
  const nodeId     = (fd.get("node_id")   as string) || "";
  const nodeType   = (fd.get("node_type") as string) || "";
  const stored: Array<{ name: string; type: string; collection?: string; key?: string }> = [];

  for (const [, rawValue] of fd.entries()) {
    const file = rawValue as unknown as File;
    if (!(file instanceof File)) continue;

    const buf = await file.arrayBuffer();

    if (nodeType === "text_input") {
      // Chunk the file via Modal — store raw chunks in KV.
      // The encoder will fine-tune on these chunks at train time, THEN embed to Actian.
      const chunkFd = new FormData();
      chunkFd.set("file", new Blob([buf], { type: file.type || "text/plain" }), file.name);
      chunkFd.set("method", "auto");
      chunkFd.set("preview_limit", "100000");

      const chunkResp = await fetch(modalUrl(c.env, "chunk"), {
        method: "POST", headers: modalHeaders(c.env), body: chunkFd,
      });
      if (!chunkResp.ok) {
        return c.json({ error: `Chunking failed (${chunkResp.status})` }, 502);
      }
      const chunkData = await chunkResp.json() as { preview?: string[] };
      const chunks    = chunkData.preview || [];

      // Save raw chunks in KV — train endpoint passes these directly to Modal
      await c.env.JOB_KV.put(
        `upload:${workflowId}:${nodeId}`,
        JSON.stringify({ type: "text", chunks }),
        { expirationTtl: 60 * 60 * 24 * 30 },
      );
      stored.push({ name: file.name, type: "text", count: chunks.length } as { name: string; type: string; collection?: string; key?: string });

    } else {
      // Binary node: store raw file in R2
      const key = `uploads/${workflowId}/${nodeId ? nodeId + "/" : ""}${file.name}`;
      await c.env.R2_MODELS.put(key, buf, {
        httpMetadata: { contentType: file.type || "application/octet-stream" },
      });

      // Save / append R2 key in KV
      const existing = (await c.env.JOB_KV.get(`upload:${workflowId}:${nodeId}`, "json")) as { keys?: string[] } | null;
      const keys = existing?.keys || [];
      keys.push(key);
      await c.env.JOB_KV.put(
        `upload:${workflowId}:${nodeId}`,
        JSON.stringify({ type: "r2", keys }),
        { expirationTtl: 60 * 60 * 24 * 30 },
      );
      stored.push({ name: file.name, type: "r2", key });
    }
  }

  return c.json({ files: stored });
});

// GET /api/models/:name/download — stream all model files as a tar archive
app.get("/api/models/:name/download", async (c) => {
  const name   = c.req.param("name");
  const listed = await c.env.R2_MODELS.list({ prefix: `models/${name}/` });

  const entries: Array<{ name: string; data: Uint8Array }> = [];
  for (const obj of listed.objects) {
    const item = await c.env.R2_MODELS.get(obj.key);
    if (!item) continue;
    const data = new Uint8Array(await item.arrayBuffer());
    const rel  = obj.key.replace(`models/${name}/`, "");
    entries.push({ name: `${name}/${rel}`, data });
  }

  if (entries.length === 0) return c.json({ error: "Model not found" }, 404);

  const tar = buildTar(entries);
  return new Response(tar, {
    headers: {
      "Content-Type":        "application/x-tar",
      "Content-Disposition": `attachment; filename="${name}.tar"`,
      "Content-Length":      String(tar.byteLength),
    },
  });
});

// POST /api/llm/chat — proxy to OpenAI or Ollama
app.post("/api/llm/chat", async (c) => {
  const body = await c.req.json<{
    provider: string;
    model?: string;
    messages: Array<{ role: string; content: string }>;
    ollamaUrl?: string;
    systemPrompt?: string;
  }>();

  const { provider, model, messages, ollamaUrl, systemPrompt } = body;
  const fullMessages = systemPrompt
    ? [{ role: "system", content: systemPrompt }, ...messages]
    : messages;

  if (provider === "openai" || provider === "openai_large") {
    const oaiModel = provider === "openai_large" ? "gpt-4o" : (model || "gpt-4o-mini");
    const resp = await fetch("https://api.openai.com/v1/chat/completions", {
      method: "POST",
      headers: {
        "Content-Type":  "application/json",
        "Authorization": `Bearer ${c.env.OPENAI_API_KEY}`,
      },
      body: JSON.stringify({ model: oaiModel, messages: fullMessages }),
    });
    if (!resp.ok) {
      const err = await resp.text();
      return c.json({ error: `OpenAI error: ${err.slice(0, 200)}` }, 502);
    }
    const data = (await resp.json()) as { choices: Array<{ message: { content: string } }> };
    return c.json({ content: data.choices[0]?.message?.content || "" });
  }

  // Ollama variants
  const baseUrl = ollamaUrl?.replace(/\/$/, "") || "http://localhost:11434";
  const ollamaModel =
    provider === "ollama_qwen"  ? "qwen2.5:0.5b"  :
    provider === "ollama_llama" ? "llama3.2:1b"    :
    (model || "llama3.2:1b");

  const resp = await fetch(`${baseUrl}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model: ollamaModel, messages: fullMessages, stream: false }),
  });
  if (!resp.ok) {
    const err = await resp.text();
    return c.json({ error: `Ollama error: ${err.slice(0, 200)}` }, 502);
  }
  const data = (await resp.json()) as { message?: { content: string } };
  return c.json({ content: data.message?.content || "" });
});

// POST /api/chat — agentic SSE streaming chat (OpenAI + tools + sub-agents + RLAIF)
app.post("/api/chat", chatHandler);

// POST /api/feedback — stores in KV and feeds RLHF in-memory store
app.post("/api/feedback", async (c) => {
  const body = await c.req.json<{
    messageId?: string;
    rating?: "up" | "down";
    feedback?: string;
    conversationSnippet?: string;
  }>();
  const id = body.messageId || crypto.randomUUID();

  // Persist to KV
  await c.env.JOB_KV.put(
    `feedback:${id}:${Date.now()}`,
    JSON.stringify({
      messageId: id,
      rating: body.rating || "up",
      feedback: body.feedback || "",
      createdAt: new Date().toISOString(),
    }),
    { expirationTtl: 60 * 60 * 24 * 30 }
  );

  // Also feed the in-memory RLHF store for system prompt enrichment
  storeFeedback({
    messageId: id,
    rating: body.rating || "up",
    feedback: body.feedback,
    conversationSnippet: body.conversationSnippet,
  });

  return c.json({ ok: true });
});

// GET /api/feedback — retrieve recent RLHF feedback entries
app.get("/api/feedback", (c) => {
  return c.json({ feedback: getRecentFeedback() });
});

// POST /api/deploy/:workflowId — persistent inference endpoint for deployment workflows
app.post("/api/deploy/:workflowId", async (c) => {
  const workflowId = c.req.param("workflowId");
  const wf = (await c.env.JOB_KV.get(`workflow:${workflowId}`, "json")) as Record<string, unknown> | null;

  const contentType = c.req.header("Content-Type") || "";
  let inboundForm: FormData | null = null;
  let inboundJson: Record<string, unknown> | null = null;
  let inboundText = "";

  if (contentType.includes("multipart/form-data")) {
    inboundForm = await c.req.formData();
  } else if (contentType.includes("application/json")) {
    inboundJson = await c.req.json<Record<string, unknown>>();
  } else {
    inboundText = await c.req.text();
  }

  let pipelineSpec: string | undefined;
  if (inboundForm) {
    const inline = inboundForm.get("pipeline") || inboundForm.get("pipeline_spec");
    if (typeof inline === "string" && inline.trim()) pipelineSpec = inline;
  }
  if (!pipelineSpec && inboundJson) {
    const inline = inboundJson.pipeline || inboundJson.pipeline_spec;
    if (typeof inline === "string" && inline.trim()) {
      pipelineSpec = inline;
    } else if (inline && typeof inline === "object") {
      pipelineSpec = JSON.stringify(inline);
    }
  }
  if (!pipelineSpec && typeof wf?.deploymentSpec === "string" && wf.deploymentSpec.trim()) {
    pipelineSpec = wf.deploymentSpec;
  }

  if (!pipelineSpec) {
    return c.json({ error: "Missing pipeline spec. Provide 'pipeline' in request body/form." }, 400);
  }

  const inputNodeId = firstInputNodeIdFromPipeline(pipelineSpec);

  const fd = new FormData();
  fd.set("pipeline", pipelineSpec);
  fd.set("workflow_id", workflowId);
  fd.set("actian_url", c.env.ACTIAN_HTTP_URL || "");
  fd.set("r2_endpoint", c.env.R2_ENDPOINT_URL      || "");
  fd.set("r2_key_id",   c.env.R2_ACCESS_KEY_ID     || "");
  fd.set("r2_secret",   c.env.R2_SECRET_ACCESS_KEY || "");
  fd.set("r2_bucket",   "hackillinois-models");

  if (inboundForm) {
    for (const [key, value] of inboundForm.entries()) {
      if (key === "pipeline" || key === "pipeline_spec") continue;
      if (typeof value === "string") {
        if (key !== "llm_config") fd.set(key, value);
        continue;
      }
      const fileValue = value as unknown as File;
      const fileName = fileValue.name || "upload.bin";

      if (key.startsWith("files[") && key.endsWith("]")) {
        fd.append(key, fileValue, fileName);
        continue;
      }

      if ((key === "file" || key === "files[]") && inputNodeId) {
        fd.append(`files[${inputNodeId}]`, fileValue, fileName);
        continue;
      }

      fd.append(key, fileValue, fileName);
    }
  }

  // Text/json payload fallback: attach as a synthetic file for text input nodes.
  const explicitInputText =
    (inboundForm?.get("input_text") as string | null) ||
    (typeof inboundJson?.input_text === "string" ? inboundJson.input_text : "");

  if (explicitInputText && inputNodeId) {
    fd.append(
      `files[${inputNodeId}]`,
      new Blob([explicitInputText], { type: "text/plain" }),
      "input.txt"
    );
  } else if (inboundJson && Object.keys(inboundJson).length > 0 && inputNodeId) {
    fd.append(
      `files[${inputNodeId}]`,
      new Blob([JSON.stringify(inboundJson)], { type: "application/json" }),
      "input.json"
    );
  } else if (!inboundForm && inboundText && inputNodeId) {
    fd.append(
      `files[${inputNodeId}]`,
      new Blob([inboundText], { type: "text/plain" }),
      "input.txt"
    );
  }

  // Call Modal for inference
  const inferResp = await fetch(modalUrl(c.env, "infer"), {
    method: "POST", headers: modalHeaders(c.env), body: fd,
  });
  if (!inferResp.ok) {
    return c.json({ error: "Inference failed", status: inferResp.status }, 502);
  }
  const result = (await inferResp.json()) as Record<string, unknown>;

  // Optional LLM augmentation from request (preferred) or workflow KV fallback.
  let llmParams: Record<string, unknown> | null = null;
  const llmConfigRaw = inboundForm?.get("llm_config");
  if (typeof llmConfigRaw === "string" && llmConfigRaw.trim()) {
    try { llmParams = JSON.parse(llmConfigRaw) as Record<string, unknown>; } catch (_) {}
  } else if (inboundJson?.llm_config && typeof inboundJson.llm_config === "object") {
    llmParams = inboundJson.llm_config as Record<string, unknown>;
  } else {
    const nodes = ((wf?.nodes as Array<{ data?: { type?: string; parameters?: Record<string, unknown> } }>) || []);
    const llmNode = nodes.find((n) => n.data?.type === "llmNode");
    if (llmNode?.data?.parameters) llmParams = llmNode.data.parameters;
  }

  if (llmParams) {
    try {
      const llmBody = {
        provider:     llmParams.provider as string || "openai",
        model:        llmParams.model as string | undefined,
        ollamaUrl:    llmParams.ollamaUrl as string | undefined,
        systemPrompt: llmParams.systemPrompt as string || "Describe the model output.",
        messages: [
          { role: "user", content: `Model output: ${JSON.stringify(result.predictions || result)}` }
        ],
      };
      const llmResp = await fetch(new URL("/api/llm/chat", c.req.url).toString(), {
        method: "POST",
        headers: { "Content-Type": "application/json", ...modalHeaders(c.env) },
        body: JSON.stringify(llmBody),
      });
      const llmData = (await llmResp.json()) as { content?: string };
      return c.json({ result, llmResponse: llmData.content || "" });
    } catch (_) {
      return c.json({ result });
    }
  }

  return c.json({ result });
});

// ═════════════════════════════════════════════════════════════════════════════
// Node catalogue — complete static definition (no Modal dependency)
// ═════════════════════════════════════════════════════════════════════════════
const NODE_CATALOGUE = {
  text_input: {
    label: "Text Input", category: "input", output_type: "text",
    accepts_files: true, description: "Upload .txt/.pdf/.md/.docx/.html files.", params: {},
  },
  image_input: {
    label: "Image Input", category: "input", output_type: "image",
    accepts_files: true, description: "Upload .png/.jpg/.jpeg image files.", params: {},
  },
  audio_input: {
    label: "Audio Input", category: "input", output_type: "audio",
    accepts_files: true, description: "Upload .wav/.mp3/.flac audio files.", params: {},
  },
  spreadsheet_input: {
    label: "Spreadsheet Input", category: "input", output_type: "tabular",
    accepts_files: true, description: "Upload .csv/.xlsx/.json tabular files.", params: {},
  },
  api_input: {
    label: "API Input", category: "input", output_type: "any",
    accepts_files: false, description: "Fetch data from an external HTTP endpoint.",
    params: {
      url:       { type: "string",  description: "Endpoint URL",          default: "" },
      method:    { type: "string",  description: "HTTP method",           default: "GET" },
      data_path: { type: "string",  description: "JSONPath into response",default: "" },
    },
  },
  chunk: {
    label: "Chunk", category: "transform", output_type: "chunks",
    accepts_files: false, description: "Split text into training chunks (runs on Modal).",
    params: {
      method: { type: "string", description: "Chunking method (auto selects via AI)", default: "auto" },
    },
  },
  image_preprocess: {
    label: "Image Preprocess", category: "transform", output_type: "image",
    accepts_files: false, description: "Resize and normalise images for CNN input.",
    params: {
      resize:    { type: "integer", description: "Target size (px)",      default: 224 },
      normalize: { type: "boolean", description: "ImageNet normalisation", default: true },
    },
  },
  audio_preprocess: {
    label: "Audio Preprocess", category: "transform", output_type: "audio",
    accepts_files: false, description: "Convert audio to log-mel spectrograms.",
    params: {
      sample_rate: { type: "integer", description: "Target sample rate", default: 16000 },
      n_mels:      { type: "integer", description: "Mel bins",           default: 80 },
    },
  },
  tabular_preprocess: {
    label: "Tabular Preprocess", category: "transform", output_type: "tabular",
    accepts_files: false, description: "Encode, scale, and split tabular data.",
    params: {
      target_column:  { type: "string",  description: "Label column name",      default: "" },
      scale_features: { type: "boolean", description: "Standard-scale numerics", default: true },
    },
  },
  text_model: {
    label: "Text Model", category: "model", output_type: "model",
    accepts_files: false,
    description: "Fine-tune a sentence-transformer (SimCSE / MNRL / LoRA / SFT). Runs on Modal GPU.",
    params: {
      base_model:    { type: "string",  description: "Model name in /vol/models",          default: "all-MiniLM-L6-v2" },
      output_name:   { type: "string",  description: "Name to save the fine-tuned model",  default: "my-text-model" },
      method:        { type: "string",  description: "simcse | mnrl | lora | sft",         default: "simcse" },
      epochs:        { type: "integer", description: "Training epochs",                    default: 3 },
      batch_size:    { type: "integer", description: "Batch size",                         default: 32 },
      learning_rate: { type: "number",  description: "Learning rate",                      default: 3e-5 },
      gpu:           { type: "string",  description: "GPU type (T4 / A10G / A100 / H100)", default: "A10G" },
    },
  },
  cnn_model: {
    label: "CNN Model", category: "model", output_type: "model",
    accepts_files: false,
    description: "Train a CNN image classifier (ResNet / VGG / from scratch). Runs on Modal GPU.",
    params: {
      base_model:    { type: "string",  description: "resnet18 | resnet50 | vgg16 | none", default: "resnet18" },
      output_name:   { type: "string",  description: "Name to save the trained model",     default: "my-cnn" },
      num_classes:   { type: "integer", description: "Number of output classes",           default: 2 },
      epochs:        { type: "integer", description: "Training epochs",                    default: 10 },
      batch_size:    { type: "integer", description: "Batch size",                         default: 32 },
      learning_rate: { type: "number",  description: "Learning rate",                      default: 1e-3 },
      transfer:      { type: "boolean", description: "Freeze backbone (transfer learning)", default: true },
      gpu:           { type: "string",  description: "GPU type",                            default: "A10G" },
    },
  },
  rnn_model: {
    label: "RNN / LSTM / GRU", category: "model", output_type: "model",
    accepts_files: false,
    description: "Train an RNN/LSTM/GRU language model on text sequences. Runs on Modal GPU.",
    params: {
      rnn_type:      { type: "string",  description: "rnn | lstm | gru",         default: "lstm" },
      output_name:   { type: "string",  description: "Name to save the model",   default: "my-rnn" },
      hidden_dim:    { type: "integer", description: "Hidden state dimension",    default: 256 },
      num_layers:    { type: "integer", description: "Recurrent layers",          default: 2 },
      bidirectional: { type: "boolean", description: "Bidirectional RNN",         default: true },
      epochs:        { type: "integer", description: "Training epochs",           default: 10 },
      batch_size:    { type: "integer", description: "Batch size",                default: 64 },
      learning_rate: { type: "number",  description: "Learning rate",             default: 1e-3 },
      gpu:           { type: "string",  description: "GPU type",                  default: "T4" },
    },
  },
  model_save: {
    label: "Model Save", category: "output", output_type: null,
    accepts_files: false,
    description: "Save the trained model to Cloudflare R2 and the Modal Volume.", params: {},
  },
  infer_output: {
    label: "Inference Output", category: "output", output_type: "infer_out",
    accepts_files: false, description: "Return predictions from the pipeline.",
    params: { top_k: { type: "integer", description: "Top-K results per input", default: 5 } },
  },
  api_output: {
    label: "API Output", category: "output", output_type: null,
    accepts_files: false, description: "POST inference results to a webhook endpoint.",
    params: { url: { type: "string", description: "Webhook URL", default: "" } },
  },
};

export default app;
