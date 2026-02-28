"""
Modal compute layer — on-demand functions only.

The Cloudflare Worker IS the backend API.
Modal is called only when Python compute is required:
  • Chunking (needs Python NLP + pypdf + docx parsers)
  • Training  (needs GPU + PyTorch + transformers)
  • Inference (needs PyTorch model loading)
  • Search    (needs PyTorch embedding model)

Each operation is an independent @modal.web_endpoint() that
cold-starts on demand and shuts down when idle. No keep_warm.

After deploying (`modal deploy modal/app.py`) Modal prints a URL
for each endpoint like:
  https://<workspace>--hackillinois-pipeline--train.modal.run
Set MODAL_BASE = https://<workspace>--hackillinois-pipeline in
your Worker's wrangler.toml, then each endpoint is reachable at
  ${MODAL_BASE}--<label>.modal.run

Seed models into the volume once:
  modal run modal/app.py::seed_models
"""

from __future__ import annotations

import base64
import json
import os
import sys
import time
import uuid
from pathlib import Path

import modal
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import StreamingResponse

# ── Infrastructure ────────────────────────────────────────────────────────────
app = modal.App("hackillinois-pipeline")

models_volume = modal.Volume.from_name("hackillinois-models", create_if_missing=True)
MODELS_DIR    = Path("/vol/models")

_log_store = modal.Dict.from_name("pipeline-logs", create_if_missing=True)
_job_store = modal.Dict.from_name("pipeline-jobs", create_if_missing=True)

# Source mounts
_src_mount = modal.Mount.from_local_file(
    Path(__file__).parent.parent / "chunk.py", remote_path="/app/chunk.py"
)
_generate_mount = modal.Mount.from_local_file(
    Path(__file__).parent / "generate.py", remote_path="/app/generate.py"
)
_pipeline_mount = modal.Mount.from_local_dir(
    Path(__file__).parent.parent / "pipeline", remote_path="/app/pipeline"
)

# Container image for all functions
_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "fastapi[standard]>=0.111",
        "python-multipart",
        "openai>=1.30",
        "transformers>=4.40",
        "torch>=2.2",
        "torchvision>=0.17",
        "torchaudio>=2.2",
        "sentence-transformers>=3.0",
        "peft>=0.11",
        "pypdf>=4.0",
        "beautifulsoup4",
        "python-docx",
        "nltk",
        "lxml",
        "huggingface_hub",
        "boto3",
        "httpx",
        "Pillow",
        "scikit-learn",
        "numpy",
    )
    .run_commands(
        'python -c "import nltk; nltk.download(\'punkt\', quiet=True); '
        'nltk.download(\'punkt_tab\', quiet=True)"'
    )
)

GPU_MAP = {"T4": "T4", "A10G": "A10G", "A100": "A100", "H100": "H100"}


# ── Logging ───────────────────────────────────────────────────────────────────
def _log(job_id: str, msg: str, done: bool = False) -> None:
    print(msg, flush=True)
    existing: list = _log_store.get(job_id, [])
    entry: dict = {"ts": time.time(), "msg": msg}
    if done:
        entry["done"] = True
    existing.append(entry)
    _log_store[job_id] = existing


# ── GPU training function (internal, not a web endpoint) ──────────────────────
@app.function(
    image=_image,
    gpu="A10G",
    volumes={str(MODELS_DIR.parent): models_volume},
    timeout=7200,
    secrets=[modal.Secret.from_name("hackillinois-secrets")],
    mounts=[_src_mount, _generate_mount, _pipeline_mount],
)
def _gpu_train(
    job_id:     str,
    spec_dict:  dict,
    files:      dict,       # node_id → list of base64 strings
    file_names: dict,       # node_id → list of file names
    r2_config:  dict,
    actian_url: str,
) -> dict:
    """Execute a training pipeline on GPU. Called by the /train endpoint."""
    sys.path.insert(0, "/app")
    from pipeline.types import PipelineSpec       # type: ignore[import]
    from pipeline.executor import execute_pipeline # type: ignore[import]

    log = lambda msg: _log(job_id, msg)
    log(f"GPU pipeline starting — job {job_id}")

    decoded_files = {
        nid: [base64.b64decode(b) for b in blobs]
        for nid, blobs in files.items()
    }

    spec = PipelineSpec.from_dict(spec_dict)
    try:
        result = execute_pipeline(
            spec=spec,
            files=decoded_files,
            file_names=file_names,
            job_id=job_id,
            log_fn=log,
            models_dir=str(MODELS_DIR),
            actian_url=actian_url,
            r2_config=r2_config,
        )
        models_volume.commit()
        _job_store[job_id] = {"status": "complete", "result": result}
        _log(job_id, "Pipeline complete.", done=True)
        return result
    except Exception as exc:
        err = f"Pipeline error: {exc}"
        _job_store[job_id] = {"status": "error", "error": err}
        _log(job_id, err, done=True)
        raise


# ═════════════════════════════════════════════════════════════════════════════
# Web endpoints — each is an independent on-demand HTTP function
# URL pattern after deploy:
#   https://<workspace>--hackillinois-pipeline--<label>.modal.run
# ═════════════════════════════════════════════════════════════════════════════

# ── /chunk — chunk a file for preview ────────────────────────────────────────
@app.function(
    image=_image,
    timeout=120,
    secrets=[modal.Secret.from_name("hackillinois-secrets")],
    mounts=[_src_mount],
)
@modal.web_endpoint(method="POST", label="chunk")
async def chunk_endpoint(
    file: UploadFile = File(...),
    method: str       = Form(default="auto"),
    method_params: str= Form(default="{}"),
    preview_limit: int= Form(default=10),
):
    """Chunk an uploaded file and return a preview. Called by the CF Worker."""
    import inspect
    import tempfile

    sys.path.insert(0, "/app")
    from chunk import CHUNKERS, agent_choose_chunker, read_file  # type: ignore[import]

    params = json.loads(method_params)
    suffix = Path(file.filename or "upload").suffix or ".txt"

    content = await file.read()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        text = read_file(tmp_path)
        if len(text.strip()) < 10:
            return {"error": "Could not extract usable text."}, 400

        if method == "auto":
            chosen, chosen_params = agent_choose_chunker(tmp_path, text[:3000])
        else:
            if method not in CHUNKERS:
                return {"error": f"Unknown method '{method}'."}, 400
            chosen, chosen_params = method, params

        chunker  = CHUNKERS[chosen]
        valid_kw = {k: v for k, v in chosen_params.items()
                    if k in inspect.signature(chunker).parameters}
        chunks   = [c.strip() for c in chunker(text, **valid_kw)
                    if len(c.strip().split()) >= 5]

        return {
            "filename":    file.filename,
            "method_used": chosen,
            "params_used": valid_kw,
            "total_chunks": len(chunks),
            "preview":     chunks[:preview_limit],
        }
    finally:
        tmp_path.unlink(missing_ok=True)


# ── /train — generate script + spawn GPU job ──────────────────────────────────
@app.function(
    image=_image,
    timeout=120,                           # fast — just spawns the GPU job
    secrets=[modal.Secret.from_name("hackillinois-secrets")],
    mounts=[_generate_mount, _pipeline_mount],
)
@modal.web_endpoint(method="POST", label="train")
async def train_endpoint(request: Request):
    """
    Accepts a multipart form with:
      pipeline     — JSON string (PipelineSpec)
      gpu          — T4 | A10G | A100 | H100
      r2_*         — R2 credentials
      actian_url   — Actian HTTP gateway URL
      files[<node_id>] — uploaded files

    Generates the training script inside the GPU function and
    spawns it asynchronously. Returns job_id immediately (202).
    """
    sys.path.insert(0, "/app")
    from pipeline.types import PipelineSpec  # type: ignore[import]

    form = await request.form()

    spec_json = form.get("pipeline")
    if not spec_json:
        return {"error": "Missing 'pipeline' field."}, 400

    try:
        spec_dict = json.loads(spec_json)
        PipelineSpec.from_dict(spec_dict)  # validate
    except Exception as e:
        return {"error": f"Invalid pipeline JSON: {e}"}, 400

    gpu_str  = (form.get("gpu") or "A10G").upper()
    gpu_spec = GPU_MAP.get(gpu_str, "A10G")
    job_id   = str(uuid.uuid4())

    r2_config = {
        "endpoint": str(form.get("r2_endpoint") or ""),
        "key_id":   str(form.get("r2_key_id")   or ""),
        "secret":   str(form.get("r2_secret")    or ""),
        "bucket":   str(form.get("r2_bucket")    or ""),
    }
    actian_url = str(form.get("actian_url") or
                     os.environ.get("ACTIAN_HTTP_URL",
                                    "https://actian-http-gate.harihara-nalamotu.workers.dev"))

    # Collect uploaded files (multipart fields named files[<node_id>])
    files_b64:  dict[str, list[str]] = {}
    file_names: dict[str, list[str]] = {}
    for key, value in form.multi_items():
        if not (key.startswith("files[") and key.endswith("]")):
            continue
        node_id = key[6:-1]
        if isinstance(value, UploadFile):
            data = await value.read()
            files_b64.setdefault(node_id, []).append(base64.b64encode(data).decode())
            file_names.setdefault(node_id, []).append(value.filename or "upload")

    # Spawn GPU function
    call = _gpu_train.with_options(gpu=gpu_spec).spawn(
        job_id=job_id,
        spec_dict=spec_dict,
        files=files_b64,
        file_names=file_names,
        r2_config=r2_config,
        actian_url=actian_url,
    )

    _job_store[job_id] = {"status": "running", "modal_call_id": call.object_id}
    _log_store[job_id] = [{"ts": time.time(), "msg": f"Job {job_id} queued on {gpu_spec}"}]

    return {"job_id": job_id, "status": "running", "gpu": gpu_spec}


# ── /infer — run inference pipeline (CPU, synchronous) ───────────────────────
@app.function(
    image=_image,
    volumes={str(MODELS_DIR.parent): models_volume},
    timeout=300,
    secrets=[modal.Secret.from_name("hackillinois-secrets")],
    mounts=[_src_mount, _generate_mount, _pipeline_mount],
)
@modal.web_endpoint(method="POST", label="infer")
async def infer_endpoint(request: Request):
    """Run an inference pipeline synchronously on CPU. Returns results directly."""
    sys.path.insert(0, "/app")
    from pipeline.types import PipelineSpec      # type: ignore[import]
    from pipeline.executor import execute_pipeline  # type: ignore[import]

    form = await request.form()
    spec_json = form.get("pipeline")
    if not spec_json:
        return {"error": "Missing 'pipeline' field."}, 400

    try:
        spec_dict = json.loads(spec_json)
        spec = PipelineSpec.from_dict(spec_dict)
    except Exception as e:
        return {"error": f"Invalid pipeline JSON: {e}"}, 400

    actian_url = str(form.get("actian_url") or
                     os.environ.get("ACTIAN_HTTP_URL",
                                    "https://actian-http-gate.harihara-nalamotu.workers.dev"))
    job_id = str(uuid.uuid4())

    files_raw:  dict[str, list[bytes]] = {}
    file_names: dict[str, list[str]]   = {}
    for key, value in form.multi_items():
        if not (key.startswith("files[") and key.endswith("]")):
            continue
        node_id = key[6:-1]
        if isinstance(value, UploadFile):
            data = await value.read()
            files_raw.setdefault(node_id, []).append(data)
            file_names.setdefault(node_id, []).append(value.filename or "upload")

    logs: list[str] = []
    result = execute_pipeline(
        spec=spec,
        files=files_raw,
        file_names=file_names,
        job_id=job_id,
        log_fn=lambda m: logs.append(m),
        models_dir=str(MODELS_DIR),
        actian_url=actian_url,
    )
    return {**result, "job_id": job_id, "logs": logs}


# ── /job-status — poll a training job ────────────────────────────────────────
@app.function(image=_image, timeout=30)
@modal.web_endpoint(method="GET", label="job-status")
def job_status_endpoint(job_id: str):
    """
    Query param: ?job_id=<id>
    Returns the job status dict.
    """
    stored = _job_store.get(job_id)
    if not stored:
        return {"status": "not_found"}

    if stored.get("status") == "running":
        call_id = stored.get("modal_call_id")
        if call_id:
            try:
                call   = modal.functions.FunctionCall.from_id(call_id)
                result = call.get(timeout=0)
                updated = {"status": "complete", "result": result}
                _job_store[job_id] = updated
                return updated
            except TimeoutError:
                pass
            except Exception as e:
                err = {"status": "error", "error": str(e)}
                _job_store[job_id] = err
                return err
    return stored


# ── /job-logs — SSE log stream ────────────────────────────────────────────────
@app.function(image=_image, timeout=3600)
@modal.web_endpoint(method="GET", label="job-logs")
def job_logs_endpoint(job_id: str):
    """
    Query param: ?job_id=<id>
    Returns a Server-Sent Events stream of log messages.
    """
    def _generate():
        cursor   = 0
        deadline = time.time() + 3600
        while time.time() < deadline:
            logs: list[dict] = _log_store.get(job_id, [])
            for entry in logs[cursor:]:
                cursor += 1
                yield f"data: {json.dumps(entry)}\n\n"
                if entry.get("done"):
                    return
            if cursor == len(logs):
                time.sleep(0.5)
        yield f"data: {json.dumps({'msg': 'Stream timeout.', 'done': True})}\n\n"

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── /search — embed query + search Actian ────────────────────────────────────
@app.function(
    image=_image,
    volumes={str(MODELS_DIR.parent): models_volume},
    timeout=60,
    secrets=[modal.Secret.from_name("hackillinois-secrets")],
)
@modal.web_endpoint(method="POST", label="search")
async def search_endpoint(request: Request):
    """
    Embed a natural-language query with the fine-tuned model and
    run nearest-neighbour search against the Actian VectorAI DB.

    Body JSON:
      query       string  — query text
      collection  string  — Actian collection name
      model_id    string  — model folder in /vol/models
      top_k       int     — results (default 10)
      with_payload bool   — include payloads (default true)
    """
    import httpx
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModel  # type: ignore[import]

    body       = await request.json()
    query      = body.get("query", "").strip()
    collection = body.get("collection", "corpus_embeddings")
    model_id   = body.get("model_id", "finetuned-MiniLM-corpus")
    top_k      = int(body.get("top_k", 10))
    with_pay   = bool(body.get("with_payload", True))

    actian_url = os.environ.get("ACTIAN_HTTP_URL",
                                "https://actian-http-gate.harihara-nalamotu.workers.dev")

    if not query:
        return {"error": "query must not be empty."}, 400

    model_path = str(MODELS_DIR / model_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok  = AutoTokenizer.from_pretrained(model_path)
    emb  = AutoModel.from_pretrained(model_path).to(device)
    emb.eval()

    def _pool(h, m):
        e = m.unsqueeze(-1).expand(h.size()).float()
        return torch.sum(h * e, 1) / torch.clamp(e.sum(1), min=1e-9)

    with torch.no_grad():
        enc  = tok(query, return_tensors="pt", truncation=True, max_length=128).to(device)
        out  = emb(**enc)
        vec  = F.normalize(_pool(out.last_hidden_state, enc["attention_mask"]), dim=-1)
        vec_list = vec.cpu().squeeze().tolist()

    resp = httpx.post(f"{actian_url}/search", json={
        "collection":   collection,
        "query":        vec_list,
        "top_k":        top_k,
        "with_payload": with_pay,
    }, timeout=30)

    results = resp.json().get("results", []) if resp.is_success else []
    return {"query": query, "collection": collection,
            "results": results, "total": len(results)}


# ── Seed models into Modal Volume (local entrypoint) ──────────────────────────
@app.function(
    image=_image,
    volumes={str(MODELS_DIR.parent): models_volume},
    mounts=[modal.Mount.from_local_dir(
        Path(__file__).parent.parent / "models",
        remote_path="/local_models",
    )],
)
def _do_seed():
    import shutil
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    copied = []
    for p in sorted(Path("/local_models").iterdir()):
        dst = MODELS_DIR / p.name
        if dst.exists():
            print(f"  skip: {p.name}")
            continue
        shutil.copytree(str(p), str(dst))
        copied.append(p.name)
        print(f"  seeded: {p.name}")
    models_volume.commit()
    return copied


@app.local_entrypoint()
def seed_models():
    """Copy ./models/* into the Modal Volume. Run once after first deploy."""
    print("Seeding models → Modal Volume …")
    copied = _do_seed.remote()
    print(f"Seeded: {copied}" if copied else "Nothing new to seed.")
