"""
Modal backend — called by the Cloudflare Worker.

Exposes HTTP endpoints (via @modal.asgi_app) for:
  POST /chunk          — chunk uploaded files
  POST /script         — generate training script from node params
  POST /train          — spawn GPU fine-tuning job, return job_id
  GET  /status/{id}    — poll job status
  GET  /logs/{id}      — SSE stream of training logs

Deploy:
  modal deploy modal/app.py

Local dev (serves on localhost:8000):
  modal serve modal/app.py

Seed local models into the Modal Volume (run once):
  modal run modal/app.py::seed_models
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import uuid
from pathlib import Path

import modal

# ── Infrastructure ────────────────────────────────────────────────────────────
app = modal.App("hackillinois-rag")

models_volume = modal.Volume.from_name("hackillinois-models", create_if_missing=True)
MODELS_DIR = Path("/vol/models")

_log_store = modal.Dict.from_name("training-logs", create_if_missing=True)
_job_store = modal.Dict.from_name("training-jobs", create_if_missing=True)

# Mount Python source files from repo root into containers
_src_mount = modal.Mount.from_local_file(
    Path(__file__).parent.parent / "chunk.py", remote_path="/app/chunk.py"
)
_gen_mount = modal.Mount.from_local_file(
    Path(__file__), remote_path="/app/modal_app.py"
)
_generate_mount = modal.Mount.from_local_file(
    Path(__file__).parent / "generate.py", remote_path="/app/generate.py"
)

# Container image shared by all functions
_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "fastapi[standard]>=0.111",
        "python-multipart",
        "openai>=1.30",
        "python-dotenv",
        "transformers>=4.40",
        "torch>=2.2",
        "sentence-transformers>=3.0",
        "peft>=0.11",
        "pypdf>=4.0",
        "beautifulsoup4",
        "python-docx",
        "nltk",
        "lxml",
        "huggingface_hub",
        "boto3",          # R2 (S3-compatible) uploads
    )
    .run_commands(
        "python -c \""
        "import nltk; "
        "nltk.download('punkt', quiet=True); "
        "nltk.download('punkt_tab', quiet=True)"
        "\""
    )
)

# ── GPU training function ─────────────────────────────────────────────────────
GPU_MAP = {"T4": "T4", "A10G": "A10G", "A100": "A100", "H100": "H100"}

@app.function(
    image=_image,
    gpu="A10G",                          # overridden per-job via .with_options()
    volumes={str(MODELS_DIR.parent): models_volume},
    timeout=7200,
    secrets=[modal.Secret.from_name("hackillinois-secrets")],
    mounts=[_src_mount, _generate_mount],
)
def _run_training(
    job_id: str,
    script: str,
    chunks: list[str],
    output_model_name: str,
    # R2 upload config
    r2_endpoint: str,
    r2_key_id: str,
    r2_secret: str,
    r2_bucket: str,
    r2_prefix: str,
) -> dict:
    """
    Executes a generated training script inside a GPU container,
    then uploads the resulting model weights to Cloudflare R2.
    """
    sys.path.insert(0, "/app")

    def log(msg: str):
        print(msg, flush=True)
        existing = _log_store.get(job_id, [])
        existing.append({"ts": time.time(), "msg": msg})
        _log_store[job_id] = existing

    log(f"[{job_id}] Starting — {len(chunks)} chunks")

    # Write chunks and script to a temp workspace
    with tempfile.TemporaryDirectory() as workspace:
        chunks_path = os.path.join(workspace, "chunks.json")
        script_path = os.path.join(workspace, "train.py")
        output_dir  = os.path.join(workspace, "output")

        with open(chunks_path, "w") as f:
            json.dump(chunks, f)

        # Patch the script's paths to the actual workspace paths
        patched = (
            script
            .replace('"chunks.json"',  f'r"{chunks_path}"')
            .replace("'chunks.json'",  f"r'{chunks_path}'")
            .replace('"./output_model"', f'r"{output_dir}"')
            .replace("'./output_model'", f"r'{output_dir}'")
        )

        with open(script_path, "w") as f:
            f.write(patched)

        log("Running training script...")
        import subprocess
        result = subprocess.run(
            ["python", script_path],
            capture_output=False,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(f"Training script exited with code {result.returncode}")

        log("Training complete. Uploading weights to R2...")

        # Upload model files to Cloudflare R2
        import boto3
        s3 = boto3.client(
            "s3",
            endpoint_url=r2_endpoint,
            aws_access_key_id=r2_key_id,
            aws_secret_access_key=r2_secret,
        )

        uploaded = []
        for root, _, files in os.walk(output_dir):
            for fname in files:
                local_path = os.path.join(root, fname)
                rel_path   = os.path.relpath(local_path, output_dir)
                r2_key     = f"{r2_prefix}/{rel_path}"
                s3.upload_file(local_path, r2_bucket, r2_key)
                uploaded.append(r2_key)
                log(f"  Uploaded: {r2_key}")

        # Write model metadata to R2
        meta = {
            "id": output_model_name,
            "is_finetuned": True,
            "job_id": job_id,
            "chunks_trained_on": len(chunks),
        }
        s3.put_object(
            Bucket=r2_bucket,
            Key=f"{r2_prefix}/meta.json",
            Body=json.dumps(meta),
            ContentType="application/json",
        )

        log(f"Upload complete — {len(uploaded)} files.")

    # ── Upload embeddings to Actian VectorAI DB ──────────────────────────────
    vdss_host = os.environ.get("VDSS_HOST", "localhost")
    vdss_port = int(os.environ.get("VDSS_PORT", "50051"))
    vectors_uploaded = 0

    try:
        import grpc
        import torch
        import torch.nn.functional as F
        from transformers import AutoTokenizer, AutoModel

        sys.path.insert(0, str(Path(output_path).parent.parent))
        from vdss_client import (
            vdss_service_pb2      as svc_pb2,
            vdss_service_pb2_grpc as svc_grpc,
            vdss_types_pb2        as types_pb2,
        )

        log(f"Connecting to Actian VectorAI DB at {vdss_host}:{vdss_port} ...")
        channel = grpc.insecure_channel(f"{vdss_host}:{vdss_port}")
        stub    = svc_grpc.VDSSServiceStub(channel)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tok    = AutoTokenizer.from_pretrained(output_path)
        emb    = AutoModel.from_pretrained(output_path).to(device)
        emb.eval()

        def _mean_pool(h, m):
            e = m.unsqueeze(-1).expand(h.size()).float()
            return torch.sum(h * e, 1) / torch.clamp(e.sum(1), min=1e-9)

        def embed_batch(texts):
            enc = tok(texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
            with torch.no_grad():
                out = emb(**enc)
            v = F.normalize(_mean_pool(out.last_hidden_state, enc["attention_mask"]), dim=-1)
            return v.cpu().tolist()

        # Probe dimension
        probe = embed_batch(["probe"])
        dim   = len(probe[0])
        collection_name = f"{output_model_name}_embeddings"

        cfg = types_pb2.CollectionConfig(
            index_driver=types_pb2.FAISS,
            index_algorithm=types_pb2.HNSW,
            storage_type=types_pb2.MEM,
            dimension=dim,
            distance_metric=types_pb2.COSINE,
            hnsw_config=types_pb2.HnswConfig(m=16, ef_construct=200, ef_search=100),
        )
        create_resp = stub.CreateCollection(svc_pb2.CreateCollectionRequest(
            collection_name=collection_name, config=cfg
        ))
        if create_resp.status.code != 0:
            stub.OpenCollection(svc_pb2.OpenCollectionRequest(collection_name=collection_name))

        log(f"Uploading {len(chunks)} vectors to collection '{collection_name}' ...")
        BATCH = 256
        for start in range(0, len(chunks), BATCH):
            batch_texts = chunks[start:start + BATCH]
            batch_vecs  = embed_batch(batch_texts)
            stub.BatchUpsert(svc_pb2.BatchUpsertRequest(
                collection_name=collection_name,
                vector_ids=[types_pb2.VectorIdentifier(u64_id=start + i) for i in range(len(batch_texts))],
                vectors=[types_pb2.Vector(data=v, dimension=dim) for v in batch_vecs],
                payloads=[types_pb2.Payload(json=json.dumps({"text": t, "id": start + i, "model": output_model_name}))
                          for i, t in enumerate(batch_texts)],
            ))
            vectors_uploaded += len(batch_texts)

        stub.Flush(svc_pb2.FlushRequest(collection_name=collection_name))
        stub.SaveSnapshot(svc_pb2.SaveSnapshotRequest(collection_name=collection_name))
        log(f"Vectors uploaded: {vectors_uploaded} → collection '{collection_name}'")

    except Exception as e:
        log(f"[WARN] Vector upload skipped: {e}")

    # Mark job done in shared store
    _job_store[job_id] = {
        "status": "complete",
        "output_model": output_model_name,
        "vectors_uploaded": vectors_uploaded,
    }
    # Append terminal log entry so SSE stream closes
    existing = _log_store.get(job_id, [])
    existing.append({"ts": time.time(), "msg": "Done.", "done": True})
    _log_store[job_id] = existing

    return {
        "status": "complete",
        "output_model": output_model_name,
        "files_uploaded": len(uploaded),
        "vectors_uploaded": vectors_uploaded,
    }


# ── ASGI app (served via Modal web endpoint) ──────────────────────────────────
@app.function(
    image=_image,
    volumes={str(MODELS_DIR.parent): models_volume},
    secrets=[modal.Secret.from_name("hackillinois-secrets")],
    mounts=[_src_mount, _generate_mount],
    keep_warm=1,
    timeout=120,
)
@modal.asgi_app()
def web() -> "fastapi.FastAPI":  # type: ignore[name-defined]
    import inspect

    import fastapi
    from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import StreamingResponse

    sys.path.insert(0, "/app")
    from chunk import CHUNKERS, TOOLS, agent_choose_chunker, read_file
    from generate import generate_script

    # ── Security middleware ───────────────────────────────────────────────────
    MODAL_SECRET = os.environ.get("MODAL_SECRET", "")

    api = FastAPI(title="Hackillinois Modal Backend", docs_url="/docs")

    api.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @api.middleware("http")
    async def require_secret(request: fastapi.Request, call_next):
        if request.method == "OPTIONS":
            return await call_next(request)
        if MODAL_SECRET and request.headers.get("X-Modal-Secret") != MODAL_SECRET:
            return fastapi.responses.JSONResponse({"error": "Unauthorized"}, status_code=401)
        return await call_next(request)

    # ── POST /chunk ───────────────────────────────────────────────────────────
    @api.post("/chunk")
    async def chunk_file(
        file: UploadFile = File(...),
        method: str = Form(default="auto"),
        method_params: str = Form(default="{}"),
        preview_limit: int = Form(default=10),
    ):
        params = json.loads(method_params)
        suffix = Path(file.filename or "upload").suffix or ".txt"

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(await file.read())
            tmp_path = Path(tmp.name)

        try:
            text = read_file(tmp_path)
            if len(text.strip()) < 10:
                raise HTTPException(400, "Could not extract usable text from the file.")

            if method == "auto":
                chosen, chosen_params = agent_choose_chunker(tmp_path, text[:3000])
            else:
                if method not in CHUNKERS:
                    raise HTTPException(400, f"Unknown method '{method}'.")
                chosen, chosen_params = method, params

            chunker = CHUNKERS[chosen]
            valid_kw = {k: v for k, v in chosen_params.items() if k in inspect.signature(chunker).parameters}
            chunks = [c.strip() for c in chunker(text, **valid_kw) if len(c.strip().split()) >= 5]

            return {
                "filename": file.filename,
                "method_used": chosen,
                "params_used": valid_kw,
                "total_chunks": len(chunks),
                "preview": chunks[:preview_limit],
            }
        finally:
            tmp_path.unlink(missing_ok=True)

    # ── POST /script ──────────────────────────────────────────────────────────
    @api.post("/script")
    async def generate(request: Request):
        """Generate a training script from node parameters without executing it."""
        params = await request.json()
        try:
            script = generate_script(params)
        except ValueError as e:
            raise HTTPException(400, str(e))
        return {
            "script": script,
            "method": params.get("method", "simcse"),
            "base_model": params.get("base_model"),
        }

    # ── POST /train ───────────────────────────────────────────────────────────
    @api.post("/train", status_code=202)
    async def start_training(
        files: list[UploadFile] = File(...),
        job_id: str             = Form(...),
        base_model: str         = Form(...),
        output_model_name: str  = Form(...),
        # Training method params (passed through to script generator)
        method: str             = Form(default="simcse"),
        epochs: int             = Form(default=3),
        batch_size: int         = Form(default=32),
        learning_rate: float    = Form(default=3e-5),
        temperature: float      = Form(default=0.05),
        warmup_steps: int       = Form(default=100),
        lora_r: int             = Form(default=16),
        lora_alpha: int         = Form(default=32),
        lora_dropout: float     = Form(default=0.1),
        gpu: str                = Form(default="A10G"),
        # Chunking
        chunking_method: str    = Form(default="auto"),
        method_params: str      = Form(default="{}"),
        # R2 upload credentials (injected by CF Worker)
        r2_endpoint: str        = Form(...),
        r2_key_id: str          = Form(...),
        r2_secret: str          = Form(...),
        r2_bucket: str          = Form(...),
        r2_model_prefix: str    = Form(...),
    ):
        base_model_path = str(MODELS_DIR / base_model)
        if not Path(base_model_path).exists():
            raise HTTPException(404, f"Base model '{base_model}' not found.")

        # ── Collect and chunk all uploaded files ──────────────────────────────
        chunk_params = json.loads(method_params)
        all_chunks: list[str] = []
        skipped: list[dict] = []

        with tempfile.TemporaryDirectory() as tmpdir:
            for upload in files:
                suffix = Path(upload.filename or "upload").suffix or ".txt"
                tmp_path = Path(tmpdir) / f"{uuid.uuid4()}{suffix}"
                tmp_path.write_bytes(await upload.read())
                try:
                    text = read_file(tmp_path)
                    if len(text.strip()) < 30:
                        raise ValueError("Extracted text too short.")
                    if chunking_method == "auto":
                        c_method, c_params = agent_choose_chunker(tmp_path, text[:3000])
                    else:
                        c_method, c_params = chunking_method, chunk_params
                    chunker = CHUNKERS[c_method]
                    valid_kw = {k: v for k, v in c_params.items() if k in inspect.signature(chunker).parameters}
                    fc = [c.strip() for c in chunker(text, **valid_kw) if len(c.strip().split()) >= 5]
                    all_chunks.extend(fc)
                except Exception as e:
                    skipped.append({"file": upload.filename, "error": str(e)[:120]})

        if not all_chunks:
            raise HTTPException(422, "No valid chunks extracted from the uploaded files.")

        # ── Generate training script ──────────────────────────────────────────
        script_params = {
            "method":         method,
            "base_model":     base_model_path,
            "output_dir":     "./output_model",
            "epochs":         epochs,
            "batch_size":     batch_size,
            "learning_rate":  learning_rate,
            "temperature":    temperature,
            "warmup_steps":   warmup_steps,
            "lora_r":         lora_r,
            "lora_alpha":     lora_alpha,
            "lora_dropout":   lora_dropout,
            "gpu":            gpu,
        }
        script = generate_script(script_params)

        # ── Spawn GPU job with requested GPU type ─────────────────────────────
        gpu_spec = GPU_MAP.get(gpu.upper(), "A10G")
        call = _run_training.with_options(gpu=gpu_spec).spawn(
            job_id=job_id,
            script=script,
            chunks=all_chunks,
            output_model_name=output_model_name,
            r2_endpoint=r2_endpoint,
            r2_key_id=r2_key_id,
            r2_secret=r2_secret,
            r2_bucket=r2_bucket,
            r2_prefix=r2_model_prefix,
        )

        _job_store[job_id] = {"status": "running", "modal_call_id": call.object_id}

        return {
            "job_id": job_id,
            "modal_call_id": call.object_id,
            "status": "running",
            "chunks_collected": len(all_chunks),
            "files_skipped": skipped,
            "script_method": method,
            "gpu": gpu_spec,
        }

    # ── GET /status/{job_id} ──────────────────────────────────────────────────
    @api.get("/status/{job_id}")
    def get_status(job_id: str):
        stored = _job_store.get(job_id)
        if not stored:
            return {"status": "not_found"}

        if stored.get("status") == "running":
            call_id = stored.get("modal_call_id")
            if call_id:
                try:
                    call = modal.functions.FunctionCall.from_id(call_id)
                    result = call.get(timeout=0)
                    updated = {"status": "complete", "result": result}
                    _job_store[job_id] = updated
                    return updated
                except TimeoutError:
                    pass
                except Exception as e:
                    error = {"status": "error", "error": str(e)}
                    _job_store[job_id] = error
                    return error

        return stored

    # ── GET /logs/{job_id} (SSE) ──────────────────────────────────────────────
    @api.get("/logs/{job_id}")
    def stream_logs(job_id: str):
        def generate():
            cursor = 0
            deadline = time.time() + 3600  # 1-hour max stream

            while time.time() < deadline:
                logs: list[dict] = _log_store.get(job_id, [])
                new_entries = logs[cursor:]

                for entry in new_entries:
                    cursor += 1
                    yield f"data: {json.dumps(entry)}\n\n"
                    if entry.get("done"):
                        return

                if not new_entries:
                    time.sleep(0.5)

            yield f"data: {json.dumps({'msg': 'Stream timeout.', 'done': True})}\n\n"

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # ── POST /search — semantic search via Actian VectorAI DB ────────────────
    @api.post("/search")
    async def search(request: Request):
        """
        Embed a query with the fine-tuned model and run nearest-neighbour
        search against the Actian VectorAI DB collection.

        Body (JSON):
          query        string   — the natural-language query
          collection   string   — collection name (default: corpus_embeddings)
          model_id     string   — model folder name (default: finetuned-MiniLM-corpus)
          top_k        int      — results to return (default: 10)
          with_payload bool     — include original text in results (default: true)
          vdss_host    string   — VectorAI DB host (default: localhost)
          vdss_port    int      — VectorAI DB port (default: 50051)
        """
        import sys
        sys.path.insert(0, "/app")

        import grpc
        import torch
        import torch.nn.functional as F
        from transformers import AutoTokenizer, AutoModel

        # Import the compiled gRPC stubs (mounted into the container)
        sys.path.insert(0, str(MODELS_DIR.parent.parent))
        from vdss_client import (
            vdss_service_pb2      as svc_pb2,
            vdss_service_pb2_grpc as svc_grpc,
            vdss_types_pb2        as types_pb2,
        )

        body = await request.json()
        query      = body.get("query", "").strip()
        collection = body.get("collection", "corpus_embeddings")
        model_id   = body.get("model_id", "finetuned-MiniLM-corpus")
        top_k      = int(body.get("top_k", 10))
        with_pay   = bool(body.get("with_payload", True))
        vdss_host  = body.get("vdss_host", os.environ.get("VDSS_HOST", "localhost"))
        vdss_port  = int(body.get("vdss_port", os.environ.get("VDSS_PORT", "50051")))

        if not query:
            raise HTTPException(400, "query must not be empty.")

        # Load model (cached between warm requests)
        model_path = str(MODELS_DIR / model_id)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        emb_model = AutoModel.from_pretrained(model_path).to(device)
        emb_model.eval()

        def _mean_pool(h, m):
            e = m.unsqueeze(-1).expand(h.size()).float()
            return torch.sum(h * e, 1) / torch.clamp(e.sum(1), min=1e-9)

        with torch.no_grad():
            enc = tokenizer(query, return_tensors="pt", truncation=True, max_length=128).to(device)
            out = emb_model(**enc)
            vec = F.normalize(_mean_pool(out.last_hidden_state, enc["attention_mask"]), dim=-1)
            vec_list = vec.cpu().squeeze().tolist()

        # Query Actian VectorAI DB
        channel = grpc.insecure_channel(f"{vdss_host}:{vdss_port}")
        stub    = svc_grpc.VDSSServiceStub(channel)

        query_vec = types_pb2.Vector(data=vec_list, dimension=len(vec_list))
        resp = stub.Search(svc_pb2.SearchRequest(
            collection_name=collection,
            query=query_vec,
            top_k=top_k,
            with_payload=with_pay,
            with_vector=False,
        ))

        results = []
        for r in resp.results:
            entry: dict = {"score": r.score, "id": r.id.u64_id}
            if with_pay and r.HasField("payload"):
                try:
                    entry["payload"] = json.loads(r.payload.json)
                except Exception:
                    entry["payload"] = r.payload.json
            results.append(entry)

        return {
            "query": query,
            "collection": collection,
            "results": results,
            "total": len(results),
        }

    return api


# ── Seed local models into Modal Volume ───────────────────────────────────────
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
            print(f"  skip (exists): {p.name}")
            continue
        print(f"  seeding: {p.name}")
        shutil.copytree(str(p), str(dst))
        copied.append(p.name)
    models_volume.commit()
    return copied


@app.local_entrypoint()
def seed_models():
    """
    Copy ./models/* into the Modal Volume.
    Run once after first deploy:
        modal run modal/app.py::seed_models
    """
    print("Seeding models → Modal Volume ...")
    copied = _do_seed.remote()
    print(f"Seeded: {copied}" if copied else "Nothing new to seed.")
