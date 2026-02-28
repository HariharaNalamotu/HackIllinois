"""
FastAPI backend for corpus fine-tuning.

Can run standalone (local dev) or be mounted inside Modal (production).
Fine-tuning backend is injected via app.state so Modal can swap in a GPU worker.

Local dev:
  uvicorn api:app --reload

Routes:
  GET  /api/health
  GET  /api/models
  DELETE /api/models/{model_id}
  GET  /api/chunking/methods
  POST /api/chunking/preview
  POST /api/finetune
  GET  /api/finetune/{job_id}
"""

import inspect
import json
import os
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Any, Callable, Optional

import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

load_dotenv()

# ── Local job store (thread-safe in-memory dict) ─────────────────────────────
class LocalJobStore:
    def __init__(self):
        self._store: dict[str, dict] = {}
        self._lock = threading.Lock()

    def get(self, job_id: str) -> dict:
        with self._lock:
            return self._store.get(job_id, {"status": "not_found"})

    def put(self, job_id: str, data: dict):
        with self._lock:
            self._store[job_id] = data

_local_job_store = LocalJobStore()


# ── Local fine-tuning (runs in a background thread) ──────────────────────────
def _simcse_train(
    chunks: list[str],
    base_model_path: str,
    output_path: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    temperature: float,
    max_length: int,
) -> dict:
    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu")
    )

    class TextDS(Dataset):
        def __init__(self, t): self.t = t
        def __len__(self): return len(self.t)
        def __getitem__(self, i): return self.t[i]

    def mean_pool(h, m):
        e = m.unsqueeze(-1).expand(h.size()).float()
        return torch.sum(h * e, 1) / torch.clamp(e.sum(1), min=1e-9)

    def simcse_loss(e1, e2, temp):
        B = e1.size(0)
        e1, e2 = F.normalize(e1, -1), F.normalize(e2, -1)
        ae = torch.cat([e1, e2])
        sim = torch.mm(ae, ae.t()) / temp
        sim = sim.masked_fill(torch.eye(2 * B, dtype=torch.bool, device=e1.device), float("-inf"))
        labels = torch.cat([torch.arange(B, 2 * B, device=e1.device), torch.arange(B, device=e1.device)])
        return F.cross_entropy(sim, labels)

    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    model = AutoModel.from_pretrained(base_model_path).to(device)
    model.train()

    bs = max(2, min(batch_size, len(chunks) // 2 if len(chunks) < batch_size * 2 else batch_size))
    loader = DataLoader(TextDS(chunks), batch_size=bs, shuffle=True, drop_last=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    history = []

    for epoch in range(1, epochs + 1):
        total = 0.0
        for batch in loader:
            def encode(s):
                enc = tokenizer(s, padding=True, truncation=True,
                                max_length=max_length, return_tensors="pt").to(device)
                return mean_pool(model(**enc).last_hidden_state, enc["attention_mask"])
            loss = simcse_loss(encode(batch), encode(batch), temperature)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total += loss.item()
        avg = total / len(loader)
        history.append({"epoch": epoch, "avg_loss": round(avg, 6)})

    Path(output_path).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    return {"status": "complete", "output_model": Path(output_path).name, "training_history": history}


def local_finetune_fn(job_id: str, job_store: LocalJobStore, **kwargs):
    """Runs fine-tuning in a background thread, updates job_store on completion."""
    def _run():
        try:
            result = _simcse_train(**kwargs)
            job_store.put(job_id, {"status": "complete", "result": result})
        except Exception as e:
            job_store.put(job_id, {"status": "error", "error": str(e)})

    job_store.put(job_id, {"status": "running"})
    threading.Thread(target=_run, daemon=True).start()
    return job_id


# ── App factory ───────────────────────────────────────────────────────────────
def create_app(
    models_dir: Path,
    spawn_finetune: Optional[Callable] = None,
    job_store: Optional[LocalJobStore] = None,
) -> FastAPI:
    """
    Build the FastAPI app.

    Args:
        models_dir:      Path to directory containing model folders.
        spawn_finetune:  Callable(job_id, **kwargs) → job_id. Defaults to local thread runner.
        job_store:       Job status store. Defaults to LocalJobStore.
    """
    from chunk import CHUNKERS, TOOLS, agent_choose_chunker, read_file

    if job_store is None:
        job_store = _local_job_store

    if spawn_finetune is None:
        def spawn_finetune(job_id: str, **kwargs):
            return local_finetune_fn(job_id, job_store, **kwargs)

    api = FastAPI(
        title="Corpus Fine-Tuning API",
        description=(
            "Upload a corpus, choose a chunking strategy (or let OpenAI decide), "
            "and fine-tune a sentence embedding model via SimCSE."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    api.add_middleware(
        CORSMiddleware,
        allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Health ────────────────────────────────────────────────────────────────
    @api.get("/api/health", tags=["System"], summary="Health check")
    def health():
        return {
            "status": "ok",
            "models_dir": str(models_dir),
            "device": (
                "cuda" if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available()
                else "cpu"
            ),
        }

    # ── Models ────────────────────────────────────────────────────────────────
    @api.get("/api/models", tags=["Models"], summary="List available models")
    def list_models():
        """
        Returns all model folders found in the models directory.
        Both base models and fine-tuned models are listed.
        `is_finetuned` is true if 'finetuned' appears in the folder name.
        """
        if not models_dir.exists():
            return {"models": [], "total": 0}

        result = []
        for p in sorted(models_dir.iterdir()):
            cfg_file = p / "config.json"
            if not (p.is_dir() and cfg_file.exists()):
                continue
            cfg = json.loads(cfg_file.read_text())
            result.append({
                "id": p.name,
                "architecture": (cfg.get("architectures") or ["unknown"])[0],
                "hidden_size": cfg.get("hidden_size"),
                "max_position_embeddings": cfg.get("max_position_embeddings"),
                "is_finetuned": "finetuned" in p.name.lower(),
            })

        return {"models": result, "total": len(result)}

    @api.delete(
        "/api/models/{model_id}",
        tags=["Models"],
        summary="Delete a fine-tuned model",
    )
    def delete_model(model_id: str):
        """Only fine-tuned models (is_finetuned=true) can be deleted."""
        target = models_dir / model_id
        if not target.exists():
            raise HTTPException(404, f"Model '{model_id}' not found.")
        if "finetuned" not in model_id.lower():
            raise HTTPException(403, "Only fine-tuned models can be deleted via the API.")
        import shutil
        shutil.rmtree(target)
        return {"deleted": model_id}

    # ── Chunking ──────────────────────────────────────────────────────────────
    @api.get("/api/chunking/methods", tags=["Chunking"], summary="List chunking methods")
    def list_methods():
        """
        Returns all available chunking methods with descriptions and parameter schemas.
        Pass method='auto' in chunking/preview or finetune to let OpenAI choose.
        """
        methods = {}
        for tool in TOOLS:
            fn = tool["function"]
            methods[fn["name"]] = {
                "description": fn["description"].strip(),
                "parameters": {
                    k: {
                        "type": v.get("type"),
                        "description": v.get("description"),
                        "default": v.get("default"),
                    }
                    for k, v in fn["parameters"].get("properties", {}).items()
                },
            }
        return {"methods": methods, "auto_available": True, "total": len(methods)}

    @api.post("/api/chunking/preview", tags=["Chunking"], summary="Preview chunks for a file")
    async def preview_chunks(
        file: UploadFile = File(..., description="File to chunk (.txt .md .pdf .html .docx .csv .json .jsonl)"),
        method: str = Form(default="auto", description="Chunking method name or 'auto'"),
        method_params: str = Form(default="{}", description="JSON string of method kwargs"),
        preview_limit: int = Form(default=10, description="Max chunks to return in preview"),
    ):
        """
        Upload a single file and get a preview of its chunks.
        Use `method='auto'` for OpenAI-powered strategy selection.
        """
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
                    raise HTTPException(400, f"Unknown method '{method}'. See GET /api/chunking/methods.")
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

    # ── Fine-tuning ───────────────────────────────────────────────────────────
    @api.post("/api/finetune", tags=["Fine-tuning"], status_code=202,
              summary="Start a fine-tuning job")
    async def start_finetune(
        files: list[UploadFile] = File(..., description="Corpus files to train on"),
        base_model: str = Form(..., description="Model ID from GET /api/models"),
        output_model_name: str = Form(..., description="Name for the fine-tuned model"),
        chunking_method: str = Form(default="auto", description="Chunking method or 'auto'"),
        method_params: str = Form(default="{}", description="JSON string of chunking kwargs"),
        epochs: int = Form(default=3, ge=1, le=20),
        batch_size: int = Form(default=32, ge=2, le=256),
        learning_rate: float = Form(default=3e-5, gt=0),
        temperature: float = Form(default=0.05, gt=0),
    ):
        """
        Upload corpus files, chunk them, and launch a SimCSE fine-tuning job.
        Returns a `job_id` — poll `GET /api/finetune/{job_id}` for status.
        """
        base_path = models_dir / base_model
        if not base_path.exists():
            raise HTTPException(404, f"Base model '{base_model}' not found. See GET /api/models.")

        output_path = models_dir / output_model_name
        if output_path.exists():
            raise HTTPException(409, f"Output model '{output_model_name}' already exists.")

        params = json.loads(method_params)
        all_chunks: list[str] = []
        skipped: list[dict] = []
        chunk_methods_used: dict[str, int] = {}

        with tempfile.TemporaryDirectory() as tmpdir:
            for upload in files:
                suffix = Path(upload.filename or "upload").suffix or ".txt"
                tmp_path = Path(tmpdir) / f"{uuid.uuid4()}{suffix}"
                tmp_path.write_bytes(await upload.read())

                try:
                    text = read_file(tmp_path)
                    if len(text.strip()) < 30:
                        raise ValueError("Extracted text too short — likely corrupted or image-only.")

                    if chunking_method == "auto":
                        method, kp = agent_choose_chunker(tmp_path, text[:3000])
                    else:
                        method, kp = chunking_method, params

                    chunker = CHUNKERS[method]
                    valid_kw = {k: v for k, v in kp.items() if k in inspect.signature(chunker).parameters}
                    fc = [c.strip() for c in chunker(text, **valid_kw) if len(c.strip().split()) >= 5]
                    all_chunks.extend(fc)
                    chunk_methods_used[method] = chunk_methods_used.get(method, 0) + len(fc)

                except Exception as e:
                    skipped.append({"file": upload.filename, "error": str(e)[:150]})

        if not all_chunks:
            raise HTTPException(422, "No valid chunks could be extracted from the uploaded files.")

        job_id = str(uuid.uuid4())
        spawn_finetune(
            job_id=job_id,
            chunks=all_chunks,
            base_model_path=str(base_path),
            output_path=str(output_path),
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            temperature=temperature,
            max_length=128,
        )

        return {
            "job_id": job_id,
            "status": "running",
            "base_model": base_model,
            "output_model": output_model_name,
            "chunks_collected": len(all_chunks),
            "chunking_methods_used": chunk_methods_used,
            "files_skipped": skipped,
            "poll_url": f"/api/finetune/{job_id}",
        }

    @api.get("/api/finetune/{job_id}", tags=["Fine-tuning"], summary="Poll job status")
    def get_job(job_id: str):
        """
        Returns current status of a fine-tuning job.
        Status values: 'running' | 'complete' | 'error' | 'not_found'
        """
        return {"job_id": job_id, **job_store.get(job_id)}

    return api


# ── Standalone entry point ───────────────────────────────────────────────────
# Used by: uvicorn api:app --reload
app = create_app(models_dir=Path(os.environ.get("MODELS_DIR", "./models")))
