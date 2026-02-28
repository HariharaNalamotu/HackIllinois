"""
Modal deployment for the corpus fine-tuning backend.

──────────────────────────────────────────────────────────────
ONE-TIME SETUP
──────────────────────────────────────────────────────────────
1. Install Modal and authenticate:
     pip install modal
     modal setup

2. Create a Modal Secret with your OpenAI key:
     modal secret create hackillinois-secrets OPENAI_API_KEY=sk-...

3. Seed your local ./models directory into the Modal Volume:
     modal run modal_app.py::seed_models

──────────────────────────────────────────────────────────────
RUNNING
──────────────────────────────────────────────────────────────
Local dev (live reload, no GPU):
     modal serve modal_app.py

Deploy to production:
     modal deploy modal_app.py

──────────────────────────────────────────────────────────────
CLOUDFLARE
──────────────────────────────────────────────────────────────
After deploying, Modal will print a URL like:
     https://your-workspace--hackillinois-rag-web.modal.run

Point your Cloudflare Worker or Pages Function at that URL,
or set it as the API_BASE_URL in your frontend .env.
Set CORS_ORIGINS in the Modal secret to restrict allowed origins.
"""

import json
import uuid
import inspect
import tempfile
from pathlib import Path

import modal

# ── Infrastructure ────────────────────────────────────────────────────────────
app = modal.App("hackillinois-rag")

# Persistent volume — stores all models across deployments
models_volume = modal.Volume.from_name("hackillinois-models", create_if_missing=True)
MODELS_DIR = Path("/vol/models")

# Mount local source files into every container
src_mount = modal.Mount.from_local_file("chunk.py",   remote_path="/app/chunk.py")
api_mount = modal.Mount.from_local_file("api.py",     remote_path="/app/api.py")

# Base container image
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "fastapi[standard]>=0.111",
        "python-multipart",
        "openai>=1.30",
        "python-dotenv",
        "transformers>=4.40",
        # torch CPU for API worker (chunking/routing), GPU variant added per-function
        "torch>=2.2",
        "pypdf>=4.0",
        "beautifulsoup4",
        "python-docx",
        "nltk",
        "lxml",
        "huggingface_hub",
    )
    .run_commands(
        "python -c \""
        "import nltk; "
        "nltk.download('punkt', quiet=True); "
        "nltk.download('punkt_tab', quiet=True)"
        "\""
    )
)


# ── GPU fine-tuning function ──────────────────────────────────────────────────
@app.function(
    image=image,
    gpu="T4",                            # upgrade to "A10G" for larger models
    volumes={str(MODELS_DIR.parent): models_volume},
    timeout=3600,
    secrets=[modal.Secret.from_name("hackillinois-secrets")],
    mounts=[src_mount],
)
def run_finetune(
    job_id: str,
    chunks: list[str],
    base_model_path: str,
    output_path: str,
    epochs: int = 3,
    batch_size: int = 32,
    learning_rate: float = 3e-5,
    temperature: float = 0.05,
    max_length: int = 128,
) -> dict:
    """SimCSE fine-tuning on a GPU worker. Called via .spawn() from the API."""
    import sys
    sys.path.insert(0, "/app")

    import torch
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    from transformers import AutoTokenizer, AutoModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{job_id}] device={device} chunks={len(chunks)} base={base_model_path}")

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
        labels = torch.cat([
            torch.arange(B, 2 * B, device=e1.device),
            torch.arange(B, device=e1.device),
        ])
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
        for step, batch in enumerate(loader, 1):
            def encode(s):
                enc = tokenizer(
                    s, padding=True, truncation=True,
                    max_length=max_length, return_tensors="pt",
                ).to(device)
                return mean_pool(model(**enc).last_hidden_state, enc["attention_mask"])
            loss = simcse_loss(encode(batch), encode(batch), temperature)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total += loss.item()
            if step % 10 == 0 or step == len(loader):
                print(f"  Epoch {epoch}/{epochs} Step {step}/{len(loader)} Loss {loss.item():.4f}")
        avg = total / len(loader)
        history.append({"epoch": epoch, "avg_loss": round(avg, 6)})
        print(f"Epoch {epoch} done — avg loss {avg:.4f}")

    Path(output_path).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    models_volume.commit()   # flush writes to volume so other containers see them

    return {
        "status": "complete",
        "output_model": Path(output_path).name,
        "training_history": history,
    }


# ── Modal job store ───────────────────────────────────────────────────────────
# Tracks job_id → Modal FunctionCall object_id so the API can poll results.
_modal_dict = modal.Dict.from_name("hackillinois-jobs", create_if_missing=True)

class ModalJobStore:
    """Persists job_id → call_object_id in a Modal Dict, polls via FunctionCall."""

    def get(self, job_id: str) -> dict:
        call_id = _modal_dict.get(job_id)
        if call_id is None:
            return {"status": "not_found"}
        try:
            call = modal.functions.FunctionCall.from_id(call_id)
            result = call.get(timeout=0)
            return {"status": "complete", "result": result}
        except TimeoutError:
            return {"status": "running"}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def put(self, job_id: str, data: dict):
        # For Modal, we only store the call_object_id — state lives in Modal
        if "call_object_id" in data:
            _modal_dict[job_id] = data["call_object_id"]


# ── Spawn wrapper (bridges api.py's interface to Modal GPU function) ───────────
def modal_spawn_finetune(job_id: str, **kwargs) -> str:
    call = run_finetune.spawn(job_id=job_id, **kwargs)
    _modal_dict[job_id] = call.object_id
    return job_id


# ── Web endpoint ──────────────────────────────────────────────────────────────
@app.function(
    image=image,
    volumes={str(MODELS_DIR.parent): models_volume},
    secrets=[modal.Secret.from_name("hackillinois-secrets")],
    mounts=[src_mount, api_mount],
    keep_warm=1,           # keep one container warm to avoid cold-start latency
    timeout=60,
)
@modal.asgi_app()
def web():
    import sys
    sys.path.insert(0, "/app")

    from api import create_app

    return create_app(
        models_dir=MODELS_DIR,
        spawn_finetune=modal_spawn_finetune,
        job_store=ModalJobStore(),
    )


# ── Seed models from local ./models → Modal Volume ────────────────────────────
@app.function(
    image=image,
    volumes={str(MODELS_DIR.parent): models_volume},
    mounts=[modal.Mount.from_local_dir("./models", remote_path="/local_models")],
)
def _do_seed():
    import shutil
    src = Path("/local_models")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    copied = []
    for model_dir in sorted(src.iterdir()):
        dst = MODELS_DIR / model_dir.name
        if dst.exists():
            print(f"  skip (exists): {model_dir.name}")
            continue
        print(f"  copying: {model_dir.name} ...")
        shutil.copytree(str(model_dir), str(dst))
        copied.append(model_dir.name)
    models_volume.commit()
    return copied


@app.local_entrypoint()
def seed_models():
    """
    Copy all folders in ./models into the Modal Volume.
    Run once after first deploy:
        modal run modal_app.py::seed_models
    """
    print("Seeding local ./models → Modal Volume ...")
    copied = _do_seed.remote()
    if copied:
        print(f"Seeded: {copied}")
    else:
        print("Nothing new to seed (all models already in volume).")
