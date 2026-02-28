"""
Model nodes — training and inference.

Each node runs in train or infer mode based on ctx["pipeline_type"].
Training generates a script, runs it as a subprocess, returns the model path.
Inference loads a saved model and returns predictions.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

from .base import BaseNode
from ..types import NodeOutput


# ── Shared helpers ────────────────────────────────────────────────────────────

def _run_script(script: str, workspace: str, log: Callable[[str], None]) -> str:
    """Write script to disk and run it; return the output_model path."""
    script_path = os.path.join(workspace, "train.py")
    with open(script_path, "w") as f:
        f.write(script)

    log("Running training script …")
    proc = subprocess.run(
        [sys.executable, script_path],
        capture_output=False,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Training script failed with exit code {proc.returncode}")
    return script_path


def _save_to_volume(src_dir: str, model_name: str, models_dir: str, log: Callable) -> str:
    """Copy trained model from workspace into the persistent models volume."""
    import shutil
    dst = os.path.join(models_dir, model_name)
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src_dir, dst)
    log(f"Model saved to volume: {dst}")
    return dst


def _upload_to_r2(model_dir: str, r2_prefix: str, r2_config: dict, log: Callable) -> list[str]:
    """Upload model directory to Cloudflare R2."""
    if not all(r2_config.get(k) for k in ("endpoint", "key_id", "secret", "bucket")):
        log("[WARN] R2 config incomplete — skipping R2 upload.")
        return []
    import boto3
    s3 = boto3.client(
        "s3",
        endpoint_url=r2_config["endpoint"],
        aws_access_key_id=r2_config["key_id"],
        aws_secret_access_key=r2_config["secret"],
    )
    uploaded: list[str] = []
    for root, _, fnames in os.walk(model_dir):
        for fname in fnames:
            local = os.path.join(root, fname)
            key   = f"{r2_prefix}/{os.path.relpath(local, model_dir)}"
            s3.upload_file(local, r2_config["bucket"], key)
            uploaded.append(key)
    log(f"Uploaded {len(uploaded)} files to R2 ({r2_prefix})")
    return uploaded


def _upload_embeddings_to_actian(
    chunks: list[str],
    model_path: str,
    collection_name: str,
    actian_url: str,
    log: Callable,
) -> int:
    """Embed chunks and upsert them into the Actian HTTP gateway."""
    import httpx
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModel  # type: ignore[import]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok  = AutoTokenizer.from_pretrained(model_path)
    emb  = AutoModel.from_pretrained(model_path).to(device)
    emb.eval()

    def _pool(h, m):
        e = m.unsqueeze(-1).expand(h.size()).float()
        return torch.sum(h * e, 1) / torch.clamp(e.sum(1), min=1e-9)

    def embed_batch(texts: list[str]) -> list[list[float]]:
        enc = tok(texts, padding=True, truncation=True, max_length=128,
                  return_tensors="pt").to(device)
        with torch.no_grad():
            out = emb(**enc)
        v = F.normalize(_pool(out.last_hidden_state, enc["attention_mask"]), dim=-1)
        return v.cpu().tolist()

    probe = embed_batch(["probe"])
    dim   = len(probe[0])

    # Create collection (ignore if already exists)
    httpx.post(f"{actian_url}/collections", json={
        "name": collection_name, "dimension": dim,
        "distance_metric": "COSINE", "storage_type": "MEM",
        "m": 16, "ef_construct": 200, "ef_search": 64,
    }, timeout=30)

    BATCH = 128
    uploaded = 0
    for start in range(0, len(chunks), BATCH):
        batch_texts = chunks[start:start + BATCH]
        batch_vecs  = embed_batch(batch_texts)
        payloads    = [{"text": t, "id": start + i, "model": collection_name}
                       for i, t in enumerate(batch_texts)]
        resp = httpx.post(f"{actian_url}/upsert", json={
            "collection": collection_name,
            "ids":        list(range(start, start + len(batch_texts))),
            "vectors":    batch_vecs,
            "payloads":   payloads,
        }, timeout=60)
        if resp.is_success:
            uploaded += len(batch_texts)
        else:
            log(f"[WARN] Actian upsert failed batch {start}: {resp.text[:200]}")

    log(f"Actian: {uploaded}/{len(chunks)} vectors → '{collection_name}'")
    return uploaded


# ── TextModelNode ─────────────────────────────────────────────────────────────

class TextModelNode(BaseNode):
    """
    Train or run inference with a sentence-transformer style model.

    Train:   chunks → SimCSE/MNRL/LoRA/SFT training → saved model
    Infer:   chunks/text → embeddings → semantic search against Actian

    Params: base_model, output_name, method, epochs, batch_size, learning_rate, gpu
    """

    def execute(self, inputs, files, ctx, log):
        sys.path.insert(0, "/app")
        from generate import generate_script  # type: ignore[import]

        if ctx["pipeline_type"] == "train":
            return self._train(inputs, ctx, log, generate_script)
        return self._infer(inputs, ctx, log)

    def _train(self, inputs, ctx, log, generate_script):
        inp = self.first_of_type(inputs, "chunks", "text")
        if inp is None:
            raise ValueError("TextModelNode (train): needs chunks or text input.")

        chunks = inp.get("chunks") or inp.get("texts", [])
        if not chunks:
            raise ValueError("TextModelNode (train): no chunks/texts to train on.")

        base_model  = self.p("base_model", "all-MiniLM-L6-v2")
        output_name = self.p("output_name", f"text-model-{ctx['job_id'][:8]}")
        workspace   = ctx["workspace"]
        models_dir  = ctx["models_dir"]

        base_path = os.path.join(models_dir, base_model)
        if not os.path.exists(base_path):
            base_path = base_model  # HuggingFace hub fallback

        output_dir = os.path.join(workspace, "output_model")
        chunks_path = os.path.join(workspace, "chunks.json")
        with open(chunks_path, "w") as f:
            json.dump(chunks, f)

        script = generate_script({
            "method":        self.p("method", "simcse"),
            "base_model":    base_path,
            "output_dir":    output_dir,
            "chunks_file":   chunks_path,
            "epochs":        self.p("epochs", 3),
            "batch_size":    self.p("batch_size", 32),
            "learning_rate": self.p("learning_rate", 3e-5),
            "temperature":   self.p("temperature", 0.05),
            "lora_r":        self.p("lora_r", 16),
            "lora_alpha":    self.p("lora_alpha", 32),
            "lora_dropout":  self.p("lora_dropout", 0.1),
            "max_length":    128,
            "weight_decay":  0.01,
            "warmup_steps":  100,
        })

        _run_script(script, workspace, log)

        # Persist to volume and R2
        model_path   = _save_to_volume(output_dir, output_name, models_dir, log)
        _upload_to_r2(output_dir, f"models/{output_name}", ctx["r2_config"], log)

        # Upload embeddings to Actian
        collection = f"{output_name}_embeddings"
        vectors_uploaded = 0
        if ctx.get("actian_url"):
            try:
                vectors_uploaded = _upload_embeddings_to_actian(
                    chunks, model_path, collection, ctx["actian_url"], log
                )
            except Exception as e:
                log(f"[WARN] Actian upload failed: {e}")

        return {
            "type":             "model",
            "model_path":       model_path,
            "model_type":       "text",
            "arch":             "text",
            "output_name":      output_name,
            "collection":       collection,
            "chunks_count":     len(chunks),
            "vectors_uploaded": vectors_uploaded,
        }

    def _infer(self, inputs, ctx, log):
        import torch
        import torch.nn.functional as F
        from transformers import AutoTokenizer, AutoModel  # type: ignore[import]
        import httpx

        inp = self.first_of_type(inputs, "chunks", "text")
        if inp is None:
            raise ValueError("TextModelNode (infer): needs chunks or text input.")

        query_texts = inp.get("chunks") or inp.get("texts", [])
        base_model  = self.p("base_model", "all-MiniLM-L6-v2")
        top_k       = int(self.p("top_k", 5))
        collection  = self.p("collection", f"{base_model}_embeddings")

        model_path = os.path.join(ctx["models_dir"], base_model)
        if not os.path.exists(model_path):
            model_path = base_model

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tok  = AutoTokenizer.from_pretrained(model_path)
        emb  = AutoModel.from_pretrained(model_path).to(device)
        emb.eval()

        def _pool(h, m):
            e = m.unsqueeze(-1).expand(h.size()).float()
            return torch.sum(h * e, 1) / torch.clamp(e.sum(1), min=1e-9)

        predictions = []
        for text in query_texts[:20]:  # cap at 20 for inference
            with torch.no_grad():
                enc = tok(text, return_tensors="pt", truncation=True, max_length=128).to(device)
                out = emb(**enc)
                vec = F.normalize(_pool(out.last_hidden_state, enc["attention_mask"]), dim=-1)
                vec_list = vec.cpu().squeeze().tolist()

            resp = httpx.post(f"{ctx['actian_url']}/search", json={
                "collection":  collection,
                "query":       vec_list,
                "top_k":       top_k,
                "with_payload": True,
            }, timeout=30)

            results = resp.json().get("results", []) if resp.is_success else []
            predictions.append({"query": text[:100], "results": results})
            log(f"  [TextModel infer] '{text[:50]}…' → {len(results)} results")

        return {
            "type":        "infer_out",
            "predictions": predictions,
            "model":       base_model,
            "collection":  collection,
        }


# ── CNNModelNode ──────────────────────────────────────────────────────────────

class CNNModelNode(BaseNode):
    """
    Train or run inference with a CNN (ResNet / VGG / from-scratch).

    Train:  image paths → CNN training → saved model
    Infer:  image paths + model path → class predictions

    Params: base_model (resnet18|resnet50|vgg16|none), output_name,
            num_classes, epochs, batch_size, learning_rate, transfer, gpu
    """

    def execute(self, inputs, files, ctx, log):
        sys.path.insert(0, "/app")
        from generate import generate_script  # type: ignore[import]

        if ctx["pipeline_type"] == "train":
            return self._train(inputs, ctx, log, generate_script)
        return self._infer(inputs, ctx, log)

    def _train(self, inputs, ctx, log, generate_script):
        inp = self.first_of_type(inputs, "image")
        if inp is None:
            raise ValueError("CNNModelNode (train): needs image input.")

        paths       = inp.get("paths", [])
        base_model  = self.p("base_model", "resnet18")
        output_name = self.p("output_name", f"cnn-model-{ctx['job_id'][:8]}")
        workspace   = ctx["workspace"]
        models_dir  = ctx["models_dir"]
        output_dir  = os.path.join(workspace, "cnn_output")

        # Save paths list for the training script
        paths_file = os.path.join(workspace, "image_paths.json")
        with open(paths_file, "w") as f:
            json.dump(paths, f)

        script = generate_script({
            "method":        "cnn",
            "base_model":    base_model,
            "output_dir":    output_dir,
            "chunks_file":   paths_file,   # reused as paths_file param
            "num_classes":   self.p("num_classes", 2),
            "epochs":        self.p("epochs", 10),
            "batch_size":    self.p("batch_size", 32),
            "learning_rate": self.p("learning_rate", 1e-3),
            "transfer":      self.p("transfer", True),
            "weight_decay":  0.01,
            "warmup_steps":  0,
        })

        _run_script(script, workspace, log)

        model_path = _save_to_volume(output_dir, output_name, models_dir, log)
        _upload_to_r2(output_dir, f"models/{output_name}", ctx["r2_config"], log)

        return {
            "type":       "model",
            "model_path": model_path,
            "model_type": "cnn",
            "arch":       "cnn",
            "output_name": output_name,
        }

    def _infer(self, inputs, ctx, log):
        import torch
        import torchvision.transforms as T  # type: ignore[import]
        from PIL import Image  # type: ignore[import]

        img_inp = self.first_of_type(inputs, "image")
        mdl_inp = self.first_of_type(inputs, "model")

        if img_inp is None:
            raise ValueError("CNNModelNode (infer): needs image input.")

        model_path = (mdl_inp or {}).get("model_path") or \
                     os.path.join(ctx["models_dir"], self.p("base_model", ""))

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model  = torch.load(os.path.join(model_path, "model.pt"),
                             map_location=device)
        model.eval()

        transform = T.Compose([
            T.Resize(224), T.CenterCrop(224), T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        predictions = []
        for src in img_inp["paths"]:
            img = Image.open(src).convert("RGB")
            tensor = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(tensor)
                probs  = torch.softmax(logits, dim=-1).cpu().squeeze().tolist()
            predictions.append({"file": Path(src).name, "probs": probs})
            log(f"  [CNN infer] {Path(src).name}: {probs}")

        return {"type": "infer_out", "predictions": predictions, "arch": "cnn"}


# ── RNNModelNode ──────────────────────────────────────────────────────────────

class RNNModelNode(BaseNode):
    """
    Train or run inference with an RNN / LSTM / GRU.

    Train:  text chunks → language model training → saved model
    Infer:  text queries → next-token completions

    Params: rnn_type (rnn|lstm|gru), output_name, vocab_size, embed_dim,
            hidden_dim, num_layers, bidirectional, epochs, batch_size, learning_rate
    """

    def execute(self, inputs, files, ctx, log):
        sys.path.insert(0, "/app")
        from generate import generate_script  # type: ignore[import]

        if ctx["pipeline_type"] == "train":
            return self._train(inputs, ctx, log, generate_script)
        return self._infer(inputs, ctx, log)

    def _train(self, inputs, ctx, log, generate_script):
        inp = self.first_of_type(inputs, "chunks", "text")
        if inp is None:
            raise ValueError("RNNModelNode (train): needs chunks or text input.")

        chunks      = inp.get("chunks") or inp.get("texts", [])
        rnn_type    = self.p("rnn_type", "lstm")
        output_name = self.p("output_name", f"rnn-model-{ctx['job_id'][:8]}")
        workspace   = ctx["workspace"]
        models_dir  = ctx["models_dir"]
        output_dir  = os.path.join(workspace, "rnn_output")

        chunks_path = os.path.join(workspace, "chunks.json")
        with open(chunks_path, "w") as f:
            json.dump(chunks, f)

        script = generate_script({
            "method":        rnn_type,   # "rnn", "lstm", "gru"
            "output_dir":    output_dir,
            "chunks_file":   chunks_path,
            "vocab_size":    self.p("vocab_size", 10000),
            "embed_dim":     self.p("embed_dim", 128),
            "hidden_dim":    self.p("hidden_dim", 256),
            "num_layers":    self.p("num_layers", 2),
            "bidirectional": self.p("bidirectional", True),
            "num_classes":   self.p("num_classes", 2),
            "epochs":        self.p("epochs", 10),
            "batch_size":    self.p("batch_size", 64),
            "learning_rate": self.p("learning_rate", 1e-3),
            "weight_decay":  0.01,
            "warmup_steps":  0,
        })

        _run_script(script, workspace, log)

        model_path = _save_to_volume(output_dir, output_name, models_dir, log)
        _upload_to_r2(output_dir, f"models/{output_name}", ctx["r2_config"], log)

        return {
            "type":       "model",
            "model_path": model_path,
            "model_type": "rnn",
            "arch":       "rnn",
            "rnn_type":   rnn_type,
            "output_name": output_name,
        }

    def _infer(self, inputs, ctx, log):
        import torch

        inp    = self.first_of_type(inputs, "chunks", "text")
        mdl_inp = self.first_of_type(inputs, "model")

        texts      = (inp or {}).get("chunks") or (inp or {}).get("texts", [])
        model_path = (mdl_inp or {}).get("model_path") or \
                     os.path.join(ctx["models_dir"], self.p("output_name", ""))

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(os.path.join(model_path, "model.pt"), map_location=device)
        model = checkpoint["model"]
        vocab = checkpoint.get("vocab", {})
        model.eval()

        predictions = []
        for text in texts[:20]:
            tokens = [vocab.get(w, 0) for w in text.lower().split()[:32]]
            if not tokens:
                continue
            t = torch.tensor([tokens], dtype=torch.long, device=device)
            with torch.no_grad():
                out = model(t)
                probs = torch.softmax(out[-1] if isinstance(out, tuple) else out, dim=-1)
                top_probs, top_ids = probs.topk(5, dim=-1)
            preds = top_probs.squeeze().tolist()
            predictions.append({"text": text[:80], "top_probs": preds})
            log(f"  [RNN infer] '{text[:40]}…'")

        return {"type": "infer_out", "predictions": predictions, "arch": "rnn"}
