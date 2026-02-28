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


# ── ObjectDetectModelNode ─────────────────────────────────────────────────────

class ObjectDetectModelNode(BaseNode):
    """
    Fine-tune YOLOS / DETR on bounding-box annotated images.

    Train:  image paths + annotations → detector → saved model
    Infer:  image paths + model path  → bounding boxes + class scores

    Params: base_model, output_name, num_classes, input_format, epochs,
            batch_size, learning_rate, gpu
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
            raise ValueError("ObjectDetectModelNode (train): needs image input.")

        workspace   = ctx["workspace"]
        models_dir  = ctx["models_dir"]
        output_dir  = os.path.join(workspace, "detect_output")
        output_name = self.p("output_name", f"detector-{ctx['job_id'][:8]}")

        # Save image paths + annotation records to a JSON file
        records_path = os.path.join(workspace, "detect_records.json")
        records = inp.get("records", [{"image": p, "annotations": []}
                                      for p in inp.get("paths", [])])
        with open(records_path, "w") as f:
            json.dump(records, f)

        script = generate_script({
            "method":        "object_detect",
            "base_model":    self.p("base_model", "hustvl/yolos-tiny"),
            "output_dir":    output_dir,
            "chunks_file":   records_path,
            "num_classes":   self.p("num_classes", 80),
            "input_format":  self.p("input_format", "coco"),
            "epochs":        self.p("epochs", 10),
            "batch_size":    self.p("batch_size", 8),
            "learning_rate": self.p("learning_rate", 5e-5),
        })

        _run_script(script, workspace, log)

        model_path = _save_to_volume(output_dir, output_name, models_dir, log)
        _upload_to_r2(output_dir, f"models/{output_name}", ctx["r2_config"], log)

        return {
            "type":        "model",
            "model_path":  model_path,
            "model_type":  "object_detect",
            "arch":        "detect",
            "output_name": output_name,
        }

    def _infer(self, inputs, ctx, log):
        import torch
        from PIL import Image  # type: ignore[import]
        from transformers import AutoImageProcessor, AutoModelForObjectDetection  # type: ignore[import]

        img_inp = self.first_of_type(inputs, "image")
        mdl_inp = self.first_of_type(inputs, "model")

        if img_inp is None:
            raise ValueError("ObjectDetectModelNode (infer): needs image input.")

        model_path = (mdl_inp or {}).get("model_path") or \
                     os.path.join(ctx["models_dir"], self.p("base_model", ""))

        device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        processor = AutoImageProcessor.from_pretrained(model_path)
        model     = AutoModelForObjectDetection.from_pretrained(model_path).to(device)
        model.eval()

        predictions = []
        for src in img_inp.get("paths", [])[:20]:
            image  = Image.open(src).convert("RGB")
            inputs = processor(images=image, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
            boxes  = outputs.pred_boxes[0].cpu().tolist()
            scores = outputs.logits[0].softmax(-1).max(-1).values.cpu().tolist()
            predictions.append({"file": os.path.basename(src), "boxes": boxes[:10], "scores": scores[:10]})
            log(f"  [ObjectDetect infer] {os.path.basename(src)}: {len(boxes)} detections")

        return {"type": "infer_out", "predictions": predictions, "arch": "detect"}


# ── AudioSpeechModelNode ──────────────────────────────────────────────────────

class AudioSpeechModelNode(BaseNode):
    """
    Fine-tune Whisper / Wav2Vec2 for STT, emotion, or classification.

    Params: base_model, task, output_name, num_classes, epochs, batch_size,
            learning_rate, gpu
    """

    def execute(self, inputs, files, ctx, log):
        sys.path.insert(0, "/app")
        from generate import generate_script  # type: ignore[import]

        if ctx["pipeline_type"] == "train":
            return self._train(inputs, ctx, log, generate_script)
        return self._infer(inputs, ctx, log)

    def _train(self, inputs, ctx, log, generate_script):
        inp = self.first_of_type(inputs, "audio")
        if inp is None:
            raise ValueError("AudioSpeechModelNode (train): needs audio input.")

        workspace   = ctx["workspace"]
        models_dir  = ctx["models_dir"]
        output_dir  = os.path.join(workspace, "audio_output")
        output_name = self.p("output_name", f"audio-model-{ctx['job_id'][:8]}")

        records_path = os.path.join(workspace, "audio_records.json")
        records = inp.get("records", inp.get("paths", []))
        with open(records_path, "w") as f:
            json.dump(records, f)

        script = generate_script({
            "method":        "audio_speech",
            "base_model":    self.p("base_model", "openai/whisper-tiny"),
            "output_dir":    output_dir,
            "chunks_file":   records_path,
            "task":          self.p("task", "transcription"),
            "num_classes":   self.p("num_classes", 2),
            "epochs":        self.p("epochs", 5),
            "batch_size":    self.p("batch_size", 16),
            "learning_rate": self.p("learning_rate", 1e-4),
        })

        _run_script(script, workspace, log)

        model_path = _save_to_volume(output_dir, output_name, models_dir, log)
        _upload_to_r2(output_dir, f"models/{output_name}", ctx["r2_config"], log)

        return {
            "type":        "model",
            "model_path":  model_path,
            "model_type":  "audio_speech",
            "arch":        "audio",
            "output_name": output_name,
        }

    def _infer(self, inputs, ctx, log):
        import torch
        import torchaudio  # type: ignore[import]
        from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq  # type: ignore[import]

        aud_inp = self.first_of_type(inputs, "audio")
        mdl_inp = self.first_of_type(inputs, "model")

        if aud_inp is None:
            raise ValueError("AudioSpeechModelNode (infer): needs audio input.")

        model_path = (mdl_inp or {}).get("model_path") or \
                     os.path.join(ctx["models_dir"], self.p("base_model", ""))

        device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        processor = AutoProcessor.from_pretrained(model_path)
        model     = AutoModelForSpeechSeq2Seq.from_pretrained(model_path).to(device)
        model.eval()

        predictions = []
        for src in aud_inp.get("paths", [])[:10]:
            wav, sr = torchaudio.load(src)
            wav = torchaudio.functional.resample(wav, sr, 16000).mean(0)
            enc = processor(wav.numpy(), sampling_rate=16000, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.no_grad():
                ids  = model.generate(**enc, max_new_tokens=128)
                text = processor.decode(ids[0], skip_special_tokens=True)
            predictions.append({"file": os.path.basename(src), "transcription": text})
            log(f"  [AudioSpeech infer] {os.path.basename(src)}: {text[:60]}")

        return {"type": "infer_out", "predictions": predictions, "arch": "audio"}


# ── AudioCNNNode ──────────────────────────────────────────────────────────────

class AudioCNNNode(BaseNode):
    """
    Train or infer with a CNN on log-mel spectrograms.

    Params: output_name, num_classes, num_layers, filters, kernel_size,
            sample_rate, n_mels, epochs, batch_size, learning_rate, gpu
    """

    def execute(self, inputs, files, ctx, log):
        sys.path.insert(0, "/app")
        from generate import generate_script  # type: ignore[import]

        if ctx["pipeline_type"] == "train":
            return self._train(inputs, ctx, log, generate_script)
        return self._infer(inputs, ctx, log)

    def _train(self, inputs, ctx, log, generate_script):
        inp = self.first_of_type(inputs, "audio")
        if inp is None:
            raise ValueError("AudioCNNNode (train): needs audio input.")

        workspace   = ctx["workspace"]
        models_dir  = ctx["models_dir"]
        output_dir  = os.path.join(workspace, "audiocnn_output")
        output_name = self.p("output_name", f"audio-cnn-{ctx['job_id'][:8]}")

        records_path = os.path.join(workspace, "audio_records.json")
        records = inp.get("records", inp.get("paths", []))
        with open(records_path, "w") as f:
            json.dump(records, f)

        script = generate_script({
            "method":        "audio_cnn",
            "output_dir":    output_dir,
            "chunks_file":   records_path,
            "num_classes":   self.p("num_classes", 2),
            "filters":       self.p("filters", "32,64,128"),
            "kernel_size":   self.p("kernel_size", 3),
            "sample_rate":   self.p("sample_rate", 16000),
            "n_mels":        self.p("n_mels", 80),
            "epochs":        self.p("epochs", 10),
            "batch_size":    self.p("batch_size", 32),
            "learning_rate": self.p("learning_rate", 1e-3),
        })

        _run_script(script, workspace, log)

        model_path = _save_to_volume(output_dir, output_name, models_dir, log)
        _upload_to_r2(output_dir, f"models/{output_name}", ctx["r2_config"], log)

        return {
            "type":        "model",
            "model_path":  model_path,
            "model_type":  "audio_cnn",
            "arch":        "audio_cnn",
            "output_name": output_name,
        }

    def _infer(self, inputs, ctx, log):
        import torch
        import torchaudio  # type: ignore[import]
        import torchaudio.transforms as AT  # type: ignore[import]

        aud_inp = self.first_of_type(inputs, "audio")
        mdl_inp = self.first_of_type(inputs, "model")

        if aud_inp is None:
            raise ValueError("AudioCNNNode (infer): needs audio input.")

        model_path = (mdl_inp or {}).get("model_path") or \
                     os.path.join(ctx["models_dir"], self.p("output_name", ""))

        cfg_path = os.path.join(model_path, "config.json")
        with open(cfg_path) as f:
            cfg = json.load(f)

        num_classes = cfg.get("num_classes", 2)
        filters     = cfg.get("filters", [32, 64, 128])
        n_mels      = cfg.get("n_mels", 80)
        sample_rate = cfg.get("sample_rate", 16000)
        kernel_size = 3

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mel_tf = AT.MelSpectrogram(sample_rate=sample_rate, n_mels=n_mels, hop_length=160).to(device)

        # Rebuild model skeleton for inference
        import torch.nn as nn

        class AudioCNN(nn.Module):
            def __init__(self):
                super().__init__()
                layers, in_ch = [], 1
                for out_ch in filters:
                    layers += [nn.Conv2d(in_ch, out_ch, kernel_size, padding=1),
                                nn.BatchNorm2d(out_ch), nn.ReLU(), nn.MaxPool2d(2)]
                    in_ch = out_ch
                self.conv = nn.Sequential(*layers)
                self.pool = nn.AdaptiveAvgPool2d((4, 4))
                self.head = nn.Linear(in_ch * 4 * 4, num_classes)
            def forward(self, wav):
                spec = mel_tf(wav).unsqueeze(1)
                spec = torch.log(spec.clamp(min=1e-9))
                x    = self.conv(spec)
                x    = self.pool(x).flatten(1)
                return self.head(x)

        model = AudioCNN().to(device)
        model.load_state_dict(torch.load(os.path.join(model_path, "model.pt"), map_location=device))
        model.eval()

        predictions = []
        for src in aud_inp.get("paths", [])[:20]:
            wav, sr = torchaudio.load(src)
            wav = torchaudio.functional.resample(wav, sr, sample_rate).mean(0).to(device)
            with torch.no_grad():
                probs = torch.softmax(model(wav.unsqueeze(0)), dim=-1).squeeze().tolist()
            predictions.append({"file": os.path.basename(src), "probs": probs})
            log(f"  [AudioCNN infer] {os.path.basename(src)}: {probs}")

        return {"type": "infer_out", "predictions": predictions, "arch": "audio_cnn"}


# ── ImageCAENode ──────────────────────────────────────────────────────────────

class ImageCAENode(BaseNode):
    """
    Train or encode images with a Convolutional AutoEncoder.

    Params: output_name, num_layers, filters, latent_dim, epochs,
            batch_size, learning_rate, gpu
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
            raise ValueError("ImageCAENode (train): needs image input.")

        workspace   = ctx["workspace"]
        models_dir  = ctx["models_dir"]
        output_dir  = os.path.join(workspace, "cae_output")
        output_name = self.p("output_name", f"cae-{ctx['job_id'][:8]}")

        paths_file = os.path.join(workspace, "image_paths.json")
        with open(paths_file, "w") as f:
            json.dump(inp.get("paths", []), f)

        script = generate_script({
            "method":        "image_cae",
            "output_dir":    output_dir,
            "chunks_file":   paths_file,
            "filters":       self.p("filters", "32,64,128"),
            "latent_dim":    self.p("latentDim", 256),
            "epochs":        self.p("epochs", 20),
            "batch_size":    self.p("batch_size", 32),
            "learning_rate": self.p("learning_rate", 1e-3),
        })

        _run_script(script, workspace, log)

        model_path = _save_to_volume(output_dir, output_name, models_dir, log)
        _upload_to_r2(output_dir, f"models/{output_name}", ctx["r2_config"], log)

        return {
            "type":        "model",
            "model_path":  model_path,
            "model_type":  "image_cae",
            "arch":        "cae",
            "output_name": output_name,
        }

    def _infer(self, inputs, ctx, log):
        import torch
        import torchvision.transforms as T  # type: ignore[import]
        from PIL import Image  # type: ignore[import]

        img_inp = self.first_of_type(inputs, "image")
        mdl_inp = self.first_of_type(inputs, "model")

        if img_inp is None:
            raise ValueError("ImageCAENode (infer): needs image input.")

        model_path = (mdl_inp or {}).get("model_path") or \
                     os.path.join(ctx["models_dir"], self.p("output_name", ""))

        cfg_path = os.path.join(model_path, "config.json")
        with open(cfg_path) as f:
            cfg = json.load(f)

        filters    = cfg.get("filters", [32, 64, 128])
        latent_dim = cfg.get("latent_dim", 256)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Rebuild encoder
        import torch.nn as nn

        class Encoder(nn.Module):
            def __init__(self):
                super().__init__()
                layers, in_ch = [], 3
                for out_ch in filters:
                    layers += [nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1), nn.ReLU()]
                    in_ch = out_ch
                self.conv = nn.Sequential(*layers)
                dummy = torch.zeros(1, 3, 128, 128)
                flat  = self.conv(dummy).flatten(1).shape[1]
                self.fc   = nn.Linear(flat, latent_dim)
            def forward(self, x):
                return self.fc(self.conv(x).flatten(1))

        checkpoint  = torch.load(os.path.join(model_path, "model.pt"), map_location=device)
        enc = Encoder().to(device)
        enc.load_state_dict(checkpoint["encoder"])
        enc.eval()

        transform = T.Compose([T.Resize((128, 128)), T.ToTensor()])

        predictions = []
        for src in img_inp.get("paths", [])[:20]:
            img   = Image.open(src).convert("RGB")
            t     = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                z = enc(t).cpu().squeeze().tolist()
            predictions.append({"file": os.path.basename(src), "latent": z[:8]})
            log(f"  [ImageCAE infer] {os.path.basename(src)}: encoded to {latent_dim}D vector")

        return {"type": "infer_out", "predictions": predictions, "arch": "cae"}


# ── TabularModelNode ──────────────────────────────────────────────────────────

class TabularModelNode(BaseNode):
    """
    Train or infer with FFNN / DNN / LSTM / GRU / RNN on tabular data.

    Params: model_type, target_column, output_name, num_layers, hidden_dim,
            num_epochs, batch_size, learning_rate, bidirectional, gpu
    """

    def execute(self, inputs, files, ctx, log):
        sys.path.insert(0, "/app")
        from generate import generate_script  # type: ignore[import]

        if ctx["pipeline_type"] == "train":
            return self._train(inputs, ctx, log, generate_script)
        return self._infer(inputs, ctx, log)

    def _train(self, inputs, ctx, log, generate_script):
        inp = self.first_of_type(inputs, "tabular")
        if inp is None:
            raise ValueError("TabularModelNode (train): needs tabular input.")

        workspace   = ctx["workspace"]
        models_dir  = ctx["models_dir"]
        output_dir  = os.path.join(workspace, "tabular_output")
        output_name = self.p("output_name", f"tabular-model-{ctx['job_id'][:8]}")

        # Write CSV from tabular data
        import csv as _csv
        csv_path = os.path.join(workspace, "tabular_data.csv")
        rows    = inp.get("rows", [])
        columns = inp.get("columns", [])
        with open(csv_path, "w", newline="") as f:
            writer = _csv.DictWriter(f, fieldnames=columns or (rows[0].keys() if rows else []))
            writer.writeheader()
            writer.writerows(rows)

        script = generate_script({
            "method":        "tabular_nn",
            "output_dir":    output_dir,
            "chunks_file":   csv_path,
            "model_type":    self.p("modelType", "ffnn"),
            "target_column": self.p("targetColumn", ""),
            "hidden_dim":    self.p("hiddenDim", 128),
            "num_layers":    self.p("numLayers", 2),
            "num_classes":   2,
            "bidirectional": self.p("bidirectional", False),
            "epochs":        self.p("numEpochs", 20),
            "batch_size":    self.p("batchSize", 64),
            "learning_rate": self.p("learningRate", 1e-3),
        })

        _run_script(script, workspace, log)

        model_path = _save_to_volume(output_dir, output_name, models_dir, log)
        _upload_to_r2(output_dir, f"models/{output_name}", ctx["r2_config"], log)

        return {
            "type":        "model",
            "model_path":  model_path,
            "model_type":  "tabular",
            "arch":        "tabular",
            "output_name": output_name,
        }

    def _infer(self, inputs, ctx, log):
        import torch
        import torch.nn as nn

        tab_inp = self.first_of_type(inputs, "tabular")
        mdl_inp = self.first_of_type(inputs, "model")

        if tab_inp is None:
            raise ValueError("TabularModelNode (infer): needs tabular input.")

        model_path = (mdl_inp or {}).get("model_path") or \
                     os.path.join(ctx["models_dir"], self.p("output_name", ""))

        cfg_path = os.path.join(model_path, "config.json")
        with open(cfg_path) as f:
            cfg = json.load(f)

        label2id   = cfg.get("label2id", {})
        id2label   = {v: k for k, v in label2id.items()}
        num_classes = cfg.get("num_classes", 2)

        checkpoint = torch.load(os.path.join(model_path, "model.pt"),
                                 map_location="cpu")
        in_dim     = checkpoint.get("in_dim", 1)
        hidden_dim = self.p("hiddenDim", 128)
        num_layers = self.p("numLayers", 2)

        model_state = checkpoint["model"]

        # Rebuild FFNN for inference
        layers = [nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.3)]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.3)]
        layers.append(nn.Linear(hidden_dim, num_classes))
        model = nn.Sequential(*layers)
        model.load_state_dict(model_state)
        model.eval()

        rows    = tab_inp.get("rows", [])
        columns = tab_inp.get("columns", [])
        target  = self.p("targetColumn", "")

        predictions = []
        for row in rows[:100]:
            vals = []
            for k in (columns or row.keys()):
                if k == target:
                    continue
                try:
                    vals.append(float(row.get(k, 0)))
                except (ValueError, TypeError):
                    vals.append(0.0)

            t = torch.tensor([vals], dtype=torch.float)
            with torch.no_grad():
                probs   = torch.softmax(model(t), dim=-1).squeeze().tolist()
                pred_id = int(torch.tensor(probs).argmax().item())
            predictions.append({
                "row":   {k: row[k] for k in list(row.keys())[:4]},
                "label": id2label.get(pred_id, str(pred_id)),
                "probs": probs,
            })

        log(f"  [TabularModel infer] {len(predictions)} predictions")
        return {"type": "infer_out", "predictions": predictions, "arch": "tabular"}
