"""
Download all static HuggingFace models to ./models/ for the pipeline.

Run once before deploying:
    python download_models.py

Then seed the Modal Volume:
    modal run modal/app.py::seed_models
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("huggingface_hub not found. Installing...")
    os.system(f"{sys.executable} -m pip install huggingface_hub")
    from huggingface_hub import snapshot_download  # type: ignore

MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# MODEL CATALOGUE
# key  = local folder name inside ./models/
# repo = HuggingFace repo ID
# ─────────────────────────────────────────────────────────────────────────────
MODELS: list[tuple[str, str, str]] = [
    # ── Text Embedding ─────────────────────────────────────────────────────
    ("all-mpnet-base-v2",
     "sentence-transformers/all-mpnet-base-v2",
     "Text embedding (~420 MB)"),
    ("bge-small-en-v1.5",
     "BAAI/bge-small-en-v1.5",
     "Text embedding (~130 MB)"),
    ("bge-base-en-v1.5",
     "BAAI/bge-base-en-v1.5",
     "Text embedding (~440 MB)"),
    ("paraphrase-multilingual-MiniLM-L12-v2",
     "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
     "Multilingual text embedding (~470 MB)"),

    # ── Object Detection ───────────────────────────────────────────────────
    ("yolos-tiny",
     "hustvl/yolos-tiny",
     "Object detection – YOLO-S Tiny (~30 MB)"),
    ("rtdetr-r18vd",
     "PekingU/rtdetr_r18vd",
     "Object detection – RT-DETR R18 (~150 MB)"),
    ("detr-resnet-50",
     "facebook/detr-resnet-50",
     "Object detection – DETR ResNet-50 (~160 MB)"),

    # ── Audio ──────────────────────────────────────────────────────────────
    ("whisper-tiny",
     "openai/whisper-tiny",
     "Speech-to-text – Whisper Tiny (~150 MB)"),
    ("wav2vec2-base",
     "facebook/wav2vec2-base",
     "Audio classification – Wav2Vec2 Base (~360 MB)"),
    ("wav2vec2-emotion",
     "superb/wav2vec2-base-superb-er",
     "Emotion recognition – Wav2Vec2 SUPERB (~360 MB)"),
]

# NOTE: Image classification backbones (ResNet-18/50, ConvNeXt-Tiny) are
# served by torchvision.models and are auto-cached by PyTorch — no manual
# download needed here.


def download_all() -> None:
    already = {p.name for p in MODELS_DIR.iterdir() if p.is_dir()}
    total = len(MODELS)
    skipped = 0

    for i, (folder, repo, desc) in enumerate(MODELS, 1):
        dst = MODELS_DIR / folder
        if folder in already:
            print(f"[{i}/{total}] skip  {folder}  ({desc})")
            skipped += 1
            continue
        print(f"[{i}/{total}] downloading  {repo}  → models/{folder}  [{desc}]")
        try:
            snapshot_download(
                repo_id=repo,
                local_dir=str(dst),
                ignore_patterns=["*.msgpack", "flax_model*", "tf_model*", "rust_model*"],
            )
            print(f"       ✓ done")
        except Exception as e:
            print(f"       ✗ FAILED: {e}")

    downloaded = total - skipped
    print(f"\nDone. {downloaded} downloaded, {skipped} skipped.")
    print(f"Models directory: {MODELS_DIR.resolve()}")
    if downloaded:
        print("\nNext step — seed models into Modal Volume:")
        print("  modal run modal/app.py::seed_models")


if __name__ == "__main__":
    download_all()
