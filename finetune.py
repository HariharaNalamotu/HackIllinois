"""
SimCSE-style unsupervised fine-tuning for MiniLM.

Training data formats accepted:
  .txt   — one text chunk per line
  .json  — list of strings, or list of dicts with a 'text' / 'content' / 'chunk' key
  .jsonl — one JSON object per line with a 'text' key
  .csv   — CSV file with a column named 'text', 'content', 'chunk', or 'sentence'

Usage:
  python finetune.py <path_to_data_file>

Example:
  python finetune.py data/chunks.txt
"""

import os
import sys
import csv
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_PATH   = "./models/all-MiniLM-L6-v2"
OUTPUT_PATH  = "./models/finetuned-MiniLM"
BATCH_SIZE   = 32
EPOCHS       = 3
LEARNING_RATE = 3e-5
MAX_LENGTH   = 128
TEMPERATURE  = 0.05   # SimCSE default; lower = harder negatives


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
TEXT_KEYS = ("text", "content", "chunk", "sentence")

def _extract_key(record: dict) -> str:
    for k in TEXT_KEYS:
        if k in record:
            return record[k]
    raise KeyError(f"No recognised text key in record. Expected one of {TEXT_KEYS}. Got: {list(record.keys())}")

def load_texts(path: str) -> list[str]:
    p = Path(path)
    ext = p.suffix.lower()

    if ext == ".txt":
        with open(p, encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]

    elif ext == ".json":
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("JSON file must contain a top-level list.")
        if len(data) == 0:
            raise ValueError("JSON list is empty.")
        if isinstance(data[0], str):
            texts = [s for s in data if s.strip()]
        elif isinstance(data[0], dict):
            texts = [_extract_key(item) for item in data]
        else:
            raise ValueError("JSON list items must be strings or dicts.")

    elif ext == ".jsonl":
        texts = []
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    texts.append(_extract_key(json.loads(line)))

    elif ext == ".csv":
        texts = []
        with open(p, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                texts.append(_extract_key(row))

    else:
        raise ValueError(f"Unsupported file extension '{ext}'. Use .txt, .json, .jsonl, or .csv.")

    if not texts:
        raise ValueError("No text chunks found in the data file.")

    return texts


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class TextDataset(Dataset):
    def __init__(self, texts: list[str]):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------
def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Masked mean pooling over token dimension."""
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    return torch.sum(last_hidden_state * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)


# ---------------------------------------------------------------------------
# SimCSE loss  (InfoNCE / NT-Xent)
# ---------------------------------------------------------------------------
def simcse_loss(emb1: torch.Tensor, emb2: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    emb1, emb2 : (B, D) — two views of the same B sentences.
    Positive pairs: (emb1[i], emb2[i])
    Negative pairs: all other cross-batch combinations.
    """
    B = emb1.size(0)

    emb1 = F.normalize(emb1, dim=-1)
    emb2 = F.normalize(emb2, dim=-1)

    # Concatenate to (2B, D) and build full similarity matrix
    all_emb = torch.cat([emb1, emb2], dim=0)                          # (2B, D)
    sim = torch.mm(all_emb, all_emb.t()) / temperature                 # (2B, 2B)

    # Mask out self-similarity (diagonal)
    mask = torch.eye(2 * B, dtype=torch.bool, device=emb1.device)
    sim = sim.masked_fill(mask, float("-inf"))

    # Targets: for row i the positive is at index i+B (and vice versa)
    labels = torch.cat([
        torch.arange(B, 2 * B, device=emb1.device),
        torch.arange(0, B,     device=emb1.device),
    ])

    return F.cross_entropy(sim, labels)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train(data_path: str):
    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # Load tokenizer + model
    print(f"Loading model from {MODEL_PATH} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModel.from_pretrained(MODEL_PATH)
    model.to(device)

    # Load data
    texts = load_texts(data_path)
    print(f"Loaded {len(texts)} text chunks from '{data_path}'")

    dataset   = TextDataset(texts)
    # drop_last=True so every batch is full (required for the contrastive objective)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    print(f"\nStarting SimCSE fine-tuning — {EPOCHS} epoch(s), batch size {BATCH_SIZE}\n")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for step, batch in enumerate(dataloader, start=1):
            # Two independent forward passes through the same sentences.
            # Because dropout is active, each pass produces a different embedding.
            def encode(sentences):
                enc = tokenizer(
                    sentences,
                    padding=True,
                    truncation=True,
                    max_length=MAX_LENGTH,
                    return_tensors="pt",
                ).to(device)
                out = model(**enc)
                return mean_pool(out.last_hidden_state, enc["attention_mask"])

            emb1 = encode(batch)
            emb2 = encode(batch)

            loss = simcse_loss(emb1, emb2, TEMPERATURE)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if step % 10 == 0 or step == len(dataloader):
                print(f"  Epoch {epoch}/{EPOCHS} | Step {step:>4}/{len(dataloader)} | Loss: {loss.item():.4f}")

        avg = total_loss / len(dataloader)
        print(f"Epoch {epoch} done — avg loss: {avg:.4f}\n")

    # Save fine-tuned encoder + tokenizer
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    model.save_pretrained(OUTPUT_PATH)
    tokenizer.save_pretrained(OUTPUT_PATH)
    print(f"Saved fine-tuned model to '{OUTPUT_PATH}'")
    print("\nTo use:")
    print("  from transformers import AutoTokenizer, AutoModel")
    print(f"  tokenizer = AutoTokenizer.from_pretrained('{OUTPUT_PATH}')")
    print(f"  model     = AutoModel.from_pretrained('{OUTPUT_PATH}')")


# ---------------------------------------------------------------------------
# Inference helper (mean-pool + normalise → ready for cosine similarity)
# ---------------------------------------------------------------------------
def embed(texts: list[str], model_path: str = OUTPUT_PATH) -> torch.Tensor:
    """Quick helper to embed texts with the fine-tuned model."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)
    model.eval()
    with torch.no_grad():
        enc = tokenizer(texts, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
        out = model(**enc)
        embs = mean_pool(out.last_hidden_state, enc["attention_mask"])
        return F.normalize(embs, dim=-1)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    train(sys.argv[1])
