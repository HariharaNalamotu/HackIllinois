"""
End-to-end pipeline: chunk corpus → fine-tune a copy of MiniLM.

Steps:
  1. Duplicate the original model (never touches the original).
  2. Sample several PDFs from corpus/ and ask OpenAI to pick the best
     chunking strategy for the whole corpus.
  3. Process every file with robust error handling; skip unreadable ones.
  4. Fine-tune the duplicated model on the collected chunks.

Usage:
  python pipeline.py [--corpus corpus/] [--sample-size 8]
"""

import os
import sys
import json
import shutil
import argparse
import textwrap
import traceback
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import openai
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel

# ── import chunkers and reader from chunk.py ────────────────────────────────
from chunk import (
    read_file,
    CHUNKERS,
    TOOLS,
    agent_choose_chunker,
)

import inspect

# ============================================================================
# Config
# ============================================================================
ORIGINAL_MODEL  = "./models/all-MiniLM-L6-v2"
FINETUNED_MODEL = "./models/finetuned-MiniLM-corpus"
CHUNKS_CACHE    = "./corpus_chunks.json"

BATCH_SIZE    = 32
EPOCHS        = 3
LEARNING_RATE = 3e-5
MAX_LENGTH    = 128
TEMPERATURE   = 0.05

client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])


# ============================================================================
# Step 1 — duplicate model
# ============================================================================
def duplicate_model():
    src = Path(ORIGINAL_MODEL)
    dst = Path(FINETUNED_MODEL)
    if not src.exists():
        print(f"[ERROR] Original model not found at '{src}'")
        sys.exit(1)
    if dst.exists():
        print(f"[INFO] Duplicate already exists at '{dst}' — reusing.")
        return
    print(f"[1/4] Duplicating model: '{src}' → '{dst}' ...")
    shutil.copytree(src, dst)
    print(f"      Done.")


# ============================================================================
# Step 2 — agentic chunking strategy decision (corpus-wide)
# ============================================================================
def pick_strategy(corpus_dir: Path, sample_size: int) -> tuple[str, dict]:
    """
    Sample up to `sample_size` readable PDFs, build a combined sample,
    and ask OpenAI to pick the best chunking strategy for the whole corpus.
    """
    pdf_files = list(corpus_dir.glob("*.pdf"))
    if not pdf_files:
        print("[ERROR] No PDF files found in corpus directory.")
        sys.exit(1)

    # Try to read sample_size files, skip unreadable ones
    samples: list[str] = []
    tried = 0
    for pdf in pdf_files:
        if len(samples) >= sample_size:
            break
        tried += 1
        try:
            text = read_file(pdf)
            if len(text.strip()) < 50:
                continue
            samples.append(f"=== {pdf.name} ===\n{text[:600]}")
        except Exception as e:
            print(f"  [SKIP sample] {pdf.name}: {e}")

    if not samples:
        print("[ERROR] Could not read any PDF files for sampling.")
        sys.exit(1)

    combined_sample = "\n\n".join(samples)[:4000]
    print(f"\n[2/4] Asking OpenAI to choose chunking strategy "
          f"(sampled {len(samples)}/{tried} files) ...")

    system_prompt = textwrap.dedent("""
        You are a text-processing expert. You will be shown samples from
        multiple PDF documents in a corpus. Choose the single chunking
        strategy that will work best across the whole corpus by calling
        exactly one of the provided tools. Choose parameters that produce
        clean, semantically coherent chunks of roughly 50–200 words.
        Do not explain yourself; just call the tool.
    """).strip()

    user_message = (
        f"Corpus directory contains {len(pdf_files)} PDF files.\n\n"
        f"--- SAMPLES FROM {len(samples)} FILES ---\n{combined_sample}\n--- END ---"
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ],
        tools=TOOLS,
        tool_choice="required",
    )

    tool_call = response.choices[0].message.tool_calls[0]
    method    = tool_call.function.name
    kwargs    = json.loads(tool_call.function.arguments) if tool_call.function.arguments else {}
    print(f"      Strategy: '{method}'  params: {kwargs}")
    return method, kwargs


# ============================================================================
# Step 3 — process every file in corpus
# ============================================================================
def collect_chunks(corpus_dir: Path, method: str, kwargs: dict) -> list[str]:
    """
    Apply the chosen chunking method to every file in the corpus.
    Skips files that are unreadable / corrupted / empty.
    Returns the full list of text chunks.
    """
    chunker = CHUNKERS[method]
    sig     = inspect.signature(chunker)
    valid_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}

    all_files = sorted(corpus_dir.iterdir())
    total     = len(all_files)
    chunks: list[str] = []
    ok, skipped = 0, 0

    print(f"\n[3/4] Processing {total} files with '{method}' chunker ...")

    for i, path in enumerate(all_files, 1):
        prefix = f"  [{i:>3}/{total}] {path.name}"
        try:
            text = read_file(path)

            # Reject nearly-empty extractions (corrupted / image-only PDFs)
            if len(text.strip()) < 30:
                raise ValueError("Extracted text too short — likely corrupted or image-only.")

            file_chunks = chunker(text, **valid_kwargs)
            # Filter chunks that are too short
            file_chunks = [c.strip() for c in file_chunks if len(c.strip().split()) >= 5]

            if not file_chunks:
                raise ValueError("No valid chunks produced.")

            chunks.extend(file_chunks)
            ok += 1
            print(f"{prefix}  ✓  {len(file_chunks)} chunks  (total: {len(chunks)})")

        except Exception as e:
            skipped += 1
            short_err = str(e).split("\n")[0][:120]
            print(f"{prefix}  ✗  SKIPPED — {short_err}")

    print(f"\n      Finished: {ok} files processed, {skipped} skipped.")
    print(f"      Total chunks collected: {len(chunks)}")

    if not chunks:
        print("[ERROR] No chunks collected — cannot fine-tune.")
        sys.exit(1)

    return chunks


# ============================================================================
# Step 4 — fine-tune the duplicated model (SimCSE)
# ============================================================================
class TextDataset(Dataset):
    def __init__(self, texts):  self.texts = texts
    def __len__(self):          return len(self.texts)
    def __getitem__(self, idx): return self.texts[idx]


def mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.unsqueeze(-1).expand(hidden.size()).float()
    return torch.sum(hidden * m, 1) / torch.clamp(m.sum(1), min=1e-9)


def simcse_loss(e1: torch.Tensor, e2: torch.Tensor, temp: float) -> torch.Tensor:
    B = e1.size(0)
    e1 = F.normalize(e1, dim=-1)
    e2 = F.normalize(e2, dim=-1)
    all_e = torch.cat([e1, e2], dim=0)
    sim   = torch.mm(all_e, all_e.t()) / temp
    mask  = torch.eye(2 * B, dtype=torch.bool, device=e1.device)
    sim   = sim.masked_fill(mask, float("-inf"))
    labels = torch.cat([
        torch.arange(B, 2 * B, device=e1.device),
        torch.arange(0, B,     device=e1.device),
    ])
    return F.cross_entropy(sim, labels)


def finetune(chunks: list[str]):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"\n[4/4] Fine-tuning on {len(chunks)} chunks  (device: {device}) ...")

    tokenizer = AutoTokenizer.from_pretrained(FINETUNED_MODEL)
    model     = AutoModel.from_pretrained(FINETUNED_MODEL).to(device)
    model.train()

    loader = DataLoader(
        TextDataset(chunks),
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
    )

    if len(chunks) < BATCH_SIZE:
        print(f"[WARN] Fewer chunks ({len(chunks)}) than batch size ({BATCH_SIZE}). "
              f"Reducing batch size to {max(2, len(chunks) // 2)}.")
        effective_bs = max(2, len(chunks) // 2)
        loader = DataLoader(TextDataset(chunks), batch_size=effective_bs,
                            shuffle=True, drop_last=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(1, EPOCHS + 1):
        total_loss = 0.0
        for step, batch in enumerate(loader, 1):
            def encode(sentences):
                enc = tokenizer(
                    sentences, padding=True, truncation=True,
                    max_length=MAX_LENGTH, return_tensors="pt",
                ).to(device)
                out = model(**enc)
                return mean_pool(out.last_hidden_state, enc["attention_mask"])

            loss = simcse_loss(encode(batch), encode(batch), TEMPERATURE)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if step % 10 == 0 or step == len(loader):
                print(f"  Epoch {epoch}/{EPOCHS} | Step {step:>4}/{len(loader)} | Loss: {loss.item():.4f}")

        print(f"  Epoch {epoch} complete — avg loss: {total_loss / len(loader):.4f}\n")

    model.save_pretrained(FINETUNED_MODEL)
    tokenizer.save_pretrained(FINETUNED_MODEL)
    print(f"Fine-tuned model saved to '{FINETUNED_MODEL}'")


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus",      default="corpus/",
                        help="Directory containing input files (default: corpus/)")
    parser.add_argument("--sample-size", type=int, default=8,
                        help="Number of files to sample when choosing chunking strategy (default: 8)")
    parser.add_argument("--skip-cache",  action="store_true",
                        help="Ignore cached chunks and re-process corpus")
    args = parser.parse_args()

    corpus_dir = Path(args.corpus)
    if not corpus_dir.exists():
        print(f"[ERROR] Corpus directory not found: '{corpus_dir}'")
        sys.exit(1)

    # Step 1 — duplicate model
    duplicate_model()

    # Step 2 — pick strategy (or load cached chunks)
    cache = Path(CHUNKS_CACHE)
    if cache.exists() and not args.skip_cache:
        print(f"\n[INFO] Loading cached chunks from '{cache}' (use --skip-cache to reprocess).")
        chunks = json.loads(cache.read_text(encoding="utf-8"))
        print(f"       {len(chunks)} chunks loaded.")
    else:
        method, kwargs = pick_strategy(corpus_dir, args.sample_size)

        # Step 3 — collect chunks
        chunks = collect_chunks(corpus_dir, method, kwargs)

        # Cache chunks so re-runs skip processing
        cache.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n      Chunks cached to '{cache}'")

    # Step 4 — fine-tune
    finetune(chunks)

    print("\nPipeline complete.")
    print(f"  Original model : {ORIGINAL_MODEL}  (unchanged)")
    print(f"  Fine-tuned model: {FINETUNED_MODEL}")


if __name__ == "__main__":
    main()
