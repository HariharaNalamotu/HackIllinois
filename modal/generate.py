"""
Training script generator.

Takes a node-parameter dict from the UI and returns a self-contained
Python training script that can be run standalone or executed by Modal.

Supported training methods:
  simcse  — Unsupervised SimCSE (same sentence, different dropout masks)
  mnrl    — MultipleNegativesRankingLoss via sentence-transformers
  lora    — Parameter-efficient LoRA fine-tuning via PEFT
  sft     — Full supervised fine-tuning via HuggingFace Trainer
"""

from __future__ import annotations
import textwrap
from datetime import datetime, timezone
from typing import Any

# ── Parameter defaults ────────────────────────────────────────────────────────
DEFAULTS: dict[str, Any] = {
    "method": "simcse",
    "base_model": "sentence-transformers/all-MiniLM-L6-v2",
    "output_dir": "./output_model",
    "chunks_file": "chunks.json",
    "epochs": 3,
    "batch_size": 32,
    "learning_rate": 3e-5,
    "warmup_steps": 100,
    "temperature": 0.05,
    "max_length": 128,
    "weight_decay": 0.01,
    # LoRA-specific
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.1,
}

# ── Template helpers ──────────────────────────────────────────────────────────
def _header(p: dict[str, Any]) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return textwrap.dedent(f'''\
        """
        Auto-generated training script
        Method:  {p["method"].upper()}
        Model:   {p["base_model"]}
        GPU:     {p.get("gpu", "auto")}
        Created: {ts}
        """
    ''')

def _common_imports() -> str:
    return textwrap.dedent('''\
        import json
        import os
        from pathlib import Path
    ''')


# ── SimCSE ────────────────────────────────────────────────────────────────────
def _simcse(p: dict[str, Any]) -> str:
    return _header(p) + _common_imports() + textwrap.dedent(f'''\
        import torch
        import torch.nn.functional as F
        from torch.utils.data import Dataset, DataLoader
        from transformers import AutoTokenizer, AutoModel

        # ── Config ────────────────────────────────────────────────────────────
        BASE_MODEL    = {p["base_model"]!r}
        OUTPUT_DIR    = {p["output_dir"]!r}
        CHUNKS_FILE   = {p["chunks_file"]!r}
        EPOCHS        = {p["epochs"]}
        BATCH_SIZE    = {p["batch_size"]}
        LEARNING_RATE = {p["learning_rate"]}
        TEMPERATURE   = {p["temperature"]}
        MAX_LENGTH    = {p["max_length"]}

        # ── Data ──────────────────────────────────────────────────────────────
        chunks = json.loads(Path(CHUNKS_FILE).read_text())
        print(f"Loaded {{len(chunks)}} chunks")

        class TextDataset(Dataset):
            def __init__(self, t): self.t = t
            def __len__(self): return len(self.t)
            def __getitem__(self, i): return self.t[i]

        # ── Model ─────────────────────────────────────────────────────────────
        device = (
            torch.device("cuda") if torch.cuda.is_available()
            else torch.device("mps") if torch.backends.mps.is_available()
            else torch.device("cpu")
        )
        print(f"Device: {{device}}")
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        model = AutoModel.from_pretrained(BASE_MODEL).to(device)
        model.train()

        # ── Training helpers ──────────────────────────────────────────────────
        def mean_pool(h, m):
            e = m.unsqueeze(-1).expand(h.size()).float()
            return torch.sum(h * e, 1) / torch.clamp(e.sum(1), min=1e-9)

        def simcse_loss(e1, e2, temp):
            B = e1.size(0)
            e1, e2 = F.normalize(e1, -1), F.normalize(e2, -1)
            ae = torch.cat([e1, e2])
            sim = torch.mm(ae, ae.t()) / temp
            sim = sim.masked_fill(
                torch.eye(2 * B, dtype=torch.bool, device=e1.device), float("-inf")
            )
            labels = torch.cat([
                torch.arange(B, 2 * B, device=e1.device),
                torch.arange(B, device=e1.device),
            ])
            return F.cross_entropy(sim, labels)

        bs = max(2, min(BATCH_SIZE, len(chunks) // 2 if len(chunks) < BATCH_SIZE * 2 else BATCH_SIZE))
        loader = DataLoader(TextDataset(chunks), batch_size=bs, shuffle=True, drop_last=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

        # ── Training loop ─────────────────────────────────────────────────────
        for epoch in range(1, EPOCHS + 1):
            total = 0.0
            for step, batch in enumerate(loader, 1):
                def encode(s):
                    enc = tokenizer(
                        s, padding=True, truncation=True,
                        max_length=MAX_LENGTH, return_tensors="pt",
                    ).to(device)
                    return mean_pool(model(**enc).last_hidden_state, enc["attention_mask"])
                loss = simcse_loss(encode(batch), encode(batch), TEMPERATURE)
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                total += loss.item()
                if step % 10 == 0 or step == len(loader):
                    print(f"Epoch {{epoch}}/{{EPOCHS}} Step {{step}}/{{len(loader)}} Loss {{loss.item():.4f}}")
            print(f"Epoch {{epoch}} done — avg loss: {{total / len(loader):.4f}}")

        # ── Save ──────────────────────────────────────────────────────────────
        Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
        model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        print(f"Model saved to {{OUTPUT_DIR}}")
    ''')


# ── MNRL (sentence-transformers MultipleNegativesRankingLoss) ─────────────────
def _mnrl(p: dict[str, Any]) -> str:
    return _header(p) + _common_imports() + textwrap.dedent(f'''\
        from sentence_transformers import SentenceTransformer, InputExample, losses
        from torch.utils.data import DataLoader

        # ── Config ────────────────────────────────────────────────────────────
        BASE_MODEL    = {p["base_model"]!r}
        OUTPUT_DIR    = {p["output_dir"]!r}
        CHUNKS_FILE   = {p["chunks_file"]!r}
        EPOCHS        = {p["epochs"]}
        BATCH_SIZE    = {p["batch_size"]}
        WARMUP_STEPS  = {p["warmup_steps"]}

        # ── Data ──────────────────────────────────────────────────────────────
        chunks = json.loads(Path(CHUNKS_FILE).read_text())
        print(f"Loaded {{len(chunks)}} chunks")
        # Each chunk is its own anchor+positive pair (unsupervised)
        examples = [InputExample(texts=[t, t]) for t in chunks]
        loader = DataLoader(examples, shuffle=True, batch_size=BATCH_SIZE)

        # ── Model + loss ──────────────────────────────────────────────────────
        model = SentenceTransformer(BASE_MODEL)
        loss_fn = losses.MultipleNegativesRankingLoss(model)

        # ── Train ─────────────────────────────────────────────────────────────
        model.fit(
            train_objectives=[(loader, loss_fn)],
            epochs=EPOCHS,
            warmup_steps=WARMUP_STEPS,
            show_progress_bar=True,
            output_path=OUTPUT_DIR,
        )
        print(f"Model saved to {{OUTPUT_DIR}}")
    ''')


# ── LoRA (PEFT) ───────────────────────────────────────────────────────────────
def _lora(p: dict[str, Any]) -> str:
    return _header(p) + _common_imports() + textwrap.dedent(f'''\
        import torch
        from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer
        from transformers import DataCollatorWithPadding
        from peft import LoraConfig, get_peft_model, TaskType
        from torch.utils.data import Dataset

        # ── Config ────────────────────────────────────────────────────────────
        BASE_MODEL    = {p["base_model"]!r}
        OUTPUT_DIR    = {p["output_dir"]!r}
        CHUNKS_FILE   = {p["chunks_file"]!r}
        EPOCHS        = {p["epochs"]}
        BATCH_SIZE    = {p["batch_size"]}
        LEARNING_RATE = {p["learning_rate"]}
        LORA_R        = {p["lora_r"]}
        LORA_ALPHA    = {p["lora_alpha"]}
        LORA_DROPOUT  = {p["lora_dropout"]}
        MAX_LENGTH    = {p["max_length"]}

        # ── Data ──────────────────────────────────────────────────────────────
        chunks = json.loads(Path(CHUNKS_FILE).read_text())
        print(f"Loaded {{len(chunks)}} chunks")
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

        class ChunkDataset(Dataset):
            def __init__(self, texts):
                self.enc = tokenizer(
                    texts, padding="max_length", truncation=True,
                    max_length=MAX_LENGTH, return_tensors="pt",
                )
                # MLM-style: use input_ids as labels
                self.enc["labels"] = self.enc["input_ids"].clone()
            def __len__(self): return len(self.enc["input_ids"])
            def __getitem__(self, i):
                return {{k: v[i] for k, v in self.enc.items()}}

        dataset = ChunkDataset(chunks)

        # ── Model + LoRA ──────────────────────────────────────────────────────
        base_model = AutoModel.from_pretrained(BASE_MODEL)
        lora_config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            target_modules=["query", "value"],   # adjust for your architecture
            bias="none",
        )
        model = get_peft_model(base_model, lora_config)
        model.print_trainable_parameters()

        # ── Train ─────────────────────────────────────────────────────────────
        args = TrainingArguments(
            output_dir=OUTPUT_DIR,
            num_train_epochs=EPOCHS,
            per_device_train_batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            weight_decay={p["weight_decay"]},
            warmup_steps={p["warmup_steps"]},
            logging_steps=10,
            save_strategy="epoch",
            fp16=torch.cuda.is_available(),
        )
        trainer = Trainer(model=model, args=args, train_dataset=dataset)
        trainer.train()

        # ── Save merged weights ───────────────────────────────────────────────
        merged = model.merge_and_unload()
        merged.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        print(f"LoRA-merged model saved to {{OUTPUT_DIR}}")
    ''')


# ── Full SFT (HuggingFace Trainer) ────────────────────────────────────────────
def _sft(p: dict[str, Any]) -> str:
    return _header(p) + _common_imports() + textwrap.dedent(f'''\
        import torch
        from transformers import (
            AutoTokenizer, AutoModelForMaskedLM,
            TrainingArguments, Trainer, DataCollatorForLanguageModeling,
        )
        from torch.utils.data import Dataset

        # ── Config ────────────────────────────────────────────────────────────
        BASE_MODEL    = {p["base_model"]!r}
        OUTPUT_DIR    = {p["output_dir"]!r}
        CHUNKS_FILE   = {p["chunks_file"]!r}
        EPOCHS        = {p["epochs"]}
        BATCH_SIZE    = {p["batch_size"]}
        LEARNING_RATE = {p["learning_rate"]}
        WEIGHT_DECAY  = {p["weight_decay"]}
        WARMUP_STEPS  = {p["warmup_steps"]}
        MAX_LENGTH    = {p["max_length"]}
        MLM_PROB      = 0.15

        # ── Data ──────────────────────────────────────────────────────────────
        chunks = json.loads(Path(CHUNKS_FILE).read_text())
        print(f"Loaded {{len(chunks)}} chunks")
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

        class ChunkDataset(Dataset):
            def __init__(self, texts):
                self.enc = tokenizer(
                    texts, padding="max_length", truncation=True,
                    max_length=MAX_LENGTH, return_tensors="pt",
                )
            def __len__(self): return len(self.enc["input_ids"])
            def __getitem__(self, i):
                return {{k: v[i] for k, v in self.enc.items()}}

        dataset = ChunkDataset(chunks)
        collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=MLM_PROB)

        # ── Model ─────────────────────────────────────────────────────────────
        model = AutoModelForMaskedLM.from_pretrained(BASE_MODEL)

        # ── Train ─────────────────────────────────────────────────────────────
        args = TrainingArguments(
            output_dir=OUTPUT_DIR,
            num_train_epochs=EPOCHS,
            per_device_train_batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            warmup_steps=WARMUP_STEPS,
            logging_steps=10,
            save_strategy="epoch",
            fp16=torch.cuda.is_available(),
        )
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=dataset,
            data_collator=collator,
        )
        trainer.train()

        model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        print(f"Model saved to {{OUTPUT_DIR}}")
    ''')


# ── Public API ────────────────────────────────────────────────────────────────
GENERATORS = {
    "simcse": _simcse,
    "mnrl":   _mnrl,
    "lora":   _lora,
    "sft":    _sft,
}


def generate_script(params: dict[str, Any]) -> str:
    """
    Generate a self-contained Python training script from node parameters.

    Args:
        params: Dict with keys matching DEFAULTS. Unknown keys are ignored.
                'method' selects the training method (simcse/mnrl/lora/sft).

    Returns:
        Python source code as a string.
    """
    p = {**DEFAULTS, **{k: v for k, v in params.items() if v is not None}}
    method = p["method"].lower()

    if method not in GENERATORS:
        raise ValueError(
            f"Unknown training method '{method}'. "
            f"Choose from: {list(GENERATORS)}"
        )

    return GENERATORS[method](p)
