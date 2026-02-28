"""
Training script generator.

Takes a node-parameter dict and returns a self-contained Python script
that can be executed standalone or inside a Modal GPU container.

Supported methods:
  Text models:   simcse | mnrl | lora | sft
  Vision models: cnn
  Sequence:      rnn | lstm | gru
"""

from __future__ import annotations

import textwrap
from datetime import datetime, timezone
from typing import Any

# ── Parameter defaults ────────────────────────────────────────────────────────
DEFAULTS: dict[str, Any] = {
    "method":        "simcse",
    "base_model":    "sentence-transformers/all-MiniLM-L6-v2",
    "output_dir":    "./output_model",
    "chunks_file":   "chunks.json",
    "epochs":        3,
    "batch_size":    32,
    "learning_rate": 3e-5,
    "warmup_steps":  100,
    "temperature":   0.05,
    "max_length":    128,
    "weight_decay":  0.01,
    # LoRA
    "lora_r":        16,
    "lora_alpha":    32,
    "lora_dropout":  0.1,
    # CNN
    "num_classes":   2,
    "transfer":      True,
    # RNN/LSTM/GRU
    "vocab_size":    10000,
    "embed_dim":     128,
    "hidden_dim":    256,
    "num_layers":    2,
    "bidirectional": True,
}


def _header(p: dict[str, Any]) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return textwrap.dedent(f'''\
        """
        Auto-generated training script
        Method:  {p["method"].upper()}
        Model:   {p.get("base_model", "custom")}
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

        BASE_MODEL    = {p["base_model"]!r}
        OUTPUT_DIR    = {p["output_dir"]!r}
        CHUNKS_FILE   = {p["chunks_file"]!r}
        EPOCHS        = {p["epochs"]}
        BATCH_SIZE    = {p["batch_size"]}
        LEARNING_RATE = {p["learning_rate"]}
        TEMPERATURE   = {p["temperature"]}
        MAX_LENGTH    = {p["max_length"]}

        chunks = json.loads(Path(CHUNKS_FILE).read_text())
        print(f"Loaded {{len(chunks)}} chunks")

        class TextDataset(Dataset):
            def __init__(self, t): self.t = t
            def __len__(self): return len(self.t)
            def __getitem__(self, i): return self.t[i]

        device = (
            torch.device("cuda") if torch.cuda.is_available()
            else torch.device("mps") if torch.backends.mps.is_available()
            else torch.device("cpu")
        )
        print(f"Device: {{device}}")
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        model = AutoModel.from_pretrained(BASE_MODEL).to(device)
        model.train()

        def mean_pool(h, m):
            e = m.unsqueeze(-1).expand(h.size()).float()
            return torch.sum(h * e, 1) / torch.clamp(e.sum(1), min=1e-9)

        def simcse_loss(e1, e2, temp):
            B = e1.size(0)
            e1, e2 = F.normalize(e1, -1), F.normalize(e2, -1)
            ae = torch.cat([e1, e2])
            sim = torch.mm(ae, ae.t()) / temp
            sim = sim.masked_fill(torch.eye(2*B, dtype=torch.bool, device=e1.device), float("-inf"))
            labels = torch.cat([torch.arange(B, 2*B, device=e1.device), torch.arange(B, device=e1.device)])
            return F.cross_entropy(sim, labels)

        bs = max(2, min(BATCH_SIZE, len(chunks) // 2 if len(chunks) < BATCH_SIZE*2 else BATCH_SIZE))
        loader = DataLoader(TextDataset(chunks), batch_size=bs, shuffle=True, drop_last=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

        for epoch in range(1, EPOCHS + 1):
            total = 0.0
            for step, batch in enumerate(loader, 1):
                def encode(s):
                    enc = tokenizer(s, padding=True, truncation=True,
                                    max_length=MAX_LENGTH, return_tensors="pt").to(device)
                    return mean_pool(model(**enc).last_hidden_state, enc["attention_mask"])
                loss = simcse_loss(encode(batch), encode(batch), TEMPERATURE)
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                total += loss.item()
                if step % 10 == 0 or step == len(loader):
                    print(f"Epoch {{epoch}}/{{EPOCHS}} Step {{step}}/{{len(loader)}} Loss {{loss.item():.4f}}")
            print(f"Epoch {{epoch}} done — avg loss: {{total/len(loader):.4f}}")

        Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
        model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        print(f"Model saved to {{OUTPUT_DIR}}")
    ''')


# ── MNRL ──────────────────────────────────────────────────────────────────────
def _mnrl(p: dict[str, Any]) -> str:
    return _header(p) + _common_imports() + textwrap.dedent(f'''\
        from sentence_transformers import SentenceTransformer, InputExample, losses
        from torch.utils.data import DataLoader

        BASE_MODEL   = {p["base_model"]!r}
        OUTPUT_DIR   = {p["output_dir"]!r}
        CHUNKS_FILE  = {p["chunks_file"]!r}
        EPOCHS       = {p["epochs"]}
        BATCH_SIZE   = {p["batch_size"]}
        WARMUP_STEPS = {p["warmup_steps"]}

        chunks   = json.loads(Path(CHUNKS_FILE).read_text())
        examples = [InputExample(texts=[t, t]) for t in chunks]
        loader   = DataLoader(examples, shuffle=True, batch_size=BATCH_SIZE)
        model    = SentenceTransformer(BASE_MODEL)
        loss_fn  = losses.MultipleNegativesRankingLoss(model)

        model.fit(
            train_objectives=[(loader, loss_fn)],
            epochs=EPOCHS,
            warmup_steps=WARMUP_STEPS,
            show_progress_bar=True,
            output_path=OUTPUT_DIR,
        )
        print(f"Model saved to {{OUTPUT_DIR}}")
    ''')


# ── LoRA ──────────────────────────────────────────────────────────────────────
def _lora(p: dict[str, Any]) -> str:
    return _header(p) + _common_imports() + textwrap.dedent(f'''\
        import torch
        from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer
        from peft import LoraConfig, get_peft_model
        from torch.utils.data import Dataset

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

        chunks    = json.loads(Path(CHUNKS_FILE).read_text())
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

        class ChunkDataset(Dataset):
            def __init__(self, texts):
                self.enc = tokenizer(texts, padding="max_length", truncation=True,
                                     max_length=MAX_LENGTH, return_tensors="pt")
                self.enc["labels"] = self.enc["input_ids"].clone()
            def __len__(self): return len(self.enc["input_ids"])
            def __getitem__(self, i): return {{k: v[i] for k, v in self.enc.items()}}

        base_model  = AutoModel.from_pretrained(BASE_MODEL)
        lora_config = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA,
                                 lora_dropout=LORA_DROPOUT,
                                 target_modules=["query", "value"], bias="none")
        model = get_peft_model(base_model, lora_config)
        model.print_trainable_parameters()

        args = TrainingArguments(
            output_dir=OUTPUT_DIR, num_train_epochs=EPOCHS,
            per_device_train_batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE,
            weight_decay={p["weight_decay"]}, warmup_steps={p["warmup_steps"]},
            logging_steps=10, save_strategy="epoch",
            fp16=torch.cuda.is_available(),
        )
        trainer = Trainer(model=model, args=args, train_dataset=ChunkDataset(chunks))
        trainer.train()
        merged = model.merge_and_unload()
        merged.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        print(f"LoRA model saved to {{OUTPUT_DIR}}")
    ''')


# ── Full SFT ──────────────────────────────────────────────────────────────────
def _sft(p: dict[str, Any]) -> str:
    return _header(p) + _common_imports() + textwrap.dedent(f'''\
        import torch
        from transformers import (AutoTokenizer, AutoModelForMaskedLM,
                                  TrainingArguments, Trainer,
                                  DataCollatorForLanguageModeling)
        from torch.utils.data import Dataset

        BASE_MODEL    = {p["base_model"]!r}
        OUTPUT_DIR    = {p["output_dir"]!r}
        CHUNKS_FILE   = {p["chunks_file"]!r}
        EPOCHS        = {p["epochs"]}
        BATCH_SIZE    = {p["batch_size"]}
        LEARNING_RATE = {p["learning_rate"]}
        WEIGHT_DECAY  = {p["weight_decay"]}
        WARMUP_STEPS  = {p["warmup_steps"]}
        MAX_LENGTH    = {p["max_length"]}

        chunks    = json.loads(Path(CHUNKS_FILE).read_text())
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

        class ChunkDataset(Dataset):
            def __init__(self, texts):
                self.enc = tokenizer(texts, padding="max_length", truncation=True,
                                     max_length=MAX_LENGTH, return_tensors="pt")
            def __len__(self): return len(self.enc["input_ids"])
            def __getitem__(self, i): return {{k: v[i] for k, v in self.enc.items()}}

        model    = AutoModelForMaskedLM.from_pretrained(BASE_MODEL)
        collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

        args = TrainingArguments(
            output_dir=OUTPUT_DIR, num_train_epochs=EPOCHS,
            per_device_train_batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY, warmup_steps=WARMUP_STEPS,
            logging_steps=10, save_strategy="epoch",
            fp16=torch.cuda.is_available(),
        )
        trainer = Trainer(model=model, args=args,
                          train_dataset=ChunkDataset(chunks), data_collator=collator)
        trainer.train()
        model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        print(f"Model saved to {{OUTPUT_DIR}}")
    ''')


# ── CNN ───────────────────────────────────────────────────────────────────────
def _cnn(p: dict[str, Any]) -> str:
    backbone   = p.get("base_model", "resnet18")
    num_classes = int(p.get("num_classes", 2))
    transfer   = bool(p.get("transfer", True))

    backbone_code = ""
    if backbone in ("resnet18", "resnet50"):
        backbone_code = textwrap.dedent(f'''\
            from torchvision.models import resnet18, resnet50, ResNet18_Weights, ResNet50_Weights
            weights = ResNet{"18" if backbone == "resnet18" else "50"}_Weights.DEFAULT if TRANSFER else None
            model   = resnet{"18" if backbone == "resnet18" else "50"}(weights=weights)
            if TRANSFER:
                for param in model.parameters(): param.requires_grad = False
            model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)
        ''')
    elif backbone == "vgg16":
        backbone_code = textwrap.dedent('''\
            from torchvision.models import vgg16, VGG16_Weights
            weights = VGG16_Weights.DEFAULT if TRANSFER else None
            model   = vgg16(weights=weights)
            if TRANSFER:
                for param in model.features.parameters(): param.requires_grad = False
            model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, NUM_CLASSES)
        ''')
    else:  # from scratch
        backbone_code = textwrap.dedent(f'''\
            model = torch.nn.Sequential(
                torch.nn.Conv2d(3, 32, 3, padding=1), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
                torch.nn.Conv2d(32, 64, 3, padding=1), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
                torch.nn.Conv2d(64, 128, 3, padding=1), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
                torch.nn.Flatten(),
                torch.nn.Linear(128 * 28 * 28, 512), torch.nn.ReLU(), torch.nn.Dropout(0.5),
                torch.nn.Linear(512, NUM_CLASSES),
            )
        ''')

    return _header(p) + _common_imports() + textwrap.dedent(f'''\
        import torch
        import torch.nn as nn
        import torchvision.transforms as T
        from torch.utils.data import Dataset, DataLoader
        from PIL import Image

        PATHS_FILE    = {p["chunks_file"]!r}
        OUTPUT_DIR    = {p["output_dir"]!r}
        NUM_CLASSES   = {num_classes}
        EPOCHS        = {p["epochs"]}
        BATCH_SIZE    = {p["batch_size"]}
        LEARNING_RATE = {p["learning_rate"]}
        TRANSFER      = {transfer}

        image_paths = json.loads(Path(PATHS_FILE).read_text())
        print(f"Found {{len(image_paths)}} images")

        # Assign synthetic labels from directory name (or all class 0 if flat)
        from pathlib import Path as _P
        def get_label(fp):
            parent = _P(fp).parent.name
            try: return int(parent)
            except ValueError: return hash(parent) % NUM_CLASSES

        transform = T.Compose([
            T.Resize(224), T.CenterCrop(224), T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        class ImgDataset(Dataset):
            def __init__(self, paths):
                self.paths  = paths
                self.labels = [get_label(p) for p in paths]
            def __len__(self): return len(self.paths)
            def __getitem__(self, i):
                img = Image.open(self.paths[i]).convert("RGB")
                return transform(img), self.labels[i]

        loader = DataLoader(ImgDataset(image_paths), batch_size=BATCH_SIZE,
                            shuffle=True, num_workers=2, pin_memory=True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {{device}}")

        {backbone_code}
        model = model.to(device)
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE
        )
        criterion = nn.CrossEntropyLoss()

        for epoch in range(1, EPOCHS + 1):
            model.train()
            total_loss, correct, total = 0.0, 0, 0
            for imgs, labels in loader:
                imgs, labels = imgs.to(device), torch.tensor(labels, device=device)
                optimizer.zero_grad()
                out  = model(imgs)
                loss = criterion(out, labels)
                loss.backward(); optimizer.step()
                total_loss += loss.item()
                correct    += (out.argmax(1) == labels).sum().item()
                total      += labels.size(0)
            print(f"Epoch {{epoch}}/{{EPOCHS}} loss={{total_loss/len(loader):.4f}} acc={{correct/total:.3f}}")

        Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
        torch.save(model, os.path.join(OUTPUT_DIR, "model.pt"))
        # Save class count for inference
        import json as _json
        _json.dump({{"num_classes": NUM_CLASSES, "backbone": {backbone!r}}},
                   open(os.path.join(OUTPUT_DIR, "config.json"), "w"))
        print(f"CNN saved to {{OUTPUT_DIR}}")
    ''')


# ── RNN / LSTM / GRU ─────────────────────────────────────────────────────────
def _rnn_family(p: dict[str, Any]) -> str:
    rnn_type     = p["method"].upper()   # RNN, LSTM, GRU
    bidirectional = bool(p.get("bidirectional", True))

    return _header(p) + _common_imports() + textwrap.dedent(f'''\
        import collections
        import torch
        import torch.nn as nn
        from torch.utils.data import Dataset, DataLoader

        CHUNKS_FILE   = {p["chunks_file"]!r}
        OUTPUT_DIR    = {p["output_dir"]!r}
        VOCAB_SIZE    = {p["vocab_size"]}
        EMBED_DIM     = {p["embed_dim"]}
        HIDDEN_DIM    = {p["hidden_dim"]}
        NUM_LAYERS    = {p["num_layers"]}
        BIDIRECTIONAL = {bidirectional}
        EPOCHS        = {p["epochs"]}
        BATCH_SIZE    = {p["batch_size"]}
        LEARNING_RATE = {p["learning_rate"]}
        SEQ_LEN       = 64           # context window
        RNN_TYPE      = "{rnn_type}"  # RNN / LSTM / GRU

        chunks = json.loads(Path(CHUNKS_FILE).read_text())
        print(f"Loaded {{len(chunks)}} chunks")

        # ── Build vocabulary ──────────────────────────────────────────────────
        counter = collections.Counter()
        for chunk in chunks:
            counter.update(chunk.lower().split())
        vocab = {{"<PAD>": 0, "<UNK>": 1}}
        for word, _ in counter.most_common(VOCAB_SIZE - 2):
            vocab[word] = len(vocab)
        print(f"Vocabulary size: {{len(vocab)}}")

        # ── Dataset ───────────────────────────────────────────────────────────
        def encode(text):
            return [vocab.get(w, 1) for w in text.lower().split()]

        sequences = []
        for chunk in chunks:
            ids = encode(chunk)
            for i in range(0, len(ids) - SEQ_LEN, SEQ_LEN // 2):
                sequences.append(ids[i : i + SEQ_LEN + 1])

        class SeqDataset(Dataset):
            def __init__(self, seqs): self.seqs = seqs
            def __len__(self): return len(self.seqs)
            def __getitem__(self, i):
                s = self.seqs[i]
                x = torch.tensor(s[:-1], dtype=torch.long)
                y = torch.tensor(s[1:],  dtype=torch.long)
                return x, y

        loader = DataLoader(SeqDataset(sequences), batch_size=BATCH_SIZE,
                            shuffle=True, drop_last=True,
                            collate_fn=lambda b: (
                                torch.nn.utils.rnn.pad_sequence([x for x,_ in b], batch_first=True),
                                torch.nn.utils.rnn.pad_sequence([y for _,y in b], batch_first=True),
                            ))

        # ── Model ─────────────────────────────────────────────────────────────
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {{device}}")

        class LMModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(len(vocab), EMBED_DIM, padding_idx=0)
                rnn_cls = getattr(nn, RNN_TYPE)
                self.rnn = rnn_cls(EMBED_DIM, HIDDEN_DIM, num_layers=NUM_LAYERS,
                                   batch_first=True, dropout=0.3 if NUM_LAYERS > 1 else 0.0,
                                   bidirectional=BIDIRECTIONAL)
                factor = 2 if BIDIRECTIONAL else 1
                self.fc = nn.Linear(HIDDEN_DIM * factor, len(vocab))

            def forward(self, x):
                emb = self.embed(x)
                out, _ = self.rnn(emb)
                return self.fc(out)

        model     = LMModel().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        criterion = nn.CrossEntropyLoss(ignore_index=0)

        # ── Training ──────────────────────────────────────────────────────────
        for epoch in range(1, EPOCHS + 1):
            model.train()
            total_loss = 0.0
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                logits = model(x)                             # (B, T, V)
                loss   = criterion(logits.view(-1, len(vocab)), y.view(-1))
                loss.backward(); optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {{epoch}}/{{EPOCHS}} loss={{total_loss/len(loader):.4f}}")

        # ── Save ──────────────────────────────────────────────────────────────
        Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
        torch.save({{"model": model, "vocab": vocab}},
                   os.path.join(OUTPUT_DIR, "model.pt"))
        import json as _json
        _json.dump({{"rnn_type": RNN_TYPE, "vocab_size": len(vocab),
                    "embed_dim": EMBED_DIM, "hidden_dim": HIDDEN_DIM}},
                   open(os.path.join(OUTPUT_DIR, "config.json"), "w"))
        print(f"{{RNN_TYPE}} model saved to {{OUTPUT_DIR}}")
    ''')


# ── Public API ────────────────────────────────────────────────────────────────
GENERATORS = {
    "simcse": _simcse,
    "mnrl":   _mnrl,
    "lora":   _lora,
    "sft":    _sft,
    "cnn":    _cnn,
    "rnn":    _rnn_family,
    "lstm":   _rnn_family,
    "gru":    _rnn_family,
}


def generate_script(params: dict[str, Any]) -> str:
    """
    Generate a self-contained Python training script from node parameters.

    Args:
        params: Dict matching DEFAULTS keys. 'method' selects the generator.

    Returns:
        Python source code as a string.
    """
    p      = {**DEFAULTS, **{k: v for k, v in params.items() if v is not None}}
    method = p["method"].lower()

    if method not in GENERATORS:
        raise ValueError(
            f"Unknown training method '{method}'. "
            f"Choose from: {list(GENERATORS)}"
        )

    return GENERATORS[method](p)
