"""
Transform nodes — preprocessing steps between input and model nodes.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any, Callable

from .base import BaseNode
from ..types import NodeOutput


class ChunkNode(BaseNode):
    """
    Splits text into training-ready chunks.

    Params:
        method:        "auto" or any CHUNKERS key (sentence, paragraph, …)
        method_params: extra kwargs passed to the chosen chunker

    Input:  {"type": "text", "texts": list[str]}
    Output: {"type": "chunks", "chunks": list[str], "method_used": str}
    """

    def execute(
        self,
        inputs: list[NodeOutput],
        files: list[str],
        ctx: dict[str, Any],
        log: Callable[[str], None],
    ) -> NodeOutput:
        import sys
        sys.path.insert(0, "/app")
        from chunk import CHUNKERS, agent_choose_chunker  # type: ignore[import]

        inp = self.first_of_type(inputs, "text")
        if inp is None:
            raise ValueError("ChunkNode requires a text input.")

        method        = self.p("method", "auto")
        method_params = self.p("method_params", {})

        all_chunks: list[str] = []
        method_used = method

        for i, text in enumerate(inp["texts"]):
            if len(text.strip()) < 30:
                continue

            if method == "auto":
                # Write text to a temp file so agent_choose_chunker can inspect it
                tmp_path = Path(ctx["workspace"]) / f"chunk_input_{i}.txt"
                tmp_path.write_text(text, encoding="utf-8")
                chosen, chosen_params = agent_choose_chunker(tmp_path, text[:3000])
                method_used = chosen
                chunker = CHUNKERS[chosen]
                valid_kw = {k: v for k, v in chosen_params.items()
                            if k in inspect.signature(chunker).parameters}
            else:
                if method not in CHUNKERS:
                    raise ValueError(f"ChunkNode: unknown method '{method}'.")
                chunker  = CHUNKERS[method]
                valid_kw = {k: v for k, v in method_params.items()
                            if k in inspect.signature(chunker).parameters}

            chunks = [c.strip() for c in chunker(text, **valid_kw)
                      if len(c.strip().split()) >= 5]
            all_chunks.extend(chunks)
            log(f"  [Chunk] text {i}: {len(chunks)} chunks (method={method_used})")

        if not all_chunks:
            raise ValueError("ChunkNode: no usable chunks produced.")

        return {"type": "chunks", "chunks": all_chunks, "method_used": method_used}


class ImagePreprocessNode(BaseNode):
    """
    Resizes and optionally normalises images.

    Params:
        resize:    Resize shorter side to N px (default 224)
        normalize: Apply ImageNet normalisation (default True)

    Input:  {"type": "image", "paths": list[str]}
    Output: {"type": "image", "paths": list[str], "preprocessed": True}
    """

    def execute(
        self,
        inputs: list[NodeOutput],
        files: list[str],
        ctx: dict[str, Any],
        log: Callable[[str], None],
    ) -> NodeOutput:
        from PIL import Image  # type: ignore[import]
        import torchvision.transforms as T  # type: ignore[import]
        import torch

        inp = self.first_of_type(inputs, "image")
        if inp is None:
            raise ValueError("ImagePreprocessNode requires an image input.")

        resize    = int(self.p("resize", 224))
        normalize = bool(self.p("normalize", True))

        tfms = [T.Resize(resize), T.CenterCrop(resize), T.ToTensor()]
        if normalize:
            tfms.append(T.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]))
        transform = T.Compose(tfms)

        out_dir = Path(ctx["workspace"]) / "preprocessed_images"
        out_dir.mkdir(exist_ok=True)

        out_paths: list[str] = []
        for src in inp["paths"]:
            p = Path(src)
            try:
                img = Image.open(p).convert("RGB")
                tensor = transform(img)
                out_p = out_dir / p.name
                # Save as tensor file for downstream CNN node
                torch.save(tensor, str(out_p) + ".pt")
                out_paths.append(str(out_p) + ".pt")
                log(f"  [ImagePreprocess] {p.name} → {resize}×{resize}")
            except Exception as e:
                log(f"  [ImagePreprocess] error {p.name}: {e}")

        return {
            "type":          "image",
            "paths":         out_paths,
            "preprocessed":  True,
            "original_paths": inp["paths"],
        }


class AudioPreprocessNode(BaseNode):
    """
    Converts audio to log-mel spectrograms.

    Params:
        sample_rate: target SR (default 16000)
        n_mels:      mel bins (default 80)

    Input:  {"type": "audio", "paths": list[str]}
    Output: {"type": "audio", "paths": list[str], "preprocessed": True}
    """

    def execute(
        self,
        inputs: list[NodeOutput],
        files: list[str],
        ctx: dict[str, Any],
        log: Callable[[str], None],
    ) -> NodeOutput:
        import torch
        import torchaudio  # type: ignore[import]
        import torchaudio.transforms as AT  # type: ignore[import]

        inp = self.first_of_type(inputs, "audio")
        if inp is None:
            raise ValueError("AudioPreprocessNode requires an audio input.")

        sr     = int(self.p("sample_rate", 16000))
        n_mels = int(self.p("n_mels", 80))

        mel_transform = AT.MelSpectrogram(sample_rate=sr, n_mels=n_mels)
        db_transform  = AT.AmplitudeToDB()

        out_dir = Path(ctx["workspace"]) / "preprocessed_audio"
        out_dir.mkdir(exist_ok=True)

        out_paths: list[str] = []
        for src in inp["paths"]:
            p = Path(src)
            try:
                wav, orig_sr = torchaudio.load(str(p))
                if orig_sr != sr:
                    resampler = AT.Resample(orig_sr, sr)
                    wav = resampler(wav)
                spec = db_transform(mel_transform(wav))
                out_p = out_dir / (p.stem + ".pt")
                torch.save(spec, str(out_p))
                out_paths.append(str(out_p))
                log(f"  [AudioPreprocess] {p.name} → spectrogram {spec.shape}")
            except Exception as e:
                log(f"  [AudioPreprocess] error {p.name}: {e}")

        return {
            "type":         "audio",
            "paths":        out_paths,
            "preprocessed": True,
        }


class TabularPreprocessNode(BaseNode):
    """
    Encodes categoricals, scales numerics, separates target column.

    Params:
        target_column:  column name to use as label (default "")
        scale_features: standard-scale numeric cols (default True)

    Input:  {"type": "tabular", "rows": list[dict], "columns": list[str]}
    Output: {"type": "tabular", "rows": ..., "columns": ...,
             "target_column": str, "feature_matrix": list[list[float]],
             "labels": list}
    """

    def execute(
        self,
        inputs: list[NodeOutput],
        files: list[str],
        ctx: dict[str, Any],
        log: Callable[[str], None],
    ) -> NodeOutput:
        import json as _json
        try:
            import numpy as np  # type: ignore[import]
            from sklearn.preprocessing import LabelEncoder, StandardScaler  # type: ignore[import]
        except ImportError as e:
            raise ImportError(f"TabularPreprocessNode needs numpy + scikit-learn: {e}")

        inp = self.first_of_type(inputs, "tabular")
        if inp is None:
            raise ValueError("TabularPreprocessNode requires a tabular input.")

        rows          = inp["rows"]
        columns       = inp["columns"]
        target_col    = self.p("target_column", "")
        scale_features= bool(self.p("scale_features", True))

        feature_cols = [c for c in columns if c != target_col] if target_col else columns
        labels: list = []

        # Build feature matrix
        X_raw = []
        for row in rows:
            X_raw.append([row.get(c, 0) for c in feature_cols])

        # Try numeric conversion; encode strings
        X = []
        encoders: dict[int, LabelEncoder] = {}
        for col_i, col in enumerate(feature_cols):
            col_vals = [r[col_i] for r in X_raw]
            try:
                col_floats = [float(v) if v not in (None, "") else 0.0 for v in col_vals]
            except (ValueError, TypeError):
                le = LabelEncoder()
                col_floats = le.fit_transform([str(v) for v in col_vals]).tolist()
                encoders[col_i] = le
            for r_i, val in enumerate(col_floats):
                if len(X) <= r_i:
                    X.append([])
                X[r_i].append(val)

        X_arr = np.array(X, dtype=np.float32)

        if scale_features and X_arr.shape[0] > 1:
            scaler = StandardScaler()
            X_arr = scaler.fit_transform(X_arr)

        if target_col and target_col in columns:
            raw_labels = [row.get(target_col) for row in rows]
            try:
                labels = [float(v) for v in raw_labels]
            except (ValueError, TypeError):
                le = LabelEncoder()
                labels = le.fit_transform([str(v) for v in raw_labels]).tolist()

        log(f"  [TabularPreprocess] {X_arr.shape[0]} rows × {X_arr.shape[1]} features, target='{target_col}'")

        return {
            "type":           "tabular",
            "rows":           rows,
            "columns":        columns,
            "feature_cols":   feature_cols,
            "target_column":  target_col,
            "feature_matrix": X_arr.tolist(),
            "labels":         labels,
        }
