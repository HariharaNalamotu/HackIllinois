"""
Core type definitions for the node-based pipeline system.

A Pipeline is a directed acyclic graph (DAG) of Nodes connected by Edges.
Data flows between nodes as NodeOutput dicts.

Node output shapes:
  text       {"type": "text",    "texts": list[str]}
  image      {"type": "image",   "paths": list[str]}
  audio      {"type": "audio",   "paths": list[str]}
  tabular    {"type": "tabular", "rows": list[dict], "columns": list[str]}
  chunks     {"type": "chunks",  "chunks": list[str]}
  model      {"type": "model",   "model_path": str, "model_type": str,
               "arch": str}                              # arch: text|cnn|rnn|tabular
  infer_out  {"type": "infer_out", "predictions": list, "labels": list[str]}
"""

from __future__ import annotations

from enum import Enum
from typing import Any


class NodeType(str, Enum):
    # ── Input nodes ───────────────────────────────────────────────────────────
    TEXT_INPUT        = "text_input"
    IMAGE_INPUT       = "image_input"
    AUDIO_INPUT       = "audio_input"
    SPREADSHEET_INPUT = "spreadsheet_input"
    API_INPUT         = "api_input"

    # ── Transform nodes ───────────────────────────────────────────────────────
    CHUNK             = "chunk"
    IMAGE_PREPROCESS  = "image_preprocess"
    AUDIO_PREPROCESS  = "audio_preprocess"
    TABULAR_PREPROCESS = "tabular_preprocess"

    # ── Model nodes (training / inference) ────────────────────────────────────
    TEXT_MODEL        = "text_model"     # sentence-transformers / MiniLM style
    CNN_MODEL         = "cnn_model"
    RNN_MODEL         = "rnn_model"      # RNN / LSTM / GRU

    # ── Output nodes ──────────────────────────────────────────────────────────
    MODEL_SAVE        = "model_save"
    INFER_OUTPUT      = "infer_output"
    API_OUTPUT        = "api_output"


# Metadata catalogue returned by GET /api/nodes/types
NODE_CATALOGUE: dict[str, dict[str, Any]] = {
    NodeType.TEXT_INPUT: {
        "label": "Text Input",
        "category": "input",
        "description": "Upload .txt / .pdf / .md / .docx / .html / .csv / .json files.",
        "accepts_files": True,
        "output_type": "text",
        "params": {},
    },
    NodeType.IMAGE_INPUT: {
        "label": "Image Input",
        "category": "input",
        "description": "Upload image files (.png, .jpg, .jpeg, .bmp, .tiff).",
        "accepts_files": True,
        "output_type": "image",
        "params": {},
    },
    NodeType.AUDIO_INPUT: {
        "label": "Audio Input",
        "category": "input",
        "description": "Upload audio files (.wav, .mp3, .flac, .ogg).",
        "accepts_files": True,
        "output_type": "audio",
        "params": {},
    },
    NodeType.SPREADSHEET_INPUT: {
        "label": "Spreadsheet Input",
        "category": "input",
        "description": "Upload .csv / .xlsx / .json tabular data files.",
        "accepts_files": True,
        "output_type": "tabular",
        "params": {},
    },
    NodeType.API_INPUT: {
        "label": "API Input",
        "category": "input",
        "description": "Pull data from an external HTTP/REST endpoint.",
        "accepts_files": False,
        "output_type": "any",
        "params": {
            "url":    {"type": "string",  "description": "Endpoint URL to GET."},
            "method": {"type": "string",  "description": "HTTP method.", "default": "GET"},
            "headers":{"type": "object",  "description": "Request headers.", "default": {}},
            "body":   {"type": "string",  "description": "Request body (POST/PUT).", "default": ""},
            "data_path": {"type": "string","description": "JSONPath into response (e.g. 'data.items').", "default": ""},
        },
    },
    NodeType.CHUNK: {
        "label": "Chunk",
        "category": "transform",
        "description": "Split text into training chunks. Method auto-selected by OpenAI if 'auto'.",
        "accepts_files": False,
        "output_type": "chunks",
        "params": {
            "method": {
                "type": "string",
                "description": "Chunking method (auto, sentence, paragraph, sliding_window, fixed_size, etc.).",
                "default": "auto",
            },
            "method_params": {"type": "object", "description": "Extra params for the chosen method.", "default": {}},
        },
    },
    NodeType.IMAGE_PREPROCESS: {
        "label": "Image Preprocess",
        "category": "transform",
        "description": "Resize, normalise and tensor-ify images for CNN training.",
        "accepts_files": False,
        "output_type": "image",
        "params": {
            "resize": {"type": "integer", "description": "Resize shorter side to N px.", "default": 224},
            "normalize": {"type": "boolean", "description": "Apply ImageNet normalisation.", "default": True},
        },
    },
    NodeType.AUDIO_PREPROCESS: {
        "label": "Audio Preprocess",
        "category": "transform",
        "description": "Convert audio to log-mel spectrograms for model input.",
        "accepts_files": False,
        "output_type": "audio",
        "params": {
            "sample_rate": {"type": "integer", "description": "Target sample rate.", "default": 16000},
            "n_mels": {"type": "integer", "description": "Number of mel bins.", "default": 80},
        },
    },
    NodeType.TABULAR_PREPROCESS: {
        "label": "Tabular Preprocess",
        "category": "transform",
        "description": "Encode categoricals, scale numerics, handle missing values.",
        "accepts_files": False,
        "output_type": "tabular",
        "params": {
            "target_column": {"type": "string", "description": "Name of the label column.", "default": ""},
            "scale_features": {"type": "boolean", "description": "Standard-scale numeric columns.", "default": True},
        },
    },
    NodeType.TEXT_MODEL: {
        "label": "Text Model",
        "category": "model",
        "description": "Fine-tune or embed with a sentence-transformer (MiniLM, BERT, etc.).",
        "accepts_files": False,
        "output_type": "model",
        "params": {
            "base_model":     {"type": "string",  "description": "Model folder name in /vol/models.", "default": "all-MiniLM-L6-v2"},
            "output_name":    {"type": "string",  "description": "Name for the saved fine-tuned model.", "default": "my-text-model"},
            "method":         {"type": "string",  "description": "Training method: simcse | mnrl | lora | sft.", "default": "simcse"},
            "epochs":         {"type": "integer", "description": "Training epochs.", "default": 3},
            "batch_size":     {"type": "integer", "description": "Batch size.", "default": 32},
            "learning_rate":  {"type": "number",  "description": "AdamW learning rate.", "default": 3e-5},
            "gpu":            {"type": "string",  "description": "GPU type: T4 | A10G | A100 | H100.", "default": "A10G"},
        },
    },
    NodeType.CNN_MODEL: {
        "label": "CNN Model",
        "category": "model",
        "description": "Train a CNN for image classification from scratch or via transfer learning.",
        "accepts_files": False,
        "output_type": "model",
        "params": {
            "base_model":     {"type": "string",  "description": "Pre-trained backbone (resnet18|resnet50|vgg16|none).", "default": "resnet18"},
            "output_name":    {"type": "string",  "description": "Name for the saved model.", "default": "my-cnn"},
            "num_classes":    {"type": "integer", "description": "Number of output classes.", "default": 2},
            "epochs":         {"type": "integer", "description": "Training epochs.", "default": 10},
            "batch_size":     {"type": "integer", "description": "Batch size.", "default": 32},
            "learning_rate":  {"type": "number",  "description": "Learning rate.", "default": 1e-3},
            "transfer":       {"type": "boolean", "description": "Freeze backbone for transfer learning.", "default": True},
            "gpu":            {"type": "string",  "description": "GPU type.", "default": "A10G"},
        },
    },
    NodeType.RNN_MODEL: {
        "label": "RNN / LSTM / GRU Model",
        "category": "model",
        "description": "Train an RNN, LSTM, or GRU on text sequences for classification.",
        "accepts_files": False,
        "output_type": "model",
        "params": {
            "rnn_type":       {"type": "string",  "description": "Architecture: rnn | lstm | gru.", "default": "lstm"},
            "output_name":    {"type": "string",  "description": "Name for the saved model.", "default": "my-rnn"},
            "vocab_size":     {"type": "integer", "description": "Vocabulary size.", "default": 10000},
            "embed_dim":      {"type": "integer", "description": "Embedding dimension.", "default": 128},
            "hidden_dim":     {"type": "integer", "description": "Hidden state size.", "default": 256},
            "num_layers":     {"type": "integer", "description": "Number of recurrent layers.", "default": 2},
            "num_classes":    {"type": "integer", "description": "Output classes.", "default": 2},
            "bidirectional":  {"type": "boolean", "description": "Use bidirectional RNN.", "default": True},
            "epochs":         {"type": "integer", "description": "Training epochs.", "default": 10},
            "batch_size":     {"type": "integer", "description": "Batch size.", "default": 64},
            "learning_rate":  {"type": "number",  "description": "Learning rate.", "default": 1e-3},
            "gpu":            {"type": "string",  "description": "GPU type.", "default": "T4"},
        },
    },
    NodeType.MODEL_SAVE: {
        "label": "Model Save",
        "category": "output",
        "description": "Save the trained model to Cloudflare R2 and the Modal Volume.",
        "accepts_files": False,
        "output_type": None,
        "params": {},
    },
    NodeType.INFER_OUTPUT: {
        "label": "Inference Output",
        "category": "output",
        "description": "Return raw predictions from the model node.",
        "accepts_files": False,
        "output_type": "infer_out",
        "params": {
            "top_k": {"type": "integer", "description": "Return top-K predictions per input.", "default": 5},
        },
    },
    NodeType.API_OUTPUT: {
        "label": "API Output",
        "category": "output",
        "description": "POST inference results to an external webhook URL.",
        "accepts_files": False,
        "output_type": None,
        "params": {
            "url":     {"type": "string", "description": "Webhook URL to POST results to."},
            "headers": {"type": "object", "description": "Extra request headers.", "default": {}},
        },
    },
}


# ── Data model ────────────────────────────────────────────────────────────────

class NodeSpec:
    """A single node in the pipeline graph."""
    def __init__(self, id: str, type: str, params: dict[str, Any]):
        self.id     = id
        self.type   = type
        self.params = params

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "type": self.type, "params": self.params}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "NodeSpec":
        return cls(id=d["id"], type=d["type"], params=d.get("params", {}))


class EdgeSpec:
    """A directed edge from one node output to another node input."""
    def __init__(self, from_id: str, to_id: str):
        self.from_id = from_id
        self.to_id   = to_id

    def to_dict(self) -> dict[str, str]:
        return {"from": self.from_id, "to": self.to_id}

    @classmethod
    def from_dict(cls, d: dict[str, str]) -> "EdgeSpec":
        return cls(from_id=d["from"], to_id=d["to"])


class PipelineSpec:
    """
    Complete pipeline specification.

    pipeline_type: "train" or "infer"
    nodes:         list of NodeSpec
    edges:         list of EdgeSpec
    """
    def __init__(
        self,
        pipeline_type: str,
        nodes: list[NodeSpec],
        edges: list[EdgeSpec],
    ):
        self.pipeline_type = pipeline_type
        self.nodes         = nodes
        self.edges         = edges

    def to_dict(self) -> dict[str, Any]:
        return {
            "pipeline_type": self.pipeline_type,
            "nodes":  [n.to_dict() for n in self.nodes],
            "edges":  [e.to_dict() for e in self.edges],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PipelineSpec":
        return cls(
            pipeline_type=d["pipeline_type"],
            nodes=[NodeSpec.from_dict(n) for n in d["nodes"]],
            edges=[EdgeSpec.from_dict(e) for e in d["edges"]],
        )


# ── Runtime data type ─────────────────────────────────────────────────────────
NodeOutput = dict[str, Any]
