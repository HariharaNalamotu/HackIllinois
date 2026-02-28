"""
Node registry — maps NodeType strings to their implementation classes.
"""

from __future__ import annotations

from ..types import NodeSpec, NodeType
from .base import BaseNode
from .input_nodes import (
    TextInputNode,
    ImageInputNode,
    AudioInputNode,
    SpreadsheetInputNode,
    APIInputNode,
)
from .transform_nodes import (
    ChunkNode,
    ImagePreprocessNode,
    AudioPreprocessNode,
    TabularPreprocessNode,
)
from .model_nodes import (
    TextModelNode,
    CNNModelNode,
    RNNModelNode,
)
from .output_nodes import (
    ModelSaveNode,
    InferOutputNode,
    APIOutputNode,
)

_REGISTRY: dict[str, type[BaseNode]] = {
    NodeType.TEXT_INPUT:          TextInputNode,
    NodeType.IMAGE_INPUT:         ImageInputNode,
    NodeType.AUDIO_INPUT:         AudioInputNode,
    NodeType.SPREADSHEET_INPUT:   SpreadsheetInputNode,
    NodeType.API_INPUT:           APIInputNode,

    NodeType.CHUNK:               ChunkNode,
    NodeType.IMAGE_PREPROCESS:    ImagePreprocessNode,
    NodeType.AUDIO_PREPROCESS:    AudioPreprocessNode,
    NodeType.TABULAR_PREPROCESS:  TabularPreprocessNode,

    NodeType.TEXT_MODEL:          TextModelNode,
    NodeType.CNN_MODEL:           CNNModelNode,
    NodeType.RNN_MODEL:           RNNModelNode,

    NodeType.MODEL_SAVE:          ModelSaveNode,
    NodeType.INFER_OUTPUT:        InferOutputNode,
    NodeType.API_OUTPUT:          APIOutputNode,
}


def create_node(spec: NodeSpec) -> BaseNode:
    """Instantiate the correct node class for the given NodeSpec."""
    cls = _REGISTRY.get(spec.type)
    if cls is None:
        raise ValueError(f"No implementation found for node type '{spec.type}'.")
    return cls(spec)


__all__ = ["create_node", "BaseNode", "_REGISTRY"]
