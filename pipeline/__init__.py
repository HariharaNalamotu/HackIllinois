"""
Node-based AI training and inference pipeline package.

Defines the data model, validation, and execution engine for
drag-and-drop pipelines that run on Modal GPU/CPU.
"""
from .types import NodeSpec, EdgeSpec, PipelineSpec, NodeOutput, NodeType
from .validator import validate_pipeline
from .executor import execute_pipeline

__all__ = [
    "NodeSpec", "EdgeSpec", "PipelineSpec", "NodeOutput", "NodeType",
    "validate_pipeline", "execute_pipeline",
]
