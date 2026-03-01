"""Tests for pipeline validation."""

import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from pipeline.types import NodeSpec, EdgeSpec, PipelineSpec
from pipeline.validator import validate_pipeline


def make_spec(nodes, edges, pipeline_type="train"):
    return PipelineSpec(
        pipeline_type=pipeline_type,
        nodes=[NodeSpec(n["id"], n["type"], n.get("params", {})) for n in nodes],
        edges=[EdgeSpec(e["from"], e["to"]) for e in edges],
    )


class TestValidation:
    def test_valid_text_pipeline(self):
        """A valid text pipeline should pass validation."""
        spec = make_spec(
            nodes=[
                {"id": "n1", "type": "text_input"},
                {"id": "n2", "type": "text_model", "params": {"base_model": "x"}},
                {"id": "n3", "type": "model_save"},
            ],
            edges=[{"from": "n1", "to": "n2"}, {"from": "n2", "to": "n3"}],
        )
        validate_pipeline(spec)  # should not raise

    def test_unknown_node_type_raises(self):
        spec = make_spec(
            nodes=[
                {"id": "n1", "type": "text_input"},
                {"id": "n2", "type": "magic_node"},  # invalid
                {"id": "n3", "type": "model_save"},
            ],
            edges=[{"from": "n1", "to": "n2"}, {"from": "n2", "to": "n3"}],
        )
        with pytest.raises(ValueError, match="unknown type 'magic_node'"):
            validate_pipeline(spec)

    def test_missing_input_node_raises(self):
        spec = make_spec(
            nodes=[{"id": "n1", "type": "model_save"}],
            edges=[],
        )
        with pytest.raises(ValueError, match="at least one input node"):
            validate_pipeline(spec)

    def test_missing_output_node_raises(self):
        spec = make_spec(
            nodes=[{"id": "n1", "type": "text_input"}],
            edges=[],
        )
        with pytest.raises(ValueError, match="at least one output node"):
            validate_pipeline(spec)

    def test_cycle_raises(self):
        spec = make_spec(
            nodes=[
                {"id": "n1", "type": "text_input"},
                {"id": "n2", "type": "text_model"},
                {"id": "n3", "type": "model_save"},
            ],
            edges=[
                {"from": "n1", "to": "n2"},
                {"from": "n2", "to": "n3"},
                {"from": "n3", "to": "n1"},  # cycle!
            ],
        )
        with pytest.raises(ValueError, match="cycle"):
            validate_pipeline(spec)

    def test_edge_references_unknown_node_raises(self):
        spec = make_spec(
            nodes=[
                {"id": "n1", "type": "text_input"},
                {"id": "n2", "type": "model_save"},
            ],
            edges=[{"from": "n1", "to": "ghost"}],
        )
        with pytest.raises(ValueError, match="unknown node id 'ghost'"):
            validate_pipeline(spec)

    def test_invalid_pipeline_type_raises(self):
        spec = make_spec(
            nodes=[
                {"id": "n1", "type": "text_input"},
                {"id": "n2", "type": "model_save"},
            ],
            edges=[],
            pipeline_type="deploy",
        )
        with pytest.raises(ValueError, match="pipeline_type must be"):
            validate_pipeline(spec)

    def test_type_incompatibility_raises(self):
        """Connecting text_input directly to cnn_model should fail (text → image mismatch)."""
        spec = make_spec(
            nodes=[
                {"id": "n1", "type": "text_input"},
                {"id": "n2", "type": "cnn_model"},
                {"id": "n3", "type": "model_save"},
            ],
            edges=[{"from": "n1", "to": "n2"}, {"from": "n2", "to": "n3"}],
        )
        with pytest.raises(ValueError, match="outputs 'text'.*expects one of"):
            validate_pipeline(spec)

    def test_valid_image_pipeline(self):
        spec = make_spec(
            nodes=[
                {"id": "n1", "type": "image_input"},
                {"id": "n2", "type": "cnn_model"},
                {"id": "n3", "type": "model_save"},
            ],
            edges=[{"from": "n1", "to": "n2"}, {"from": "n2", "to": "n3"}],
        )
        validate_pipeline(spec)  # should not raise
