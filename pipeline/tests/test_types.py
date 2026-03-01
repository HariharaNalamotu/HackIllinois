"""Tests for pipeline type system."""

import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from pipeline.types import NodeSpec, EdgeSpec, PipelineSpec, NodeType, NODE_CATALOGUE


class TestNodeType:
    def test_all_expected_types_exist(self):
        expected = [
            "text_input", "image_input", "audio_input", "spreadsheet_input",
            "chunk", "text_model", "cnn_model", "model_save", "infer_output",
            "object_detect_model", "audio_model", "audio_cnn", "image_cae",
            "tabular_model",
        ]
        for t in expected:
            assert t in [e.value for e in NodeType], f"Missing NodeType: {t}"

    def test_node_catalogue_covers_all_input_types(self):
        """Every input type must be in the catalogue."""
        for nt in [NodeType.TEXT_INPUT, NodeType.IMAGE_INPUT, NodeType.AUDIO_INPUT, NodeType.SPREADSHEET_INPUT]:
            assert nt in NODE_CATALOGUE or nt.value in NODE_CATALOGUE


class TestPipelineSpec:
    def test_from_dict_roundtrip(self):
        d = {
            "pipeline_type": "train",
            "nodes": [
                {"id": "n1", "type": "text_input", "params": {}},
                {"id": "n2", "type": "text_model", "params": {"base_model": "bge-small-en-v1.5"}},
                {"id": "n3", "type": "model_save", "params": {}},
            ],
            "edges": [
                {"from": "n1", "to": "n2"},
                {"from": "n2", "to": "n3"},
            ],
        }
        spec = PipelineSpec.from_dict(d)
        assert spec.pipeline_type == "train"
        assert len(spec.nodes) == 3
        assert len(spec.edges) == 2
        assert spec.nodes[0].id == "n1"
        assert spec.nodes[0].type == "text_input"
        assert spec.edges[0].from_id == "n1"
        assert spec.edges[0].to_id == "n2"

    def test_to_dict_preserves_data(self):
        spec = PipelineSpec(
            pipeline_type="infer",
            nodes=[NodeSpec("a", "text_input", {"key": "val"})],
            edges=[EdgeSpec("a", "b")],
        )
        d = spec.to_dict()
        assert d["pipeline_type"] == "infer"
        assert d["nodes"][0]["id"] == "a"
        assert d["nodes"][0]["params"]["key"] == "val"
        assert d["edges"][0]["from"] == "a"
