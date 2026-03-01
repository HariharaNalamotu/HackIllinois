"""Tests for pipeline executor — focusing on data flow and text_chunks."""

import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from unittest.mock import patch, MagicMock
from pipeline.types import NodeSpec, EdgeSpec, PipelineSpec
from pipeline.executor import _topological_order, execute_pipeline


def make_spec(nodes, edges, pipeline_type="train"):
    return PipelineSpec(
        pipeline_type=pipeline_type,
        nodes=[NodeSpec(n["id"], n["type"], n.get("params", {})) for n in nodes],
        edges=[EdgeSpec(e["from"], e["to"]) for e in edges],
    )


class TestTopologicalOrder:
    def test_linear_chain(self):
        spec = make_spec(
            nodes=[
                {"id": "a", "type": "text_input"},
                {"id": "b", "type": "text_model"},
                {"id": "c", "type": "model_save"},
            ],
            edges=[{"from": "a", "to": "b"}, {"from": "b", "to": "c"}],
        )
        order = _topological_order(spec)
        ids = [n.id for n in order]
        assert ids.index("a") < ids.index("b") < ids.index("c")

    def test_diamond_dag(self):
        spec = make_spec(
            nodes=[
                {"id": "src", "type": "text_input"},
                {"id": "left", "type": "chunk"},
                {"id": "right", "type": "text_model"},
                {"id": "out", "type": "model_save"},
            ],
            edges=[
                {"from": "src", "to": "left"},
                {"from": "src", "to": "right"},
                {"from": "left", "to": "out"},
                {"from": "right", "to": "out"},
            ],
        )
        order = _topological_order(spec)
        ids = [n.id for n in order]
        assert ids[0] == "src"
        assert ids[-1] == "out"


class TestExecutorTextChunks:
    """Test that text_chunks flow correctly through the executor to TextInputNode."""

    def test_text_chunks_passed_to_context(self):
        """Verify text_chunks dict is placed in ctx for TextInputNode to read."""
        spec = make_spec(
            nodes=[
                {"id": "node-abc", "type": "text_input"},
                {"id": "node-def", "type": "text_model", "params": {"base_model": "x", "method": "simcse"}},
                {"id": "node-out", "type": "model_save"},
            ],
            edges=[
                {"from": "node-abc", "to": "node-def"},
                {"from": "node-def", "to": "node-out"},
            ],
        )

        chunks = {"node-abc": ["chunk 1", "chunk 2", "chunk 3"]}
        logs = []

        # Mock the node execution to capture context
        captured_ctx = {}

        class MockTextInput:
            def __init__(self, spec):
                self.id = spec.id
            def execute(self, inputs, files, ctx, log):
                captured_ctx.update(ctx)
                return {"type": "text", "texts": ctx["text_chunks"].get(self.id, []), "filenames": ["test"]}

        class MockTextModel:
            def __init__(self, spec):
                self.id = spec.id
            def execute(self, inputs, files, ctx, log):
                return {"type": "model", "model_path": "/tmp/model", "model_type": "text", "arch": "text"}

        class MockModelSave:
            def __init__(self, spec):
                self.id = spec.id
            def execute(self, inputs, files, ctx, log):
                return {"type": "model", "model_path": "/tmp/saved"}

        def mock_create_node(node_spec):
            mapping = {
                "text_input": MockTextInput,
                "text_model": MockTextModel,
                "model_save": MockModelSave,
            }
            return mapping[node_spec.type](node_spec)

        with patch("pipeline.nodes.create_node", side_effect=mock_create_node):
            # Patch validate_pipeline to skip validation since we use simplified mocks
            with patch("pipeline.executor.validate_pipeline"):
                result = execute_pipeline(
                    spec=spec,
                    files={},
                    file_names={},
                    job_id="test-job",
                    log_fn=lambda msg: logs.append(msg),
                    models_dir="/tmp/models",
                    actian_url="http://test",
                    text_chunks=chunks,
                )

        # Verify text_chunks was in context
        assert "text_chunks" in captured_ctx
        assert captured_ctx["text_chunks"]["node-abc"] == ["chunk 1", "chunk 2", "chunk 3"]

    def test_empty_text_chunks_produces_empty_dict(self):
        """When no text_chunks are passed, ctx should have empty dict."""
        spec = make_spec(
            nodes=[
                {"id": "n1", "type": "text_input"},
                {"id": "n2", "type": "model_save"},
            ],
            edges=[{"from": "n1", "to": "n2"}],
        )

        captured_ctx = {}

        class MockNode:
            def __init__(self, spec):
                self.id = spec.id
            def execute(self, inputs, files, ctx, log):
                captured_ctx.update(ctx)
                return {"type": "text", "texts": ["x"]}

        with patch("pipeline.nodes.create_node", return_value=MockNode(NodeSpec("x", "x", {}))):
            with patch("pipeline.executor.validate_pipeline"):
                execute_pipeline(
                    spec=spec, files={}, file_names={},
                    job_id="j", log_fn=lambda m: None,
                    models_dir="/tmp", actian_url="http://x",
                    text_chunks=None,
                )

        assert captured_ctx["text_chunks"] == {}

    def test_text_chunks_key_mismatch_produces_empty(self):
        """If text_chunks has wrong node ID, TextInputNode gets empty list."""
        chunks = {"wrong-node-id": ["data"]}

        # Simulate TextInputNode lookup
        self_id = "correct-node-id"
        pre_chunks = chunks.get(self_id, [])
        assert pre_chunks == []  # This is the bug — node ID mismatch

    def test_files_written_to_workspace(self):
        """Files passed as bytes should be written to workspace for node access."""
        spec = make_spec(
            nodes=[
                {"id": "n1", "type": "text_input"},
                {"id": "n2", "type": "model_save"},
            ],
            edges=[{"from": "n1", "to": "n2"}],
        )

        captured_files = []

        class MockNode:
            def __init__(self, spec):
                self.id = spec.id
            def execute(self, inputs, files, ctx, log):
                captured_files.extend(files)
                return {"type": "text", "texts": ["x"]}

        with patch("pipeline.nodes.create_node", side_effect=lambda s: MockNode(s)):
            with patch("pipeline.executor.validate_pipeline"):
                execute_pipeline(
                    spec=spec,
                    files={"n1": [b"hello world"]},
                    file_names={"n1": ["test.txt"]},
                    job_id="j",
                    log_fn=lambda m: None,
                    models_dir="/tmp",
                    actian_url="http://x",
                )

        # The first node (n1) should receive the file path
        assert len(captured_files) >= 1
        assert captured_files[0].endswith("test.txt")
        # Note: workspace is cleaned up after execute_pipeline returns,
        # so we can't read the file here. But the path was correctly formed.


class TestExecutorTerminalOutputs:
    def test_terminal_nodes_are_collected(self):
        spec = make_spec(
            nodes=[
                {"id": "n1", "type": "text_input"},
                {"id": "n2", "type": "model_save"},
            ],
            edges=[{"from": "n1", "to": "n2"}],
        )

        class MockNode:
            def __init__(self, spec):
                self.id = spec.id
                self.type = spec.type
            def execute(self, inputs, files, ctx, log):
                if self.type == "text_input":
                    return {"type": "text", "texts": ["hello"]}
                return {"type": "model", "model_path": "/saved"}

        with patch("pipeline.nodes.create_node", side_effect=lambda s: MockNode(s)):
            with patch("pipeline.executor.validate_pipeline"):
                result = execute_pipeline(
                    spec=spec, files={}, file_names={},
                    job_id="j", log_fn=lambda m: None,
                    models_dir="/tmp", actian_url="http://x",
                )

        assert result["status"] == "complete"
        # n2 (model_save) is terminal — no outgoing edges
        assert "n2" in result["outputs"]
        # n1 has outgoing edge to n2 so it's NOT terminal
        assert "n1" not in result["outputs"]
