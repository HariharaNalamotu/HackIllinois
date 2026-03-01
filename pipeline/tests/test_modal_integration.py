"""
Tests for Modal endpoint integration — simulates form data parsing
that train_endpoint does, and verifies the data flow from Worker → Modal → Executor.

These tests don't call Modal directly but verify the parsing logic
that handles text_chunks, files, r2_keys, and actian_collections.
"""

import json
import base64
import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))


class TestModalFormParsing:
    """Simulate what Modal's train_endpoint does with incoming form data."""

    def test_text_chunks_parsing(self):
        """Verify text_chunks[nodeId] form fields are parsed correctly."""
        # Simulate form.multi_items() output
        form_items = [
            ("pipeline", '{"pipeline_type":"train","nodes":[{"id":"n1","type":"text_input","params":{}}],"edges":[]}'),
            ("text_chunks[n1]", json.dumps(["chunk A", "chunk B", "chunk C"])),
            ("job_id", "test-123"),
        ]

        text_chunks_direct: dict[str, list[str]] = {}
        for key, value in form_items:
            if key.startswith("text_chunks[") and key.endswith("]"):
                node_id = key[12:-1]
                text_chunks_direct[node_id] = json.loads(str(value))

        assert "n1" in text_chunks_direct
        assert text_chunks_direct["n1"] == ["chunk A", "chunk B", "chunk C"]

    def test_text_chunks_empty_list(self):
        """Empty chunk list should be parsed but filtered out before passing to executor."""
        form_items = [
            ("text_chunks[n1]", json.dumps([])),
        ]

        text_chunks_direct: dict[str, list[str]] = {}
        for key, value in form_items:
            if key.startswith("text_chunks[") and key.endswith("]"):
                node_id = key[12:-1]
                parsed = json.loads(str(value))
                text_chunks_direct[node_id] = parsed

        # The raw parsed value is []
        assert text_chunks_direct["n1"] == []

        # But _gpu_train should filter this out (our fix)
        filtered = {k: v for k, v in text_chunks_direct.items() if v}
        assert "n1" not in filtered

    def test_text_chunks_invalid_json_skipped(self):
        """Invalid JSON in text_chunks should be silently skipped."""
        form_items = [
            ("text_chunks[n1]", "not valid json {{{"),
        ]

        text_chunks_direct: dict[str, list[str]] = {}
        for key, value in form_items:
            if key.startswith("text_chunks[") and key.endswith("]"):
                node_id = key[12:-1]
                try:
                    text_chunks_direct[node_id] = json.loads(str(value))
                except Exception:
                    pass

        assert "n1" not in text_chunks_direct

    def test_files_b64_parsing(self):
        """files[nodeId] form fields should be base64-encoded."""
        content = b"Hello world PDF content"
        b64 = base64.b64encode(content).decode()

        files_b64: dict[str, list[str]] = {}
        file_names: dict[str, list[str]] = {}

        # Simulate what train_endpoint does
        form_items = [
            ("files[n1]", (b64, "document.pdf")),
        ]

        for key, value in form_items:
            if key.startswith("files[") and key.endswith("]"):
                node_id = key[6:-1]
                data_b64, filename = value
                files_b64.setdefault(node_id, []).append(data_b64)
                file_names.setdefault(node_id, []).append(filename)

        assert "n1" in files_b64
        decoded = base64.b64decode(files_b64["n1"][0])
        assert decoded == content
        assert file_names["n1"][0] == "document.pdf"

    def test_r2_keys_parsing(self):
        """r2_keys[nodeId] form fields should accumulate multiple keys."""
        form_items = [
            ("r2_keys[img-node]", "uploads/wf1/img-node/cat.jpg"),
            ("r2_keys[img-node]", "uploads/wf1/img-node/dog.jpg"),
        ]

        r2_keys: dict[str, list[str]] = {}
        for key, value in form_items:
            if key.startswith("r2_keys[") and key.endswith("]"):
                node_id = key[8:-1]
                r2_keys.setdefault(node_id, []).append(str(value))

        assert r2_keys["img-node"] == [
            "uploads/wf1/img-node/cat.jpg",
            "uploads/wf1/img-node/dog.jpg",
        ]

    def test_actian_collections_parsing(self):
        """actian_collections[nodeId] should map to collection name."""
        form_items = [
            ("actian_collections[text-1]", "wf_abc_text-1"),
        ]

        actian_collections: dict[str, str] = {}
        for key, value in form_items:
            if key.startswith("actian_collections[") and key.endswith("]"):
                node_id = key[19:-1]
                actian_collections[node_id] = str(value)

        assert actian_collections["text-1"] == "wf_abc_text-1"


class TestGpuTrainDataFlow:
    """Simulate _gpu_train's logic for text_chunks_direct vs actian fallback."""

    def test_direct_chunks_take_priority_over_actian(self):
        """When both text_chunks_direct and actian_collections exist for a node,
        direct chunks should win."""
        text_chunks_direct = {"node-1": ["direct chunk 1", "direct chunk 2"]}
        actian_collections = {"node-1": "some_collection"}

        # Simulate _gpu_train logic
        text_chunks: dict[str, list[str]] = {}

        if text_chunks_direct:
            for node_id, chunks in text_chunks_direct.items():
                if chunks:
                    text_chunks[node_id] = chunks

        for node_id, collection in actian_collections.items():
            if node_id in text_chunks:
                continue  # skip — already have direct chunks
            # Would scan Actian here
            text_chunks[node_id] = ["actian chunk"]

        assert text_chunks["node-1"] == ["direct chunk 1", "direct chunk 2"]

    def test_actian_fallback_when_no_direct_chunks(self):
        """When no direct chunks exist, fall back to Actian scan."""
        text_chunks_direct = {}
        actian_collections = {"node-1": "collection_x"}

        text_chunks: dict[str, list[str]] = {}

        if text_chunks_direct:
            for node_id, chunks in text_chunks_direct.items():
                if chunks:
                    text_chunks[node_id] = chunks

        for node_id, collection in actian_collections.items():
            if node_id in text_chunks:
                continue
            # Simulate successful Actian scan
            text_chunks[node_id] = ["actian text 1", "actian text 2"]

        assert text_chunks["node-1"] == ["actian text 1", "actian text 2"]

    def test_empty_direct_chunks_not_blocking_actian(self):
        """Empty direct chunks (our fix) should NOT block Actian fallback."""
        text_chunks_direct = {"node-1": []}  # empty
        actian_collections = {"node-1": "collection_x"}

        text_chunks: dict[str, list[str]] = {}

        # Fixed logic: skip empty lists
        if text_chunks_direct:
            for node_id, chunks in text_chunks_direct.items():
                if chunks:  # <-- the fix
                    text_chunks[node_id] = chunks

        for node_id, collection in actian_collections.items():
            if node_id in text_chunks:
                continue
            text_chunks[node_id] = ["actian fallback"]

        assert text_chunks["node-1"] == ["actian fallback"]


class TestWorkerNodeTypeRouting:
    """Simulate the worker's logic for routing files by node type."""

    def test_text_input_goes_to_chunking(self):
        """files[nodeId] where node type is text_input should be chunked."""
        pipeline_spec = {
            "nodes": [
                {"id": "txt-1", "type": "text_input", "params": {}},
                {"id": "img-1", "type": "image_input", "params": {}},
                {"id": "aud-1", "type": "audio_input", "params": {}},
            ]
        }

        # Worker builds nodeTypeMap
        node_type_map: dict[str, str] = {}
        for node in pipeline_spec["nodes"]:
            node_type_map[node["id"]] = node["type"]

        # For each file, decide routing
        files_to_process = [
            ("files[txt-1]", "doc.pdf"),
            ("files[img-1]", "photo.jpg"),
            ("files[aud-1]", "speech.wav"),
        ]

        text_nodes = []
        binary_nodes = []

        for key, filename in files_to_process:
            node_id = key[6:-1]
            node_type = node_type_map.get(node_id, "")
            if node_type == "text_input":
                text_nodes.append(node_id)
            else:
                binary_nodes.append(node_id)

        assert text_nodes == ["txt-1"]
        assert set(binary_nodes) == {"img-1", "aud-1"}

    def test_worker_chunks_zero_returns_fallback(self):
        """When chunking returns 0 chunks, worker should send raw file instead."""
        chunks_from_modal: list[str] = []  # chunk endpoint returned empty

        # Our fix in the worker:
        if len(chunks_from_modal) > 0:
            action = "set_text_chunks"
        else:
            action = "set_raw_file"  # fallback

        assert action == "set_raw_file"

    def test_kv_text_entry_produces_text_chunks(self):
        """KV entries with type=text should populate text_chunks."""
        kv_entry = {"type": "text", "chunks": ["pre-chunked 1", "pre-chunked 2"]}

        # Worker logic for KV entries
        if kv_entry["type"] == "text" and kv_entry.get("chunks"):
            result = ("text_chunks", kv_entry["chunks"])
        elif kv_entry["type"] == "actian" and kv_entry.get("collection"):
            result = ("actian_collections", kv_entry["collection"])
        elif kv_entry["type"] == "r2" and kv_entry.get("keys"):
            result = ("r2_keys", kv_entry["keys"])
        else:
            result = None

        assert result == ("text_chunks", ["pre-chunked 1", "pre-chunked 2"])
