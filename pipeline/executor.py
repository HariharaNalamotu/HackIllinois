"""
Pipeline execution engine.

Runs a validated PipelineSpec by executing nodes in topological order,
passing NodeOutput data between them.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Callable

from .types import NodeSpec, PipelineSpec, NodeOutput
from .validator import validate_pipeline


def _topological_order(spec: PipelineSpec) -> list[NodeSpec]:
    """Return nodes sorted in topological execution order."""
    node_by_id = {n.id: n for n in spec.nodes}
    in_degree: dict[str, int] = defaultdict(int)
    children: dict[str, list[str]] = defaultdict(list)

    for edge in spec.edges:
        children[edge.from_id].append(edge.to_id)
        in_degree[edge.to_id] += 1
    for n in spec.nodes:
        in_degree.setdefault(n.id, 0)

    queue = deque([n.id for n in spec.nodes if in_degree[n.id] == 0])
    order: list[NodeSpec] = []
    while queue:
        nid = queue.popleft()
        order.append(node_by_id[nid])
        for child in children[nid]:
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(child)

    return order


def execute_pipeline(
    spec: PipelineSpec,
    files: dict[str, list[bytes]],       # node_id -> list of raw file bytes
    file_names: dict[str, list[str]],    # node_id -> list of file names
    job_id: str,
    log_fn: Callable[[str], None],
    models_dir: str,
    actian_url: str,
    r2_config: dict | None = None,
) -> dict[str, Any]:
    """
    Execute a validated PipelineSpec.

    Args:
        spec:        The pipeline to execute.
        files:       Dict mapping node_id → list of raw file bytes uploaded for that node.
        file_names:  Dict mapping node_id → list of original file names.
        job_id:      Unique job identifier (used for logging and namespacing).
        log_fn:      Callable(str) that persists log messages for SSE streaming.
        models_dir:  Absolute path to the models directory (e.g. /vol/models).
        actian_url:  Base URL of the Actian HTTP gateway.
        r2_config:   Optional R2 upload config dict (endpoint, key_id, secret, bucket).

    Returns:
        Dict with 'status', 'outputs' (terminal nodes), and 'all_outputs'.
    """
    validate_pipeline(spec)
    order = _topological_order(spec)

    node_outputs: dict[str, NodeOutput] = {}

    def get_inputs(node_id: str) -> list[NodeOutput]:
        return [
            node_outputs[edge.from_id]
            for edge in spec.edges
            if edge.to_id == node_id and edge.from_id in node_outputs
        ]

    # Import here so Modal containers don't need the package at import time
    from .nodes import create_node

    with tempfile.TemporaryDirectory() as workspace:
        ctx: dict[str, Any] = {
            "job_id":        job_id,
            "workspace":     workspace,
            "models_dir":    models_dir,
            "actian_url":    actian_url,
            "r2_config":     r2_config or {},
            "pipeline_type": spec.pipeline_type,
        }

        for node_spec in order:
            log_fn(f"[{node_spec.id}] type={node_spec.type} ...")

            # Save uploaded files for this node into the workspace
            node_files: list[str] = []
            if node_spec.id in files:
                node_dir = Path(workspace) / node_spec.id
                node_dir.mkdir(exist_ok=True)
                raw_bytes_list = files[node_spec.id]
                raw_names_list = file_names.get(
                    node_spec.id,
                    [f"file_{i}" for i in range(len(raw_bytes_list))],
                )
                for fname, fdata in zip(raw_names_list, raw_bytes_list):
                    fp = node_dir / Path(fname).name
                    fp.write_bytes(fdata)
                    node_files.append(str(fp))

            inputs = get_inputs(node_spec.id)

            node = create_node(node_spec)
            t0 = time.time()
            output = node.execute(inputs=inputs, files=node_files, ctx=ctx, log=log_fn)
            elapsed = time.time() - t0

            node_outputs[node_spec.id] = output
            log_fn(f"[{node_spec.id}] done ({elapsed:.1f}s) → type={output.get('type', '?')}")

    # Collect terminal outputs (nodes with no outgoing edges)
    has_outgoing = {e.from_id for e in spec.edges}
    terminals = {
        n.id: node_outputs[n.id]
        for n in spec.nodes
        if n.id not in has_outgoing and n.id in node_outputs
    }

    return {
        "status":      "complete",
        "outputs":     terminals,
        "all_outputs": {k: _serialisable(v) for k, v in node_outputs.items()},
    }


def _serialisable(obj: Any) -> Any:
    """Strip non-serialisable values (e.g. large byte arrays) before JSON storage."""
    if isinstance(obj, dict):
        return {k: _serialisable(v) for k, v in obj.items() if k != "_raw"}
    if isinstance(obj, list):
        return [_serialisable(x) for x in obj]
    return obj
