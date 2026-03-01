"""
Pipeline validation.

Checks structural correctness of a PipelineSpec before execution:
  - Node types are valid
  - The graph is a valid DAG (no cycles)
  - Exactly one source and one terminal node
  - Edge data-type compatibility
"""

from __future__ import annotations

from collections import defaultdict, deque

from .types import NodeSpec, PipelineSpec, NodeType, NODE_CATALOGUE


# Which output types each node type accepts as input
_INPUT_ACCEPTS: dict[str, set[str]] = {
    NodeType.CHUNK:             {"text"},
    NodeType.IMAGE_PREPROCESS:  {"image"},
    NodeType.AUDIO_PREPROCESS:  {"audio"},
    NodeType.TABULAR_PREPROCESS: {"tabular"},
    NodeType.TEXT_MODEL:        {"chunks", "text"},
    NodeType.CNN_MODEL:         {"image"},
    NodeType.RNN_MODEL:         {"chunks", "text"},
    NodeType.OBJECT_DETECT_MODEL: {"image"},
    NodeType.AUDIO_MODEL:       {"audio"},
    NodeType.AUDIO_CNN:         {"audio"},
    NodeType.IMAGE_CAE:         {"image"},
    NodeType.TABULAR_MODEL:     {"tabular"},
    NodeType.MODEL_SAVE:        {"model"},
    NodeType.INFER_OUTPUT:      {"infer_out", "text", "image", "audio", "tabular", "chunks", "model"},
    NodeType.API_OUTPUT:        {"model", "infer_out"},
}

# Input nodes (no incoming edges needed)
_INPUT_NODE_TYPES = {
    NodeType.TEXT_INPUT,
    NodeType.IMAGE_INPUT,
    NodeType.AUDIO_INPUT,
    NodeType.SPREADSHEET_INPUT,
    NodeType.API_INPUT,
}


def validate_pipeline(spec: PipelineSpec) -> None:
    """
    Raise ValueError with a descriptive message if the pipeline is invalid.
    """
    node_ids  = {n.id for n in spec.nodes}
    node_by_id: dict[str, NodeSpec] = {n.id: n for n in spec.nodes}

    # ── 1. All node types must be known ───────────────────────────────────────
    valid_types = {t.value for t in NodeType}
    for node in spec.nodes:
        if node.type not in valid_types:
            raise ValueError(
                f"Node '{node.id}' has unknown type '{node.type}'. "
                f"Valid types: {sorted(valid_types)}"
            )

    # ── 2. All edge endpoints must exist ──────────────────────────────────────
    for edge in spec.edges:
        for eid in (edge.from_id, edge.to_id):
            if eid not in node_ids:
                raise ValueError(f"Edge references unknown node id '{eid}'.")

    # ── 3. Build adjacency for cycle check (Kahn's algorithm) ─────────────────
    in_degree: dict[str, int] = defaultdict(int)
    children:  dict[str, list[str]] = defaultdict(list)
    for edge in spec.edges:
        children[edge.from_id].append(edge.to_id)
        in_degree[edge.to_id] += 1
    for nid in node_ids:
        in_degree.setdefault(nid, 0)

    queue = deque([nid for nid, deg in in_degree.items() if deg == 0])
    visited = 0
    while queue:
        nid = queue.popleft()
        visited += 1
        for child in children[nid]:
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(child)

    if visited != len(node_ids):
        raise ValueError("Pipeline graph contains a cycle.")

    # ── 4. Must have at least one input node and one output node ──────────────
    has_input  = any(n.type in _INPUT_NODE_TYPES for n in spec.nodes)
    has_output = any(
        n.type in {NodeType.MODEL_SAVE, NodeType.INFER_OUTPUT, NodeType.API_OUTPUT}
        for n in spec.nodes
    )
    if not has_input:
        raise ValueError("Pipeline must contain at least one input node.")
    if not has_output:
        raise ValueError("Pipeline must contain at least one output node.")

    # ── 5. Data-type compatibility along edges ────────────────────────────────
    def _out_type(node: NodeSpec) -> str:
        cat = NODE_CATALOGUE.get(node.type, {})
        return cat.get("output_type", "any") or "any"

    for edge in spec.edges:
        src_type = _out_type(node_by_id[edge.from_id])
        dst_accepts = _INPUT_ACCEPTS.get(node_by_id[edge.to_id].type)
        if dst_accepts is None:
            continue  # output nodes accept anything
        if src_type != "any" and src_type not in dst_accepts:
            raise ValueError(
                f"Edge {edge.from_id} → {edge.to_id}: "
                f"node '{edge.from_id}' outputs '{src_type}' but "
                f"'{edge.to_id}' (type={node_by_id[edge.to_id].type}) "
                f"expects one of {sorted(dst_accepts)}."
            )

    # ── 6. Pipeline type specific rules ───────────────────────────────────────
    if spec.pipeline_type not in ("train", "infer"):
        raise ValueError(
            f"pipeline_type must be 'train' or 'infer', got '{spec.pipeline_type}'."
        )
