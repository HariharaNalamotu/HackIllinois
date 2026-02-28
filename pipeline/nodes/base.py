"""Base class for all pipeline nodes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable

from ..types import NodeSpec, NodeOutput


class BaseNode(ABC):
    def __init__(self, spec: NodeSpec):
        self.spec   = spec
        self.id     = spec.id
        self.params = spec.params

    def p(self, key: str, default: Any = None) -> Any:
        """Helper to fetch a param with a fallback default."""
        return self.params.get(key, default)

    @abstractmethod
    def execute(
        self,
        inputs: list[NodeOutput],
        files:  list[str],
        ctx:    dict[str, Any],
        log:    Callable[[str], None],
    ) -> NodeOutput:
        """
        Execute this node.

        Args:
            inputs:  List of NodeOutput dicts from upstream nodes.
            files:   List of local file paths uploaded for this node.
            ctx:     Execution context (job_id, workspace, models_dir, etc.).
            log:     Logging callback.

        Returns:
            A NodeOutput dict.
        """
        ...

    # ── Helpers ───────────────────────────────────────────────────────────────
    @staticmethod
    def first_of_type(inputs: list[NodeOutput], *types: str) -> NodeOutput | None:
        """Return the first input whose 'type' matches any of the given types."""
        for inp in inputs:
            if inp.get("type") in types:
                return inp
        return None
