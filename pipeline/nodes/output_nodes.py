"""
Output nodes — terminal nodes that persist results or expose APIs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from .base import BaseNode
from ..types import NodeOutput


class ModelSaveNode(BaseNode):
    """
    Saves a trained model to R2 and records metadata.
    This is typically a no-op because model nodes already upload to R2.
    It serves as a semantic terminal marker in the pipeline graph.

    Input:  {"type": "model", "model_path": str, ...}
    Output: {"type": "saved", "model_path": str, "output_name": str}
    """

    def execute(
        self,
        inputs: list[NodeOutput],
        files: list[str],
        ctx: dict[str, Any],
        log: Callable[[str], None],
    ) -> NodeOutput:
        inp = self.first_of_type(inputs, "model")
        if inp is None:
            raise ValueError("ModelSaveNode: requires a model input.")

        model_path  = inp.get("model_path", "")
        user_name   = self.p("model_name", None)
        output_name = user_name or inp.get("output_name", Path(model_path).name if model_path else "unknown")
        log(f"  [ModelSave] model '{output_name}' at {model_path}")

        return {
            "type":        "saved",
            "model_path":  model_path,
            "output_name": output_name,
            "arch":        inp.get("arch", "unknown"),
            "collection":  inp.get("collection", ""),
        }


class InferOutputNode(BaseNode):
    """
    Returns inference predictions as the pipeline result.

    Params: top_k (default 5)

    Input:  {"type": "infer_out", "predictions": list}
    Output: {"type": "infer_out", "predictions": list (top-k trimmed)}
    """

    def execute(
        self,
        inputs: list[NodeOutput],
        files: list[str],
        ctx: dict[str, Any],
        log: Callable[[str], None],
    ) -> NodeOutput:
        if not inputs:
            raise ValueError("InferOutputNode: no input received.")

        # Accept infer_out with predictions (from model nodes)
        inp = self.first_of_type(inputs, "infer_out")
        if inp is not None:
            top_k       = int(self.p("top_k", 5))
            predictions = inp.get("predictions", [])
            trimmed = []
            for pred in predictions:
                if isinstance(pred, dict) and "results" in pred:
                    pred = {**pred, "results": pred["results"][:top_k]}
                trimmed.append(pred)
            log(f"  [InferOutput] {len(trimmed)} predictions returned")
            return {
                "type":        "infer_out",
                "predictions": trimmed,
                "total":       len(trimmed),
            }

        # Accept any other input type — pass through as result
        inp = inputs[0]
        log(f"  [InferOutput] pass-through input type='{inp.get('type')}'")
        return {
            "type":        "infer_out",
            "result":      inp,
        }


class APIOutputNode(BaseNode):
    """
    POSTs inference results to a webhook endpoint.

    Params: url (required), headers ({})

    Input:  {"type": "infer_out", ...}
    Output: {"type": "api_posted", "status_code": int, "url": str}
    """

    def execute(
        self,
        inputs: list[NodeOutput],
        files: list[str],
        ctx: dict[str, Any],
        log: Callable[[str], None],
    ) -> NodeOutput:
        import httpx

        inp = self.first_of_type(inputs, "infer_out", "model", "saved")
        if inp is None:
            raise ValueError("APIOutputNode: requires an infer_out or model input.")

        url     = self.p("url", "")
        headers = self.p("headers", {})

        if not url:
            raise ValueError("APIOutputNode: 'url' param is required.")

        payload = {
            "job_id":  ctx.get("job_id"),
            "results": inp,
        }

        log(f"  [APIOutput] posting to {url}")
        resp = httpx.post(url, json=payload, headers=headers, timeout=30)
        log(f"  [APIOutput] response: {resp.status_code}")

        return {
            "type":        "api_posted",
            "url":         url,
            "status_code": resp.status_code,
        }
