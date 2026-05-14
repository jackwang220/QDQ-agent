from __future__ import annotations

from operator import add
from typing import Annotated, Optional
from typing_extensions import TypedDict


class QDQState(TypedDict):
    # ── Input ────────────────────────────────────────────────────────────────
    config_path: str
    config: dict

    # ── Pipeline tracking ────────────────────────────────────────────────────
    current_stage: str
    errors: Annotated[list[str], add]

    # ── Resolved paths (filled by load_config node) ──────────────────────────
    model_txt_path: str
    excel_path: str
    raw_onnx_path: str
    int8_onnx_path: str
    qdq_onnx_path: str
    implicit_onnx_path: str
    final_onnx_path: str

    # ── Step 1 output ────────────────────────────────────────────────────────
    excel_unknown_nodes: list[dict]   # [{layer, component, role_type, raw_line}]

    # ── Detect layer ─────────────────────────────────────────────────────────
    detect_layer: Optional[int]       # None = not yet determined
    model_txt_content: str

    # ── Agent suggestions (pre-HITL) ─────────────────────────────────────────
    suggested_detect_layer: Optional[int]
    suggested_patterns: list[dict]    # [{node_name, suggested_pattern, reason}]

    # ── Final ────────────────────────────────────────────────────────────────
    success: bool
