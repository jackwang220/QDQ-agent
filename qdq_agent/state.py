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
    excel_unknown_nodes: list[dict]

    # ── Detect layer ─────────────────────────────────────────────────────────
    detect_layer: Optional[int]
    model_txt_content: str

    # ── Quantizer mode (inferred from model.txt, confirmed by HITL) ─────────────
    quantizer_mode: Optional[str]

    # ── Layer structure (inferred from model.txt) ─────────────────────────────
    layer_constants: dict  # {concat:[], maxpool:[], upsample:[], sppcspc:int|None, repconv:[]}

    # ── Agent suggestions (pre-HITL) ─────────────────────────────────────────
    suggested_detect_layer: Optional[int]
    suggested_patterns: list[dict]
    suggested_extra_qdq: list[dict]   # [{name, op_type, layer, fl, reason}]

    # ── Auto-generated postprocess config ────────────────────────────────────
    postprocess_config: dict          # {transpose_configs, outputs_to_remove, nodes_to_remove, extra_qdq_nodes}

    # ── Confirmed input bias (set by HITL, overrides config) ─────────────────
    input_bias: Optional[float]

    # ── Final ────────────────────────────────────────────────────────────────
    success: bool
