from __future__ import annotations

from langgraph.graph import StateGraph, END

from qdq_agent.llm import trace_node
from qdq_agent.state import QDQState
from qdq_agent.nodes.pipeline import (
    load_config_node,
    run_step1_node,
    run_step2_node,
    run_step3_node,
    run_step4_node,
    run_step5_node,
    run_step6_node,
)
from qdq_agent.nodes.agent import (
    load_model_txt_node,
    infer_detect_layer_node,
    review_detect_layer_node,
    suggest_unknown_fix_node,
    review_unknown_fix_node,
)


# ── Routing functions ─────────────────────────────────────────────────────────

def _route_after_step1(state: QDQState) -> str:
    if state.get("errors"):
        return "error"
    if state.get("excel_unknown_nodes"):
        return "suggest_unknown_fix"
    return "check_detect_layer"


def _route_after_unknown_review(state: QDQState) -> str:
    stage = state.get("current_stage", "")
    if stage == "unknown_fix_skipped":
        return "check_detect_layer"
    # Human applied fix — re-run step 1
    return "run_step1"


def _route_check_detect_layer(state: QDQState) -> str:
    if state.get("detect_layer") is None:
        return "load_model_txt"
    return "run_step2"


def _route_after_step(state: QDQState) -> str:
    if state.get("errors"):
        return "error"
    return "continue"


def _error_node(state: QDQState) -> dict:
    errors = state.get("errors", [])
    for e in errors:
        print(f"  [ERROR] {e}")
    return {"current_stage": "failed", "success": False}


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    g = StateGraph(QDQState)

    def add(name: str, fn):
        g.add_node(name, trace_node(name)(fn))

    # Pipeline nodes
    add("load_config", load_config_node)
    add("run_step1", run_step1_node)
    add("run_step2", run_step2_node)
    add("run_step3", run_step3_node)
    add("run_step4", run_step4_node)
    add("run_step5", run_step5_node)
    add("run_step6", run_step6_node)

    # Agent nodes
    add("load_model_txt", load_model_txt_node)
    add("infer_detect_layer", infer_detect_layer_node)
    add("review_detect_layer", review_detect_layer_node)   # HITL
    add("suggest_unknown_fix", suggest_unknown_fix_node)
    add("review_unknown_fix", review_unknown_fix_node)     # HITL

    # Terminal
    add("error", _error_node)

    # ── Edges ──────────────────────────────────────────────────────────────
    g.set_entry_point("load_config")
    g.add_edge("load_config", "run_step1")

    # After step 1: check for unknown nodes
    g.add_conditional_edges(
        "run_step1",
        _route_after_step1,
        {
            "suggest_unknown_fix": "suggest_unknown_fix",
            "check_detect_layer": "check_detect_layer",
            "error": "error",
        },
    )

    # Unknown fix loop
    g.add_edge("suggest_unknown_fix", "review_unknown_fix")
    g.add_conditional_edges(
        "review_unknown_fix",
        _route_after_unknown_review,
        {
            "run_step1": "run_step1",
            "check_detect_layer": "check_detect_layer",
        },
    )

    # Detect layer: check → (infer if missing) → run_step2
    g.add_node("check_detect_layer", lambda s: {})  # passthrough
    g.add_conditional_edges(
        "check_detect_layer",
        _route_check_detect_layer,
        {
            "load_model_txt": "load_model_txt",
            "run_step2": "run_step2",
        },
    )
    g.add_edge("load_model_txt", "infer_detect_layer")
    g.add_conditional_edges(
        "infer_detect_layer",
        _route_after_step,
        {"continue": "review_detect_layer", "error": "error"},
    )
    g.add_edge("review_detect_layer", "run_step2")

    # Linear pipeline: step2 → step3 → step4 → step5 → step6
    for src, dst in [
        ("run_step2", "run_step3"),
        ("run_step3", "run_step4"),
        ("run_step4", "run_step5"),
        ("run_step5", "run_step6"),
    ]:
        g.add_conditional_edges(
            src,
            _route_after_step,
            {"continue": dst, "error": "error"},
        )

    g.add_edge("run_step6", END)
    g.add_edge("error", END)

    return g


def create_app(checkpointer=None):
    graph = build_graph()
    return graph.compile(checkpointer=checkpointer)
