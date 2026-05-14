"""Agent nodes — LLM inference + HITL interrupt points."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

from langgraph.types import interrupt

from qdq_agent.llm import get_client, robust_generate
from qdq_agent.prompts.detect_layer import DETECT_LAYER_PROMPT
from qdq_agent.prompts.fix_unknown import FIX_UNKNOWN_PROMPT
from qdq_agent.state import QDQState

_MODEL = os.getenv("MODEL_NAME", "gpt-4o-mini")


# ── detect_layer ─────────────────────────────────────────────────────────────

def load_model_txt_node(state: QDQState) -> dict:
    """Read model.txt into state so the agent can analyze it."""
    path = state["model_txt_path"]
    if not Path(path).exists():
        return {"errors": [f"model.txt not found: {path}"], "model_txt_content": ""}
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    return {"model_txt_content": content, "current_stage": "model_txt_loaded"}


def infer_detect_layer_node(state: QDQState) -> dict:
    """Infer the IDetect layer index: regex first, LLM as fallback."""
    print("  [Agent] Inferring detect_layer from model.txt...")
    model_txt = state["model_txt_content"]
    if not model_txt:
        return {"errors": ["model_txt_content is empty — cannot infer detect_layer"]}

    # Fast path: regex scan for IDetect/Detect layer number
    matches = re.findall(r"\((\d+)\):\s*(?:Quant)?(?:IDetect|IKeypoint|Detect)\b", model_txt)
    if len(matches) == 1:
        suggested = int(matches[0])
        print(f"  [Agent] Found detect_layer via regex = {suggested}")
        return {"suggested_detect_layer": suggested, "current_stage": "detect_layer_suggested"}

    # Fallback: ask LLM with the last 8000 chars (IDetect is near the end)
    print("  [Agent] Regex inconclusive, asking LLM...")
    excerpt = model_txt[-8000:] if len(model_txt) > 8000 else model_txt
    prompt = DETECT_LAYER_PROMPT.format(model_txt=excerpt)

    client = get_client()
    raw = robust_generate(client, _MODEL, prompt, trace_name="infer_detect_layer")

    try:
        text = raw.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        data = json.loads(text.strip())
        suggested = int(data["detect_layer"])
        print(f"  [Agent] LLM suggested detect_layer = {suggested}")
        return {"suggested_detect_layer": suggested, "current_stage": "detect_layer_suggested"}
    except Exception as e:
        return {"errors": [f"Failed to parse detect_layer response: {e}\nRaw: {raw[:300]}"]}


def review_detect_layer_node(state: QDQState) -> dict:
    """HITL: show the agent's suggestion to the human and wait for approval."""
    suggestion = state["suggested_detect_layer"]

    # interrupt() pauses the graph and sends data to the caller.
    # The caller resumes with Command(resume=<user_input>).
    human_input: str = interrupt({
        "type": "detect_layer_review",
        "message": (
            f"Agent suggests detect_layer = {suggestion}.\n"
            "Press Enter to accept, or type a different number:"
        ),
        "suggestion": suggestion,
    })

    if human_input and human_input.strip().isdigit():
        final = int(human_input.strip())
        print(f"  [HITL] Human overrode detect_layer: {suggestion} -> {final}")
    else:
        final = suggestion
        print(f"  [HITL] Human accepted detect_layer = {final}")

    return {"detect_layer": final, "current_stage": "detect_layer_confirmed"}


# ── unknown nodes ─────────────────────────────────────────────────────────────

def suggest_unknown_fix_node(state: QDQState) -> dict:
    """LLM analyzes unknown nodes in the Excel and suggests pattern fixes."""
    unknown_nodes = state["excel_unknown_nodes"]
    model_txt = state["model_txt_content"]

    print(f"  [Agent] Analyzing {len(unknown_nodes)} unknown node(s)...")

    unknown_str = json.dumps(unknown_nodes, indent=2, ensure_ascii=False)
    # Provide relevant excerpt from model.txt (first 4000 chars is usually enough)
    excerpt = model_txt[:4000] if model_txt else "(model.txt not loaded)"

    prompt = FIX_UNKNOWN_PROMPT.format(
        unknown_nodes=unknown_str,
        model_txt_excerpt=excerpt,
    )

    client = get_client()
    raw = robust_generate(client, _MODEL, prompt, trace_name="suggest_unknown_fix")

    try:
        text = raw.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        suggestions = json.loads(text.strip())
        print(f"  [Agent] Got {len(suggestions)} suggestion(s)")
        return {"suggested_patterns": suggestions, "current_stage": "unknown_fix_suggested"}
    except Exception as e:
        return {"errors": [f"Failed to parse unknown-fix response: {e}\nRaw: {raw[:300]}"]}


def review_unknown_fix_node(state: QDQState) -> dict:
    """HITL: show unknown-node suggestions to the human."""
    suggestions = state["suggested_patterns"]

    human_input: str = interrupt({
        "type": "unknown_fix_review",
        "message": (
            "Agent suggests the following fixes for unknown role_types.\n"
            "Please apply them to export_model_excel.py manually, then press Enter to re-run Step 1.\n"
            "Type 'skip' to continue without fixing."
        ),
        "suggestions": suggestions,
    })

    if human_input and human_input.strip().lower() == "skip":
        print("  [HITL] Human chose to skip unknown-fix and continue.")
        return {"excel_unknown_nodes": [], "current_stage": "unknown_fix_skipped"}

    print("  [HITL] Human confirmed fix applied. Re-running Step 1...")
    return {"current_stage": "unknown_fix_confirmed"}
