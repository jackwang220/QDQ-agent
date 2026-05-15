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


# ── infer_layer_constants ─────────────────────────────────────────────────────

def infer_layer_constants_node(state: QDQState) -> dict:
    """Parse model.txt to infer CONCAT/MAXPOOL/UPSAMPLE/SPPCSPC/RepConv layer indices."""
    model_txt = state["model_txt_content"]
    if not model_txt:
        print("  [Agent] model.txt empty, skipping layer constant inference")
        return {"layer_constants": {}, "current_stage": "layer_constants_skipped"}

    concat, maxpool, upsample, repconv = [], [], [], []
    sppcspc = None

    # Match top-level Sequential children: exactly 4-space indent + (N): ClassName
    pattern = re.compile(r'^ {4}\((\d+)\):\s+([\w.]+)', re.MULTILINE)
    for m in pattern.finditer(model_txt):
        idx, cls = int(m.group(1)), m.group(2)
        if re.match(r'Concat', cls):
            concat.append(idx)
        elif re.match(r'(?:MaxPool|MP)\b', cls):
            maxpool.append(idx)
        elif re.match(r'(?:nn\.Upsample|Upsample|UP)\b', cls):
            upsample.append(idx)
        elif re.match(r'SPPCSPC', cls, re.IGNORECASE):
            sppcspc = idx
        elif re.match(r'RepConv', cls, re.IGNORECASE):
            repconv.append(idx)

    constants = {
        "concat": concat,
        "maxpool": maxpool,
        "upsample": upsample,
        "sppcspc": sppcspc,
        "repconv": repconv,
    }
    print(f"  [Agent] Layer constants: SPPCSPC={sppcspc}, RepConv={repconv}")
    print(f"          Concat={concat}")
    print(f"          MaxPool={maxpool}, Upsample={upsample}")
    return {"layer_constants": constants, "current_stage": "layer_constants_inferred"}


# ── scan_qdq_coverage ─────────────────────────────────────────────────────────

def scan_qdq_coverage_node(state: QDQState) -> dict:
    """After Step 4: scan QDQ ONNX to find Conv/Act/Concat/Resize nodes missing Q/DQ."""
    try:
        import onnx
        import pandas as pd
    except ImportError as e:
        print(f"  [Agent] scan_qdq_coverage skipped: {e}")
        return {"suggested_extra_qdq": [], "current_stage": "qdq_coverage_skipped"}

    qdq_path = state["qdq_onnx_path"]
    excel_path = state["excel_path"]
    detect_layer = state["detect_layer"]

    if not Path(qdq_path).exists():
        return {"errors": [f"QDQ ONNX not found: {qdq_path}"], "current_stage": "qdq_coverage_failed"}

    print(f"  [Agent] Scanning QDQ coverage: {qdq_path}")
    model = onnx.load(qdq_path)
    graph = model.graph

    # Collect tensors that are already fed into QuantizeLinear
    quantized_inputs = {n.input[0] for n in graph.node if n.op_type == "QuantizeLinear"}

    # Build output tensor → producer node map
    output_to_node: dict[str, object] = {}
    for node in graph.node:
        for out in node.output:
            output_to_node[out] = node

    # Build FL lookup from Excel: layer_num → output FL
    fl_lookup: dict[int, float] = {}
    try:
        df = pd.read_excel(excel_path)
        for _, row in df.iterrows():
            layer = row.get("layer_number")
            role = str(row.get("quantizer_role", "")).lower()
            fl = row.get("fl")
            if pd.notna(layer) and pd.notna(fl) and "output" in role:
                fl_lookup[int(layer)] = float(fl)
    except Exception as e:
        print(f"  [Agent] Warning: could not load Excel FL values: {e}")

    TARGET_OPS = {"Conv", "LeakyRelu", "Relu", "Sigmoid", "Concat", "Add", "Resize", "MaxPool"}
    missing: list[dict] = []
    seen_names: set[str] = set()

    for node in graph.node:
        if node.op_type not in TARGET_OPS or not node.name or not node.output:
            continue
        if node.name in seen_names:
            continue

        output = node.output[0]
        if output in quantized_inputs:
            continue  # already covered

        layer_m = re.search(r'model\.(\d+)', node.name)
        layer_num = int(layer_m.group(1)) if layer_m else None

        # Skip detect_layer — handled by implicit_topo
        if layer_num == detect_layer:
            continue

        # Resolve FL value
        fl: float | None = fl_lookup.get(layer_num)

        if fl is None and node.op_type in {"Resize", "MaxPool", "Upsample"}:
            # Inherit output FL from the preceding node
            input_tensor = node.input[0]
            prev = output_to_node.get(input_tensor)
            if prev:
                prev_layer_m = re.search(r'model\.(\d+)', prev.name or "")
                if prev_layer_m:
                    fl = fl_lookup.get(int(prev_layer_m.group(1)))

        seen_names.add(node.name)
        missing.append({
            "name": node.name,
            "op_type": node.op_type,
            "layer": layer_num,
            "fl": fl,
            "reason": f"{node.op_type} at layer {layer_num} has no Q/DQ on output",
        })

    print(f"  [Agent] Found {len(missing)} node(s) missing Q/DQ coverage")
    return {"suggested_extra_qdq": missing, "current_stage": "qdq_coverage_scanned"}


def review_qdq_coverage_node(state: QDQState) -> dict:
    """HITL: show missing Q/DQ nodes and let human confirm/skip."""
    suggestions = state.get("suggested_extra_qdq", [])

    if not suggestions:
        print("  [Agent] No missing Q/DQ nodes — skipping review")
        return {"current_stage": "qdq_coverage_confirmed"}

    # Format for display
    lines = []
    for i, s in enumerate(suggestions, 1):
        fl_str = f"{s['fl']}" if s["fl"] is not None else "UNKNOWN (set manually)"
        lines.append(f"  [{i}] {s['name']}")
        lines.append(f"      op={s['op_type']}  layer={s['layer']}  fl={fl_str}")
        lines.append(f"      reason: {s['reason']}")

    human_input: str = interrupt({
        "type": "qdq_coverage_review",
        "message": (
            "Agent found nodes missing Q/DQ coverage.\n"
            "These will be added as extra_qdq_nodes in postprocess.\n"
            "Press Enter to accept all, or type 'skip' to skip all.\n"
            "To override FLs, type comma-separated values matching the list order\n"
            "(e.g. '4,4,3,3' — use '-' to keep suggestion):\n\n"
            + "\n".join(lines)
        ),
        "suggestions": suggestions,
    })

    inp = (human_input or "").strip().lower()

    if inp == "skip":
        print("  [HITL] Skipped extra Q/DQ insertion")
        return {"suggested_extra_qdq": [], "current_stage": "qdq_coverage_skipped"}

    # Parse optional FL overrides
    if inp and inp != "":
        parts = [p.strip() for p in inp.split(",")]
        for i, part in enumerate(parts):
            if i < len(suggestions) and part != "-":
                try:
                    suggestions[i]["fl"] = float(part)
                except ValueError:
                    pass

    # Drop entries with unknown FL
    valid = [s for s in suggestions if s["fl"] is not None]
    dropped = len(suggestions) - len(valid)
    if dropped:
        print(f"  [HITL] Dropped {dropped} node(s) with unknown FL")

    print(f"  [HITL] Confirmed {len(valid)} extra Q/DQ node(s)")
    return {"suggested_extra_qdq": valid, "current_stage": "qdq_coverage_confirmed"}


# ── infer_postprocess ─────────────────────────────────────────────────────────

def infer_postprocess_node(state: QDQState) -> dict:
    """After Step 5: auto-determine transpose_configs, outputs_to_remove, nodes_to_remove."""
    try:
        import onnx
    except ImportError as e:
        print(f"  [Agent] infer_postprocess skipped: {e}")
        return {"postprocess_config": state["config"].get("postprocess", {}),
                "current_stage": "postprocess_inferred"}

    implicit_path = state["implicit_onnx_path"]
    detect_layer = state["detect_layer"]
    extra_qdq = state.get("suggested_extra_qdq", [])

    if not Path(implicit_path).exists():
        return {"errors": [f"implicit ONNX not found: {implicit_path}"],
                "current_stage": "postprocess_infer_failed"}

    print(f"  [Agent] Inferring postprocess config from {implicit_path}")
    model = onnx.load(implicit_path)
    graph = model.graph

    # 1. Transpose target nodes — implicit_output_dequant_{layer}_im{i}_mul
    pat = re.compile(rf'implicit_output_dequant_{detect_layer}_im(\d+)_mul')
    targets: list[tuple[int, str]] = []
    for node in graph.node:
        m = pat.match(node.name or "")
        if m:
            targets.append((int(m.group(1)), node.name))
    targets.sort()

    transpose_configs = [
        {"target_node": name, "perm": [0, 2, 3, 1], "output_name": f"transposed_output_{i}"}
        for i, (_, name) in enumerate(targets)
    ]
    print(f"  [Agent] Found {len(transpose_configs)} transpose target(s)")

    # 2. nodes_to_remove — Reshape/Transpose in detect_layer
    nodes_to_remove = [
        node.name for node in graph.node
        if node.op_type in {"Reshape", "Transpose"}
        and re.search(rf'/model\.{detect_layer}/', node.name or "")
    ]
    print(f"  [Agent] nodes_to_remove: {nodes_to_remove}")

    # 3. outputs_to_remove — outputs produced by the to-be-removed nodes or purely numeric names
    remove_set = set(nodes_to_remove)
    node_to_outputs = {n.name: list(n.output) for n in graph.node}
    outputs_to_remove = []
    for out in graph.output:
        name = out.name
        # Numeric tensor IDs (like "455", "469")
        if name.isdigit():
            outputs_to_remove.append(name)
            continue
        # Outputs produced by a node we're removing
        for nname in remove_set:
            if name in node_to_outputs.get(nname, []):
                outputs_to_remove.append(name)
                break
        # "output" — the original raw detect head output
        if name == "output":
            outputs_to_remove.append(name)
    outputs_to_remove = list(dict.fromkeys(outputs_to_remove))  # deduplicate, preserve order
    print(f"  [Agent] outputs_to_remove: {outputs_to_remove}")

    # Build final postprocess config (merge extra_qdq from scan step)
    extra_qdq_nodes = [{"name": s["name"], "fl": s["fl"]} for s in extra_qdq if s.get("fl") is not None]

    postprocess_config = {
        "extra_qdq_nodes": extra_qdq_nodes,
        "transpose_configs": transpose_configs,
        "outputs_to_remove": outputs_to_remove,
        "nodes_to_remove": nodes_to_remove,
    }

    return {"postprocess_config": postprocess_config, "current_stage": "postprocess_inferred"}

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
