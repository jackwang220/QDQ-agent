"""Pipeline nodes — each node runs one script step and returns a state delta."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from qdq_agent.state import QDQState


def _run(cmd: list[str], cwd: str | None = None) -> tuple[bool, str, str]:
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    ok = result.returncode == 0
    if not ok:
        print(f"  [FAIL exit={result.returncode}] {' '.join(cmd)}")
        print(result.stderr[-2000:])
    return ok, result.stdout, result.stderr


# ── load_config ──────────────────────────────────────────────────────────────

def load_config_node(state: QDQState) -> dict:
    import yaml
    config_path = state["config_path"]
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    output_dir = cfg["paths"]["output_dir"]
    inter = cfg["intermediate"]

    detect_layer = cfg["model"].get("detect_layer")  # may be None

    # "auto" → derive model_txt path from weights stem inside output_dir
    raw_model_txt = cfg["paths"].get("model_txt", "auto")
    if raw_model_txt == "auto":
        weights_stem = Path(cfg["model"]["weights"]).stem
        model_txt_path = str(Path(output_dir) / f"{weights_stem}_model.txt")
    else:
        model_txt_path = raw_model_txt

    return {
        "config": cfg,
        "model_txt_path": model_txt_path,
        "excel_path": cfg["paths"]["excel_output"],
        "raw_onnx_path": str(Path(output_dir) / inter["raw_onnx"]),
        "int8_onnx_path": str(Path(output_dir) / inter["int8_onnx"]),
        "qdq_onnx_path": str(Path(output_dir) / inter["qdq_onnx"]),
        "implicit_onnx_path": str(Path(output_dir) / inter["implicit_onnx"]),
        "final_onnx_path": str(Path(output_dir) / inter["final_onnx"]),
        "detect_layer": detect_layer,
        "excel_unknown_nodes": [],
        "suggested_detect_layer": None,
        "suggested_patterns": [],
        "model_txt_content": "",
        "success": False,
        "current_stage": "config_loaded",
    }


# ── Step 0: generate_model_txt (auto, skipped if file already exists) ────────

def run_step0_node(state: QDQState) -> dict:
    model_txt = state["model_txt_path"]

    if Path(model_txt).exists():
        print(f"  [Step 0] model.txt already exists, skipping: {model_txt}")
        return {"current_stage": "step0_skipped"}

    cfg = state["config"]
    code_dir = str(Path(cfg["paths"]["code_dir"]).resolve())
    yolov7_dir = cfg["paths"].get("yolov7_dir", ".")
    script = str(Path(code_dir) / "generate_model_txt.py")
    weights = str(Path(cfg["model"]["weights"]).resolve())
    model_txt_abs = str(Path(model_txt).resolve())

    print(f"  [Step 0] generate_model_txt: {weights} -> {model_txt}")
    Path(model_txt_abs).parent.mkdir(parents=True, exist_ok=True)

    ok, stdout, stderr = _run(
        [sys.executable, script, "--weights", weights, "--output", model_txt_abs],
        cwd=yolov7_dir,
    )
    if not ok:
        return {"errors": [f"Step 0 failed: {stderr[-500:]}"], "current_stage": "step0_failed"}

    return {"current_stage": "step0_done"}


# ── Step 1: export_model_excel ───────────────────────────────────────────────

def run_step1_node(state: QDQState) -> dict:
    cfg = state["config"]
    code_dir = cfg["paths"]["code_dir"]
    script = str(Path(code_dir) / "export_model_excel.py")
    model_txt = state["model_txt_path"]
    excel_out = state["excel_path"]

    print(f"  [Step 1] export_model_excel: {model_txt} -> {excel_out}")

    if not Path(model_txt).exists():
        return {"errors": [f"model_txt not found: {model_txt}"], "current_stage": "step1_failed"}

    # Build CLI args from inferred layer constants
    constants = state.get("layer_constants") or {}
    cmd = [sys.executable, script, model_txt, excel_out]
    if constants.get("concat"):
        cmd += ["--concat-layers", ",".join(map(str, constants["concat"]))]
    if constants.get("maxpool"):
        cmd += ["--maxpool-layers", ",".join(map(str, constants["maxpool"]))]
    if constants.get("upsample"):
        cmd += ["--upsample-layers", ",".join(map(str, constants["upsample"]))]

    ok, stdout, stderr = _run(cmd)

    if not ok:
        return {"errors": [f"Step 1 failed: {stderr[-500:]}"], "current_stage": "step1_failed"}

    # Check for unknown role_types in output
    unknown_nodes = []
    for line in stdout.splitlines():
        if "unknown" in line.lower() and "role_type" in line.lower():
            unknown_nodes.append({"raw_line": line.strip()})

    return {
        "excel_unknown_nodes": unknown_nodes,
        "current_stage": "step1_done",
    }


# ── Step 2: export_quant_fused ───────────────────────────────────────────────

def run_step2_node(state: QDQState) -> dict:
    cfg = state["config"]
    code_dir = str(Path(cfg["paths"]["code_dir"]).resolve())
    yolov7_dir = cfg["paths"].get("yolov7_dir", ".")
    script = str(Path(code_dir) / "export_quant_fused.py")
    # Use absolute path for weights so it resolves correctly from yolov7_dir cwd
    weights = str(Path(cfg["model"]["weights"]).resolve())
    img_size = str(cfg["model"]["img_size"])
    output_dir = cfg["paths"]["output_dir"]
    raw_onnx = state["raw_onnx_path"]

    print(f"  [Step 2] export_quant_fused: {weights}")
    print(f"  cwd: {yolov7_dir}")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Run from yolov7_dir so that `import models` resolves correctly
    ok, stdout, stderr = _run(
        [sys.executable, script, "--weights", weights, "--img_size", img_size, "--ezquant_int8"],
        cwd=yolov7_dir,
    )
    if not ok:
        return {"errors": [f"Step 2 failed: {stderr[-500:]}"], "current_stage": "step2_failed"}

    # The script outputs a *_sim_annotate.onnx next to the weights file
    candidates = list(Path(weights).parent.glob("*_sim_annotate.onnx"))
    if candidates:
        shutil.copy(candidates[0], raw_onnx)
        print(f"  Copied {candidates[0]} -> {raw_onnx}")
    else:
        return {"errors": ["Step 2: could not find *_sim_annotate.onnx output"], "current_stage": "step2_failed"}

    return {"current_stage": "step2_done"}


# ── Step 3: model_fp32_int8 ──────────────────────────────────────────────────

def run_step3_node(state: QDQState) -> dict:
    cfg = state["config"]
    code_dir = cfg["paths"]["code_dir"]
    script = str(Path(code_dir) / "model_fp32_int8.py")
    input_onnx = state["raw_onnx_path"]
    output_onnx = state["int8_onnx_path"]

    print(f"  [Step 3] model_fp32_int8: {input_onnx} -> {output_onnx}")

    if not Path(input_onnx).exists():
        return {"errors": [f"Step 3: input not found: {input_onnx}"], "current_stage": "step3_failed"}

    ok, _, stderr = _run([sys.executable, script, "--input", input_onnx, "--output", output_onnx])
    if not ok:
        return {"errors": [f"Step 3 failed: {stderr[-500:]}"], "current_stage": "step3_failed"}

    return {"current_stage": "step3_done"}


# ── Step 4: onnx_view_wrap_all_repconv_topo ──────────────────────────────────

def run_step4_node(state: QDQState) -> dict:
    cfg = state["config"]
    code_dir = cfg["paths"]["code_dir"]
    script = str(Path(code_dir) / "onnx_view_wrap_all_repconv_topo.py")
    input_onnx = state["int8_onnx_path"]
    output_onnx = state["qdq_onnx_path"]
    excel = state["excel_path"]

    print(f"  [Step 4] insert_qdq: {input_onnx} -> {output_onnx}")

    if not Path(input_onnx).exists():
        return {"errors": [f"Step 4: input not found: {input_onnx}"], "current_stage": "step4_failed"}

    constants = state.get("layer_constants") or {}
    cmd = [
        sys.executable, script,
        "--model_path", input_onnx,
        "--output_path", output_onnx,
        "--quant_info", excel,
    ]
    if constants.get("sppcspc") is not None:
        cmd += ["--sppcspc-layer", str(constants["sppcspc"])]
    if constants.get("repconv"):
        cmd += ["--repconv-layers", ",".join(map(str, constants["repconv"]))]
    if state.get("detect_layer") is not None:
        cmd += ["--detection-layer", str(state["detect_layer"])]

    ok, _, stderr = _run(cmd)
    if not ok:
        return {"errors": [f"Step 4 failed: {stderr[-500:]}"], "current_stage": "step4_failed"}

    return {"current_stage": "step4_done"}


# ── Step 5: implicit_topo ────────────────────────────────────────────────────

def run_step5_node(state: QDQState) -> dict:
    cfg = state["config"]
    code_dir = cfg["paths"]["code_dir"]
    script = str(Path(code_dir) / "implicit_topo.py")
    input_onnx = state["qdq_onnx_path"]
    output_onnx = state["implicit_onnx_path"]
    excel = state["excel_path"]
    detect_layer = state["detect_layer"]

    print(f"  [Step 5] implicit: detect_layer={detect_layer}")

    if not Path(input_onnx).exists():
        return {"errors": [f"Step 5: input not found: {input_onnx}"], "current_stage": "step5_failed"}

    ok, _, stderr = _run([
        sys.executable, script,
        "--model_path", input_onnx,
        "--output_path", output_onnx,
        "--quant_info", excel,
        "--detect_layer", str(detect_layer),
    ])
    if not ok:
        return {"errors": [f"Step 5 failed: {stderr[-500:]}"], "current_stage": "step5_failed"}

    return {"current_stage": "step5_done"}


# ── Step 6: modify_model_topo (import functions directly) ────────────────────

def run_step6_node(state: QDQState) -> dict:
    cfg = state["config"]
    code_dir = str(Path(cfg["paths"]["code_dir"]).resolve())
    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)

    input_onnx = state["implicit_onnx_path"]
    output_onnx = state["final_onnx_path"]
    # Use auto-generated postprocess config if available, else fall back to yaml
    pp = state.get("postprocess_config") or cfg.get("postprocess", {})

    print(f"  [Step 6] postprocess: {input_onnx} -> {output_onnx}")

    if not Path(input_onnx).exists():
        return {"errors": [f"Step 6: input not found: {input_onnx}"], "current_stage": "step6_failed"}

    try:
        import onnx
        from modify_model_topo import (
            add_input_bias_node,
            add_qdq_after_nodes,
            add_transpose_after_node,
            remove_outputs_by_names,
            remove_nodes_by_names,
        )

        model = onnx.load(input_onnx)

        input_bias = cfg["model"].get("input_bias", -0.5)
        if input_bias != 0.0:
            model = add_input_bias_node(model, bias_value=input_bias)

        extra_nodes = pp.get("extra_qdq_nodes", [])
        if extra_nodes:
            model = add_qdq_after_nodes(
                model,
                [n["name"] for n in extra_nodes],
                [float(n["fl"]) for n in extra_nodes],
            )

        for i, tc in enumerate(pp.get("transpose_configs", [])):
            add_transpose_after_node(
                model,
                tc["target_node"],
                tc["perm"],
                f"transpose_{i}",
                tc["output_name"],
            )

        remove_outputs_by_names(model, pp.get("outputs_to_remove", []))
        remove_nodes_by_names(model, pp.get("nodes_to_remove", []))

        onnx.save(model, output_onnx)
        print(f"  Saved: {output_onnx}")

    except Exception as e:
        return {"errors": [f"Step 6 failed: {e}"], "current_stage": "step6_failed"}

    return {"current_stage": "step6_done", "success": True}
