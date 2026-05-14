"""
QDQ Pipeline Runner

Usage:
    python pipeline_runner.py --config pipeline_config.yaml
    python pipeline_runner.py --config pipeline_config.yaml --start-from step3
"""

import argparse
import logging
import subprocess
import sys
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("pipeline_run.log"),
    ],
)
log = logging.getLogger(__name__)

STEPS = ["step1", "step2", "step3", "step4", "step5", "step6"]


@dataclass
class StepResult:
    name: str
    success: bool
    stdout: str = ""
    stderr: str = ""
    needs_agent: bool = False
    agent_reason: str = ""


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_subprocess(cmd: list[str], cwd: str = ".") -> tuple[bool, str, str]:
    log.info("Running: %s", " ".join(cmd))
    result = subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        log.error("FAILED (exit %d)\n%s", result.returncode, result.stderr)
    return result.returncode == 0, result.stdout, result.stderr


# ── Step 1: export_model_excel.py ───────────────────────────────────────────

def step1_export_excel(cfg: dict) -> StepResult:
    log.info("=== Step 1: Export FL values to Excel ===")

    model_txt = cfg["paths"]["model_txt"]
    excel_out = cfg["paths"]["excel_output"]
    code_dir = cfg["paths"]["code_dir"]
    script = str(Path(code_dir) / "export_model_excel.py")

    if not Path(model_txt).exists():
        return StepResult(
            name="step1",
            success=False,
            needs_agent=False,
            agent_reason="",
            stderr=f"model_txt not found: {model_txt}",
        )

    ok, stdout, stderr = run_subprocess(
        [sys.executable, script, model_txt, excel_out]
    )

    result = StepResult(name="step1", success=ok, stdout=stdout, stderr=stderr)

    if ok and "unknown" in stdout.lower():
        result.needs_agent = True
        result.agent_reason = "Excel 有 unknown role_type，需要 agent 分析並建議 pattern 補法"

    return result


# ── Step 2: export_quant_fused.py ───────────────────────────────────────────

def step2_export_onnx(cfg: dict) -> StepResult:
    log.info("=== Step 2: Export quantized fused ONNX ===")

    weights = cfg["model"]["weights"]
    img_size = str(cfg["model"]["img_size"])
    code_dir = cfg["paths"]["code_dir"]
    output_dir = cfg["paths"]["output_dir"]
    raw_onnx = cfg["intermediate"]["raw_onnx"]
    script = str(Path(code_dir) / "export_quant_fused.py")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    ok, stdout, stderr = run_subprocess(
        [
            sys.executable, script,
            "--weights", weights,
            "--img", img_size,
            "--ezquant_int8",
        ]
    )

    if ok:
        # copy output to output_dir
        import shutil
        for candidate in Path(".").glob("*_sim_annotate.onnx"):
            dest = Path(output_dir) / raw_onnx
            shutil.copy(candidate, dest)
            log.info("Copied %s -> %s", candidate, dest)
            break

    return StepResult(name="step2", success=ok, stdout=stdout, stderr=stderr)


# ── Step 3: model_fp32_int8.py ──────────────────────────────────────────────

def step3_fp32_to_int8(cfg: dict) -> StepResult:
    log.info("=== Step 3: FP32 -> INT8 datatype conversion ===")

    output_dir = cfg["paths"]["output_dir"]
    raw_onnx = Path(output_dir) / cfg["intermediate"]["raw_onnx"]
    int8_onnx = Path(output_dir) / cfg["intermediate"]["int8_onnx"]
    code_dir = cfg["paths"]["code_dir"]
    script = str(Path(code_dir) / "model_fp32_int8.py")

    if not raw_onnx.exists():
        return StepResult(
            name="step3",
            success=False,
            stderr=f"Input ONNX not found: {raw_onnx}",
        )

    ok, stdout, stderr = run_subprocess(
        [
            sys.executable, script,
            "--input", str(raw_onnx),
            "--output", str(int8_onnx),
        ]
    )

    return StepResult(name="step3", success=ok, stdout=stdout, stderr=stderr)


# ── Step 4: onnx_view_wrap_all_repconv_topo.py ──────────────────────────────

def step4_insert_qdq(cfg: dict) -> StepResult:
    log.info("=== Step 4: Insert main Q/DQ nodes ===")

    output_dir = cfg["paths"]["output_dir"]
    int8_onnx = Path(output_dir) / cfg["intermediate"]["int8_onnx"]
    qdq_onnx = Path(output_dir) / cfg["intermediate"]["qdq_onnx"]
    excel = cfg["paths"]["excel_output"]
    code_dir = cfg["paths"]["code_dir"]
    script = str(Path(code_dir) / "onnx_view_wrap_all_repconv_topo.py")

    ok, stdout, stderr = run_subprocess(
        [
            sys.executable, script,
            "--model_path", str(int8_onnx),
            "--output_path", str(qdq_onnx),
            "--quant_info", excel,
        ],
        cwd=output_dir,
    )

    return StepResult(name="step4", success=ok, stdout=stdout, stderr=stderr)


# ── Step 5: implicit_topo.py ────────────────────────────────────────────────

def step5_implicit(cfg: dict) -> StepResult:
    log.info("=== Step 5: Handle implicit nodes ===")

    detect_layer = cfg["model"].get("detect_layer")
    output_dir = cfg["paths"]["output_dir"]
    qdq_onnx = Path(output_dir) / cfg["intermediate"]["qdq_onnx"]
    implicit_onnx = Path(output_dir) / cfg["intermediate"]["implicit_onnx"]
    excel = cfg["paths"]["excel_output"]
    code_dir = cfg["paths"]["code_dir"]
    script = str(Path(code_dir) / "implicit_topo.py")

    if detect_layer is None:
        return StepResult(
            name="step5",
            success=False,
            needs_agent=True,
            agent_reason="detect_layer 未設定，需要 agent 從 model.txt 推斷 IDetect layer index",
        )

    ok, stdout, stderr = run_subprocess(
        [
            sys.executable, script,
            "--model_path", str(qdq_onnx),
            "--output_path", str(implicit_onnx),
            "--quant_info", excel,
            "--detect_layer", str(detect_layer),
        ],
        cwd=output_dir,
    )

    return StepResult(name="step5", success=ok, stdout=stdout, stderr=stderr)


# ── Step 6: modify_model_topo.py (函式直接呼叫) ─────────────────────────────

def step6_postprocess(cfg: dict) -> StepResult:
    log.info("=== Step 6: Post-process (bias, Q/DQ patch, transpose, cleanup) ===")

    output_dir = cfg["paths"]["output_dir"]
    implicit_onnx = Path(output_dir) / cfg["intermediate"]["implicit_onnx"]
    final_onnx = Path(output_dir) / cfg["intermediate"]["final_onnx"]
    pp = cfg["postprocess"]

    # 把 code_dir 加入 path 才能 import
    code_dir = str(Path(cfg["paths"]["code_dir"]).resolve())
    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)

    try:
        import onnx
        from modify_model_topo import (
            add_input_bias_node,
            add_qdq_after_nodes,
            add_transpose_after_node,
            remove_outputs_by_names,
            remove_nodes_by_names,
        )

        model = onnx.load(str(implicit_onnx))

        # 1. input bias
        input_bias = cfg["model"].get("input_bias", -0.5)
        if input_bias != 0.0:
            model = add_input_bias_node(model, bias_value=input_bias)

        # 2. 補插 Q/DQ
        extra_nodes = pp.get("extra_qdq_nodes", [])
        if extra_nodes:
            node_names = [n["name"] for n in extra_nodes]
            fl_values = [float(n["fl"]) for n in extra_nodes]
            model = add_qdq_after_nodes(model, node_names, fl_values)

        # 3. Transpose
        for tc in pp.get("transpose_configs", []):
            add_transpose_after_node(
                model,
                tc["target_node"],
                tc["perm"],
                f"transpose_{tc['output_name']}",
                tc["output_name"],
            )

        # 4. cleanup
        remove_outputs_by_names(model, pp.get("outputs_to_remove", []))
        remove_nodes_by_names(model, pp.get("nodes_to_remove", []))

        onnx.save(model, str(final_onnx))
        log.info("Saved final model: %s", final_onnx)
        return StepResult(name="step6", success=True)

    except Exception as e:
        return StepResult(name="step6", success=False, stderr=str(e))


# ── Runner ───────────────────────────────────────────────────────────────────

STEP_FNS = {
    "step1": step1_export_excel,
    "step2": step2_export_onnx,
    "step3": step3_fp32_to_int8,
    "step4": step4_insert_qdq,
    "step5": step5_implicit,
    "step6": step6_postprocess,
}


def run_pipeline(cfg: dict, start_from: str = "step1") -> None:
    started = False
    for step_name in STEPS:
        if step_name == start_from:
            started = True
        if not started:
            log.info("Skipping %s", step_name)
            continue

        result = STEP_FNS[step_name](cfg)

        if result.needs_agent:
            log.warning("[AGENT NEEDED] %s: %s", result.name, result.agent_reason)
            log.warning("Pipeline paused. Run agent to resolve, then re-run with --start-from %s", result.name)
            sys.exit(2)

        if not result.success:
            log.error("[FAILED] %s", result.name)
            if result.stderr:
                log.error(result.stderr)
            sys.exit(1)

        log.info("[OK] %s", result.name)

    log.info("=== Pipeline complete ===")


def main():
    parser = argparse.ArgumentParser(description="QDQ Pipeline Runner")
    parser.add_argument("--config", default="pipeline_config.yaml", help="Path to config YAML")
    parser.add_argument(
        "--start-from",
        default="step1",
        choices=STEPS,
        help="Resume pipeline from a specific step",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_pipeline(cfg, start_from=args.start_from)


if __name__ == "__main__":
    main()
