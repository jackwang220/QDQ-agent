"""
QDQ Agent — entry point.

Usage:
    python -m qdq_agent.main --config pipeline_config.yaml
    python -m qdq_agent.main --config pipeline_config.yaml --thread-id my_run_01
"""
from __future__ import annotations

import argparse
import atexit
import os
import signal
import time

from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from qdq_agent.graph import create_app
from qdq_agent.llm import flush_langfuse, get_langfuse_client
from qdq_agent.state import QDQState


def _handle_interrupt(interrupt_data: dict) -> str:
    """Print the interrupt message and collect human input from stdin."""
    itype = interrupt_data.get("type", "unknown")
    message = interrupt_data.get("message", "Agent needs input:")

    print("\n" + "=" * 60)
    print(f"[HUMAN IN THE LOOP] ({itype})")
    print(message)

    if itype == "detect_layer_review":
        suggestion = interrupt_data.get("suggestion")
        print(f"  Agent suggestion: {suggestion}")

    elif itype == "unknown_fix_review":
        suggestions = interrupt_data.get("suggestions", [])
        print("\n  Suggested fixes:")
        for i, s in enumerate(suggestions, 1):
            print(f"  [{i}] Layer: {s.get('layer', '?')} | Component: {s.get('component', '?')}")
            print(f"      role_type  : {s.get('role_type', '?')}")
            print(f"      pattern    : {s.get('suggested_pattern', '?')}")
            print(f"      reason     : {s.get('reason', '?')}")

    print("=" * 60)
    try:
        return input("> ").strip()
    except EOFError:
        return ""


def _run_pipeline(app, initial_state: QDQState, thread_config: dict) -> QDQState:
    """Stream the graph, handling HITL interrupts in a loop."""
    cmd = initial_state
    final_state = initial_state
    start = time.time()

    while True:
        interrupted = False

        for event in app.stream(cmd, config=thread_config, stream_mode="values"):
            stage = event.get("current_stage", "")
            elapsed = time.time() - start
            print(f"  [{int(elapsed // 60):02d}:{int(elapsed % 60):02d}] {stage}")
            final_state = event

            # Check for interrupt events
            interrupts = event.get("__interrupt__", [])
            if interrupts:
                interrupt_data = interrupts[0].value
                user_input = _handle_interrupt(interrupt_data)
                cmd = Command(resume=user_input)
                interrupted = True
                break

        if not interrupted:
            break

    return final_state


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="QDQ Agent")
    parser.add_argument("--config", default="pipeline_config.yaml", help="Path to pipeline_config.yaml")
    parser.add_argument("--thread-id", default="qdq_run_01", help="Checkpoint thread ID (for HITL resume)")
    args = parser.parse_args()

    # Register Langfuse flush on exit/signal
    atexit.register(flush_langfuse)
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, lambda s, f: (flush_langfuse(), os._exit(128 + s)))
        except (ValueError, OSError):
            pass

    # MemorySaver keeps state in-process (supports HITL interrupt/resume).
    # Swap for SqliteSaver("qdq.db") to persist across restarts.
    checkpointer = MemorySaver()
    app = create_app(checkpointer=checkpointer)

    thread_config = {"configurable": {"thread_id": args.thread_id}}

    initial_state: QDQState = {
        "config_path": args.config,
        "config": {},
        "current_stage": "init",
        "errors": [],
        "model_txt_path": "",
        "excel_path": "",
        "raw_onnx_path": "",
        "int8_onnx_path": "",
        "qdq_onnx_path": "",
        "implicit_onnx_path": "",
        "final_onnx_path": "",
        "excel_unknown_nodes": [],
        "detect_layer": None,
        "model_txt_content": "",
        "layer_constants": {},
        "suggested_detect_layer": None,
        "suggested_patterns": [],
        "suggested_extra_qdq": [],
        "postprocess_config": {},
        "success": False,
    }

    langfuse = get_langfuse_client()
    start_time = time.time()

    print(f"\nStarting QDQ Agent (thread_id={args.thread_id})")
    if langfuse:
        print("  Langfuse tracing enabled")

        # Export Mermaid graph
        try:
            mermaid = app.get_graph().draw_mermaid()
            with open("qdq_pipeline_graph.html", "w", encoding="utf-8") as f:
                f.write(f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
</head><body><div class="mermaid">{mermaid}</div>
<script>mermaid.initialize({{startOnLoad:true}});</script></body></html>""")
            print("  Graph saved -> qdq_pipeline_graph.html")
        except Exception:
            pass

        with langfuse.start_as_current_observation(
            name="qdq_pipeline",
            as_type="span",
            input={"config": args.config, "thread_id": args.thread_id},
        ) as root_span:
            final_state = _run_pipeline(app, initial_state, thread_config)
            success = final_state.get("success", False)
            errors = final_state.get("errors", [])
            root_span.update(
                output={"success": success, "errors": errors, "stage": final_state.get("current_stage")},
                level="DEFAULT" if success else "ERROR",
                status_message=None if success else "; ".join(errors),
            )
            try:
                langfuse.score_current_trace(
                    name="success",
                    value=1.0 if success else 0.0,
                    data_type="BOOLEAN",
                )
            except Exception:
                pass
        flush_langfuse()
    else:
        final_state = _run_pipeline(app, initial_state, thread_config)

    elapsed = time.time() - start_time
    print(f"\nTotal time: {int(elapsed // 60)}m {int(elapsed % 60)}s")

    if final_state.get("success"):
        print(f"Done. Final model: {final_state.get('final_onnx_path')}")
    else:
        print("Pipeline did not complete successfully.")
        for e in final_state.get("errors", []):
            print(f"  - {e}")


if __name__ == "__main__":
    main()
