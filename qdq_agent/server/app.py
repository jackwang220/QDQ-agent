"""QDQ Agent — FastAPI server wrapping the pipeline with a web HITL interface.

Run with:
    python -m qdq_agent.server            (default port 8000)
    python -m qdq_agent.server --port 9000

The original CLI (python -m qdq_agent.main) is unchanged — this is additive.

NOTE: Pydantic models must be at module level (not inside create_app()) so that
Pydantic v2 can fully resolve them at import time.
"""
from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import Body, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, RedirectResponse
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from pydantic import BaseModel

from qdq_agent.graph import create_app as create_graph
from qdq_agent.state import QDQState
from qdq_agent.server.panel import PANEL_HTML
from qdq_agent.server.session import PipelineSession

# ── Request models (module-level — required for Pydantic v2 ForwardRef resolution) ──

class RunRequest(BaseModel):
    config_path: str = "pipeline_config.yaml"
    model_txt: str = ""              # overrides paths.model_txt in YAML; empty = use YAML
    detect_layer: Optional[int] = None  # overrides model.detect_layer; None = auto
    thread_id: str = "qdq_run_01"


class RespondRequest(BaseModel):
    response: str = ""


# ── Session singleton ────────────────────────────────────────────────────────

_session = PipelineSession()

_INITIAL_STATE_TEMPLATE: QDQState = {
    "config_path": "",
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
    "quantizer_mode": None,
    "input_bias": None,
    "success": False,
}


def _run_pipeline_thread(req: RunRequest, sess: PipelineSession) -> None:
    """Runs the LangGraph pipeline in a daemon thread.

    HITL interrupts are forwarded to the browser via session.put_interrupt()
    instead of reading from stdin.
    """
    load_dotenv()
    checkpointer = MemorySaver()
    graph_app = create_graph(checkpointer=checkpointer)
    thread_config = {"configurable": {"thread_id": req.thread_id}}

    initial_state = dict(_INITIAL_STATE_TEMPLATE)
    initial_state["config_path"] = req.config_path
    if req.model_txt:
        initial_state["model_txt_path"] = req.model_txt
    if req.detect_layer is not None:
        initial_state["detect_layer"] = req.detect_layer

    cmd = initial_state
    final_state: dict = initial_state
    start = time.time()

    try:
        while True:
            interrupted = False
            for event in graph_app.stream(cmd, config=thread_config, stream_mode="values"):
                stage = event.get("current_stage", "")
                if stage:
                    sess.put_stage(stage, time.time() - start)
                final_state = event

                interrupts = event.get("__interrupt__", [])
                if interrupts:
                    interrupt_data = interrupts[0].value
                    sess.put_interrupt(interrupt_data)
                    user_input = sess.wait_for_response()
                    cmd = Command(resume=user_input)
                    interrupted = True
                    break

            if not interrupted:
                break

        success = bool(final_state.get("success", False))
        errors = list(final_state.get("errors", []))
        sess.put_done(success, errors)

    except Exception as exc:
        sess.put_error(f"{type(exc).__name__}: {exc}")


# ── FastAPI app ──────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(
        title="QDQ Agent Panel",
        version="1.0.0",
        description="Web HITL interface for the QDQ quantization pipeline.",
    )

    @app.on_event("startup")
    async def _set_loop() -> None:
        _session._loop = asyncio.get_event_loop()

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse(url="/panel", status_code=302)

    @app.get("/panel", response_class=HTMLResponse, include_in_schema=False)
    def panel() -> str:
        return PANEL_HTML

    @app.get("/healthz")
    def healthz() -> dict:
        return {"ok": True}

    @app.get("/status")
    def get_status() -> dict:
        return _session.snapshot()

    @app.post("/run")
    def start_run(req: RunRequest = Body()) -> dict:
        snap = _session.snapshot()
        if snap["status"] in ("running", "paused"):
            raise HTTPException(409, "Pipeline is already running.")
        if not Path(req.config_path).exists():
            raise HTTPException(400, f"Config file not found: {req.config_path}")
        if req.model_txt and not Path(req.model_txt).exists():
            raise HTTPException(400, f"model.txt not found: {req.model_txt}")
        _session.reset()
        t = threading.Thread(
            target=_run_pipeline_thread,
            args=(req, _session),
            daemon=True,
        )
        t.start()
        return {"status": "started", "config_path": req.config_path, "thread_id": req.thread_id}

    @app.post("/respond")
    def respond(req: RespondRequest = Body()) -> dict:
        ok = _session.put_response(req.response)
        if not ok:
            raise HTTPException(409, "No HITL interrupt is pending.")
        return {"ok": True}

    @app.websocket("/events")
    async def ws_events(ws: WebSocket) -> None:
        await ws.accept()
        q: asyncio.Queue = asyncio.Queue()
        _session.add_listener(q)

        with _session._lock:
            past = list(_session.events)
        for event in past:
            await ws.send_json(event)

        await ws.send_json({"type": "status", **_session.snapshot()})

        try:
            while True:
                try:
                    msg = await asyncio.wait_for(q.get(), timeout=25)
                    await ws.send_json(msg)
                except asyncio.TimeoutError:
                    await ws.send_json({"type": "ping"})
        except WebSocketDisconnect:
            pass
        finally:
            _session.remove_listener(q)

    return app


app = create_app()
