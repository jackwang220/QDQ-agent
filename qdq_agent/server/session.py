"""Thread-safe session object bridging the pipeline thread and the async FastAPI layer.

The pipeline runs in a daemon thread (sync, blocking LangGraph stream).
FastAPI handlers and WebSocket clients live in the asyncio event loop.

Bridge:
  thread → session.put_*()  → _broadcast() → asyncio.run_coroutine_threadsafe → WS queues
  client → POST /respond    → session.put_response() → _response_queue.put()
  thread ← session.wait_for_response() ← _response_queue.get()  (blocks until client submits)
"""
from __future__ import annotations

import asyncio
import queue
import threading
from typing import Optional


class PipelineSession:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._response_queue: queue.Queue[str] = queue.Queue(maxsize=1)
        self._listeners: list[asyncio.Queue] = []
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._reset_locked()

    def _reset_locked(self) -> None:
        self.status: str = "idle"          # idle | running | paused | done | error
        self.events: list[dict] = []
        self.current_interrupt: Optional[dict] = None
        self.final_success: Optional[bool] = None
        self.final_errors: list[str] = []
        self.final_stage: str = ""

    def reset(self) -> None:
        with self._lock:
            self._reset_locked()
        while True:
            try:
                self._response_queue.get_nowait()
            except queue.Empty:
                break

    # ── listener management ───────────────────────────────────────────────────

    def add_listener(self, q: asyncio.Queue) -> None:
        with self._lock:
            self._listeners.append(q)

    def remove_listener(self, q: asyncio.Queue) -> None:
        with self._lock:
            if q in self._listeners:
                self._listeners.remove(q)

    def _broadcast(self, event: dict) -> None:
        with self._lock:
            self.events.append(event)
            listeners = list(self._listeners)
            loop = self._loop
        if loop:
            for q in listeners:
                asyncio.run_coroutine_threadsafe(q.put(event), loop)

    # ── called from pipeline thread ───────────────────────────────────────────

    def put_stage(self, stage: str, elapsed: float) -> None:
        with self._lock:
            self.final_stage = stage
        self._broadcast({"type": "stage", "stage": stage, "elapsed": round(elapsed, 1)})

    def put_interrupt(self, data: dict) -> None:
        with self._lock:
            self.status = "paused"
            self.current_interrupt = data
        self._broadcast({"type": "interrupt", "data": data})

    def wait_for_response(self) -> str:
        """Block the pipeline thread until the user submits a response via HTTP."""
        return self._response_queue.get()

    def put_done(self, success: bool, errors: list[str] | None = None) -> None:
        with self._lock:
            self.status = "done"
            self.final_success = success
            self.final_errors = errors or []
        self._broadcast({"type": "done", "success": success, "errors": errors or []})

    def put_error(self, msg: str) -> None:
        with self._lock:
            self.status = "error"
        self._broadcast({"type": "error", "error": msg})

    # ── called from HTTP handler ──────────────────────────────────────────────

    def put_response(self, user_input: str) -> bool:
        """Return False if not currently paused (idempotent-safe)."""
        with self._lock:
            if self.status != "paused":
                return False
            self.status = "running"
            self.current_interrupt = None
        self._response_queue.put(user_input)
        self._broadcast({"type": "resumed", "response": user_input})
        return True

    # ── snapshot for /status ──────────────────────────────────────────────────

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "status": self.status,
                "interrupt": self.current_interrupt,
                "final_success": self.final_success,
                "final_errors": list(self.final_errors),
                "final_stage": self.final_stage,
            }
