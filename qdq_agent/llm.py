"""LLM client + Langfuse tracing — mirrors the Rust-translator pattern."""
from __future__ import annotations

import json
import os
import re
import time

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

DEFAULT_API_KEY = os.getenv("API_KEY", "")
DEFAULT_BASE_URL = os.getenv("BASE_URL", "https://api.openai.com/v1")
DEFAULT_MODEL = os.getenv("MODEL_NAME", "gpt-4o-mini")
DEFAULT_LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "120"))
DEFAULT_LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "4096"))


def get_langfuse_client():
    secret_key = os.getenv("LANGFUSE_SECRET_KEY", "")
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    base_url = os.getenv("LANGFUSE_BASE_URL", "")
    if not (secret_key and public_key and base_url):
        return None
    try:
        from langfuse import get_client
        return get_client()
    except Exception:
        return None


def get_client() -> OpenAI:
    return OpenAI(
        api_key=os.getenv("API_KEY", DEFAULT_API_KEY),
        base_url=os.getenv("BASE_URL", DEFAULT_BASE_URL),
        timeout=DEFAULT_LLM_TIMEOUT,
        max_retries=0,
    )


def robust_generate(
    client: OpenAI,
    model: str,
    prompt: str,
    max_retries: int = 3,
    trace_name: str = "llm_call",
) -> str:
    """Generate a completion with retry and optional Langfuse logging."""
    langfuse = get_langfuse_client()
    delay = 5

    for attempt in range(max_retries):
        try:
            kwargs = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "timeout": DEFAULT_LLM_TIMEOUT,
            }
            if DEFAULT_LLM_MAX_TOKENS > 0:
                kwargs["max_tokens"] = DEFAULT_LLM_MAX_TOKENS

            if langfuse:
                with langfuse.start_as_current_observation(
                    name=trace_name,
                    as_type="generation",
                    model=model,
                    input=prompt,
                ) as gen:
                    response = client.chat.completions.create(**kwargs)
                    msg = response.choices[0].message
                    text = re.sub(r"<think>.*?</think>", "", msg.content or "", flags=re.DOTALL).strip()
                    usage = response.usage
                    gen.update(
                        output=text,
                        usage_details={
                            "input": getattr(usage, "prompt_tokens", 0) if usage else 0,
                            "output": getattr(usage, "completion_tokens", 0) if usage else 0,
                        },
                        metadata={"attempt": attempt + 1},
                    )
            else:
                response = client.chat.completions.create(**kwargs)
                raw_content = response.choices[0].message.content or ""
                text = re.sub(r"<think>.*?</think>", "", raw_content, flags=re.DOTALL).strip()

            return text

        except Exception as e:
            print(f"  API error: {e}")
            if attempt < max_retries - 1:
                print(f"  Retrying in {delay}s...")
                time.sleep(delay)
                delay *= 2
            else:
                raise


def flush_langfuse():
    lf = get_langfuse_client()
    if lf:
        lf.flush()


# ── Langfuse node tracing decorator ─────────────────────────────────────────

_SUMMARY_KEYS = ("current_stage", "detect_layer", "suggested_detect_layer", "success")


def _summarise_state(state: dict) -> dict:
    out = {k: state[k] for k in _SUMMARY_KEYS if k in state}
    unknowns = state.get("excel_unknown_nodes", [])
    if unknowns:
        out["unknown_count"] = len(unknowns)
    return out


def _summarise_delta(delta: dict) -> dict:
    out = {}
    for k, v in delta.items():
        if isinstance(v, (str,)) and len(v) > 200:
            out[f"{k}_chars"] = len(v)
        elif isinstance(v, (int, float, bool, list, dict)) or v is None:
            out[k] = v
        else:
            out[k] = str(v)[:200]
    return out


def trace_node(name: str):
    """Decorator that wraps a LangGraph node in a Langfuse span."""
    def deco(fn):
        def wrapped(state):
            lf = get_langfuse_client()
            if not lf:
                return fn(state)
            with lf.start_as_current_observation(
                name=name,
                as_type="span",
                input=_summarise_state(state),
            ) as span:
                try:
                    result = fn(state)
                except Exception as e:
                    span.update(level="ERROR", status_message=f"{type(e).__name__}: {e}")
                    raise
                span.update(output=_summarise_delta(result))
                return result
        wrapped.__name__ = getattr(fn, "__name__", name)
        wrapped.__wrapped__ = fn
        return wrapped
    return deco
