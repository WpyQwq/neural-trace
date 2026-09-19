"""Local telemetry service for Neural Trace.

The service exposes one stable SSE contract for the browser. It uses the
actual local checkpoint when the optional model runtime can load it; otherwise
it emits clearly labelled mock events so the UI remains usable during download.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .model_runtime import ModelRuntime

MODEL_DIR = Path(os.getenv("MODEL_DIR", r"D:\watch\_LLM\_think"))
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen3.5-4B")
runtime = ModelRuntime(MODEL_DIR, MODEL_NAME)


class StreamRequest(BaseModel):
    prompt: str = "解释一下量子纠缠"
    mode: str = "forward"
    image_data: str | None = None

app = FastAPI(title="Neural Trace Telemetry", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "connected": runtime.ready,
        "downloaded": runtime.checkpoint_ready,
        "model": MODEL_NAME,
        "modelPath": str(runtime.checkpoint_path or MODEL_DIR) if MODEL_DIR.exists() else "waiting for local model",
        "mode": "model" if runtime.ready else "mock",
        "runtime": runtime.status,
        "error": runtime.error,
        "architecture": runtime.architecture,
    }


async def mock_trace(prompt: str, mode: str, image_data: str | None = None) -> AsyncIterator[str]:
    tokens = list(prompt[:10]) or ["·"]
    stages = ["L01", "L04", "L08", "L12", "L16", "L20", "L24", "L28"]
    if mode == "backward":
        stages = list(reversed(stages))
    if image_data:
        vision_payload = {
            "step": 0,
            "kind": "vision_encoder",
            "value": 0.72,
            "shape": [1,  vision_payload_size(image_data), 3],
            "source": "mock",
            "mode": mode,
        }
        yield f"data: {json.dumps(vision_payload, ensure_ascii=False)}\n\n"
    for step in range(48):
        if mode == "backward":
            kind = "gradient"
        elif mode == "attention":
            kind = "attention_output"
        elif mode == "state":
            kind = "delta_state"
        elif step % 16 == 11:
            kind = "mtp_logits"
        elif step % 12 == 3:
            kind = "router_weights"
        elif step % 12 == 7:
            kind = "expert_mixture"
        else:
            kind = "full_attention" if step % 4 == 0 else "delta_state"
        payload = {
            "step": step,
            "token": tokens[step % len(tokens)],
            "layer": stages[step % len(stages)],
            "kind": kind,
            "value": round(0.2 + ((step * 17) % 70) / 100, 3),
            "source": "mock",
            "mode": mode,
        }
        yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
        await asyncio.sleep(0.08)
    suffix = "，并经过视觉编码分支" if image_data else ""
    answer = f"演示输出：已完成对“{prompt}”的可观测计算{suffix}。真实模型接入后，这里会替换为本地 Qwen3.5-4B 的生成结果。"
    yield f"data: {json.dumps({'step': 48, 'kind': 'generation', 'text': answer, 'value': 1.0, 'source': 'mock', 'mode': mode}, ensure_ascii=False)}\n\n"


def vision_payload_size(image_data: str) -> int:
    return 196


async def runtime_trace(prompt: str, mode: str, image_data: str | None = None) -> AsyncIterator[str]:
    events = await asyncio.to_thread(runtime.start_trace, prompt, mode, image_data)
    if events is None:
        if runtime.error:
            yield f"data: {json.dumps({'step': 0, 'kind': 'runtime_error', 'error': runtime.error, 'source': 'model_runtime', 'mode': mode}, ensure_ascii=False)}\n\n"
        async for event in mock_trace(prompt, mode, image_data):
            yield event
        return

    while True:
        payload = await asyncio.to_thread(events.get)
        if payload.get("kind") == "done":
            break
        yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@app.get("/api/stream")
async def stream(
    prompt: str = Query(default="解释一下量子纠缠"),
    mode: str = Query(default="forward"),
) -> StreamingResponse:
    allowed_modes = {"forward", "backward", "attention", "state"}
    safe_mode = mode if mode in allowed_modes else "forward"
    return StreamingResponse(
        runtime_trace(prompt, safe_mode),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@app.post("/api/stream")
async def stream_post(request: StreamRequest) -> StreamingResponse:
    safe_mode = request.mode if request.mode in {"forward", "backward", "attention", "state"} else "forward"
    return StreamingResponse(
        runtime_trace(request.prompt.strip() or "解释一下量子纠缠", safe_mode, request.image_data),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )
