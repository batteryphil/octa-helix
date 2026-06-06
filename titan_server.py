"""
Titan FastAPI Server — Fast + Deep Think endpoints
  POST /generate        → full JSON (fast)
  POST /generate_deep   → full JSON (deep think — includes thoughts)
  POST /stream          → SSE fast streaming
  POST /stream_deep     → SSE deep think: yields thoughts first, then tokens
  GET  /telemetry       → last inference arm weights
  GET  /health          → model status
"""
import json, time, asyncio, os
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from titan_inference import TitanInference

app    = FastAPI(title="Titan MoE Reasoning Engine", version="2.0")
engine = TitanInference()

app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

_model_loaded = False
_load_error   = ""


@app.on_event("startup")
async def startup():
    global _model_loaded, _load_error
    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, engine.load)
        _model_loaded = True
    except Exception as e:
        _load_error = str(e)
        print(f"[Server] Load failed: {e}")


class GenerateRequest(BaseModel):
    prompt:             str
    max_new_tokens:     Optional[int]   = 512
    temperature:        Optional[float] = 0.72
    top_p:              Optional[float] = 0.92
    repetition_penalty: Optional[float] = 1.15
    show_arm_weights:   Optional[bool]  = True
    deep_think:         Optional[bool]  = False
    thought_tokens:     Optional[int]   = 60


@app.get("/health")
async def health():
    import torch
    telem_path = os.path.join(os.path.dirname(__file__), "monitor_ui", "telemetry.json")
    step, phase = 0, "unknown"
    try:
        with open(telem_path) as f:
            t = json.load(f); step = t.get("step",0); phase = t.get("phase","?")
    except Exception: pass
    return {
        "status": "ready" if _model_loaded else "loading",
        "error": _load_error,
        "device": str(torch.device('cuda' if torch.cuda.is_available() else 'cpu')),
        "vram_gb": round(torch.cuda.memory_allocated()/1e9, 2) if torch.cuda.is_available() else 0,
        "training_step": step, "phase": phase,
    }


@app.get("/telemetry")
async def telemetry():
    return JSONResponse({"arm_weights": engine.last_arm_weights,
                         "top_arms": engine.last_top_arms,
                         "thoughts": engine.last_thoughts})


@app.post("/generate")
async def generate(req: GenerateRequest):
    if not _model_loaded:
        return JSONResponse({"error": "Model not loaded"}, status_code=503)
    loop = asyncio.get_event_loop()
    text, arm_log, thoughts = await loop.run_in_executor(
        None, lambda: engine.generate(
            req.prompt, deep_think=req.deep_think,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature, top_p=req.top_p,
            repetition_penalty=req.repetition_penalty,
        )
    )
    return {
        "response": text, "prompt": req.prompt,
        "thoughts": thoughts if req.show_arm_weights else [],
        "top_arms": engine.last_top_arms if req.show_arm_weights else [],
        "arm_weights": engine.last_arm_weights if req.show_arm_weights else [],
    }


@app.post("/stream")
async def stream_fast(req: GenerateRequest):
    """Fast streaming — SSE token stream."""
    if not _model_loaded:
        async def err():
            yield 'data: {"error":"Model not loaded"}\n\n'
        return StreamingResponse(err(), media_type="text/event-stream")

    async def gen():
        loop = asyncio.get_event_loop()
        q: asyncio.Queue = asyncio.Queue()

        def run():
            try:
                for tok, arm_info in engine.stream(
                    req.prompt,
                    max_new_tokens=req.max_new_tokens,
                    temperature=req.temperature, top_p=req.top_p,
                    repetition_penalty=req.repetition_penalty,
                ):
                    loop.call_soon_threadsafe(q.put_nowait, ("token", tok, arm_info))
                loop.call_soon_threadsafe(q.put_nowait, ("done", None, None))
            except Exception as e:
                loop.call_soon_threadsafe(q.put_nowait, ("error", str(e), None))

        loop.run_in_executor(None, run)
        while True:
            kind, tok, info = await q.get()
            if kind == "done":
                yield f"data: {json.dumps({'done': True, 'top_arms': engine.last_top_arms})}\n\n"
                break
            elif kind == "error":
                yield f"data: {json.dumps({'error': tok})}\n\n"
                break
            else:
                clean = tok.replace("<|endoftext|>", "")
                yield f"data: {json.dumps({'token': clean, 'arm_info': info if req.show_arm_weights else {}})}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.post("/stream_deep")
async def stream_deep(req: GenerateRequest):
    """
    Deep Think streaming — yields arm thoughts first, then tokens.
    SSE event types:
      {"type":"thinking_start"}
      {"type":"thought", "arm":N, "label":"...", "weight":0.xx, "thought":"..."}
      {"type":"thinking_done", "thought_count":N}
      {"type":"token", "token":"...", "arm_info":{...}}
      {"type":"done", "top_arms":[...]}
    """
    if not _model_loaded:
        async def err():
            yield 'data: {"type":"error","msg":"Model not loaded"}\n\n'
        return StreamingResponse(err(), media_type="text/event-stream")

    async def gen():
        loop = asyncio.get_event_loop()

        # Phase 1: Gather thoughts (blocking, but fast — 60 tokens per arm)
        yield f"data: {json.dumps({'type':'thinking_start'})}\n\n"
        thoughts = await loop.run_in_executor(
            None, lambda: engine.deep_think_thoughts(
                req.prompt,
                thought_tokens=req.thought_tokens or 60,
            )
        )
        for t in thoughts:
            yield f"data: {json.dumps({'type':'thought', **t})}\n\n"
        yield f"data: {json.dumps({'type':'thinking_done','thought_count':len(thoughts)})}\n\n"

        # Phase 2: Stream final answer
        q: asyncio.Queue = asyncio.Queue()
        def run():
            try:
                for tok, arm_info in engine.stream(
                    req.prompt,
                    context_thoughts=thoughts,
                    max_new_tokens=req.max_new_tokens,
                    temperature=req.temperature, top_p=req.top_p,
                    repetition_penalty=req.repetition_penalty,
                ):
                    loop.call_soon_threadsafe(q.put_nowait, ("token", tok, arm_info))
                loop.call_soon_threadsafe(q.put_nowait, ("done", None, None))
            except Exception as e:
                loop.call_soon_threadsafe(q.put_nowait, ("error", str(e), None))

        loop.run_in_executor(None, run)
        while True:
            kind, tok, info = await q.get()
            if kind == "done":
                yield f"data: {json.dumps({'type':'done','top_arms':engine.last_top_arms})}\n\n"
                break
            elif kind == "error":
                yield f"data: {json.dumps({'type':'error','msg':tok})}\n\n"
                break
            else:
                clean = tok.replace("<|endoftext|>", "")
                yield f"data: {json.dumps({'type':'token','token':clean,'arm_info':info if req.show_arm_weights else {}})}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("titan_server:app", host="0.0.0.0", port=8000, reload=False)
