"""TTS routes: prepare conditionals, enqueue jobs, WebSocket progress."""

from __future__ import annotations

import asyncio
import json
import time
import uuid

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException

from ..state import app_state
from ..api_types import (
    TTSPrepareRequest, TTSPrepareResponse,
    TTSEnqueueRequest, TTSEnqueueResponse,
    TTSJobStatus, TTSResultResponse,
)
from ..models import build_conditionals_from_ve, run_tts_job, numpy_audio_to_b64

router = APIRouter(prefix="/api/tts", tags=["tts"])

# In-memory job tracking
_jobs: dict[str, dict] = {}  # job_id -> {status, text, result_audio_b64, duration_s, error}
_job_order: list[str] = []


@router.post("/prepare-conditionals", response_model=TTSPrepareResponse)
async def prepare_conditionals(req: TTSPrepareRequest):
    cond_id = build_conditionals_from_ve(
        app_state,
        req.ve,
        req.cp,
        mel=req.mel,
        exaggeration=req.exaggeration,
        audio_id=req.audio_id,
        t3_tokens=req.t3_tokens,
    )
    return TTSPrepareResponse(conditionals_id=cond_id)


@router.post("/enqueue", response_model=TTSEnqueueResponse)
async def enqueue_tts(req: TTSEnqueueRequest):
    if req.conditionals_id not in app_state.conditionals_cache:
        raise HTTPException(404, "Conditionals not found or expired")

    job_id = uuid.uuid4().hex[:12]
    job = {
        "status": "queued",
        "text": req.text,
        "result_audio_b64": None,
        "duration_s": None,
        "error": None,
        "request": req,
    }
    _jobs[job_id] = job
    _job_order.append(job_id)

    await app_state.tts_queue.put(job_id)
    position = sum(1 for jid in _job_order if _jobs[jid]["status"] in ("queued", "running"))

    return TTSEnqueueResponse(job_id=job_id, position=position)


@router.get("/queue", response_model=list[TTSJobStatus])
async def get_queue():
    result = []
    for jid in _job_order[-50:]:  # Last 50 jobs
        job = _jobs[jid]
        result.append(TTSJobStatus(
            job_id=jid,
            status=job["status"],
            text=job["text"],
            error=job["error"],
        ))
    return result


@router.get("/result/{job_id}", response_model=TTSResultResponse)
async def get_result(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job["status"] != "done":
        raise HTTPException(400, f"Job status is {job['status']}")
    return TTSResultResponse(
        audio_b64=job["result_audio_b64"],
        duration_s=job["duration_s"],
    )


async def tts_worker():
    """Background worker that processes TTS jobs from the queue."""
    while True:
        job_id = await app_state.tts_queue.get()
        job = _jobs.get(job_id)
        if not job:
            continue

        job["status"] = "running"
        await _broadcast({"type": "job_started", "job_id": job_id})

        try:
            req = job["request"]
            audio_np, duration_s = await run_tts_job(
                app_state,
                job_id,
                req.text,
                req.conditionals_id,
                exaggeration=req.exaggeration,
                cfg_weight=req.cfg_weight,
                temperature=req.temperature,
                repetition_penalty=req.repetition_penalty,
                min_p=req.min_p,
                top_p=req.top_p,
            )
            audio_b64 = numpy_audio_to_b64(audio_np)
            job["status"] = "done"
            job["result_audio_b64"] = audio_b64
            job["duration_s"] = duration_s
            await _broadcast({
                "type": "job_done",
                "job_id": job_id,
                "audio_b64": audio_b64,
                "duration_s": duration_s,
            })
        except Exception as e:
            job["status"] = "error"
            job["error"] = str(e)
            await _broadcast({
                "type": "job_error",
                "job_id": job_id,
                "error": str(e),
            })

        # Broadcast queue update
        await _broadcast({
            "type": "queue_update",
            "queue": [
                {"job_id": jid, "status": _jobs[jid]["status"], "text": _jobs[jid]["text"]}
                for jid in _job_order[-20:]
            ],
        })


async def _broadcast(msg: dict):
    """Send message to all connected WebSocket clients."""
    data = json.dumps(msg)
    dead = set()
    for ws in app_state.ws_connections:
        try:
            await ws.send_text(data)
        except Exception:
            dead.add(ws)
    app_state.ws_connections -= dead


async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for TTS progress updates."""
    await websocket.accept()
    app_state.ws_connections.add(websocket)
    try:
        while True:
            # Keep connection alive; client can send pings
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        app_state.ws_connections.discard(websocket)
