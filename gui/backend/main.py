"""Charvogen — Character Voice Generator API server.

Usage:
    python -m gui.backend [--turbo] [--host 0.0.0.0] [--port 8910]
"""

from __future__ import annotations

import argparse
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .state import app_state, PROJECT_ROOT
from .routes import sae, vae, mel, audio, tts, annotations, project


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: load models
    app_state.load_models()
    # Start TTS background worker
    worker_task = asyncio.create_task(tts.tts_worker())
    yield
    # Shutdown
    worker_task.cancel()


app = FastAPI(title="Charvogen", lifespan=lifespan)

# CORS for dev (Vite dev server on :5173)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API routes
app.include_router(sae.router)
app.include_router(vae.router)
app.include_router(mel.router)
app.include_router(audio.router)
app.include_router(tts.router)
app.include_router(annotations.router)
app.include_router(project.router)

# WebSocket
app.websocket("/ws/tts")(tts.websocket_endpoint)

# Emotion columns endpoint
@app.get("/api/emotions")
async def get_emotion_columns():
    from models.constants import EMOTION_COLUMNS, METRIC_COLUMNS
    return {"emotions": EMOTION_COLUMNS, "metrics": METRIC_COLUMNS}


# Serve built frontend as static files (production mode)
_DIST = PROJECT_ROOT / "frontend" / "dist"
if _DIST.exists():
    app.mount("/", StaticFiles(directory=str(_DIST), html=True), name="static")


def main():
    parser = argparse.ArgumentParser(description="Charvogen API server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8910)
    parser.add_argument("--turbo", action="store_true",
                        help="Use ChatterboxTurboTTS instead of ChatterboxTTS")
    args = parser.parse_args()

    # Set before uvicorn imports the app (lifespan reads this)
    app_state.use_turbo = args.turbo

    import uvicorn
    uvicorn.run(
        "gui.backend.main:app",
        host=args.host,
        port=args.port,
        reload=False,
    )


if __name__ == "__main__":
    main()
