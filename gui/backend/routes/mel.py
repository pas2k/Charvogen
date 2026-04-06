"""Mel routes: generate mel spectrogram, convert to audio."""

from fastapi import APIRouter

from ..state import app_state
from ..api_types import (
    MelGenerateRequest, MelGenerateResponse,
    MelToAudioRequest, MelToAudioResponse,
)
from ..models import generate_mel, numpy_audio_to_b64, mel_to_b64_audio

router = APIRouter(prefix="/api/mel", tags=["mel"])


@router.post("/generate", response_model=MelGenerateResponse)
async def mel_generate(req: MelGenerateRequest):
    mel_np, audio_np = generate_mel(
        app_state,
        req.ve,
        req.cp,
        n_frames=req.n_frames,
        ddim_steps=req.ddim_steps,
        guidance_scale=req.guidance_scale,
        seed_strength=req.seed_strength,
    )
    audio_b64 = numpy_audio_to_b64(audio_np)
    mel_list = mel_np.tolist()
    return MelGenerateResponse(mel=mel_list, audio_b64=audio_b64)


@router.post("/to-audio", response_model=MelToAudioResponse)
async def mel_to_audio(req: MelToAudioRequest):
    import numpy as np
    mel_np = np.array(req.mel, dtype=np.float32)
    audio_b64 = mel_to_b64_audio(app_state, mel_np)
    return MelToAudioResponse(audio_b64=audio_b64)
