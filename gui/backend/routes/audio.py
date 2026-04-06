"""Audio routes: upload, analyze reference, extract embeddings/mel."""

import uuid

import librosa
import numpy as np
from fastapi import APIRouter, UploadFile, File, HTTPException

from ..state import app_state
from ..api_types import (
    AudioUploadResponse,
    ExtractEmbeddingsResponse,
    ExtractMelResponse,
    AnalyzeReferenceResponse,
    T3TokensResponse,
)
from ..models import extract_embeddings_from_audio, analyze_reference

router = APIRouter(prefix="/api/audio", tags=["audio"])


def _is_wem(contents: bytes) -> bool:
    """Detect Wwise .wem files.

    WEM can be RIFF-based (but with Wwise-specific codecs like Vorbis/Opus
    that ffmpeg can't decode) or RIFX (big-endian).  We peek at the RIFF
    sub-format: standard WAV has 'WAVE' at offset 8, but Wwise WEMs that
    *look* like RIFF still need vgmstream because they use custom codecs.
    We let vgmstream handle any RIFF that isn't plain PCM/IEEE WAV.

    Non-RIFF WEMs (e.g. Ogg-wrapped Wwise Vorbis) also exist; the upload
    filename suffix is checked as a fallback in the caller.
    """
    if len(contents) < 16:
        return False
    # RIFX (big-endian Wwise) is always WEM
    if contents[:4] == b"RIFX":
        return True
    # RIFF + non-WAVE subtype
    if contents[:4] == b"RIFF" and contents[8:12] != b"WAVE":
        return True
    return False


def _transcode_wem(contents: bytes) -> tuple[np.ndarray, int]:
    """Transcode WEM bytes to WAV via vgmstream-cli, then load with librosa."""
    import shutil
    import subprocess
    import tempfile
    import os

    from ..state import PROJECT_ROOT
    project_root = str(PROJECT_ROOT.parent)  # charvogen/
    exe = (
        shutil.which("vgmstream-cli")
        or shutil.which("vgmstream_cli")
        or next(
            (p for p in [
                os.path.join(project_root, "vgmstream-cli.exe"),
                os.path.join(project_root, "vgmstream-cli"),
            ] if os.path.isfile(p)),
            None,
        )
    )
    if exe is None:
        raise HTTPException(
            422,
            "vgmstream-cli not found on PATH — required for .wem playback. "
            "Install from https://vgmstream.org/",
        )

    fd_in, in_path = tempfile.mkstemp(suffix=".wem")
    out_path = in_path.replace(".wem", ".wav")
    try:
        os.write(fd_in, contents)
        os.close(fd_in)
        result = subprocess.run(
            [exe, "-o", out_path, in_path],
            capture_output=True, timeout=30,
        )
        if result.returncode != 0:
            detail = result.stderr.decode(errors="replace")[:500]
            raise HTTPException(422, f"vgmstream failed: {detail}")
        audio_np, sr = librosa.load(out_path, sr=None, mono=True)
    finally:
        for p in (in_path, out_path):
            try:
                os.unlink(p)
            except OSError:
                pass

    return audio_np.astype(np.float32), int(sr)


def _load_audio_from_upload(
    contents: bytes, filename: str = "",
) -> tuple[np.ndarray, int]:
    """Load audio bytes into (mono float32 ndarray, sample_rate).

    Writes to a temp file so librosa/ffmpeg can handle arbitrary formats.
    Uses vgmstream-cli for .wem files (Wwise game audio).
    """
    import tempfile
    import os

    # WEM detection: magic bytes or file extension fallback
    if _is_wem(contents) or filename.lower().endswith(".wem"):
        return _transcode_wem(contents)

    # Determine suffix from magic bytes for better ffmpeg detection
    suffix = ".bin"
    if contents[:4] == b"fLaC":
        suffix = ".flac"
    elif contents[:4] == b"RIFF":
        suffix = ".wav"
    elif contents[:3] == b"ID3" or contents[:2] == b"\xff\xfb":
        suffix = ".mp3"
    elif contents[:4] == b"OggS":
        suffix = ".ogg"

    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    try:
        os.write(fd, contents)
        os.close(fd)
        # librosa uses soundfile for wav/flac, audioread+ffmpeg for others
        audio_np, sr = librosa.load(tmp_path, sr=None, mono=True)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    return audio_np.astype(np.float32), int(sr)


@router.post("/upload", response_model=AudioUploadResponse)
async def upload_audio(file: UploadFile = File(...)):
    contents = await file.read()
    audio_np, sr = _load_audio_from_upload(contents, file.filename or "")
    duration_s = len(audio_np) / sr

    audio_id = uuid.uuid4().hex[:12]
    app_state.uploaded_audio[audio_id] = (audio_np, sr)

    return AudioUploadResponse(audio_id=audio_id, duration_s=duration_s)


@router.post("/analyze", response_model=AnalyzeReferenceResponse)
async def analyze_audio(file: UploadFile = File(...)):
    """Upload + full analysis in one shot: VE, CP, SAE, emotions, mel."""
    contents = await file.read()
    audio_np, sr = _load_audio_from_upload(contents, file.filename or "")

    result = analyze_reference(app_state, audio_np, sr)

    return AnalyzeReferenceResponse(
        audio_id=result["audio_id"],
        ve=result["ve"],
        cp=result["cp"],
        latent=result["latent"],
        ve_activations=result["ve_activations"],
        cp_activations=result["cp_activations"],
        emotions=result["emotions"],
        mel=result["mel"],
    )


@router.post("/{audio_id}/t3-tokens", response_model=T3TokensResponse)
async def extract_t3_tokens(audio_id: str):
    if audio_id not in app_state.uploaded_audio:
        raise HTTPException(404, f"Audio {audio_id} not found")
    audio_np, audio_sr = app_state.uploaded_audio[audio_id]
    from chatterbox.models.s3tokenizer import S3_SR
    ref_16k = librosa.resample(audio_np, orig_sr=audio_sr, target_sr=S3_SR)
    ref_16k = ref_16k[:app_state.tts.ENC_COND_LEN]
    plen = app_state.tts.t3.hp.speech_cond_prompt_len
    tokens, _ = app_state.tts.s3gen.tokenizer.forward([ref_16k], max_len=plen)
    return T3TokensResponse(tokens=tokens.squeeze().tolist())


@router.post("/extract-embeddings", response_model=ExtractEmbeddingsResponse)
async def extract_embeddings(audio_id: str):
    if audio_id not in app_state.uploaded_audio:
        raise HTTPException(404, f"Audio {audio_id} not found")

    audio_np, sr = app_state.uploaded_audio[audio_id]
    result = extract_embeddings_from_audio(app_state, audio_np, sr)

    return ExtractEmbeddingsResponse(
        ve=result["ve"],
        cp_80=result["cp_80"],
        cp_192=result["cp_192"],
    )


@router.post("/extract-mel", response_model=ExtractMelResponse)
async def extract_mel(audio_id: str):
    if audio_id not in app_state.uploaded_audio:
        raise HTTPException(404, f"Audio {audio_id} not found")

    audio_np, sr = app_state.uploaded_audio[audio_id]

    from chatterbox.models.s3gen import S3GEN_SR
    from chatterbox.models.s3gen.utils.mel import mel_spectrogram
    import torch

    if sr != S3GEN_SR:
        audio_np = librosa.resample(audio_np, orig_sr=sr, target_sr=S3GEN_SR)

    mel = mel_spectrogram(torch.from_numpy(audio_np).float())  # (1, 80, T)
    mel = mel.squeeze(0)  # (80, T)

    return ExtractMelResponse(mel=mel.tolist())
