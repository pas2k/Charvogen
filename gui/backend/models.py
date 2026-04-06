"""Model inference wrappers — all the actual computation lives here."""

from __future__ import annotations

import base64
import io
import uuid
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from .state import AppState


def _all_columns():
    """Return all 61 conditioning columns (55 emotions + 6 metrics)."""
    from models.constants import EMOTION_COLUMNS, METRIC_COLUMNS
    return EMOTION_COLUMNS + METRIC_COLUMNS


def build_emotion_tensors(
    emotions_dict: dict[str, float],
    mask_dict: dict[str, int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build emotion and mask tensors from dicts (all 61 columns)."""
    columns = _all_columns()
    emotion_dim = len(columns)
    emo = torch.zeros(1, emotion_dim, device=device)
    mask = torch.zeros(1, emotion_dim, device=device)

    for name, val in emotions_dict.items():
        if name in columns:
            idx = columns.index(name)
            emo[0, idx] = val

    for name, val in mask_dict.items():
        if name in columns:
            idx = columns.index(name)
            mask[0, idx] = float(val)

    return emo, mask


@torch.inference_mode()
def full_pipeline(
    state: AppState,
    ve_features: dict[str, float],
    cp_features: dict[str, float],
    emotions_dict: dict[str, float],
    mask_dict: dict[str, int],
) -> tuple[list[float], list[float], list[float]]:
    """Dual RotSAE features + emotions -> VE (256d) + CP (80d).

    Returns (latent, ve, cp) as Python lists.
    """
    # Build VE h vector -> decode
    h_ve = torch.zeros(1, state.rotsae_ve.n_planes, device=state.device)
    for idx_str, val in ve_features.items():
        idx = int(idx_str)
        if 0 <= idx < state.rotsae_ve.n_planes:
            h_ve[0, idx] = val
    ve_latent = state.rotsae_ve.decode(h_ve)  # (1, ve_dim+cp_dim)
    ve_mu = ve_latent[:, :state.sae_ve_dim]   # (1, 128)

    # Build CP h vector -> decode
    h_cp = torch.zeros(1, state.rotsae_cp.n_planes, device=state.device)
    for idx_str, val in cp_features.items():
        idx = int(idx_str)
        if 0 <= idx < state.rotsae_cp.n_planes:
            h_cp[0, idx] = val
    cp_latent = state.rotsae_cp.decode(h_cp)  # (1, ve_dim+cp_dim)
    cp_mu = cp_latent[:, state.sae_ve_dim:]   # (1, 32)

    latent = torch.cat([ve_mu, cp_mu], dim=-1)  # (1, 160)

    # Build emotion tensors
    emo, mask = build_emotion_tensors(emotions_dict, mask_dict, state.device)

    # Decode through VAE
    ve = state.ve_vae.decode(ve_mu, emo * mask, mask)       # (1, 256)
    cp = state.cp_vae.decode(cp_mu, ve, emo * mask, mask)   # (1, 80)

    return (
        latent.squeeze(0).cpu().tolist(),
        ve.squeeze(0).cpu().tolist(),
        cp.squeeze(0).cpu().tolist(),
    )


@torch.inference_mode()
def random_k(
    state: AppState,
    ve_features: dict[str, float],
    cp_features: dict[str, float],
    emotions_dict: dict[str, float],
    mask_dict: dict[str, int],
    kappa: float = 100.0,
    subspace: str = "all",
) -> dict:
    """Perturb current voice via vMF sampling, interpret back to SAE + emotions.

    Args:
        subspace: "all" | "ve" | "cp" — which latent subspace(s) to perturb.

    Pipeline:
      1. SAE decode → ve_mu (128d), cp_mu (32d)
      2. vMF sample around selected mu(s)
      3. Inverse RotSAE: perturbed latent → SAE activations
      4. VAE decode perturbed latent → embeddings
      5. VAE encode embeddings → emotion predictions
    """
    from models.vmf import VonMisesFisher

    # Step 1: SAE decode to get current latent
    h_ve = torch.zeros(1, state.rotsae_ve.n_planes, device=state.device)
    for idx_str, val in ve_features.items():
        idx = int(idx_str)
        if 0 <= idx < state.rotsae_ve.n_planes:
            h_ve[0, idx] = val
    ve_latent = state.rotsae_ve.decode(h_ve)
    ve_mu = ve_latent[:, :state.sae_ve_dim]   # (1, 128)

    h_cp = torch.zeros(1, state.rotsae_cp.n_planes, device=state.device)
    for idx_str, val in cp_features.items():
        idx = int(idx_str)
        if 0 <= idx < state.rotsae_cp.n_planes:
            h_cp[0, idx] = val
    cp_latent = state.rotsae_cp.decode(h_cp)
    cp_mu = cp_latent[:, state.sae_ve_dim:]   # (1, 32)

    # Step 2: vMF perturbation (only on selected subspace)
    # Normalize mu before vMF — RotSAE decode slices aren't unit-norm,
    # and VonMisesFisher's Householder reflection assumes ‖mu‖=1.
    kappa_t = torch.tensor([[kappa]], device=state.device)

    if subspace in ("all", "ve"):
        ve_perturbed = F.normalize(VonMisesFisher(F.normalize(ve_mu, dim=-1), kappa_t).rsample(), dim=-1)
    else:
        ve_perturbed = ve_mu

    if subspace in ("all", "cp"):
        cp_perturbed = F.normalize(VonMisesFisher(F.normalize(cp_mu, dim=-1), kappa_t).rsample(), dim=-1)
    else:
        cp_perturbed = cp_mu

    perturbed_latent = torch.cat([ve_perturbed, cp_perturbed], dim=-1)  # (1, 160)

    # Step 3: Inverse RotSAE → SAE activations
    _, h_ve_new, _ = state.rotsae_ve(perturbed_latent)
    h_ve_sq = h_ve_new.squeeze(0)
    ve_active = h_ve_sq.abs() > 1e-8
    new_ve_activations = {str(i.item()): round(h_ve_sq[i].item(), 4)
                          for i in ve_active.nonzero(as_tuple=True)[0]}

    _, h_cp_new, _ = state.rotsae_cp(perturbed_latent)
    h_cp_sq = h_cp_new.squeeze(0)
    cp_active = h_cp_sq.abs() > 1e-8
    new_cp_activations = {str(i.item()): round(h_cp_sq[i].item(), 4)
                          for i in cp_active.nonzero(as_tuple=True)[0]}

    # Step 4: Decode perturbed latent through VAE to get embeddings
    emo, mask = build_emotion_tensors(emotions_dict, mask_dict, state.device)
    ve_embed = state.ve_vae.decode(ve_perturbed, emo * mask, mask)     # (1, 256)
    cp_embed = state.cp_vae.decode(cp_perturbed, ve_embed, emo * mask, mask)  # (1, 80)

    # Step 5: Predict emotions from perturbed embeddings
    columns = _all_columns()
    emotion_dim = len(columns)
    zeros_emo = torch.zeros(1, emotion_dim, device=state.device)
    zeros_mask = torch.zeros(1, emotion_dim, device=state.device)

    _, _, ve_emo_pred = state.ve_vae.encode(ve_embed, zeros_emo, zeros_mask)
    _, _, cp_emo_pred = state.cp_vae.encode(cp_embed, ve_embed, zeros_emo, zeros_mask)
    avg_emo_pred = (ve_emo_pred + cp_emo_pred) / 2.0

    emo_values = avg_emo_pred.squeeze(0).cpu()
    emotions_out = {name: round(emo_values[i].item(), 4) for i, name in enumerate(columns)}

    return {
        "ve_activations": new_ve_activations,
        "cp_activations": new_cp_activations,
        "emotions": emotions_out,
        "latent": perturbed_latent.squeeze(0).cpu().tolist(),
        "ve": ve_embed.squeeze(0).cpu().tolist(),
        "cp": cp_embed.squeeze(0).cpu().tolist(),
    }


@torch.inference_mode()
def random_k_batch(
    state: AppState,
    ve_features: dict[str, float],
    cp_features: dict[str, float],
    emotions_dict: dict[str, float],
    mask_dict: dict[str, int],
    kappa: float = 100.0,
    subspace: str = "all",
    n_samples: int = 5,
) -> list[dict]:
    """Generate n_samples vMF perturbations from the same base state."""
    from models.vmf import VonMisesFisher

    # Decode current SAE → latent (same as random_k)
    h_ve = torch.zeros(1, state.rotsae_ve.n_planes, device=state.device)
    for idx_str, val in ve_features.items():
        idx = int(idx_str)
        if 0 <= idx < state.rotsae_ve.n_planes:
            h_ve[0, idx] = val
    ve_mu = state.rotsae_ve.decode(h_ve)[:, :state.sae_ve_dim]

    h_cp = torch.zeros(1, state.rotsae_cp.n_planes, device=state.device)
    for idx_str, val in cp_features.items():
        idx = int(idx_str)
        if 0 <= idx < state.rotsae_cp.n_planes:
            h_cp[0, idx] = val
    cp_mu = state.rotsae_cp.decode(h_cp)[:, state.sae_ve_dim:]

    # Normalize mu before vMF — RotSAE decode slices aren't unit-norm
    kappa_t = torch.tensor([[kappa]], device=state.device)
    ve_mu_n = F.normalize(ve_mu, dim=-1)
    cp_mu_n = F.normalize(cp_mu, dim=-1)
    columns = _all_columns()
    emotion_dim = len(columns)
    zeros_emo = torch.zeros(1, emotion_dim, device=state.device)
    zeros_mask = torch.zeros(1, emotion_dim, device=state.device)

    samples = []
    for _ in range(n_samples):
        ve_p = F.normalize(VonMisesFisher(ve_mu_n, kappa_t).rsample(), dim=-1) if subspace in ("all", "ve") else ve_mu
        cp_p = F.normalize(VonMisesFisher(cp_mu_n, kappa_t).rsample(), dim=-1) if subspace in ("all", "cp") else cp_mu
        perturbed = torch.cat([ve_p, cp_p], dim=-1)

        # Inverse RotSAE → activations
        _, h_ve_new, _ = state.rotsae_ve(perturbed)
        h_sq = h_ve_new.squeeze(0)
        ve_acts = {str(i.item()): round(h_sq[i].item(), 4)
                   for i in (h_sq.abs() > 1e-8).nonzero(as_tuple=True)[0]}

        _, h_cp_new, _ = state.rotsae_cp(perturbed)
        h_sq = h_cp_new.squeeze(0)
        cp_acts = {str(i.item()): round(h_sq[i].item(), 4)
                   for i in (h_sq.abs() > 1e-8).nonzero(as_tuple=True)[0]}

        # Decode → embeddings
        emo, mask = build_emotion_tensors(emotions_dict, mask_dict, state.device)
        ve_embed = state.ve_vae.decode(ve_p, emo * mask, mask)
        cp_embed = state.cp_vae.decode(cp_p, ve_embed, emo * mask, mask)

        # Predict emotions
        _, _, ve_emo_pred = state.ve_vae.encode(ve_embed, zeros_emo, zeros_mask)
        _, _, cp_emo_pred = state.cp_vae.encode(cp_embed, ve_embed, zeros_emo, zeros_mask)
        avg_emo = ((ve_emo_pred + cp_emo_pred) / 2.0).squeeze(0).cpu()
        emotions_out = {name: round(avg_emo[j].item(), 4) for j, name in enumerate(columns)}

        samples.append({
            "ve_activations": ve_acts,
            "cp_activations": cp_acts,
            "emotions": emotions_out,
            "latent": perturbed.squeeze(0).cpu().tolist(),
            "ve": ve_embed.squeeze(0).cpu().tolist(),
            "cp": cp_embed.squeeze(0).cpu().tolist(),
        })

    return samples


@torch.inference_mode()
def generate_mel(
    state: AppState,
    ve: list[float],
    cp: list[float],
    n_frames: int = 256,
    ddim_steps: int = 50,
    guidance_scale: float = 3.5,
    seed_strength: float = 0.075,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate mel spectrogram from VE+CP embeddings via SDEdit.

    Uses the preloaded seed mel as a structural scaffold. seed_strength
    controls how much of the seed's structure is preserved:
      0.0 = pure noise (no seed), 1.0 = no change (keep seed exactly).
    Default 0.075 barely affects the output but avoids noisy artefacts.

    Returns (mel_np, audio_np) where mel is (80, T) and audio is (samples,).
    """
    from models.mel.sample import (
        generate_mel as _generate_mel,
        generate_mel_sdedit as _generate_mel_sdedit,
        mel_to_audio as _mel_to_audio,
    )

    ve_t = torch.tensor(ve, device=state.device)
    cp_t = torch.tensor(cp, device=state.device)

    if state.seed_mel is not None and seed_strength > 0:
        # Prepare seed mel: crop or pad to n_frames
        seed = state.seed_mel  # (80, T_seed)
        T_seed = seed.shape[1]
        if T_seed >= n_frames:
            source_mel = seed[:, :n_frames]
        else:
            source_mel = F.pad(seed, (0, n_frames - T_seed))
        source_mask = torch.ones(1, n_frames, device=state.device)

        # SDEdit strength = 1 - seed_strength
        # (strength=1 => full noise => no seed; strength=0 => keep seed)
        sdedit_strength = 1.0 - seed_strength

        mel = _generate_mel_sdedit(
            state.mel_denoiser,
            state.mel_diffusion,
            source_mel, source_mask,
            cp_t, ve_t,
            state.device,
            strength=sdedit_strength,
            ddim_steps=ddim_steps,
            guidance_scale=guidance_scale,
        )
    else:
        mel = _generate_mel(
            state.mel_denoiser,
            state.mel_diffusion,
            cp_t, ve_t,
            n_frames,
            state.device,
            ddim_steps=ddim_steps,
            guidance_scale=guidance_scale,
        )

    audio = _mel_to_audio(mel, state.vocoder, state.device)

    return mel.cpu().numpy(), audio


def numpy_audio_to_b64(audio_np: np.ndarray, sr: int = 24000) -> str:
    """Encode numpy audio to base64 WAV string."""
    import soundfile as sf
    buf = io.BytesIO()
    sf.write(buf, audio_np, sr, format="WAV", subtype="PCM_16")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def mel_to_b64_audio(state: AppState, mel_np: np.ndarray) -> str:
    """Convert mel numpy array to base64 WAV string via vocoder."""
    from models.mel.sample import mel_to_audio as _mel_to_audio
    mel_t = torch.tensor(mel_np, device=state.device)
    audio = _mel_to_audio(mel_t, state.vocoder, state.device)
    return numpy_audio_to_b64(audio)


@torch.inference_mode()
def extract_embeddings_from_audio(
    state: AppState,
    audio_np: np.ndarray,
    sr: int,
) -> dict:
    """Extract VE + CP embeddings from audio waveform.

    Returns dict with 've', 'cp_80', 'cp_192' as lists, plus
    's3gen_ref_dict' for TTS conditioning.
    """
    import librosa
    from chatterbox.models.s3tokenizer import S3_SR
    from chatterbox.models.s3gen import S3GEN_SR

    # Resample to needed rates
    if sr != S3GEN_SR:
        s3gen_wav = librosa.resample(audio_np, orig_sr=sr, target_sr=S3GEN_SR)
    else:
        s3gen_wav = audio_np
    if sr != S3_SR:
        ref_16k = librosa.resample(audio_np, orig_sr=sr, target_sr=S3_SR)
    else:
        ref_16k = audio_np

    # Trim lengths
    s3gen_wav = s3gen_wav[:state.tts.DEC_COND_LEN]
    ref_16k = ref_16k[:state.tts.ENC_COND_LEN]

    # S3Gen embeddings (includes camplus 192d)
    s3gen_ref_dict = state.tts.s3gen.embed_ref(s3gen_wav, S3GEN_SR, device=state.device)

    # VE embedding
    ve_embed = torch.from_numpy(
        state.tts.ve.embeds_from_wavs([ref_16k], sample_rate=S3_SR)
    )
    ve_embed = ve_embed.mean(axis=0, keepdim=True).to(state.device)

    # Camplus 192d -> 80d projection via s3gen's spk_embed_affine_layer
    cp_192 = s3gen_ref_dict["embedding"]  # (1, 192)
    cp_192_norm = F.normalize(cp_192, p=2, dim=1)
    cp_80 = state.tts.s3gen.flow.spk_embed_affine_layer(cp_192_norm)  # (1, 80)

    # Speech cond prompt tokens for T3
    t3_cond_prompt_tokens = None
    if plen := state.tts.t3.hp.speech_cond_prompt_len:
        s3_tokzr = state.tts.s3gen.tokenizer
        t3_cond_prompt_tokens, _ = s3_tokzr.forward(
            [ref_16k[:state.tts.ENC_COND_LEN]], max_len=plen
        )
        t3_cond_prompt_tokens = torch.atleast_2d(t3_cond_prompt_tokens).to(state.device)

    return {
        "ve": ve_embed.squeeze(0).cpu().tolist(),
        "cp_80": cp_80.squeeze(0).cpu().tolist(),
        "cp_192": cp_192.squeeze(0).cpu().tolist(),
        "s3gen_ref_dict": s3gen_ref_dict,
        "ve_tensor": ve_embed,
        "t3_cond_prompt_tokens": t3_cond_prompt_tokens,
    }


@torch.inference_mode()
def analyze_reference(
    state: AppState,
    audio_np: np.ndarray,
    sr: int,
) -> dict:
    """Full analysis of reference audio: extract everything the UI needs.

    Pipeline:
      audio → VE (256d), CP_192 (192d), CP_80 (80d)
      VE → ve_vae.encode → ve_mu (128d) + emotion predictions
      CP_80 → cp_vae.encode → cp_mu (32d) + emotion predictions
      [ve_mu, cp_mu] → dual RotSAE.encode → signed angle activations
      audio → mel_spectrogram → mel (80, T)

    Returns dict with all results for populating every frontend panel.
    """
    import librosa
    from chatterbox.models.s3tokenizer import S3_SR
    from chatterbox.models.s3gen import S3GEN_SR
    from chatterbox.models.s3gen.utils.mel import mel_spectrogram

    columns = _all_columns()
    emotion_dim = len(columns)

    # --- Resample ---
    if sr != S3GEN_SR:
        s3gen_wav = librosa.resample(audio_np, orig_sr=sr, target_sr=S3GEN_SR)
    else:
        s3gen_wav = audio_np.copy()
    if sr != S3_SR:
        ref_16k = librosa.resample(audio_np, orig_sr=sr, target_sr=S3_SR)
    else:
        ref_16k = audio_np.copy()

    s3gen_wav = s3gen_wav[:state.tts.DEC_COND_LEN]
    ref_16k = ref_16k[:state.tts.ENC_COND_LEN]

    # --- Extract embeddings ---
    s3gen_ref_dict = state.tts.s3gen.embed_ref(s3gen_wav, S3GEN_SR, device=state.device)

    ve_embed = torch.from_numpy(
        state.tts.ve.embeds_from_wavs([ref_16k], sample_rate=S3_SR)
    )
    ve_embed = ve_embed.mean(axis=0, keepdim=True).to(state.device)  # (1, 256)

    cp_192 = s3gen_ref_dict["embedding"]  # (1, 192)
    cp_192_norm = F.normalize(cp_192, p=2, dim=1)
    cp_80 = state.tts.s3gen.flow.spk_embed_affine_layer(cp_192_norm)  # (1, 80)

    # --- VAE encode (with zero emotions to get clean latents) ---
    zeros_emo = torch.zeros(1, emotion_dim, device=state.device)
    zeros_mask = torch.zeros(1, emotion_dim, device=state.device)

    ve_mu, ve_kappa, ve_emo_pred = state.ve_vae.encode(ve_embed, zeros_emo, zeros_mask)
    cp_mu, cp_kappa, cp_emo_pred = state.cp_vae.encode(cp_80, ve_embed, zeros_emo, zeros_mask)

    # --- Average emotion predictions from both heads ---
    avg_emo_pred = (ve_emo_pred + cp_emo_pred) / 2.0  # (1, emotion_dim)

    # --- RotSAE encode (dual) ---
    latent = torch.cat([ve_mu, cp_mu], dim=-1)  # (1, 160)

    # VE RotSAE
    _, h_ve, _ = state.rotsae_ve(latent)  # (1, n_planes)
    h_ve_sq = h_ve.squeeze(0)
    ve_active = h_ve_sq.abs() > 1e-8  # signed angles
    ve_activations = {str(i.item()): round(h_ve_sq[i].item(), 4)
                      for i in ve_active.nonzero(as_tuple=True)[0]}

    # CP RotSAE
    _, h_cp, _ = state.rotsae_cp(latent)  # (1, n_planes)
    h_cp_sq = h_cp.squeeze(0)
    cp_active = h_cp_sq.abs() > 1e-8
    cp_activations = {str(i.item()): round(h_cp_sq[i].item(), 4)
                      for i in cp_active.nonzero(as_tuple=True)[0]}

    # --- Extract mel spectrogram (S3Gen format) ---
    mel = mel_spectrogram(torch.from_numpy(s3gen_wav).float())  # (1, 80, T)
    mel = mel.squeeze(0)  # (80, T)

    # --- Build emotion values dict ---
    emo_values = avg_emo_pred.squeeze(0).cpu()
    emotions_dict = {}
    for i, name in enumerate(columns):
        emotions_dict[name] = round(emo_values[i].item(), 4)

    # --- Speech cond prompt tokens (for TTS conditioning later) ---
    t3_cond_prompt_tokens = None
    if plen := state.tts.t3.hp.speech_cond_prompt_len:
        s3_tokzr = state.tts.s3gen.tokenizer
        t3_cond_prompt_tokens, _ = s3_tokzr.forward(
            [ref_16k[:state.tts.ENC_COND_LEN]], max_len=plen
        )
        t3_cond_prompt_tokens = torch.atleast_2d(t3_cond_prompt_tokens).to(state.device)

    # --- Cache the s3gen ref dict for TTS (keyed by audio hash) ---
    # Store internally for later use by prepare-conditionals
    audio_id = uuid.uuid4().hex[:12]
    state.uploaded_audio[audio_id] = (audio_np, sr)

    return {
        "audio_id": audio_id,
        # Embeddings
        "ve": ve_embed.squeeze(0).cpu().tolist(),
        "cp": cp_80.squeeze(0).cpu().tolist(),
        # Latents
        "latent": latent.squeeze(0).cpu().tolist(),
        "ve_mu": ve_mu.squeeze(0).cpu().tolist(),
        "cp_mu": cp_mu.squeeze(0).cpu().tolist(),
        # RotSAE activations (dual)
        "ve_activations": ve_activations,
        "cp_activations": cp_activations,
        # Emotions (predicted)
        "emotions": emotions_dict,
        # Mel
        "mel": mel.cpu().tolist(),
    }


@torch.inference_mode()
def build_conditionals_from_ve(
    state: AppState,
    ve: list[float],
    cp: list[float] | None = None,
    mel: list[list[float]] | None = None,
    exaggeration: float = 0.5,
    audio_id: str | None = None,
    t3_tokens: list[int] | None = None,
) -> str:
    """Build T3Cond + S3Gen ref_dict from designed voice components.

    Four independent conditioning sources (see generate_mixed_voice.py):
      1. VE (256d) → T3Cond.speaker_emb  (always from designed voice)
      2. CP (80d)  → s3gen_ref_dict["embedding"]  (overrides camplus)
      3. Mel       → vocode → embed_ref → prompt_feat + prompt_token
      4. T3 tokens → from audio_id upload or seed default

    Returns conditionals_id for caching.
    """
    import time
    import librosa
    if state.use_turbo:
        from chatterbox.tts_turbo import Conditionals
    else:
        from chatterbox.tts import Conditionals
    from chatterbox.models.t3.modules.cond_enc import T3Cond
    from chatterbox.models.s3tokenizer import S3_SR
    from chatterbox.models.s3gen import S3GEN_SR

    ve_t = torch.tensor(ve, device=state.device).unsqueeze(0)  # (1, 256)

    # --- T3 tokens (source 4) — pre-computed or from audio ---
    if t3_tokens is not None:
        t3_tok = torch.tensor(t3_tokens, device=state.device).unsqueeze(0)
    elif audio_id and audio_id in state.uploaded_audio:
        audio_np, audio_sr = state.uploaded_audio[audio_id]
        ref_16k = librosa.resample(audio_np, orig_sr=audio_sr, target_sr=S3_SR)
        ref_16k = ref_16k[:state.tts.ENC_COND_LEN]
        if plen := state.tts.t3.hp.speech_cond_prompt_len:
            s3_tokzr = state.tts.s3gen.tokenizer
            tokens, _ = s3_tokzr.forward([ref_16k], max_len=plen)
            t3_tok = torch.atleast_2d(tokens).to(state.device)
        else:
            raise ValueError("Model has no speech_cond_prompt_len configured")
    else:
        raise ValueError(
            "T3 speech prompt required. Provide t3_tokens or upload audio "
            "in the T3 Prompt panel."
        )

    # --- S3Gen ref_dict (sources 2+3) — requires mel ---
    if mel is None:
        raise ValueError(
            "Mel spectrogram is required. Generate one in the Mel panel first."
        )
    from models.mel.sample import mel_to_audio as _mel_to_audio
    mel_t = torch.tensor(mel, device=state.device)  # (80, T)
    vocoded_audio = _mel_to_audio(mel_t, state.vocoder, state.device)
    vocoded_audio = vocoded_audio[:state.tts.DEC_COND_LEN]
    s3gen_ref_dict = state.tts.s3gen.embed_ref(vocoded_audio, S3GEN_SR, device=state.device)

    # Override embedding with designed CP (80d — flow skips projection for 80d)
    if cp is not None:
        cp_t = torch.tensor(cp, device=state.device).unsqueeze(0)  # (1, 80)
        s3gen_ref_dict["embedding"] = cp_t

    # --- Build conditionals ---
    t3_cond = T3Cond(
        speaker_emb=ve_t,
        cond_prompt_speech_tokens=t3_tok,
        emotion_adv=exaggeration * torch.ones(1, 1, 1),
    ).to(device=state.device)

    conds = Conditionals(t3_cond, s3gen_ref_dict)

    # Cache
    cond_id = uuid.uuid4().hex[:12]
    state.conditionals_cache[cond_id] = (conds, time.time())

    # Cleanup old entries (>30min)
    now = time.time()
    expired = [k for k, (_, t) in state.conditionals_cache.items() if now - t > 1800]
    for k in expired:
        del state.conditionals_cache[k]

    return cond_id


async def run_tts_job(
    state: AppState,
    job_id: str,
    text: str,
    conditionals_id: str,
    exaggeration: float = 0.5,
    cfg_weight: float = 0.5,
    temperature: float = 0.8,
    repetition_penalty: float = 1.2,
    min_p: float = 0.05,
    top_p: float = 1.0,
) -> tuple[np.ndarray, float]:
    """Run TTS generation. Returns (audio_np, duration_s)."""
    import asyncio

    conds, _ = state.conditionals_cache.get(conditionals_id, (None, None))
    if conds is None:
        raise ValueError(f"Conditionals {conditionals_id} not found or expired")

    state.tts.conds = conds

    # Run in thread executor since it's GPU-bound
    loop = asyncio.get_event_loop()
    wav = await loop.run_in_executor(
        None,
        lambda: state.tts.generate(
            text=text,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            min_p=min_p,
            top_p=top_p,
        ),
    )

    audio_np = wav.squeeze().numpy()
    duration_s = len(audio_np) / state.tts.sr
    return audio_np, duration_s
