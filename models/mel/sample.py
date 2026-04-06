"""Mel spectrogram generation and vocoding — inference only.

Extracted from mel_train/sample.py. CLI / dataset code removed.
"""

import torch

from .diffusion import GaussianDiffusion
from .model import MelDenoiser


def load_model_from_checkpoint(checkpoint_path, device):
    """Load MelDenoiser and GaussianDiffusion from a checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    model_config = config['model']

    model = MelDenoiser(
        mel_channels=model_config['mel_channels'],
        spk_dim=model_config['spk_dim'],
        ve_dim=model_config['ve_dim'],
        channels=tuple(model_config['channels']),
        num_mid_blocks=model_config['num_mid_blocks'],
        n_blocks=model_config['n_blocks'],
        num_heads=model_config['num_heads'],
        attention_head_dim=model_config['attention_head_dim'],
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    diffusion = GaussianDiffusion(
        timesteps=config['timesteps'],
    ).to(device)

    print(f"Loaded model from step {checkpoint['step']}, "
          f"best_val_loss={checkpoint.get('best_val_loss', 'N/A')}")
    return model, diffusion, config


def generate_mel(model, diffusion, camplus, voice_encoder, output_frames, device,
                 ddim_steps=50, guidance_scale=3.5, eta=0.0):
    """Generate a mel spectrogram from speaker embeddings.

    Args:
        model: Trained MelDenoiser.
        diffusion: GaussianDiffusion instance.
        camplus: (80,) projected camplus embedding.
        voice_encoder: (256,) voice encoder embedding.
        output_frames: Number of output mel frames.
        device: Torch device.

    Returns:
        mel: (80, T) generated mel spectrogram.
    """
    camplus = camplus.unsqueeze(0).to(device)
    voice_encoder = voice_encoder.unsqueeze(0).to(device)

    cond = {'camplus': camplus, 'voice_encoder': voice_encoder}
    shape = (1, 80, output_frames)
    mask = torch.ones(1, 1, output_frames, device=device)

    mel = diffusion.ddim_sample(
        model, shape, cond, mask=mask,
        n_steps=ddim_steps, eta=eta, guidance_scale=guidance_scale,
    )

    return mel.squeeze(0)


def generate_mel_sdedit(model, diffusion, source_mel, source_mask, camplus, voice_encoder,
                        device, strength=0.3, ddim_steps=50, guidance_scale=3.5, eta=0.0):
    """SDEdit: noise a source mel and denoise with target speaker conditioning.

    Args:
        source_mel: (80, T) source mel spectrogram.
        source_mask: (1, T) binary mask for source mel.
        camplus: (80,) target speaker camplus embedding.
        voice_encoder: (256,) target speaker voice encoder embedding.
        strength: 0-1, fraction of diffusion schedule to noise through.

    Returns:
        mel: (80, T) generated mel spectrogram.
    """
    source_mel = source_mel.unsqueeze(0).to(device)
    source_mask = source_mask.unsqueeze(0).to(device)
    camplus = camplus.unsqueeze(0).to(device)
    voice_encoder = voice_encoder.unsqueeze(0).to(device)

    t_start = int(strength * (diffusion.num_timesteps - 1))
    t_start = max(1, min(t_start, diffusion.num_timesteps - 1))

    cond = {'camplus': camplus, 'voice_encoder': voice_encoder}

    mel = diffusion.ddim_sample_from_mel(
        model, source_mel, cond, mask=source_mask,
        t_start=t_start, n_steps=ddim_steps, eta=eta, guidance_scale=guidance_scale,
    )

    return mel.squeeze(0)


def load_vocoder(device):
    """Load HiFTGenerator vocoder from S3Gen weights."""
    from safetensors.torch import load_file
    from huggingface_hub import hf_hub_download
    from chatterbox.models.s3gen import S3Gen

    ckpt_path = hf_hub_download(repo_id="ResembleAI/chatterbox", filename="s3gen.safetensors")
    s3gen = S3Gen()
    s3gen.load_state_dict(load_file(ckpt_path), strict=False)
    s3gen.to(device).eval()

    return s3gen.mel2wav


def mel_to_audio(mel, vocoder, device):
    """Convert mel spectrogram to audio using HiFTGenerator vocoder.

    Args:
        mel: (80, T) mel spectrogram tensor.
        vocoder: HiFTGenerator instance.

    Returns:
        waveform: (samples,) numpy array at 24kHz.
    """
    with torch.no_grad():
        mel_input = mel.unsqueeze(0).to(device)
        audio, _ = vocoder.inference(mel_input)
        audio = audio.squeeze().cpu().numpy()

    return audio
