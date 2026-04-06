"""Emotion-conditioned vMF VAEs — inference only.

Extracted from vecamp_train/train_vmfecvae.py (model classes)
and vecamp_train/encode_decode_ecvae.py (checkpoint loading).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vmf import VonMisesFisher


class vMFVoiceEncoderVAE(nn.Module):
    """vMF VAE for voice_encoder reconstruction conditioned on masked emotions.

    Encoder: [voice_encoder(256) | emotions*mask(56) | mask(56)] -> hidden -> mu, kappa
    Decoder: [z(latent) | emotions*mask(56) | mask(56)] -> hidden -> voice_encoder(256)
    """

    def __init__(self, latent_dim: int = 128, hidden_dim: int = 512,
                 dropout: float = 0.1, voice_encoder_dim: int = 256,
                 emotion_dim: int = 56, deterministic: bool = False,
                 decoder_depth: int = 2, decoder_dropout: float = None,
                 decoder_skip: bool = False):
        super().__init__()
        self.voice_encoder_dim = voice_encoder_dim
        self.emotion_dim = emotion_dim
        self.deterministic = deterministic
        self.latent_dim = latent_dim
        self.kappa_max = 700.0
        self.decoder_skip = decoder_skip
        if decoder_dropout is None:
            decoder_dropout = dropout

        enc_input = voice_encoder_dim + emotion_dim * 2
        dec_input = latent_dim + emotion_dim * 2

        self.encoder = nn.Sequential(
            nn.Linear(enc_input, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.fc_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc_kappa = nn.Linear(hidden_dim, 1)
        self.fc_emotion = nn.Linear(hidden_dim, emotion_dim)

        dec_layers = []
        dec_layers += [nn.Linear(dec_input, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(decoder_dropout)]
        for _ in range(decoder_depth - 1):
            dec_layers += [nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(decoder_dropout)]
        dec_layers.append(nn.Linear(hidden_dim, voice_encoder_dim))
        self.decoder = nn.Sequential(*dec_layers)

        if decoder_skip:
            self.skip_proj = nn.Linear(latent_dim, voice_encoder_dim)

    def encode(self, voice_encoder: torch.Tensor, masked_emotions: torch.Tensor,
               mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.encoder(torch.cat([voice_encoder, masked_emotions, mask], dim=-1))
        mu = F.normalize(self.fc_mean(h), dim=-1, eps=1e-8)
        kappa = F.softplus(self.fc_kappa(h)) + 1.0
        kappa = torch.clamp(kappa, max=self.kappa_max)
        emotion_pred = self.fc_emotion(h)
        return mu, kappa, emotion_pred

    def decode(self, z: torch.Tensor, masked_emotions: torch.Tensor,
               mask: torch.Tensor) -> torch.Tensor:
        raw = self.decoder(torch.cat([z, masked_emotions, mask], dim=-1))
        if self.decoder_skip:
            raw = raw + self.skip_proj(z)
        return F.normalize(F.softplus(raw), dim=-1, eps=1e-8)

    def forward(self, voice_encoder: torch.Tensor, masked_emotions: torch.Tensor,
                mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, kappa, emotion_pred = self.encode(voice_encoder, masked_emotions, mask)

        if self.training and not self.deterministic:
            q_z = VonMisesFisher(mu, kappa)
            z = q_z.rsample()
        else:
            z = mu

        reconstructed = self.decode(z, masked_emotions, mask)
        return reconstructed, mu, kappa, emotion_pred


class vMFCamplusVAE(nn.Module):
    """vMF VAE for camplus reconstruction conditioned on bottlenecked voice_encoder + masked emotions.

    Operates in S3Gen's projected 80d camplus space.
    Encoder: [camplus_80d(80) | ve_bottleneck(16) | emotions*mask(56) | mask(56)] -> hidden -> mu, kappa
    Decoder: [z(latent) | ve_bottleneck(16) | emotions*mask(56) | mask(56)] -> hidden -> camplus_80d(80)
    """

    def __init__(self, latent_dim: int = 128, hidden_dim: int = 512,
                 dropout: float = 0.1, camplus_dim: int = 80,
                 voice_encoder_dim: int = 256, emotion_dim: int = 56,
                 ve_bottleneck_dim: int = 16, deterministic: bool = False):
        super().__init__()
        self.camplus_dim = camplus_dim
        self.voice_encoder_dim = voice_encoder_dim
        self.emotion_dim = emotion_dim
        self.deterministic = deterministic
        self.latent_dim = latent_dim
        self.ve_bottleneck_dim = ve_bottleneck_dim
        self.kappa_max = 700.0

        self.ve_bottleneck = nn.Linear(voice_encoder_dim, ve_bottleneck_dim)

        enc_input = camplus_dim + ve_bottleneck_dim + emotion_dim * 2
        dec_input = latent_dim + ve_bottleneck_dim + emotion_dim * 2

        self.encoder = nn.Sequential(
            nn.Linear(enc_input, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.fc_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc_kappa = nn.Linear(hidden_dim, 1)
        self.fc_emotion = nn.Linear(hidden_dim, emotion_dim)

        self.decoder = nn.Sequential(
            nn.Linear(dec_input, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, camplus_dim),
        )

    def encode(self, camplus: torch.Tensor, voice_encoder: torch.Tensor,
               masked_emotions: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ve_cond = self.ve_bottleneck(voice_encoder.detach())
        h = self.encoder(torch.cat([camplus, ve_cond, masked_emotions, mask], dim=-1))
        mu = F.normalize(self.fc_mean(h), dim=-1, eps=1e-8)
        kappa = F.softplus(self.fc_kappa(h)) + 1.0
        kappa = torch.clamp(kappa, max=self.kappa_max)
        emotion_pred = self.fc_emotion(h)
        return mu, kappa, emotion_pred

    def decode(self, z: torch.Tensor, voice_encoder: torch.Tensor,
               masked_emotions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        ve_cond = self.ve_bottleneck(voice_encoder.detach())
        return self.decoder(torch.cat([z, ve_cond, masked_emotions, mask], dim=-1))

    def forward(self, camplus: torch.Tensor, voice_encoder: torch.Tensor,
                masked_emotions: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, kappa, emotion_pred = self.encode(camplus, voice_encoder, masked_emotions, mask)

        if self.training and not self.deterministic:
            q_z = VonMisesFisher(mu, kappa)
            z = q_z.rsample()
        else:
            z = mu

        reconstructed = self.decode(z, voice_encoder, masked_emotions, mask)
        return reconstructed, mu, kappa, emotion_pred


def load_ecvae_from_checkpoint(path: str, device: torch.device):
    """Load both VAE models from a single ECVAE checkpoint.

    Returns (ve_model, cp_model, config).
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    config = ckpt['config']

    ve_model = vMFVoiceEncoderVAE(
        latent_dim=config['ve_latent_dim'],
        hidden_dim=config['hidden_dim'],
        emotion_dim=config['emotion_dim'],
        decoder_depth=config.get('ve_decoder_depth', 2),
        decoder_dropout=config.get('decoder_dropout'),
        decoder_skip=config.get('decoder_skip', False),
    ).to(device)
    ve_model.load_state_dict(ckpt['ve_model_state_dict'])
    ve_model.eval()

    cp_model = vMFCamplusVAE(
        latent_dim=config['cp_latent_dim'],
        hidden_dim=config['hidden_dim'],
        emotion_dim=config['emotion_dim'],
    ).to(device)
    cp_model.load_state_dict(ckpt['cp_model_state_dict'])
    cp_model.eval()

    return ve_model, cp_model, config
