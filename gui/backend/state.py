"""Singleton application state: loaded models, TTS queue, WebSocket connections."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import torch

# Resolve project root (gui/)
_HERE = Path(__file__).resolve().parent
PROJECT_ROOT = _HERE.parent
WEIGHTS_DIR = PROJECT_ROOT / "weights"
ANNOTATIONS_DIR = PROJECT_ROOT / "annotations"
PROJECTS_DIR = PROJECT_ROOT / "projects"

# Ensure dirs exist
ANNOTATIONS_DIR.mkdir(exist_ok=True)
PROJECTS_DIR.mkdir(exist_ok=True)

HF_REPO_ID = "pas2k/charvogen"


def _hf_weight(repo_path: str) -> str:
    """Resolve a weight file: download from HF cache if needed, return local path."""
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=HF_REPO_ID, filename=repo_path)


class AppState:
    """Holds all loaded models and runtime state."""

    def __init__(self):
        self.device: torch.device | None = None
        self.rotsae_ve = None
        self.rotsae_cp = None
        self.sae_ve_dim: int = 128
        self.sae_cp_dim: int = 32
        self.ve_vae = None
        self.cp_vae = None
        self.vae_config: dict = {}
        self.mel_denoiser = None
        self.mel_diffusion = None
        self.mel_config: dict = {}
        self.tts = None
        self.vocoder = None
        self.ve_planes_meta: list[dict] = []
        self.cp_planes_meta: list[dict] = []
        self.ve_hierarchy: dict = {}
        self.cp_hierarchy: dict = {}
        self.ve_rotsae_config: dict = {}
        self.cp_rotsae_config: dict = {}
        self.seed_mel: torch.Tensor | None = None  # (80, T) log-mel of seed audio for SDEdit scaffold

        # TTS queue
        self.tts_queue: asyncio.Queue | None = None
        self.ws_connections: set = set()

        # Caches
        self.conditionals_cache: dict = {}  # id -> (Conditionals, timestamp)
        self.uploaded_audio: dict = {}  # id -> (waveform_np, sr)
        self.use_turbo: bool = False

    def load_models(self):
        """Load all models at startup."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[charvogen] Using device: {self.device}")

        # --- Load RotSAEs (weights from HF, configs shipped locally) ---
        from models.rotsae import load_rotsae_from_checkpoint

        # VE-only RotSAE
        ve_rotsae_path = _hf_weight("weights/rotsae/veonly/latest_rotsae.pt")
        print(f"[charvogen] Loading VE RotSAE from {ve_rotsae_path}...")
        self.rotsae_ve = load_rotsae_from_checkpoint(ve_rotsae_path, self.device)
        self.rotsae_ve.eval()
        with open(WEIGHTS_DIR / "rotsae" / "veonly" / "config.json") as f:
            ve_analysis = json.load(f)
        self.ve_planes_meta = ve_analysis.get("planes", [])
        self.ve_hierarchy = ve_analysis.get("hierarchy", {})
        self.ve_rotsae_config = ve_analysis.get("metadata", {})
        self.sae_ve_dim = self.ve_rotsae_config.get("ve_dim", 128)
        self.sae_cp_dim = self.ve_rotsae_config.get("cp_dim", 32)
        ve_dist_path = WEIGHTS_DIR / "rotsae" / "veonly" / "ve_plane_distributions.json"
        if ve_dist_path.exists():
            with open(ve_dist_path) as f:
                ve_distributions = json.load(f)
            for plane in self.ve_planes_meta:
                plane["angle_distribution"] = ve_distributions.get(str(plane["index"]), {})

        # CP-only RotSAE
        cp_rotsae_path = _hf_weight("weights/rotsae/cponly/latest_rotsae.pt")
        print(f"[charvogen] Loading CP RotSAE from {cp_rotsae_path}...")
        self.rotsae_cp = load_rotsae_from_checkpoint(cp_rotsae_path, self.device)
        self.rotsae_cp.eval()
        with open(WEIGHTS_DIR / "rotsae" / "cponly" / "config.json") as f:
            cp_analysis = json.load(f)
        self.cp_planes_meta = cp_analysis.get("planes", [])
        self.cp_hierarchy = cp_analysis.get("hierarchy", {})
        self.cp_rotsae_config = cp_analysis.get("metadata", {})

        # --- Load ECVAE ---
        ecvae_path = _hf_weight("weights/vmfecvae/latest.pt")
        print(f"[charvogen] Loading ECVAE from {ecvae_path}...")
        from models.ecvae import load_ecvae_from_checkpoint
        self.ve_vae, self.cp_vae, self.vae_config = load_ecvae_from_checkpoint(
            ecvae_path, self.device
        )

        # --- Load Mel Denoiser ---
        mel_path = _hf_weight("weights/mel/latest.pt")
        print(f"[charvogen] Loading MelDenoiser from {mel_path}...")
        from models.mel.sample import load_model_from_checkpoint as load_mel
        self.mel_denoiser, self.mel_diffusion, self.mel_config = load_mel(
            mel_path, self.device
        )

        # --- Load ChatterboxTTS ---
        if self.use_turbo:
            print("[charvogen] Loading ChatterboxTurboTTS...")
            from chatterbox.tts_turbo import ChatterboxTurboTTS
            self.tts = ChatterboxTurboTTS.from_pretrained(device=str(self.device))
        else:
            print("[charvogen] Loading ChatterboxTTS...")
            from chatterbox.tts import ChatterboxTTS
            self.tts = ChatterboxTTS.from_pretrained(device=str(self.device))

        # Patch flow.inference to handle pre-projected 80d embeddings
        self._patch_flow_inference()

        # Reuse s3gen vocoder for mel preview
        self.vocoder = self.tts.s3gen.mel2wav

        # --- Load seed mel for SDEdit ---
        seed_path = _hf_weight("weights/3025-12971-0001.flac")
        print(f"[charvogen] Loading seed audio from {seed_path}...")
        import librosa
        from chatterbox.models.s3gen.utils.mel import mel_spectrogram

        seed_wav, _ = librosa.load(seed_path, sr=24000, mono=True)
        seed_mel = mel_spectrogram(seed_wav)  # (1, 80, T)
        self.seed_mel = seed_mel.squeeze(0).to(self.device)  # (80, T)
        print(f"[charvogen] Seed mel shape: {self.seed_mel.shape}")

        # Init TTS queue
        self.tts_queue = asyncio.Queue()

        print("[charvogen] All models loaded successfully!")

    def _patch_flow_inference(self):
        """Monkey-patch CausalMaskedDiffWithXvec.inference to skip
        spk_embed_affine_layer when embedding is already 80-d.

        The charvogen VAE pipeline produces 80-d CP embeddings that have
        already been projected. The stock turbo flow.inference() always
        runs embedding through Linear(192->80), which crashes on 80-d
        input. This patch adds a shape guard identical to the one in the
        user-modified chatterbox (non-turbo) flow.py.
        """
        import types
        import logging
        import torch
        import torch.nn.functional as F
        from chatterbox.models.s3gen.utils.mask import make_pad_mask
        import chatterbox.models.s3gen.flow as _flow_mod

        _repeat_batch_dim = _flow_mod._repeat_batch_dim
        _logger = logging.getLogger("charvogen.flow_patch")

        @torch.inference_mode()
        def _patched_inference(
            self_flow,
            token,
            token_len,
            prompt_token,
            prompt_token_len,
            prompt_feat,
            prompt_feat_len,
            embedding,
            finalize,
            n_timesteps=10,
            noised_mels=None,
            meanflow=False,
        ):
            B = token.size(0)

            # --- PATCHED: skip projection if embedding is already 80-d ---
            embedding = torch.atleast_2d(embedding)
            if embedding.shape[-1] == self_flow.output_size:
                pass  # already projected (from charvogen VAE)
            else:
                embedding = F.normalize(embedding, dim=1)
                embedding = self_flow.spk_embed_affine_layer(embedding)

            prompt_token = _repeat_batch_dim(prompt_token, B, ndim=2)
            prompt_token_len = _repeat_batch_dim(prompt_token_len, B, ndim=1)
            prompt_feat = _repeat_batch_dim(prompt_feat, B, ndim=3)
            prompt_feat_len = _repeat_batch_dim(prompt_feat_len, B, ndim=1)
            embedding = _repeat_batch_dim(embedding, B, ndim=2)

            token, token_len = (
                torch.concat([prompt_token, token], dim=1),
                prompt_token_len + token_len,
            )
            mask = (~make_pad_mask(token_len)).unsqueeze(-1).to(embedding)

            if (token >= self_flow.vocab_size).any():
                _logger.error(
                    f"{token.max()}>{self_flow.vocab_size}\n"
                    "out-of-range special tokens found in flow, fix inputs!"
                )
            token = self_flow.input_embedding(token.long()) * mask

            h, h_masks = self_flow.encoder(token, token_len)
            if finalize is False:
                h = h[:, :-self_flow.pre_lookahead_len * self_flow.token_mel_ratio]

            h_lengths = h_masks.sum(dim=-1).squeeze(dim=-1)
            mel_len1 = prompt_feat.shape[1]
            mel_len2 = h.shape[1] - prompt_feat.shape[1]
            h = self_flow.encoder_proj(h)

            conds = torch.zeros(
                [B, mel_len1 + mel_len2, self_flow.output_size],
                device=token.device,
            ).to(h.dtype)
            conds[:, :mel_len1] = prompt_feat
            conds = conds.transpose(1, 2)

            mask = (~make_pad_mask(h_lengths)).unsqueeze(1).to(h)
            if mask.shape[0] != B:
                mask = mask.repeat(B, 1, 1)

            feat, _ = self_flow.decoder(
                mu=h.transpose(1, 2).contiguous(),
                mask=mask,
                spks=embedding,
                cond=conds,
                n_timesteps=n_timesteps,
                noised_mels=noised_mels,
                meanflow=meanflow,
            )
            feat = feat[:, :, mel_len1:]
            assert feat.shape[2] == mel_len2
            return feat, None

        flow = self.tts.s3gen.flow
        flow.inference = types.MethodType(_patched_inference, flow)
        print("[charvogen] Patched flow.inference to skip projection for 80-d embeddings")


# Module-level singleton
app_state = AppState()
