"""1D UNet Mel Denoiser for DDPM mel spectrogram generation.

Architecture mirrors S3Gen's Decoder (matcha/decoder.py) but adapted:
- Removes `mu` input (no text encoder output)
- Adds voice_encoder FiLM conditioning via TimestepEmbedding.cond_proj_dim
- Input is noisy mel + broadcast camplus → (160, T)
"""

import torch
import torch.nn as nn
from einops import pack, rearrange, repeat

from chatterbox.models.s3gen.matcha.decoder import (
    SinusoidalPosEmb,
    TimestepEmbedding,
    ResnetBlock1D,
    Block1D,
    Downsample1D,
    Upsample1D,
)
from chatterbox.models.s3gen.matcha.transformer import BasicTransformerBlock


class MelDenoiser(nn.Module):
    """1D UNet denoiser that predicts noise from noisy mel + speaker embeddings.

    Conditioning:
    - Camplus (80d): broadcast-concatenated with noisy mel along channel dim
    - Voice encoder (256d): injected via FiLM in the time embedding MLP
    """

    def __init__(
        self,
        mel_channels: int = 80,
        spk_dim: int = 80,
        ve_dim: int = 256,
        channels: tuple = (256, 256),
        num_mid_blocks: int = 12,
        n_blocks: int = 4,
        num_heads: int = 8,
        attention_head_dim: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()

        channels = tuple(channels)
        in_channels = mel_channels + spk_dim  # 80 + 80 = 160

        # Time embedding with voice encoder FiLM
        self.time_embeddings = SinusoidalPosEmb(in_channels)
        time_embed_dim = channels[0] * 4
        self.time_mlp = TimestepEmbedding(
            in_channels=in_channels,
            time_embed_dim=time_embed_dim,
            act_fn="silu",
            cond_proj_dim=ve_dim,
        )

        self.down_blocks = nn.ModuleList([])
        self.mid_blocks = nn.ModuleList([])
        self.up_blocks = nn.ModuleList([])

        # --- Down blocks ---
        output_channel = in_channels
        for i in range(len(channels)):
            input_channel = output_channel
            output_channel = channels[i]
            is_last = i == len(channels) - 1

            resnet = ResnetBlock1D(
                dim=input_channel, dim_out=output_channel, time_emb_dim=time_embed_dim,
            )
            transformer_blocks = nn.ModuleList([
                BasicTransformerBlock(
                    dim=output_channel,
                    num_attention_heads=num_heads,
                    attention_head_dim=attention_head_dim,
                    dropout=dropout,
                    activation_fn="snakebeta",
                )
                for _ in range(n_blocks)
            ])
            downsample = (
                Downsample1D(output_channel)
                if not is_last
                else nn.Conv1d(output_channel, output_channel, 3, padding=1)
            )
            self.down_blocks.append(nn.ModuleList([resnet, transformer_blocks, downsample]))

        # --- Mid blocks ---
        for i in range(num_mid_blocks):
            resnet = ResnetBlock1D(
                dim=channels[-1], dim_out=channels[-1], time_emb_dim=time_embed_dim,
            )
            transformer_blocks = nn.ModuleList([
                BasicTransformerBlock(
                    dim=channels[-1],
                    num_attention_heads=num_heads,
                    attention_head_dim=attention_head_dim,
                    dropout=dropout,
                    activation_fn="snakebeta",
                )
                for _ in range(n_blocks)
            ])
            self.mid_blocks.append(nn.ModuleList([resnet, transformer_blocks]))

        # --- Up blocks ---
        up_channels = channels[::-1] + (channels[0],)
        for i in range(len(up_channels) - 1):
            input_channel = up_channels[i]
            output_channel = up_channels[i + 1]
            is_last = i == len(up_channels) - 2

            resnet = ResnetBlock1D(
                dim=2 * input_channel,
                dim_out=output_channel,
                time_emb_dim=time_embed_dim,
            )
            transformer_blocks = nn.ModuleList([
                BasicTransformerBlock(
                    dim=output_channel,
                    num_attention_heads=num_heads,
                    attention_head_dim=attention_head_dim,
                    dropout=dropout,
                    activation_fn="snakebeta",
                )
                for _ in range(n_blocks)
            ])
            upsample = (
                Upsample1D(output_channel, use_conv_transpose=True)
                if not is_last
                else nn.Conv1d(output_channel, output_channel, 3, padding=1)
            )
            self.up_blocks.append(nn.ModuleList([resnet, transformer_blocks, upsample]))

        self.final_block = Block1D(channels[-1], channels[-1])
        self.final_proj = nn.Conv1d(channels[-1], mel_channels, 1)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x_noisy, mask, t, camplus, voice_encoder):
        """Forward pass: predict noise from noisy mel + conditioning.

        Args:
            x_noisy: Noisy mel spectrogram (B, 80, T).
            mask: Binary mask (B, 1, T).
            t: Diffusion timesteps (B,).
            camplus: Projected camplus embedding (B, 80).
            voice_encoder: Voice encoder embedding (B, 256).

        Returns:
            Predicted noise (B, 80, T).
        """
        t_emb = self.time_embeddings(t).to(x_noisy.dtype)
        t_emb = self.time_mlp(t_emb, condition=voice_encoder)

        spks = repeat(camplus, "b c -> b c t", t=x_noisy.shape[-1])
        x = torch.cat([x_noisy, spks], dim=1)

        # --- Down path ---
        hiddens = []
        masks = [mask]
        for resnet, transformer_blocks, downsample in self.down_blocks:
            mask_down = masks[-1]
            x = resnet(x, mask_down, t_emb)
            x = rearrange(x, "b c t -> b t c")
            mask_down_flat = rearrange(mask_down, "b 1 t -> b t")
            for transformer_block in transformer_blocks:
                x = transformer_block(
                    hidden_states=x,
                    attention_mask=mask_down_flat,
                    timestep=t_emb,
                )
            x = rearrange(x, "b t c -> b c t")
            hiddens.append(x)
            x = downsample(x * mask_down)
            masks.append(mask_down[:, :, ::2])

        masks = masks[:-1]
        mask_mid = masks[-1]

        # --- Mid path ---
        for resnet, transformer_blocks in self.mid_blocks:
            x = resnet(x, mask_mid, t_emb)
            x = rearrange(x, "b c t -> b t c")
            mask_mid_flat = rearrange(mask_mid, "b 1 t -> b t")
            for transformer_block in transformer_blocks:
                x = transformer_block(
                    hidden_states=x,
                    attention_mask=mask_mid_flat,
                    timestep=t_emb,
                )
            x = rearrange(x, "b t c -> b c t")

        # --- Up path ---
        for resnet, transformer_blocks, upsample in self.up_blocks:
            mask_up = masks.pop()
            x = pack([x, hiddens.pop()], "b * t")[0]
            x = resnet(x, mask_up, t_emb)
            x = rearrange(x, "b c t -> b t c")
            mask_up_flat = rearrange(mask_up, "b 1 t -> b t")
            for transformer_block in transformer_blocks:
                x = transformer_block(
                    hidden_states=x,
                    attention_mask=mask_up_flat,
                    timestep=t_emb,
                )
            x = rearrange(x, "b t c -> b c t")
            x = upsample(x * mask_up)

        x = self.final_block(x, mask_up)
        output = self.final_proj(x * mask_up)
        return output * mask
