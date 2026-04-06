"""Gaussian diffusion core: DDPM forward process, training loss, DDIM sampling.

Adapted from AudioLDM's DDPM but standalone and minimal.
Linear beta schedule matching AudioLDM (beta_start=0.0015, beta_end=0.0195, T=1000).
"""

import torch
import torch.nn as nn
import numpy as np


class GaussianDiffusion(nn.Module):
    """DDPM noise schedule, forward process, training loss, and DDIM sampling."""

    def __init__(
        self,
        timesteps: int = 1000,
        linear_start: float = 0.0015,
        linear_end: float = 0.0195,
        parameterization: str = 'eps',
    ):
        super().__init__()
        assert parameterization in ('eps', 'x0')
        self.parameterization = parameterization
        self.num_timesteps = timesteps

        betas = np.linspace(linear_start ** 0.5, linear_end ** 0.5, timesteps, dtype=np.float64) ** 2
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

        self.register_buffer('betas', torch.tensor(betas, dtype=torch.float32))
        self.register_buffer('alphas_cumprod', torch.tensor(alphas_cumprod, dtype=torch.float32))
        self.register_buffer('alphas_cumprod_prev', torch.tensor(alphas_cumprod_prev, dtype=torch.float32))
        self.register_buffer('sqrt_alphas_cumprod', torch.tensor(np.sqrt(alphas_cumprod), dtype=torch.float32))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.tensor(np.sqrt(1.0 - alphas_cumprod), dtype=torch.float32))
        self.register_buffer('posterior_variance', torch.tensor(posterior_variance, dtype=torch.float32))
        self.register_buffer('posterior_mean_coef1', torch.tensor(
            betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod), dtype=torch.float32,
        ))
        self.register_buffer('posterior_mean_coef2', torch.tensor(
            (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod), dtype=torch.float32,
        ))

    def _extract(self, a, t, x_shape):
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alpha = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alpha = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        return sqrt_alpha * x_start + sqrt_one_minus_alpha * noise

    def p_losses(self, model, x_start, t, cond, mask=None):
        noise = torch.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise=noise)

        noise_pred = model(x_t, mask, t, cond['camplus'], cond['voice_encoder'])

        if self.parameterization == 'eps':
            target = noise
        else:
            target = x_start

        if mask is not None:
            diff = (noise_pred - target) ** 2 * mask
            loss = diff.sum() / (mask.sum() * x_start.shape[1])
        else:
            loss = torch.nn.functional.mse_loss(noise_pred, target)

        return loss

    def _predict_x0_from_eps(self, x_t, t, eps):
        sqrt_alpha = self._extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_one_minus_alpha = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        return (x_t - sqrt_one_minus_alpha * eps) / sqrt_alpha

    @torch.no_grad()
    def ddim_sample(
        self,
        model,
        shape,
        cond,
        mask=None,
        n_steps: int = 50,
        eta: float = 0.0,
        guidance_scale: float = 3.5,
    ):
        device = self.betas.device
        batch_size = shape[0]

        step_size = self.num_timesteps // n_steps
        timesteps = torch.arange(0, self.num_timesteps, step_size, device=device).flip(0)

        if mask is None:
            mask = torch.ones(batch_size, 1, shape[2], device=device)

        x = torch.randn(shape, device=device)

        for i, t_val in enumerate(timesteps):
            t = torch.full((batch_size,), t_val, device=device, dtype=torch.long)

            eps_cond = model(x, mask, t, cond['camplus'], cond['voice_encoder'])

            if guidance_scale != 1.0:
                zeros_cp = torch.zeros_like(cond['camplus'])
                zeros_ve = torch.zeros_like(cond['voice_encoder'])
                eps_uncond = model(x, mask, t, zeros_cp, zeros_ve)
                eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
            else:
                eps = eps_cond

            x0_pred = self._predict_x0_from_eps(x, t, eps)

            if i < len(timesteps) - 1:
                t_next = timesteps[i + 1]
                alpha = self.alphas_cumprod[t_val]
                alpha_next = self.alphas_cumprod[t_next]

                sigma = eta * torch.sqrt(
                    (1 - alpha_next) / (1 - alpha) * (1 - alpha / alpha_next)
                )
                noise = torch.randn_like(x) if eta > 0 else torch.zeros_like(x)

                x = (
                    torch.sqrt(alpha_next) * x0_pred
                    + torch.sqrt(1 - alpha_next - sigma ** 2) * eps
                    + sigma * noise
                )
            else:
                x = x0_pred

            x = x * mask

        return x

    @torch.no_grad()
    def ddim_sample_from_mel(
        self,
        model,
        x_start,
        cond,
        mask=None,
        t_start: int = 300,
        n_steps: int = 50,
        eta: float = 0.0,
        guidance_scale: float = 3.5,
    ):
        device = self.betas.device
        batch_size = x_start.shape[0]

        t = torch.full((batch_size,), t_start, device=device, dtype=torch.long)
        x = self.q_sample(x_start, t)

        step_size = self.num_timesteps // n_steps
        all_timesteps = torch.arange(0, self.num_timesteps, step_size, device=device).flip(0)
        timesteps = all_timesteps[all_timesteps <= t_start]

        if mask is None:
            mask = torch.ones(batch_size, 1, x_start.shape[2], device=device)

        for i, t_val in enumerate(timesteps):
            t = torch.full((batch_size,), t_val, device=device, dtype=torch.long)

            eps_cond = model(x, mask, t, cond['camplus'], cond['voice_encoder'])

            if guidance_scale != 1.0:
                zeros_cp = torch.zeros_like(cond['camplus'])
                zeros_ve = torch.zeros_like(cond['voice_encoder'])
                eps_uncond = model(x, mask, t, zeros_cp, zeros_ve)
                eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
            else:
                eps = eps_cond

            x0_pred = self._predict_x0_from_eps(x, t, eps)

            if i < len(timesteps) - 1:
                t_next = timesteps[i + 1]
                alpha = self.alphas_cumprod[t_val]
                alpha_next = self.alphas_cumprod[t_next]

                sigma = eta * torch.sqrt(
                    (1 - alpha_next) / (1 - alpha) * (1 - alpha / alpha_next)
                )
                noise = torch.randn_like(x) if eta > 0 else torch.zeros_like(x)

                x = (
                    torch.sqrt(alpha_next) * x0_pred
                    + torch.sqrt(1 - alpha_next - sigma ** 2) * eps
                    + sigma * noise
                )
            else:
                x = x0_pred

            x = x * mask

        return x
