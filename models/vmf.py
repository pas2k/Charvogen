"""Von Mises-Fisher distribution and helpers (extracted from vecamp_train/train_vmfvae.py).

Inference-only: no training code, no VAE classes.
"""

import math

import numpy as np
import torch
from scipy import special as sp_special


# ---------------------------------------------------------------------------
# HypersphericalUniform distribution
# ---------------------------------------------------------------------------

class HypersphericalUniform(torch.distributions.Distribution):
    """Uniform distribution on the (d-1)-sphere S^{d-1} embedded in R^d."""

    arg_constraints = {}
    has_rsample = False

    def __init__(self, dim: int, device='cpu', dtype=torch.float32, validate_args=None):
        self._dim = dim
        self._device = device
        self._dtype = dtype
        super().__init__(
            event_shape=torch.Size([dim]),
            validate_args=validate_args,
        )

    @property
    def dim(self):
        return self._dim

    def _log_surface_area(self) -> float:
        """log surface area of S^{d-1}: log(2 * pi^{d/2} / Gamma(d/2))."""
        d = self._dim
        return math.log(2) + (d / 2) * math.log(math.pi) - math.lgamma(d / 2)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        return torch.full(
            x.shape[:-1], -self._log_surface_area(),
            device=x.device, dtype=x.dtype,
        )

    def entropy(self) -> torch.Tensor:
        return torch.tensor(
            self._log_surface_area(), device=self._device, dtype=self._dtype,
        )

    def sample(self, shape=torch.Size()):
        v = torch.randn(*shape, self._dim, device=self._device, dtype=self._dtype)
        return v / v.norm(dim=-1, keepdim=True)


# ---------------------------------------------------------------------------
# Bessel function helpers
# ---------------------------------------------------------------------------

class _IveFunction(torch.autograd.Function):
    """Differentiable exponentially-scaled modified Bessel function ive(v, kappa)."""

    @staticmethod
    def forward(ctx, order, kappa):
        ctx.order = order
        ctx.save_for_backward(kappa)
        kappa_np = kappa.detach().cpu().numpy().astype(np.float64)
        ive_val = sp_special.ive(order, kappa_np)
        return torch.tensor(ive_val, device=kappa.device, dtype=kappa.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        order = ctx.order
        kappa, = ctx.saved_tensors
        kappa_np = kappa.detach().cpu().numpy().astype(np.float64)

        ive_val = sp_special.ive(order, kappa_np)
        ive_prev = sp_special.ive(order - 1, kappa_np)
        kappa_np_safe = np.maximum(kappa_np, 1e-10)
        grad_np = ive_prev - ive_val * (order / kappa_np_safe + 1.0)

        grad_kappa = torch.tensor(grad_np, device=kappa.device, dtype=kappa.dtype)
        return None, grad_output * grad_kappa


def _ive(order: float, kappa: torch.Tensor) -> torch.Tensor:
    """Differentiable ive(order, kappa) with exact forward and analytical backward."""
    return _IveFunction.apply(order, kappa)


def _log_bessel_ive(order: float, kappa: torch.Tensor) -> torch.Tensor:
    """Differentiable log ive(order, kappa)."""
    ive_val = _ive(order, kappa)
    return torch.log(torch.clamp(ive_val, min=1e-30))


# ---------------------------------------------------------------------------
# VonMisesFisher distribution
# ---------------------------------------------------------------------------

class VonMisesFisher(torch.distributions.Distribution):
    """Von Mises-Fisher distribution on S^{d-1}.

    Parameterized by:
        loc  (mu):  unit mean direction,  shape (*, d)
        scale (kappa): concentration > 0, shape (*, 1)
    """

    arg_constraints = {
        'loc': torch.distributions.constraints.real,
        'scale': torch.distributions.constraints.positive,
    }
    has_rsample = True

    def __init__(self, loc: torch.Tensor, scale: torch.Tensor, validate_args=None):
        self.loc = loc
        self.scale = scale
        self._dim = loc.shape[-1]
        batch_shape = loc.shape[:-1]
        event_shape = torch.Size([self._dim])
        super().__init__(batch_shape, event_shape, validate_args=False)

    @property
    def dim(self):
        return self._dim

    def _log_normalization(self) -> torch.Tensor:
        d = self._dim
        kappa = self.scale.squeeze(-1)

        half_d = d / 2.0
        order = half_d - 1.0

        log_ive = _log_bessel_ive(order, kappa)
        log_bessel = log_ive + kappa

        log_c = order * torch.log(torch.clamp(kappa, min=1e-10)) \
                - half_d * math.log(2 * math.pi) \
                - log_bessel

        return log_c

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        dot = (self.loc * x).sum(dim=-1)
        return self.scale.squeeze(-1) * dot + self._log_normalization()

    def entropy(self) -> torch.Tensor:
        d = self._dim
        kappa = self.scale.squeeze(-1)

        order = d / 2.0 - 1.0
        log_ive_num = _log_bessel_ive(order + 1.0, kappa)
        log_ive_den = _log_bessel_ive(order, kappa)
        A_d = torch.exp(log_ive_num - log_ive_den)

        return -self._log_normalization() - kappa * A_d

    def rsample(self, sample_shape=torch.Size()) -> torch.Tensor:
        """Reparameterized sampling via Wood (1994) rejection sampling
        + Householder reflection to align with mu.
        """
        shape = self._extended_shape(sample_shape)
        batch_shape = shape[:-1]
        d = self._dim

        kappa = self.scale.squeeze(-1)
        kappa = kappa.expand(batch_shape)

        w = self._sample_w(kappa, batch_shape)

        v = torch.randn(*batch_shape, d - 1, device=self.loc.device, dtype=self.loc.dtype)
        v = v / v.norm(dim=-1, keepdim=True)

        w_ = w.unsqueeze(-1)
        t = torch.sqrt(torch.clamp(1 - w_ ** 2, min=1e-10))
        z = torch.cat([t * v, w_], dim=-1)

        z = self._householder_rotation(z)

        return z

    def _sample_w(self, kappa: torch.Tensor, batch_shape: torch.Size) -> torch.Tensor:
        """Sample the marginal w using rejection sampling (Wood 1994)."""
        d = self._dim
        device = kappa.device
        dtype = kappa.dtype

        b = (-2.0 * kappa + torch.sqrt(4.0 * kappa ** 2 + (d - 1) ** 2)) / (d - 1)
        a = ((d - 1) + 2.0 * kappa + torch.sqrt(4.0 * kappa ** 2 + (d - 1) ** 2)) / 4.0
        c = kappa * b + (d - 1) * torch.log(1.0 - b ** 2 + 1e-10)

        w = torch.zeros(batch_shape, device=device, dtype=dtype)
        done = torch.zeros(batch_shape, device=device, dtype=torch.bool)

        max_iter = 200
        for _ in range(max_iter):
            if done.all():
                break

            eps = torch.distributions.Beta(
                torch.tensor((d - 1.0) / 2.0, device=device, dtype=dtype),
                torch.tensor((d - 1.0) / 2.0, device=device, dtype=dtype),
            ).sample(batch_shape)

            w_proposal = (1.0 - (1.0 + b) * eps) / (1.0 - (1.0 - b) * eps)

            t_val = kappa * w_proposal + (d - 1) * torch.log(1.0 - b * w_proposal + 1e-10) - c
            u = torch.rand(batch_shape, device=device, dtype=dtype)
            accept = torch.log(u + 1e-10) < t_val

            new_accept = accept & ~done
            w = torch.where(new_accept, w_proposal, w)
            done = done | new_accept

        return w

    def _householder_rotation(self, z: torch.Tensor) -> torch.Tensor:
        """Rotate z so that e_d maps to mu via Householder reflection."""
        d = self._dim
        mu = self.loc

        if z.dim() > mu.dim():
            extra = z.dim() - mu.dim()
            for _ in range(extra):
                mu = mu.unsqueeze(0)
            mu = mu.expand_as(z)

        e_d = torch.zeros_like(mu)
        e_d[..., -1] = 1.0

        u = e_d - mu
        u_norm = u.norm(dim=-1, keepdim=True)
        needs_rotation = (u_norm.squeeze(-1) > 1e-6)

        u_hat = u / (u_norm + 1e-10)

        dot = (u_hat * z).sum(dim=-1, keepdim=True)
        z_rotated = z - 2 * dot * u_hat

        if not needs_rotation.all():
            z_rotated = torch.where(needs_rotation.unsqueeze(-1), z_rotated, z)

        return z_rotated


# Register KL divergence between vMF and HypersphericalUniform
@torch.distributions.kl.register_kl(VonMisesFisher, HypersphericalUniform)
def _kl_vmf_uniform(p: VonMisesFisher, q: HypersphericalUniform) -> torch.Tensor:
    """KL(vMF || Uniform) = -H[vMF] + H[Uniform] = -entropy(vMF) + log(surface_area)."""
    return -p.entropy() + q.entropy()
