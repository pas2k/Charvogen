"""Givens Rotation SAE — inference only (extracted from vecamp_train/train_rotsae.py).

No training loop, loss functions, or data loading.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class GivensRotationSAE(nn.Module):
    """Sparse autoencoder using Givens rotations on product of spheres.

    Each "feature" is a rotation plane (u, v) + angle theta, not an additive
    direction vector. The decoder composes K active Givens rotations from a
    learned base direction mu0, preserving unit norm exactly.

    Product-of-spheres: VE and CP planes are independent subspaces.
    Angles can be negative (rotation direction matters). No ReLU.
    """

    def __init__(self, ve_dim: int = 128, cp_dim: int = 32,
                 n_planes: int = 1024, k: int = 20,
                 k_ve: int | None = None, k_cp: int | None = None,
                 enc_hidden: int | None = None,
                 freeze_mu0: bool = False,
                 jumprelu: bool = False,
                 init_threshold: float = 0.01,
                 bandwidth: float = 0.01,
                 gated_sae: bool = False,
                 init_gate_bias: float = 0.0,
                 gate_wnorm: bool = False,
                 ve_only: bool = False,
                 cp_only: bool = False):
        super().__init__()
        self.ve_dim = ve_dim
        self.cp_dim = cp_dim
        self.n_planes = n_planes
        self.k = k
        self.enc_hidden = enc_hidden
        self.freeze_mu0 = freeze_mu0
        self.jumprelu = jumprelu
        self.bandwidth = bandwidth
        self.gated_sae = gated_sae
        self.gate_wnorm = gate_wnorm
        self.ve_only = ve_only
        self.cp_only = cp_only

        # Plane budget and encoder input dim
        if ve_only:
            self.input_dim = ve_dim
            self.n_ve_planes = n_planes
            self.n_cp_planes = 0
        elif cp_only:
            self.input_dim = cp_dim
            self.n_ve_planes = 0
            self.n_cp_planes = n_planes
        else:
            self.input_dim = ve_dim + cp_dim
            self.n_ve_planes = round(n_planes * ve_dim / (ve_dim + cp_dim))
            self.n_cp_planes = n_planes - self.n_ve_planes

        # K budget split
        if ve_only:
            self.k_ve = k
            self.k_cp = 0
        elif cp_only:
            self.k_ve = 0
            self.k_cp = k
        elif k_ve is not None and k_cp is not None:
            self.k_ve = k_ve
            self.k_cp = k_cp
        else:
            self.k_ve = max(1, round(k * self.n_ve_planes / n_planes))
            self.k_cp = max(1, k - self.k_ve)
            if self.k_ve + self.k_cp > k:
                self.k_ve = k - self.k_cp

        # Base directions on each sphere
        if freeze_mu0:
            self.register_buffer('mu0_ve', torch.randn(ve_dim))
            self.register_buffer('mu0_cp', torch.randn(cp_dim))
        else:
            self.mu0_ve = nn.Parameter(torch.randn(ve_dim))
            self.mu0_cp = nn.Parameter(torch.randn(cp_dim))

        # Rotation planes: orthonormal (u, v) pairs per subspace
        self.planes_u_ve = nn.Parameter(torch.randn(self.n_ve_planes, ve_dim))
        self.planes_v_ve = nn.Parameter(torch.randn(self.n_ve_planes, ve_dim))
        self.planes_u_cp = nn.Parameter(torch.randn(self.n_cp_planes, cp_dim))
        self.planes_v_cp = nn.Parameter(torch.randn(self.n_cp_planes, cp_dim))

        if gated_sae:
            self.W_gate = nn.Linear(self.input_dim, n_planes)
            self.W_angle = nn.Linear(self.input_dim, n_planes)
            self.encoder = None
            self.log_threshold = None
            self.W_hidden = None
            self.W_skip = None
            nn.init.kaiming_uniform_(self.W_gate.weight)
            nn.init.kaiming_uniform_(self.W_angle.weight)
            nn.init.constant_(self.W_gate.bias, init_gate_bias)
        elif jumprelu:
            self.encoder = nn.Linear(self.input_dim, n_planes)
            self.log_threshold = nn.Parameter(
                torch.full((n_planes,), math.log(init_threshold)))
            self.W_hidden = None
            self.W_gate = None
            self.W_angle = None
            self.W_skip = None
            nn.init.kaiming_uniform_(self.encoder.weight)
        elif enc_hidden is not None and enc_hidden > 0:
            self.encoder = None
            self.log_threshold = None
            self.W_hidden = nn.Linear(self.input_dim, enc_hidden)
            self.W_gate = nn.Linear(enc_hidden, n_planes)
            self.W_angle = nn.Linear(enc_hidden, n_planes)
            self.W_skip = nn.Linear(self.input_dim, n_planes)
            nn.init.kaiming_uniform_(self.W_hidden.weight)
            nn.init.kaiming_uniform_(self.W_gate.weight)
            nn.init.kaiming_uniform_(self.W_angle.weight)
            nn.init.zeros_(self.W_skip.weight)
            nn.init.zeros_(self.W_skip.bias)
        elif enc_hidden is not None:
            self.W_hidden = None
            self.W_gate = nn.Linear(self.input_dim, n_planes)
            self.W_angle = nn.Linear(self.input_dim, n_planes)
            self.W_skip = None
            self.encoder = None
            self.log_threshold = None
            nn.init.kaiming_uniform_(self.W_gate.weight)
            nn.init.kaiming_uniform_(self.W_angle.weight)
        else:
            self.encoder = nn.Linear(self.input_dim, n_planes)
            self.log_threshold = None
            self.W_hidden = None
            self.W_gate = None
            self.W_angle = None
            self.W_skip = None
            nn.init.kaiming_uniform_(self.encoder.weight)

        # Orthonormalize planes and normalize mu0
        self.normalize_planes_()

    @torch.no_grad()
    def normalize_planes_(self):
        """Gram-Schmidt on (u, v) pairs + normalize mu0."""
        for u, v in [(self.planes_u_ve, self.planes_v_ve),
                     (self.planes_u_cp, self.planes_v_cp)]:
            if u.shape[0] == 0:
                continue
            u.data = F.normalize(u.data, dim=1)
            v.data = v.data - (v.data * u.data).sum(1, keepdim=True) * u.data
            v.data = F.normalize(v.data, dim=1)
        if not self.freeze_mu0:
            self.mu0_ve.data = F.normalize(self.mu0_ve.data, dim=0)
            self.mu0_cp.data = F.normalize(self.mu0_cp.data, dim=0)

    @torch.no_grad()
    def init_from_data(self, train_data: torch.Tensor):
        """Initialize mu0 from mean of training latents."""
        ve = train_data[:, :self.ve_dim]
        cp = train_data[:, self.ve_dim:]
        self.mu0_ve.data = F.normalize(ve.mean(0), dim=0)
        self.mu0_cp.data = F.normalize(cp.mean(0), dim=0)

    def _split_topk(self, angles: torch.Tensor):
        """Split angles into VE/CP and apply topk. Handles k=0 safely."""
        B = angles.shape[0]
        ve_angles_all = angles[:, :self.n_ve_planes]
        cp_angles_all = angles[:, self.n_ve_planes:]

        if self.k_ve > 0:
            _, ve_idx = ve_angles_all.abs().topk(self.k_ve, dim=-1)
            ve_angles = ve_angles_all.gather(1, ve_idx)
        else:
            ve_idx = angles.new_zeros(B, 0, dtype=torch.long)
            ve_angles = angles.new_zeros(B, 0)

        if self.k_cp > 0:
            _, cp_idx = cp_angles_all.abs().topk(self.k_cp, dim=-1)
            cp_angles = cp_angles_all.gather(1, cp_idx)
        else:
            cp_idx = angles.new_zeros(B, 0, dtype=torch.long)
            cp_angles = angles.new_zeros(B, 0)

        return ve_idx, ve_angles, cp_idx, cp_angles

    def encode(self, z: torch.Tensor):
        """Encode to sparse rotation angles.

        Returns:
            h: (B, n_planes) sparse angles (sign preserved)
            ve_idx, ve_angles, cp_idx, cp_angles: active plane info
            l0: differentiable L0 (Gated SAE / JumpReLU, else None)
        """
        if self.gated_sae:
            if self.gate_wnorm:
                gate_pre = F.linear(z, F.normalize(self.W_gate.weight, dim=1),
                                    self.W_gate.bias)
            else:
                gate_pre = self.W_gate(z)
            angle_scores = self.W_angle(z)

            soft_gate = torch.sigmoid(gate_pre / self.bandwidth)
            hard_gate = (gate_pre > 0).float()
            gate = soft_gate + (hard_gate - soft_gate).detach()

            angles = gate * angle_scores
            l0 = gate.sum(-1).mean()

            ve_idx, ve_angles, cp_idx, cp_angles = self._split_topk(angles)
            return angles, ve_idx, ve_angles, cp_idx, cp_angles, l0

        if self.jumprelu:
            scores = self.encoder(z)
            threshold = self.log_threshold.exp()

            soft = torch.sigmoid((scores.abs() - threshold) / self.bandwidth)
            hard = (scores.abs() > threshold).float()
            active = soft + (hard - soft).detach()

            angles = scores * active
            l0 = active.sum(-1).mean()

            ve_idx, ve_angles, cp_idx, cp_angles = self._split_topk(angles)
            return angles, ve_idx, ve_angles, cp_idx, cp_angles, l0

        if self.enc_hidden is not None and self.W_hidden is not None:
            hidden = F.relu(self.W_hidden(z))
            gate_scores = self.W_gate(hidden)
            angle_scores = self.W_angle(hidden) + self.W_skip(z)
        elif self.W_gate is not None:
            gate_scores = self.W_gate(z)
            angle_scores = self.W_angle(z)
        else:
            gate_scores = None
            angle_scores = None

        if gate_scores is not None:
            angles = angle_scores
            B = z.shape[0]
            ve_gate = gate_scores[:, :self.n_ve_planes]
            cp_gate = gate_scores[:, self.n_ve_planes:]
            ve_angle_all = angle_scores[:, :self.n_ve_planes]
            cp_angle_all = angle_scores[:, self.n_ve_planes:]

            if self.k_ve > 0:
                _, ve_idx = ve_gate.topk(self.k_ve, dim=-1)
                ve_angles = ve_angle_all.gather(1, ve_idx)
            else:
                ve_idx = z.new_zeros(B, 0, dtype=torch.long)
                ve_angles = z.new_zeros(B, 0)

            if self.k_cp > 0:
                _, cp_idx = cp_gate.topk(self.k_cp, dim=-1)
                cp_angles = cp_angle_all.gather(1, cp_idx)
            else:
                cp_idx = z.new_zeros(B, 0, dtype=torch.long)
                cp_angles = z.new_zeros(B, 0)
        else:
            scores = self.encoder(z)
            angles = scores
            ve_idx, ve_angles, cp_idx, cp_angles = self._split_topk(scores)

        # Build full sparse h for tracking / loss
        B = z.shape[0]
        ve_h = torch.zeros(B, self.n_ve_planes, device=z.device)
        cp_h = torch.zeros(B, self.n_cp_planes, device=z.device)
        if self.k_ve > 0:
            ve_h.scatter_(1, ve_idx, ve_angles)
        if self.k_cp > 0:
            cp_h.scatter_(1, cp_idx, cp_angles)
        h = torch.cat([ve_h, cp_h], dim=-1)

        return h, ve_idx, ve_angles, cp_idx, cp_angles, None

    def _decode_subspace(
        self,
        mu0: torch.Tensor,
        planes_u: torch.Tensor,
        planes_v: torch.Tensor,
        active_idx: torch.Tensor,
        active_angles: torch.Tensor,
        k_sub: int,
        B: int,
    ) -> torch.Tensor:
        """Apply K Givens rotations in one subspace."""
        if k_sub == 0:
            return mu0.unsqueeze(0).expand(B, -1)

        z = mu0.unsqueeze(0).expand(B, -1).clone()

        sort_perm = active_idx.argsort(dim=-1)
        active_idx = active_idx.gather(1, sort_perm)
        active_angles = active_angles.gather(1, sort_perm)

        for step in range(k_sub):
            idx = active_idx[:, step]
            theta = active_angles[:, step]
            u = planes_u[idx]
            v = planes_v[idx]

            zu = (z * u).sum(-1, keepdim=True)
            zv = (z * v).sum(-1, keepdim=True)
            cos_t = theta.cos().unsqueeze(-1)
            sin_t = theta.sin().unsqueeze(-1)

            z = z + (cos_t - 1) * (zu * u + zv * v) + sin_t * (zu * v - zv * u)

        return z

    def decode_from_active(
        self,
        ve_idx: torch.Tensor,
        ve_angles: torch.Tensor,
        cp_idx: torch.Tensor,
        cp_angles: torch.Tensor,
        B: int,
    ) -> torch.Tensor:
        """Decode from pre-gathered active planes and angles."""
        ve_hat = self._decode_subspace(
            self.mu0_ve, self.planes_u_ve, self.planes_v_ve,
            ve_idx, ve_angles, self.k_ve, B)
        cp_hat = self._decode_subspace(
            self.mu0_cp, self.planes_u_cp, self.planes_v_cp,
            cp_idx, cp_angles, self.k_cp, B)
        return torch.cat([ve_hat, cp_hat], dim=-1)

    def decode(self, h: torch.Tensor) -> torch.Tensor:
        """Decode from full sparse h vector (for analysis / manipulation)."""
        B = h.shape[0]
        ve_idx, ve_angles, cp_idx, cp_angles = self._split_topk(h)
        return self.decode_from_active(ve_idx, ve_angles, cp_idx, cp_angles, B)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Returns (z_hat, h, l0)."""
        if self.ve_only:
            z_enc = z[:, :self.ve_dim]
        elif self.cp_only:
            z_enc = z[:, self.ve_dim:]
        else:
            z_enc = z
        h, ve_idx, ve_angles, cp_idx, cp_angles, l0 = self.encode(z_enc)
        z_hat = self.decode_from_active(ve_idx, ve_angles, cp_idx, cp_angles, z.shape[0])
        return z_hat, h, l0


def load_rotsae_from_checkpoint(
    path: str, device: torch.device
) -> GivensRotationSAE:
    """Load Rotation SAE from checkpoint."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt['config']
    model = GivensRotationSAE(
        ve_dim=cfg['ve_dim'],
        cp_dim=cfg['cp_dim'],
        n_planes=cfg['n_planes'],
        k=cfg['k'],
        k_ve=cfg.get('k_ve'),
        k_cp=cfg.get('k_cp'),
        enc_hidden=cfg.get('enc_hidden'),
        freeze_mu0=cfg.get('freeze_mu0', False),
        gated_sae=cfg.get('gated_sae', False),
        gate_wnorm=cfg.get('gate_wnorm', False),
        ve_only=cfg.get('ve_only', False),
        cp_only=cfg.get('cp_only', False),
        jumprelu=cfg.get('jumprelu', False),
        bandwidth=cfg.get('bandwidth', 0.01),
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    return model
