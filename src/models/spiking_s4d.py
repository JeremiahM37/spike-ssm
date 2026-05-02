"""SpikingS4D: spiking state-space model based on Shen et al., AAAI 2025.

This is a clean re-implementation of SpikingSSMs that removes the S4 framework
dependency. Source: https://github.com/shenshuaijie/SDN

Key components:
1. S4DKernel — diagonal SSM convolution kernel (from S4D, Gu et al.)
2. BinaryLIF / SLTTLIF — binary leaky integrate-and-fire neurons with learnable threshold
3. SS4D — single layer combining S4D kernel + LIF + output projection
4. SpikingS4D — stack of SS4D layers (no embedding/head; this is the body)
5. SpikingS4DLM — language model wrapper with embedding and head

We add LeakyTernaryLIF and top-K sparsity variants as drop-in replacements
for the binary LIF, enabling additive ablation: baseline → +leaky → +topK → +TWN.
"""

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Surrogate gradient (from SpikingSSMs)
# ============================================================================

class _PiecewiseQuadratic(torch.autograd.Function):
    """Surrogate gradient for binary spike. Forward: Heaviside.
    Backward: piecewise quadratic, gradient zero outside [-1, 1].
    """
    @staticmethod
    def forward(ctx, x):
        if x.requires_grad:
            ctx.save_for_backward(x)
        return (x >= 0).to(x)

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        x_abs = x.abs()
        mask = x_abs > 1
        grad_x = (grad_output * (-x_abs + 1.0)).masked_fill_(mask, 0)
        return grad_x


def piecewise_quadratic_surrogate(x):
    return _PiecewiseQuadratic.apply(x)


# ============================================================================
# S4D Kernel — diagonal state-space convolution kernel (from S4D, Gu et al.)
# ============================================================================

class S4DKernel(nn.Module):
    """Generate convolution kernel from diagonal SSM parameters.

    Re-implementation of S4D kernel without S4 framework dependency.
    Reference: https://arxiv.org/abs/2206.11893
    """

    def __init__(self, d_model, d_state=64, dt_min=0.001, dt_max=0.1, channels=1, lr=None):
        super().__init__()
        H = d_model
        N = d_state

        # Sample dt log-uniformly
        log_dt = torch.rand(H) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)

        # Initialize C with random complex values (stored as real for autograd)
        C = torch.randn(channels, H, N // 2, dtype=torch.cfloat)
        self.C = nn.Parameter(torch.view_as_real(C))

        # A is initialized to log(0.5) (real part) + i*pi*k (imag part)
        log_A_real = torch.log(0.5 * torch.ones(H, N // 2))
        A_imag = math.pi * torch.arange(N // 2).float().unsqueeze(0).expand(H, -1).contiguous()

        self.log_dt = nn.Parameter(log_dt)
        self.log_A_real = nn.Parameter(log_A_real)
        self.A_imag = nn.Parameter(A_imag)

    def forward(self, L):
        """Returns kernel of shape (channels, H, L)."""
        dt = torch.exp(self.log_dt)  # (H,)
        C = torch.view_as_complex(self.C)  # (channels, H, N//2)
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag  # (H, N//2)

        dtA = A * dt.unsqueeze(-1)  # (H, N//2)
        # K[h, n, l] = exp(l * dtA[h, n])
        K = dtA.unsqueeze(-1) * torch.arange(L, device=A.device)  # (H, N//2, L)
        C = C * (torch.exp(dtA) - 1.0) / A
        K = 2 * torch.einsum("chn, hnl -> chl", C, torch.exp(K)).real
        return K  # (channels, H, L)


# ============================================================================
# Spiking neurons (from SpikingSSMs + our extensions)
# ============================================================================

class BPTTBinaryLIF(nn.Module):
    """Binary leaky integrate-and-fire neuron, BPTT-trained.

    From SpikingSSMs: u[t] = tau*u[t-1] + x[t]; spike if u > vth.
    This is the SpikingSSMs baseline neuron.
    """
    def __init__(self, tau=0.125, vth=1.0, v_r=0.0):
        super().__init__()
        self.tau = tau
        self.vth = vth
        self.v_r = v_r

    def forward(self, x):
        # x: (B, H, L) — process along last dim (time)
        u = torch.zeros_like(x[..., 0])
        out = []
        for i in range(x.size(-1)):
            u = u * self.tau + x[..., i]
            s = piecewise_quadratic_surrogate(u - self.vth)
            out.append(s)
            u = (1 - s.detach()) * u + s.detach() * self.v_r
        return torch.stack(out, -1)


class SLTTBinaryLIF(nn.Module):
    """Binary LIF with stop-gradient on temporal recurrence (SLTT).

    From SpikingSSMs: faster training than BPTT by detaching membrane state.
    """
    def __init__(self, tau=0.125, vth=1.0, v_r=0.0):
        super().__init__()
        self.tau = tau
        self.vth = vth
        self.v_r = v_r

    def forward(self, x):
        u = torch.zeros_like(x[..., 0])
        out = []
        for i in range(x.size(-1)):
            u = u.detach() * self.tau + x[..., i]
            s = piecewise_quadratic_surrogate(u - self.vth)
            out.append(s)
            u = (1 - s.detach()) * u + s.detach() * self.v_r
        return torch.stack(out, -1)


class TernaryLIFNeuron(nn.Module):
    """Ternary LIF: spikes in {-1, 0, +1} based on signed threshold."""
    def __init__(self, tau=0.125, vth=1.5, v_r=0.0):
        super().__init__()
        self.tau = tau
        self.vth = vth
        self.v_r = v_r

    def forward(self, x):
        u = torch.zeros_like(x[..., 0])
        out = []
        for i in range(x.size(-1)):
            u = u.detach() * self.tau + x[..., i]
            s_pos = piecewise_quadratic_surrogate(u - self.vth)
            s_neg = piecewise_quadratic_surrogate(-u - self.vth)
            s = s_pos - s_neg  # in {-1, 0, +1}
            out.append(s)
            u = (1 - s.abs().detach()) * u + s.abs().detach() * self.v_r
        return torch.stack(out, -1)


class LeakyTernaryLIFNeuron(nn.Module):
    """Our LeakyTernaryLIF: out = α*spike + (1-α)*silu(x).
    Drop-in replacement for SpikingSSMs neurons.
    """
    def __init__(self, tau=0.125, vth=1.5, v_r=0.0, init_alpha=0.95):
        super().__init__()
        self.lif = TernaryLIFNeuron(tau=tau, vth=vth, v_r=v_r)
        self.alpha_logit = nn.Parameter(
            torch.tensor(math.log(init_alpha / (1 - init_alpha))))

    @property
    def alpha(self):
        return torch.sigmoid(self.alpha_logit)

    def forward(self, x):
        spikes = self.lif(x)
        a = self.alpha
        return a * spikes + (1 - a) * F.silu(x)


class TopKLeakyTernaryLIFNeuron(nn.Module):
    """LeakyTernaryLIF with top-K sparsity enforcement.
    Only top K% of neurons (by |membrane|) fire per timestep.
    """
    def __init__(self, tau=0.125, vth=1.5, v_r=0.0, init_alpha=0.95, top_k_frac=0.3):
        super().__init__()
        self.tau = tau
        self.vth = vth
        self.v_r = v_r
        self.top_k_frac = top_k_frac
        self.alpha_logit = nn.Parameter(
            torch.tensor(math.log(init_alpha / (1 - init_alpha))))

    @property
    def alpha(self):
        return torch.sigmoid(self.alpha_logit)

    def forward(self, x):
        # x: (B, H, L) — process along time dim
        B, H, L = x.shape
        u = torch.zeros(B, H, device=x.device, dtype=x.dtype)
        out_spikes = []
        for i in range(L):
            u = u.detach() * self.tau + x[..., i]
            # Top-K mask: only top K% of channels fire (per batch position)
            k = max(1, int(H * self.top_k_frac))
            abs_u = u.abs()
            _, topk_idx = abs_u.topk(k, dim=-1)  # (B, k)
            topk_mask = torch.zeros_like(u, dtype=torch.bool)
            topk_mask.scatter_(-1, topk_idx, True)

            # Ternary spike with surrogate, only for top-K
            s_pos = piecewise_quadratic_surrogate(u - self.vth)
            s_neg = piecewise_quadratic_surrogate(-u - self.vth)
            s = (s_pos - s_neg) * topk_mask.float()
            out_spikes.append(s)
            u = (1 - s.abs().detach()) * u + s.abs().detach() * self.v_r

        spikes = torch.stack(out_spikes, -1)  # (B, H, L)
        a = self.alpha
        return a * spikes + (1 - a) * F.silu(x)


# ============================================================================
# SpikingS4D layer
# ============================================================================

class SpikingS4DLayer(nn.Module):
    """One SpikingS4D layer: S4D kernel → conv → learnable threshold → LIF → output projection.

    Adapted from external/SDN/models/spike/ss4d.py without the S4 framework dependency.
    Default neuron is the SpikingSSMs binary LIF (BPTT or SLTT).
    """
    def __init__(self, d_model, d_state=64, dropout=0.0, neuron_type="sltt",
                 learnable_vth=True, init_alpha=0.95, top_k_frac=None):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.D = nn.Parameter(torch.randn(d_model))

        if learnable_vth:
            # Per-channel learnable threshold scaling (from SpikingSSMs)
            self.ln_vth = nn.Parameter(torch.zeros(d_model, 1))

        self.kernel = S4DKernel(d_model, d_state=d_state)

        # Neuron selection
        if neuron_type == "bptt":
            self.neuron = BPTTBinaryLIF()
        elif neuron_type == "sltt":
            self.neuron = SLTTBinaryLIF()
        elif neuron_type == "ternary":
            self.neuron = TernaryLIFNeuron()
        elif neuron_type == "leaky_ternary":
            self.neuron = LeakyTernaryLIFNeuron(init_alpha=init_alpha)
        elif neuron_type == "topk_leaky_ternary":
            assert top_k_frac is not None
            self.neuron = TopKLeakyTernaryLIFNeuron(init_alpha=init_alpha, top_k_frac=top_k_frac)
        else:
            raise ValueError(f"Unknown neuron_type: {neuron_type}")

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Output projection: GLU-style mixing (from SpikingSSMs)
        self.output_linear = nn.Sequential(
            nn.Conv1d(d_model, 2 * d_model, kernel_size=1),
            nn.GLU(dim=-2),
        )

        self.learnable_vth = learnable_vth

    def forward(self, u):
        """u: (B, H, L)."""
        B, H, L = u.shape

        # 1. S4D convolution
        k = self.kernel(L=L)  # (1, H, L)
        k_f = torch.fft.rfft(k, n=2 * L)  # (1, H, L+1)
        u_f = torch.fft.rfft(u, n=2 * L)  # (B, H, L+1)
        y = torch.einsum("bhl,chl->bchl", u_f, k_f)
        y = torch.fft.irfft(y, n=2 * L)[..., :L]  # (B, 1, H, L)
        y = y.squeeze(1)  # (B, H, L)

        # 2. Skip connection (D term): D is (H,), broadcast over (B, H, L)
        y = y + u * self.D.unsqueeze(0).unsqueeze(-1)

        # 3. Learnable threshold scaling
        if self.learnable_vth:
            y = y / torch.exp(self.ln_vth)

        # 4. Spiking neuron
        y = self.neuron(y)

        # 5. Dropout + output projection
        y = self.dropout(y)
        y = self.output_linear(y)
        return y


# ============================================================================
# SpikingS4D body and language model wrapper
# ============================================================================

@dataclass
class SpikingS4DConfig:
    vocab_size: int = 50277
    d_model: int = 256
    n_layers: int = 4
    d_state: int = 64
    dropout: float = 0.1
    prenorm: bool = True
    neuron_type: str = "sltt"      # baseline; or 'leaky_ternary', 'topk_leaky_ternary'
    init_alpha: float = 0.95
    top_k_frac: Optional[float] = None
    twn_weights: bool = False      # if True, replace nn.Linear/Conv1d weights with TWN
    max_seq_len: int = 512


class SpikingS4DBody(nn.Module):
    """Stack of SpikingS4D layers with residual connections + layer norms."""
    def __init__(self, config: SpikingS4DConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([
            SpikingS4DLayer(
                d_model=config.d_model,
                d_state=config.d_state,
                dropout=config.dropout,
                neuron_type=config.neuron_type,
                init_alpha=config.init_alpha,
                top_k_frac=config.top_k_frac,
            )
            for _ in range(config.n_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(config.d_model) for _ in range(config.n_layers)
        ])
        self.dropouts = nn.ModuleList([
            nn.Dropout(config.dropout) for _ in range(config.n_layers)
        ])

    def forward(self, x):
        """x: (B, L, d_model)."""
        x = x.transpose(-1, -2)  # (B, d_model, L)
        for layer, norm, drop in zip(self.layers, self.norms, self.dropouts):
            z = x
            if self.config.prenorm:
                z = norm(z.transpose(-1, -2)).transpose(-1, -2)
            z, *_ = (layer(z),)
            z = drop(z)
            x = z + x
            if not self.config.prenorm:
                x = norm(x.transpose(-1, -2)).transpose(-1, -2)
        x = x.transpose(-1, -2)  # (B, L, d_model)
        return x


class SpikingS4DLM(nn.Module):
    """Language model: embedding + SpikingS4D body + LM head."""
    def __init__(self, config: SpikingS4DConfig):
        super().__init__()
        self.config = config
        self.emb = nn.Embedding(config.vocab_size, config.d_model)
        self.emb_drop = nn.Dropout(config.dropout)
        self.body = SpikingS4DBody(config)
        self.ln_out = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(self, input_ids):
        x = self.emb_drop(self.emb(input_ids))
        x = self.body(x)
        x = self.ln_out(x)
        logits = self.head(x)
        return _Output(logits=logits)

    def get_alphas(self):
        """Return per-layer alpha values (only meaningful for leaky variants)."""
        alphas = []
        for layer in self.body.layers:
            if hasattr(layer.neuron, 'alpha'):
                a = layer.neuron.alpha
                alphas.append(a.item() if torch.is_tensor(a) else a)
        return alphas


@dataclass
class _Output:
    logits: torch.Tensor
