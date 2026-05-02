"""SpikeMamba: Spiking adaptation of the Mamba (S4/SSM) architecture.

Adapted from:
  - state-spaces/mamba (Apache 2.0, Copyright 2023 Albert Gu, Tri Dao)
    https://github.com/state-spaces/mamba
    Block structure: in_proj, x_proj, dt_proj, A_log, D, out_proj,
    selective SSM scan, 1D causal convolution

Modifications from upstream:
  - Replaced SiLU activations with LIF spiking neurons
  - Added ternary spike support {-1, 0, +1}
  - Added LeakyTernaryLIF (learned alpha mixing)
  - Added DynamicLeakyTernaryLIF (per-token gating)
  - Added continuous gate option (hybrid-gate design)
  - Sequential SSM scan (no custom CUDA kernel)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional
from dataclasses import dataclass

from .snn_layers import spike_fn


# --- Surrogate gradient functions ---
class SigmoidSurrogate(torch.autograd.Function):
    scale = 5.0
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x >= 0).float()
    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        sx = torch.sigmoid(SigmoidSurrogate.scale * x)
        return grad_output * SigmoidSurrogate.scale * sx * (1 - sx)

class TriangularSurrogate(torch.autograd.Function):
    scale = 1.0
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x >= 0).float()
    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        return grad_output * torch.clamp(1 - torch.abs(x) * TriangularSurrogate.scale, min=0)

class STESurrogate(torch.autograd.Function):
    """Straight-through estimator."""
    @staticmethod
    def forward(ctx, x):
        return (x >= 0).float()
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def _get_spike_fn(name="cauchy"):
    if name == "cauchy":
        return spike_fn
    elif name == "sigmoid":
        return SigmoidSurrogate.apply
    elif name == "triangular":
        return TriangularSurrogate.apply
    elif name == "ste":
        return STESurrogate.apply
    return spike_fn


@dataclass
class SpikeMambaConfig:
    """Configuration for SpikeMamba model."""
    n_layers: int = 12
    d_model: int = 768
    d_state: int = 16        # SSM state dimension
    d_conv: int = 4          # Local conv width
    expand: int = 2          # Inner dimension expansion factor
    vocab_size: int = 50280  # Match mamba-130m tokenizer
    ctx_len: int = 256
    # Spiking parameters
    spike_beta: float = 0.9
    spike_threshold: float = 1.0
    adaptive_threshold: bool = True
    ternary: bool = False    # Use ternary {-1, 0, +1} spikes
    ternary_threshold: float = 1.5
    leaky_ternary: bool = False   # Use LeakyTernaryLIF (learned spike/continuous blend)
    leaky_alpha_init: float = 0.95  # Initial alpha for LeakyTernaryLIF
    dynamic_alpha: bool = False    # Use DynamicLeakyTernaryLIF (per-token gate)
    dynamic_alpha_init_bias: float = 3.0  # Init bias for dynamic gate (3.0 ≈ 95% spiking)
    continuous_gate: bool = False  # Keep gate branch continuous (proven better for ternary)
    soft_reset: bool = False       # Soft reset: mem *= (1-spike.abs()) instead of hard subtract
    surrogate: str = "cauchy"      # Surrogate gradient: "cauchy", "sigmoid", "triangular", "ste"
    # Regularization
    dropout: float = 0.0


class AdaptiveThresholdLIF(nn.Module):
    """LIF neuron with per-neuron adaptive thresholds."""

    def __init__(self, n_neurons: int, beta: float = 0.9, base_threshold: float = 1.0,
                 soft_reset: bool = False, surrogate: str = "cauchy"):
        super().__init__()
        self.beta = beta
        self.base_threshold = base_threshold
        self.soft_reset = soft_reset
        self._spike_fn = _get_spike_fn(surrogate)
        self.threshold_offset = nn.Parameter(torch.zeros(n_neurons))
        self.register_buffer("threshold_trace", torch.zeros(n_neurons))

    @property
    def effective_threshold(self):
        return self.base_threshold + torch.tanh(self.threshold_offset) * 0.5

    def forward(self, x, mem=None):
        if mem is None:
            mem = torch.zeros_like(x)
        mem = self.beta * mem + x
        threshold = self.effective_threshold
        while threshold.dim() < mem.dim():
            threshold = threshold.unsqueeze(0)
        spikes = self._spike_fn(mem - threshold)
        if self.soft_reset:
            mem = mem * (1 - spikes.abs())
        else:
            mem = mem - spikes * threshold
        with torch.no_grad():
            fire_rate = spikes.mean(dim=tuple(range(spikes.dim() - 1)))
            self.threshold_trace = 0.99 * self.threshold_trace + 0.01 * fire_rate
        return spikes, mem


class TernaryLIF(nn.Module):
    """Ternary-Integer LIF: outputs {-1, 0, +1} spikes.

    Positive spike when membrane > threshold, negative when < -threshold.
    Carries signed information while staying sparse.
    """

    class _TernarySpike(torch.autograd.Function):
        scale = 2.0
        @staticmethod
        def forward(ctx, membrane_potential, threshold):
            ctx.save_for_backward(membrane_potential)
            pos = (membrane_potential > threshold).float()
            neg = (membrane_potential < -threshold).float()
            return pos - neg
        @staticmethod
        def backward(ctx, grad_output):
            (mem,) = ctx.saved_tensors
            grad = grad_output / (1 + (math.pi * mem * TernaryLIF._TernarySpike.scale) ** 2)
            return grad, None

    def __init__(self, n_neurons: int, beta: float = 0.9, base_threshold: float = 1.5,
                 soft_reset: bool = False):
        super().__init__()
        self.beta = beta
        self.base_threshold = base_threshold
        self.soft_reset = soft_reset
        self.threshold_offset = nn.Parameter(torch.zeros(n_neurons))
        self.register_buffer("threshold_trace", torch.zeros(n_neurons))

    @property
    def effective_threshold(self):
        return self.base_threshold + torch.tanh(self.threshold_offset) * 0.5

    def forward(self, x, mem=None):
        if mem is None:
            mem = torch.zeros_like(x)
        mem = self.beta * mem + x
        threshold = self.effective_threshold
        while threshold.dim() < mem.dim():
            threshold = threshold.unsqueeze(0)
        spikes = TernaryLIF._TernarySpike.apply(mem, threshold)
        if self.soft_reset:
            mem = mem * (1 - spikes.abs())
        else:
            mem = mem - spikes * threshold
        with torch.no_grad():
            self.threshold_trace = 0.99 * self.threshold_trace + 0.01 * spikes.abs().mean(
                dim=tuple(range(spikes.dim() - 1)))
        return spikes, mem


class LeakyTernaryLIF(nn.Module):
    """Ternary LIF with learned continuous leak per layer.

    Output: out = alpha * spike + (1 - alpha) * silu(x)

    The learned alpha controls the spike/continuous balance per layer.
    Empirically, the model learns:
    - Early layers: alpha ≈ 0.4-0.5 (mostly continuous, feature extraction)
    - Later layers: alpha ≈ 0.97-0.99 (almost pure spike, classification)

    This subsumes selective spiking — the model discovers the optimal
    balance via gradient descent rather than manual layer selection.

    Hardware: ~95% of activation energy is neuromorphic (add/subtract).
    The ~5% continuous correction is small MAC overhead.
    """

    def __init__(self, n_neurons: int, beta: float = 0.9, base_threshold: float = 1.5,
                 soft_reset: bool = True, init_alpha: float = 0.95):
        super().__init__()
        self.lif = TernaryLIF(n_neurons, beta, base_threshold, soft_reset)
        self.alpha_logit = nn.Parameter(
            torch.tensor(math.log(init_alpha / (1 - init_alpha))))
        self.register_buffer("threshold_trace", torch.zeros(n_neurons))
        self.threshold_offset = self.lif.threshold_offset

    @property
    def alpha(self):
        return torch.sigmoid(self.alpha_logit)

    @property
    def effective_threshold(self):
        return self.lif.effective_threshold

    def forward(self, x, mem=None):
        spikes, mem = self.lif(x, mem)
        a = self.alpha
        out = a * spikes + (1 - a) * F.silu(x)
        with torch.no_grad():
            self.threshold_trace = self.lif.threshold_trace
        return out, mem


class DynamicLeakyTernaryLIF(nn.Module):
    """Ternary LIF with per-token dynamic alpha gating.

    Instead of a fixed learned alpha per layer, computes alpha dynamically
    based on input content:
        gate = sigmoid(linear(x) + bias)
        out = gate * spike + (1 - gate) * silu(x)

    This allows the model to decide per-token whether to spike (high gate)
    or stay continuous (low gate). Rare/complex tokens may need continuous
    precision while common patterns can spike efficiently.

    The gate network adds minimal parameters (d_model + 1 per layer).
    """

    def __init__(self, n_neurons: int, beta: float = 0.9, base_threshold: float = 1.5,
                 soft_reset: bool = True, init_bias: float = 3.0):
        super().__init__()
        self.lif = TernaryLIF(n_neurons, beta, base_threshold, soft_reset)
        # Gate network: project input to scalar gate per token
        self.gate_proj = nn.Linear(n_neurons, 1, bias=True)
        # Initialize bias high so gate starts near 1.0 (mostly spiking)
        nn.init.zeros_(self.gate_proj.weight)
        self.gate_proj.bias.data.fill_(init_bias)
        self.register_buffer("threshold_trace", torch.zeros(n_neurons))
        self.threshold_offset = self.lif.threshold_offset

    @property
    def alpha(self):
        """Return mean gate value from last forward pass (for monitoring)."""
        return getattr(self, '_last_gate_mean', torch.tensor(0.95))

    @property
    def effective_threshold(self):
        return self.lif.effective_threshold

    def forward(self, x, mem=None):
        spikes, mem = self.lif(x, mem)
        # Per-token gate: [B, T, 1]
        gate = torch.sigmoid(self.gate_proj(x.detach()))  # detach to avoid gate influencing feature learning
        out = gate * spikes + (1 - gate) * F.silu(x)
        with torch.no_grad():
            self._last_gate_mean = gate.mean().item()
            self.threshold_trace = self.lif.threshold_trace
        return out, mem


class _ContinuousPassthrough(nn.Module):
    """No spiking — continuous pass-through for hybrid gate."""
    def __init__(self, n_neurons):
        super().__init__()
        # Keep dummy params so spike_neurons property works
        self.threshold_offset = nn.Parameter(torch.zeros(1))
        self.register_buffer("threshold_trace", torch.zeros(1))
    @property
    def effective_threshold(self):
        return torch.tensor(1.0)
    def forward(self, x, mem=None):
        # Apply SiLU like original Mamba gate
        return F.silu(x), torch.zeros(1, device=x.device)


def _make_lif(n_neurons, config, is_gate=False):
    """Create the appropriate LIF neuron based on config."""
    if is_gate and config.continuous_gate:
        return _ContinuousPassthrough(n_neurons)
    if config.dynamic_alpha and not is_gate:
        return DynamicLeakyTernaryLIF(n_neurons, config.spike_beta, config.ternary_threshold,
                                      soft_reset=config.soft_reset, init_bias=config.dynamic_alpha_init_bias)
    if config.leaky_ternary and not is_gate:
        return LeakyTernaryLIF(n_neurons, config.spike_beta, config.ternary_threshold,
                              soft_reset=config.soft_reset, init_alpha=config.leaky_alpha_init)
    if config.ternary:
        return TernaryLIF(n_neurons, config.spike_beta, config.ternary_threshold,
                         soft_reset=config.soft_reset)
    elif config.adaptive_threshold:
        return AdaptiveThresholdLIF(n_neurons, config.spike_beta, config.spike_threshold,
                                   soft_reset=config.soft_reset, surrogate=config.surrogate)
    else:
        from .snn_layers import LIFNeuron
        return LIFNeuron(beta=config.spike_beta, threshold=config.spike_threshold)


class SpikeMambaBlock(nn.Module):
    """One SpikeMamba block: norm → expand → conv1d → SSM → LIF → project.

    Follows Mamba's architecture but replaces SiLU activations with LIF neurons.
    The SSM (selective state space) uses a simplified first-order formulation
    that maps naturally to LIF membrane dynamics.
    """

    def __init__(self, config: SpikeMambaConfig):
        super().__init__()
        self.d_model = config.d_model
        d_inner = config.d_model * config.expand
        self.d_inner = d_inner
        self.d_state = config.d_state
        self.d_conv = config.d_conv

        self.norm = nn.LayerNorm(config.d_model)

        # Input projection: x → (z, x_ssm) where z is the gate branch
        self.in_proj = nn.Linear(config.d_model, d_inner * 2, bias=False)

        # 1D causal convolution (local context, like original Mamba)
        self.conv1d = nn.Conv1d(
            d_inner, d_inner, config.d_conv,
            padding=config.d_conv - 1, groups=d_inner, bias=True
        )

        # SSM parameters: discretized A, B, C, D
        # A is diagonal (like LIF decay), initialized to be stable
        self.A_log = nn.Parameter(torch.log(torch.arange(1, config.d_state + 1).float().repeat(d_inner, 1)))
        self.D = nn.Parameter(torch.ones(d_inner))

        # Input-dependent B, C, dt (selective mechanism)
        self.x_proj = nn.Linear(d_inner, config.d_state * 2 + 1, bias=False)  # B, C, dt
        self.dt_proj = nn.Linear(1, d_inner, bias=True)

        # Output projection
        self.out_proj = nn.Linear(d_inner, config.d_model, bias=False)

        # LIF neurons replace SiLU activation
        self.lif_gate = _make_lif(d_inner, config, is_gate=True)
        self.lif_out = _make_lif(config.d_model, config, is_gate=False)

        self.drop = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()

    def forward(self, x, mem_gate=None, mem_out=None, return_spikes=False):
        residual = x
        x = self.norm(x)

        # Input projection: split into SSM path and gate path
        xz = self.in_proj(x)
        x_ssm, z = xz.chunk(2, dim=-1)  # each [B, T, d_inner]

        # 1D causal conv on SSM path
        x_conv = x_ssm.transpose(1, 2)  # [B, d_inner, T]
        x_conv = self.conv1d(x_conv)[:, :, :x.size(1)]  # causal: trim to T
        x_conv = x_conv.transpose(1, 2)  # [B, T, d_inner]

        # Selective SSM
        y = self._selective_ssm(x_conv)

        # Gate with LIF spike (replaces SiLU)
        z_spike, mem_gate = self.lif_gate(z, mem_gate)
        y = y * z_spike

        # Output projection + residual + LIF
        out = self.out_proj(y)
        s_out, mem_out = self.lif_out(residual + out, mem_out)
        result = residual + self.drop(s_out)

        if return_spikes:
            return result, mem_gate, mem_out, (z_spike, s_out)
        return result, mem_gate, mem_out

    def _selective_ssm(self, x):
        """Simplified selective SSM scan.

        Uses input-dependent discretization (the 'selective' part of Mamba).
        A is diagonal → equivalent to d_state independent first-order systems.
        """
        B_sz, T, D = x.shape
        N = self.d_state

        # Input-dependent parameters
        x_dbl = self.x_proj(x)  # [B, T, 2*N + 1]
        B_inp = x_dbl[:, :, :N]           # [B, T, N]
        C_inp = x_dbl[:, :, N:2*N]        # [B, T, N]
        dt_raw = x_dbl[:, :, 2*N:2*N+1]   # [B, T, 1]

        # Discretize dt
        dt = F.softplus(self.dt_proj(dt_raw))  # [B, T, D]

        # A matrix (diagonal, negative for stability)
        A = -torch.exp(self.A_log)  # [D, N]

        # Discretize A and B using zero-order hold
        dA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))  # [B, T, D, N]
        dB = dt.unsqueeze(-1) * B_inp.unsqueeze(2)  # [B, T, D, N]  (broadcast)

        # Sequential scan (can't parallelize without custom CUDA kernel)
        h = torch.zeros(B_sz, D, N, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(T):
            h = dA[:, t] * h + dB[:, t] * x[:, t].unsqueeze(-1)
            y_t = (h * C_inp[:, t].unsqueeze(1)).sum(-1)  # [B, D]
            ys.append(y_t)

        y = torch.stack(ys, dim=1)  # [B, T, D]
        y = y + x * self.D.unsqueeze(0).unsqueeze(0)  # skip connection
        return y


class SpikeMambaModel(nn.Module):
    """Complete SpikeMamba language model.

    A spiking adaptation of Mamba that:
    1. Uses 1D causal convolution for local context
    2. Uses selective SSM with LIF-equivalent state dynamics
    3. Replaces SiLU gating with LIF neurons (binary or ternary spikes)
    4. Processes tokens with constant memory (O(1) per step, neuromorphic-compatible)
    """

    def __init__(self, config: SpikeMambaConfig):
        super().__init__()
        self.config = config

        self.emb = nn.Embedding(config.vocab_size, config.d_model)
        self.emb_ln = nn.LayerNorm(config.d_model)
        self.emb_drop = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()

        self.blocks = nn.ModuleList([
            SpikeMambaBlock(config) for _ in range(config.n_layers)
        ])

        self.ln_out = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    @property
    def model_blocks(self):
        return self.blocks

    @property
    def spike_neurons(self):
        neurons = []
        for block in self.blocks:
            neurons.append(block.lif_gate)
            neurons.append(block.lif_out)
        return neurons

    def forward(self, input_ids, targets=None, return_hidden_states=False,
                return_spike_rates=False):
        B, T = input_ids.shape
        x = self.emb_drop(self.emb_ln(self.emb(input_ids)))

        hidden_states = [] if return_hidden_states else None
        spike_rates = [] if return_spike_rates else None

        for block in self.blocks:
            x, _, _, spikes = block(x, return_spikes=True)
            if return_hidden_states:
                hidden_states.append(x)
            if return_spike_rates:
                spike_rates.append(spikes[1].mean(dim=1))

        x = self.ln_out(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )

        from .outputs import ModelOutput
        return ModelOutput(
            logits=logits,
            hidden_states=hidden_states,
            spike_rates=spike_rates,
            loss=loss,
        )

    def get_hidden_dim(self) -> int:
        return self.config.d_model

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
