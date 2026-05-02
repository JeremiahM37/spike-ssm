"""Evaluation metrics for spiking language models."""

import torch
import torch.nn.functional as F
import math
from typing import Optional


def perplexity(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute perplexity from model logits and target token IDs.

    Args:
        logits: [batch, seq_len, vocab_size]
        targets: [batch, seq_len] token IDs
    """
    logits_flat = logits.reshape(-1, logits.size(-1))
    targets_flat = targets.reshape(-1)
    loss = F.cross_entropy(logits_flat, targets_flat, reduction="mean")
    return math.exp(loss.item())


def bits_per_character(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute bits-per-character (BPC) for character-level language models."""
    logits_flat = logits.reshape(-1, logits.size(-1))
    targets_flat = targets.reshape(-1)
    loss = F.cross_entropy(logits_flat, targets_flat, reduction="mean")
    return loss.item() / math.log(2)


def spike_sparsity(spike_tensor: torch.Tensor) -> float:
    """Fraction of activations that are zero (no spike fired).

    Higher = more sparse = more energy efficient on neuromorphic hardware.
    """
    total = spike_tensor.numel()
    if total == 0:
        return 0.0
    zeros = (spike_tensor == 0).sum().item()
    return zeros / total


def synops_count(spike_tensor: torch.Tensor, weight_tensor: torch.Tensor) -> int:
    """Count synaptic operations: only non-zero spikes cause weight reads.

    This is the primary energy metric for neuromorphic hardware.
    """
    active_spikes = (spike_tensor != 0).sum().item()
    fan_out = weight_tensor.shape[-1] if weight_tensor.dim() >= 2 else 1
    return int(active_spikes * fan_out)


def model_size_bytes(model: torch.nn.Module, bits: int = 32) -> int:
    """Total model weight memory at given precision."""
    total_params = sum(p.numel() for p in model.parameters())
    return total_params * bits // 8


def forgetting_metric(
    perplexity_before: float, perplexity_after: float
) -> float:
    """Measure catastrophic forgetting as relative perplexity increase.

    Returns fraction increase. 0.0 = no forgetting, 0.1 = 10% worse.
    """
    if perplexity_before == 0:
        return 0.0
    return max(0.0, (perplexity_after - perplexity_before) / perplexity_before)


def adaptation_gain(
    perplexity_frozen: float, perplexity_adapted: float
) -> float:
    """Measure adaptation benefit as relative perplexity reduction.

    Returns fraction decrease. 0.15 = 15% better than frozen.
    """
    if perplexity_frozen == 0:
        return 0.0
    return max(0.0, (perplexity_frozen - perplexity_adapted) / perplexity_frozen)


def firing_rate_per_layer(
    spike_trains: list[torch.Tensor],
) -> list[float]:
    """Average firing rate for each layer's spike train."""
    rates = []
    for spikes in spike_trains:
        rate = spikes.float().mean().item()
        rates.append(rate)
    return rates
