"""Standardized model output containers."""

from dataclasses import dataclass, field
from typing import Optional
import torch


@dataclass
class ModelOutput:
    """Output from forward pass."""
    logits: torch.Tensor
    hidden_states: Optional[list] = None
    spike_rates: Optional[list] = None
    loss: Optional[torch.Tensor] = None
