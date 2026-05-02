"""Custom SNN layers: LIF neurons, surrogate gradients, spiking activations."""

import torch
import torch.nn as nn
import math


class SurrogateSpike(torch.autograd.Function):
    """Surrogate gradient for non-differentiable spike function.

    Forward: Heaviside step (binary spike when membrane > threshold)
    Backward: Arctangent surrogate gradient (smooth approximation)
    """

    scale = 2.0  # Controls surrogate gradient sharpness (was 25.0 — caused vanishing gradients through deep networks)

    @staticmethod
    def forward(ctx, membrane_potential):
        ctx.save_for_backward(membrane_potential)
        return (membrane_potential > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        (membrane_potential,) = ctx.saved_tensors
        grad = grad_output / (1 + (math.pi * membrane_potential * SurrogateSpike.scale) ** 2)
        return grad


spike_fn = SurrogateSpike.apply


class LIFNeuron(nn.Module):
    """Leaky Integrate-and-Fire neuron layer.

    Dynamics per timestep:
        U[t] = beta * U[t-1] + W @ X[t]  (leak + integrate)
        S[t] = Heaviside(U[t] - threshold)  (fire)
        U[t] = U[t] * (1 - S[t])  (reset after spike)

    Args:
        beta: Membrane potential decay factor (0-1). Higher = more memory.
        threshold: Firing threshold voltage.
        reset_mechanism: "subtract" (soft reset) or "zero" (hard reset).
    """

    def __init__(self, beta=0.9, threshold=1.0, reset_mechanism="subtract"):
        super().__init__()
        self.beta = beta
        self.threshold = threshold
        self.reset_mechanism = reset_mechanism

    def forward(self, x, mem=None):
        """
        Args:
            x: Input current [batch, features] or [batch, seq, features]
            mem: Previous membrane potential (None = start from zero)

        Returns:
            spikes: Binary spike tensor, same shape as x
            mem: Updated membrane potential
        """
        if mem is None:
            mem = torch.zeros_like(x)

        mem = self.beta * mem + x
        spikes = spike_fn(mem - self.threshold)

        if self.reset_mechanism == "subtract":
            mem = mem - spikes * self.threshold
        else:
            mem = mem * (1 - spikes)

        return spikes, mem


class IFNeuron(nn.Module):
    """Integrate-and-Fire neuron (no leak). Simpler than LIF.

    Used in ANN-to-SNN conversion (FAS/LAS style) where the accumulated
    spike count over T timesteps approximates the original ANN activation.
    """

    def __init__(self, threshold=1.0):
        super().__init__()
        self.threshold = threshold

    def forward(self, x, mem=None):
        if mem is None:
            mem = torch.zeros_like(x)

        mem = mem + x
        spikes = spike_fn(mem - self.threshold)
        mem = mem - spikes * self.threshold
        return spikes, mem


class SpikingLinear(nn.Module):
    """Linear layer followed by LIF neuron."""

    def __init__(self, in_features, out_features, beta=0.9, threshold=1.0):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.neuron = LIFNeuron(beta=beta, threshold=threshold)

    def forward(self, x, mem=None):
        current = self.linear(x)
        spikes, mem = self.neuron(current, mem)
        return spikes, mem


class SpikingBlock(nn.Module):
    """A block of SpikingLinear layers with residual connections.

    Used for building generic SNN models (sensor tasks, etc.).
    SpikeGPT uses its own RWKV-based blocks instead.
    """

    def __init__(self, dim, hidden_dim, beta=0.9, threshold=1.0):
        super().__init__()
        self.layer1 = SpikingLinear(dim, hidden_dim, beta, threshold)
        self.layer2 = SpikingLinear(hidden_dim, dim, beta, threshold)

    def forward(self, x, mem1=None, mem2=None):
        spikes1, mem1 = self.layer1(x, mem1)
        spikes2, mem2 = self.layer2(spikes1, mem2)
        # Residual: add input spikes to output
        out = x + spikes2
        return out, mem1, mem2
