"""Tests for hardware constraint checking and profiling."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from src.hardware_sim.kv260_profile import (
    KV260Budget, profile_spiking_lm, check_constraints,
)
from src.hardware_sim.fixed_point import FixedPointSimulator
from src.models.quantized_snn import quantize_tensor
import torch


@pytest.fixture
def budget():
    return KV260Budget()


def test_profile_45m_fits(budget):
    profile = profile_spiking_lm(
        n_layers=12, n_embd=768, vocab_size=50277,
        ctx_len=512, total_params_m=45, budget=budget,
    )
    assert profile.fits_in_ddr
    assert not profile.fits_in_bram  # 45MB > 5MB BRAM
    violations = check_constraints(profile, budget)
    assert len(violations) == 0  # Should fit


def test_profile_260m_fits_ddr(budget):
    profile = profile_spiking_lm(
        n_layers=24, n_embd=1024, vocab_size=50277,
        ctx_len=512, total_params_m=260, budget=budget,
    )
    assert profile.fits_in_ddr


def test_quantize_tensor_int8():
    t = torch.randn(100, 100) * 5
    q, scale = quantize_tensor(t, bits=8)
    # All values should be within INT8 range * scale
    max_val = 127 * scale
    assert q.abs().max().item() <= max_val + 1e-6


def test_quantize_tensor_preserves_shape():
    t = torch.randn(32, 64)
    q, _ = quantize_tensor(t, bits=8)
    assert q.shape == t.shape


def test_fixed_point_membrane():
    sim = FixedPointSimulator(weight_bits=8, membrane_bits=16)
    weights = torch.randn(32, 16)
    spikes = (torch.rand(4, 16) > 0.5).float()  # Binary
    membrane = sim.simulate_membrane_accumulation(weights, spikes)
    assert membrane.shape == (4, 32)


def test_energy_savings():
    sim = FixedPointSimulator()
    savings = sim.estimate_energy_savings(spike_sparsity=0.7, baseline_ops=1000000)
    assert savings["energy_reduction_pct"] > 90  # Binary + 70% sparsity → huge savings
    assert savings["snn_ops"] < savings["baseline_ops"]
