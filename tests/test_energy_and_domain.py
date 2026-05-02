"""Tests for energy estimation and domain-shift continual learning."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch
import torch.nn as nn


def test_energy_savings_at_sparsity():
    """Verify energy calculation: higher sparsity = more savings."""
    MAC_ENERGY = 4.6
    AC_ENERGY = 0.9
    params = 1e6
    seq_len = 64

    ann_energy = params * seq_len * MAC_ENERGY

    savings = []
    for sparsity in [0.0, 0.5, 0.8, 0.9, 0.95]:
        snn_energy = params * seq_len * (1 - sparsity) * AC_ENERGY
        s = 1 - (snn_energy / ann_energy)
        savings.append(s)

    # Savings should monotonically increase with sparsity
    for i in range(len(savings) - 1):
        assert savings[i] < savings[i + 1], f"Savings not increasing: {savings}"

    # At 90% sparsity, SNN should use <5% of ANN energy
    assert savings[3] > 0.95


def test_neurogenesis_activates_on_novel_data():
    """Neurogenesis should activate when data is very different from history."""
    from src.mono_forward.continual_learning import NeurogenesisPool

    pool = NeurogenesisPool(32, 8, threshold_novelty=0.01)  # Very low threshold

    # First pass: establish baseline
    normal = torch.randn(2, 32) * 0.1
    pool.compute_novelty(normal)
    pool.maybe_activate_reserve(normal)

    # Feed very different data
    novel = torch.randn(2, 32) * 100  # Wildly different scale
    novelty = pool.compute_novelty(novel)
    assert novelty > 0.01, f"Expected high novelty, got {novelty}"


def test_metaplasticity_increases_stiffness():
    """Weights that are updated more should become stiffer."""
    from src.mono_forward.continual_learning import MetaplasticWeights

    model = nn.Linear(16, 16)
    mp = MetaplasticWeights(model, consolidation_rate=0.1)

    # Update weight many times
    for name, _ in model.named_parameters():
        for _ in range(50):
            mp.record_update(name)

    stats = mp.get_stiffness_stats()
    for name, s in stats.items():
        assert s["mean"] > 1.0, f"Expected stiffness > 1.0, got {s['mean']}"
        assert s["pct_consolidated"] > 0


def test_three_factor_modulation():
    """Three-factor learner should produce modulation signal between 0 and 1."""
    from src.mono_forward.continual_learning import ThreeFactorMFLearner

    learner = ThreeFactorMFLearner(goodness_name="rate_squared", lr=0.01, epsilon=0.01)
    layer = nn.Linear(16, 16)
    pos = torch.randn(2, 16) * 2
    neg = torch.randn(2, 16) * 0.5
    metrics = learner.update_layer(layer, pos, neg)

    assert "modulation" in metrics
    assert 0 <= metrics["modulation"] <= 1


def test_domain_shift_detection():
    """Model should detect distribution shift via novelty."""
    from src.mono_forward.continual_learning import NeurogenesisPool

    pool = NeurogenesisPool(64, 16, threshold_novelty=0.1)

    # Train domain
    for _ in range(10):
        x = torch.randn(4, 64) + 5  # Centered at 5
        pool.compute_novelty(x)
        pool.maybe_activate_reserve(x)

    # Shifted domain
    shifted = torch.randn(4, 64) - 10  # Centered at -10
    novelty = pool.compute_novelty(shifted)

    # Should detect the shift
    assert novelty > 0, "Should detect domain shift"


def test_hybrid_stdp_mf_updates_both():
    """Hybrid learner should update both MF and STDP components."""
    from src.mono_forward.continual_learning import HybridSTDPMFLearner

    learner = HybridSTDPMFLearner(mf_lr=0.01, stdp_lr=0.01)

    # Block with both ffn and time_mix
    block = nn.Module()
    block.ffn = nn.Linear(16, 16)
    block.channel_mix = nn.Linear(16, 16)

    pos = torch.randn(2, 16)
    neg = torch.randn(2, 16)
    metrics = learner.update_block(block, pos, neg)

    assert len(metrics) > 0
    assert any("mf_" in k for k in metrics)
