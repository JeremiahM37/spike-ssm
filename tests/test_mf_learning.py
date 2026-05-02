"""Tests for Mono-Forward learning rule."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch
import torch.nn as nn
from src.mono_forward.goodness import (
    spike_count_goodness, rate_squared_goodness,
    temporal_coherence_goodness, get_goodness_fn,
)
from src.mono_forward.local_loss import mf_local_loss, contrastive_goodness_loss
from src.mono_forward.mf_learning import MFLearner, MFPerturbationLearner
from src.models.snn_layers import LIFNeuron


def test_spike_count_goodness():
    spikes = torch.tensor([[1, 0, 1, 0, 1], [0, 0, 1, 0, 0]], dtype=torch.float)
    g = spike_count_goodness(spikes)
    assert g.item() == pytest.approx(2.0)  # (3 + 1) / 2


def test_rate_squared_goodness():
    rates = torch.tensor([[0.5, 0.5], [0.0, 1.0]], dtype=torch.float)
    g = rate_squared_goodness(rates)
    # (0.25 + 0.25) + (0.0 + 1.0) = 1.5, / 2 batches = 0.75
    assert g.item() == pytest.approx(0.75)


def test_temporal_coherence_goodness_3d():
    # Coherent: all ones → high autocorrelation
    spikes_coherent = torch.ones(1, 10, 5)
    # Random: mix → lower autocorrelation
    spikes_random = torch.randint(0, 2, (1, 10, 5)).float()
    g_coh = temporal_coherence_goodness(spikes_coherent)
    g_rand = temporal_coherence_goodness(spikes_random)
    assert g_coh.item() >= g_rand.item()


def test_mf_local_loss():
    pos_g = torch.tensor(5.0)
    neg_g = torch.tensor(2.0)
    loss = mf_local_loss(pos_g, neg_g)
    assert loss.item() > 0
    assert loss.item() < 1.0  # Should be small when pos >> neg


def test_mf_local_loss_wrong_order():
    pos_g = torch.tensor(2.0)
    neg_g = torch.tensor(5.0)
    loss = mf_local_loss(pos_g, neg_g)
    # Loss should be large when neg > pos
    loss_correct = mf_local_loss(torch.tensor(5.0), torch.tensor(2.0))
    assert loss.item() > loss_correct.item()


def test_mf_learner_updates_weights():
    layer = nn.Linear(16, 16)
    neuron = LIFNeuron(threshold=0.1)  # Low threshold so spikes fire
    learner = MFLearner(goodness_name="spike_count", lr=0.1)

    weights_before = layer.weight.data.clone()
    pos = torch.randn(2, 16) * 3  # Strong signal to trigger spikes
    neg = torch.randn(2, 16) * 0.1

    metrics = learner.update_layer(layer, pos, neg, neuron)
    assert "loss" in metrics
    assert not torch.equal(weights_before, layer.weight.data)


def test_mf_perturbation_updates_weights():
    layer = nn.Linear(16, 16)
    learner = MFPerturbationLearner(
        goodness_name="spike_count", lr=0.01,
        epsilon=0.01, subset_fraction=0.1,
    )

    weights_before = layer.weight.data.clone()
    pos = torch.randn(2, 16)
    neg = torch.randn(2, 16)

    metrics = learner.update_layer(layer, pos, neg)
    assert "baseline_gap" in metrics
    assert "weights_updated" in metrics
    assert metrics["weights_updated"] > 0


def test_get_goodness_fn():
    for name in ["spike_count", "rate_squared", "temporal_coherence"]:
        fn = get_goodness_fn(name)
        result = fn(torch.rand(2, 10))
        assert isinstance(result.item(), float)

    with pytest.raises(ValueError):
        get_goodness_fn("nonexistent")
