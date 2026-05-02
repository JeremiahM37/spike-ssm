"""Tests for SpikeGPT wrapper — model loading, inference, generation."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch
from src.models.spikegpt_wrapper import SpikeGPTWrapper, SpikeGPTConfig


@pytest.fixture
def small_config():
    return SpikeGPTConfig(
        n_layers=2, n_embd=64, vocab_size=100,
        ctx_len=32, model_name="test_tiny",
    )


@pytest.fixture
def model(small_config):
    return SpikeGPTWrapper(small_config)


def test_model_loads(model):
    assert model is not None
    assert model.get_num_params() > 0


def test_forward_shape(model, small_config):
    input_ids = torch.randint(0, small_config.vocab_size, (2, 16))
    output = model(input_ids)
    assert output.logits.shape == (2, 16, small_config.vocab_size)


def test_forward_with_targets(model, small_config):
    input_ids = torch.randint(0, small_config.vocab_size, (2, 16))
    targets = torch.randint(0, small_config.vocab_size, (2, 16))
    output = model(input_ids, targets=targets)
    assert output.loss is not None
    assert output.loss.item() > 0


def test_hidden_states_returned(model, small_config):
    input_ids = torch.randint(0, small_config.vocab_size, (2, 16))
    output = model(input_ids, return_hidden_states=True)
    assert output.hidden_states is not None
    assert len(output.hidden_states) == small_config.n_layers


def test_spike_rates_returned(model, small_config):
    input_ids = torch.randint(0, small_config.vocab_size, (2, 16))
    output = model(input_ids, return_spike_rates=True)
    assert output.spike_rates is not None
    assert len(output.spike_rates) == small_config.n_layers
    for sr in output.spike_rates:
        assert (sr >= 0).all() and (sr <= 1).all()


def test_generate(model, small_config):
    prompt = torch.randint(0, small_config.vocab_size, (1, 5))
    generated = model.generate(prompt, max_new_tokens=10)
    assert generated.shape[1] == 15  # 5 prompt + 10 generated


def test_layer_params(model, small_config):
    params = model.get_layer_params()
    assert len(params) == small_config.n_layers
    assert all(p > 0 for p in params)
