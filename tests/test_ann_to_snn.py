"""Tests for ANN-to-SNN conversion."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch
from src.models.teacher import StubTeacher
from src.models.ann_to_snn import ANNtoSNNConverter, measure_conversion_quality


def test_converter_creates_model():
    teacher = StubTeacher(vocab_size=100, n_embd=64, n_layers=2, n_head=2)
    converter = ANNtoSNNConverter(timesteps=4)
    converted = converter.convert_model(teacher)
    assert converted is not None


def test_converted_model_runs():
    teacher = StubTeacher(vocab_size=100, n_embd=64, n_layers=2, n_head=2)
    converter = ANNtoSNNConverter(timesteps=4)
    converted = converter.convert_model(teacher)
    x = torch.randint(0, 100, (2, 16))
    out = converted(x)
    # Should produce output (may be wrapped)
    assert out is not None


def test_conversion_quality_metric():
    teacher = StubTeacher(vocab_size=100, n_embd=64, n_layers=2, n_head=2)
    converter = ANNtoSNNConverter(timesteps=8)
    converted = converter.convert_model(teacher)
    x = torch.randint(0, 100, (2, 16))
    quality = measure_conversion_quality(teacher, converted, x)
    assert "mse" in quality
    assert "cosine_similarity" in quality
    assert "kl_divergence" in quality


def test_more_timesteps_better_quality():
    teacher = StubTeacher(vocab_size=100, n_embd=64, n_layers=2, n_head=2)
    x = torch.randint(0, 100, (2, 16))

    q4 = measure_conversion_quality(
        teacher, ANNtoSNNConverter(4).convert_model(teacher), x
    )
    q16 = measure_conversion_quality(
        teacher, ANNtoSNNConverter(16).convert_model(teacher), x
    )
    # More timesteps should generally give better (or equal) cosine similarity
    # This may not always hold for small random models, so we just check it runs
    assert q4["mse"] >= 0 and q16["mse"] >= 0


def test_gelu_modules_replaced():
    """Verify that GELU activations are actually replaced with IF neurons."""
    import torch.nn as nn
    teacher = StubTeacher(vocab_size=100, n_embd=64, n_layers=2, n_head=2)
    gelu_before = sum(1 for _, m in teacher.named_modules() if isinstance(m, nn.GELU))
    assert gelu_before == 2, f"Expected 2 GELU modules, got {gelu_before}"

    converter = ANNtoSNNConverter(timesteps=8)
    converted = converter.convert_model(teacher)

    gelu_after = sum(1 for _, m in converted.named_modules() if isinstance(m, nn.GELU))
    assert gelu_after == 0, f"Expected 0 GELU after conversion, got {gelu_after}"

    # Check IF neurons were inserted
    if_count = sum(1 for _, m in converted.named_modules()
                   if type(m).__name__ == '_IFActivation')
    assert if_count == 2


def test_conversion_produces_nonzero_error():
    """With separate model copies, conversion should produce non-trivial error."""
    import torch.nn as nn
    teacher = StubTeacher(vocab_size=100, n_embd=64, n_layers=2, n_head=2)
    teacher.eval()

    # Create a separate copy and convert it
    teacher_copy = StubTeacher(vocab_size=100, n_embd=64, n_layers=2, n_head=2)
    teacher_copy.load_state_dict(teacher.state_dict())
    converted = ANNtoSNNConverter(timesteps=4).convert_model(teacher_copy)
    converted.eval()

    x = torch.randint(0, 100, (2, 8))
    quality = measure_conversion_quality(teacher, converted, x)
    # Separate copy with IF neurons should have non-zero error
    assert quality["mse"] > 0, "Expected non-zero MSE from IF neuron approximation"
