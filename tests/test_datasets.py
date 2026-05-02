"""Tests for PennTreebankDataset and Enwik8Dataset loaders."""

import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.data_loaders import (
    PennTreebankDataset,
    Enwik8Dataset,
    create_dataloader,
)


# ---------------------------------------------------------------------------
# Penn Treebank tests
# ---------------------------------------------------------------------------

class TestPennTreebankDataset:
    """Tests for the character-level PTB dataset."""

    @pytest.fixture(scope="class")
    def ptb_train(self):
        return PennTreebankDataset(split="train", seq_len=32)

    @pytest.fixture(scope="class")
    def ptb_val(self):
        return PennTreebankDataset(split="validation", seq_len=32)

    @pytest.fixture(scope="class")
    def ptb_test(self):
        return PennTreebankDataset(split="test", seq_len=32)

    def test_loads_without_error(self, ptb_train):
        """PTB train split loads successfully."""
        assert ptb_train is not None
        assert len(ptb_train) > 0

    def test_all_splits_load(self, ptb_train, ptb_val, ptb_test):
        """All three PTB splits load."""
        assert len(ptb_train) > 0
        assert len(ptb_val) > 0
        assert len(ptb_test) > 0
        # Train should be largest
        assert len(ptb_train) > len(ptb_val)
        assert len(ptb_train) > len(ptb_test)

    def test_shapes(self, ptb_train):
        """Returned tensors have correct shapes."""
        item = ptb_train[0]
        assert "input_ids" in item
        assert "labels" in item
        assert item["input_ids"].shape == (32,)
        assert item["labels"].shape == (32,)
        assert item["input_ids"].dtype == torch.long
        assert item["labels"].dtype == torch.long

    def test_labels_shifted(self, ptb_train):
        """Labels are shifted by one position relative to input_ids."""
        item = ptb_train[0]
        item2 = ptb_train[1]
        # Labels[i] should equal input_ids from position i+1
        # Since chunk = data[idx:idx+seq_len+1], input_ids = chunk[:-1], labels = chunk[1:]
        # So labels[0] == data[idx+1] and for the next sample input_ids[0] == data[idx+1]
        assert item["labels"][0].item() == item2["input_ids"][0].item()

    def test_vocab_size_reasonable(self, ptb_train):
        """PTB character vocab should be small (roughly 50-100 unique chars)."""
        vs = ptb_train.actual_vocab_size
        assert 30 <= vs <= 200, f"PTB char vocab size {vs} seems unreasonable"

    def test_create_dataloader(self, ptb_train):
        """create_dataloader works with PTB dataset."""
        loader = create_dataloader(ptb_train, batch_size=2, shuffle=False)
        batch = next(iter(loader))
        assert batch["input_ids"].shape == (2, 32)
        assert batch["labels"].shape == (2, 32)

    def test_invalid_split_raises(self):
        """Invalid split name raises ValueError."""
        with pytest.raises(ValueError):
            PennTreebankDataset(split="invalid")

    def test_decode(self, ptb_train):
        """decode method returns a string."""
        item = ptb_train[0]
        decoded = ptb_train.decode(item["input_ids"])
        assert isinstance(decoded, str)
        assert len(decoded) == 32


# ---------------------------------------------------------------------------
# Enwik8 tests
# ---------------------------------------------------------------------------

class TestEnwik8Dataset:
    """Tests for the byte-level enwik8 dataset."""

    @pytest.fixture(scope="class")
    def enwik8_train(self):
        return Enwik8Dataset(split="train", seq_len=32)

    @pytest.fixture(scope="class")
    def enwik8_val(self):
        return Enwik8Dataset(split="validation", seq_len=32)

    @pytest.fixture(scope="class")
    def enwik8_test(self):
        return Enwik8Dataset(split="test", seq_len=32)

    def test_loads_without_error(self, enwik8_train):
        """Enwik8 train split loads successfully."""
        assert enwik8_train is not None
        assert len(enwik8_train) > 0

    def test_all_splits_load(self, enwik8_train, enwik8_val, enwik8_test):
        """All three enwik8 splits load."""
        assert len(enwik8_train) > 0
        assert len(enwik8_val) > 0
        assert len(enwik8_test) > 0
        # Train (90M) >> val (5M) > test (5M)
        assert len(enwik8_train) > len(enwik8_val)

    def test_split_sizes(self, enwik8_train, enwik8_val, enwik8_test):
        """Split sizes match expected 90M/5M/5M byte boundaries."""
        seq_len = 32
        # len = data_size - seq_len - 1
        train_bytes = len(enwik8_train) + seq_len + 1
        val_bytes = len(enwik8_val) + seq_len + 1
        test_bytes = len(enwik8_test) + seq_len + 1
        assert train_bytes == 90_000_000
        assert val_bytes == 5_000_000
        assert test_bytes == 5_000_000

    def test_shapes(self, enwik8_train):
        """Returned tensors have correct shapes."""
        item = enwik8_train[0]
        assert "input_ids" in item
        assert "labels" in item
        assert item["input_ids"].shape == (32,)
        assert item["labels"].shape == (32,)
        assert item["input_ids"].dtype == torch.long
        assert item["labels"].dtype == torch.long

    def test_byte_range(self, enwik8_train):
        """All token values should be in valid byte range [0, 255]."""
        item = enwik8_train[0]
        assert item["input_ids"].min() >= 0
        assert item["input_ids"].max() <= 255

    def test_vocab_size_reasonable(self, enwik8_train):
        """Enwik8 byte vocab should be <= 256."""
        vs = enwik8_train.actual_vocab_size
        assert 50 <= vs <= 256, f"Enwik8 byte vocab size {vs} seems unreasonable"

    def test_create_dataloader(self, enwik8_train):
        """create_dataloader works with enwik8 dataset."""
        loader = create_dataloader(enwik8_train, batch_size=2, shuffle=False)
        batch = next(iter(loader))
        assert batch["input_ids"].shape == (2, 32)
        assert batch["labels"].shape == (2, 32)

    def test_invalid_split_raises(self):
        """Invalid split name raises ValueError."""
        with pytest.raises(ValueError):
            Enwik8Dataset(split="invalid")

    def test_decode(self, enwik8_train):
        """decode method returns a string."""
        item = enwik8_train[0]
        decoded = enwik8_train.decode(item["input_ids"])
        assert isinstance(decoded, str)
        assert len(decoded) == 32
