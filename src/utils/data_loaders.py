"""Data loading utilities for language model training and evaluation."""

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional
import os


class CharDataset(Dataset):
    """Character-level dataset (for enwik8-style training)."""

    def __init__(self, data: str, seq_len: int = 512):
        self.seq_len = seq_len
        self.chars = sorted(set(data))
        self.char_to_idx = {c: i for i, c in enumerate(self.chars)}
        self.idx_to_char = {i: c for i, c in enumerate(self.chars)}
        self.vocab_size = len(self.chars)
        self.data = torch.tensor(
            [self.char_to_idx[c] for c in data], dtype=torch.long
        )

    def __len__(self):
        return max(0, len(self.data) - self.seq_len - 1)

    def __getitem__(self, idx):
        chunk = self.data[idx : idx + self.seq_len + 1]
        return {
            "input_ids": chunk[:-1],
            "labels": chunk[1:],
        }

    def decode(self, indices):
        return "".join(self.idx_to_char.get(i.item(), "?") for i in indices)


class SyntheticLMDataset(Dataset):
    """Synthetic dataset for pipeline testing (no download needed)."""

    def __init__(self, vocab_size: int = 50277, seq_len: int = 128, num_samples: int = 1000):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Deterministic pseudo-random based on index
        gen = torch.Generator().manual_seed(idx)
        tokens = torch.randint(0, self.vocab_size, (self.seq_len + 1,), generator=gen)
        return {
            "input_ids": tokens[:-1],
            "labels": tokens[1:],
        }


def load_enwik8(data_dir: str = "data/raw", seq_len: int = 512) -> tuple[CharDataset, CharDataset, CharDataset]:
    """Load enwik8 dataset, split into train/val/test.

    Downloads if not present. enwik8 = first 100MB of Wikipedia XML.
    """
    data_path = Path(data_dir) / "enwik8"

    if not data_path.exists():
        print(f"enwik8 not found at {data_path}.")
        print("Download from: http://mattmahoney.net/dc/enwik8.zip")
        print("Using synthetic data instead.")
        return (
            SyntheticLMDataset(256, seq_len, 5000),
            SyntheticLMDataset(256, seq_len, 500),
            SyntheticLMDataset(256, seq_len, 500),
        )

    with open(data_path, "r", errors="replace") as f:
        data = f.read()

    # Standard split: 90M train, 5M val, 5M test
    n = len(data)
    train_data = data[: int(n * 0.9)]
    val_data = data[int(n * 0.9) : int(n * 0.95)]
    test_data = data[int(n * 0.95) :]

    return (
        CharDataset(train_data, seq_len),
        CharDataset(val_data, seq_len),
        CharDataset(test_data, seq_len),
    )


class WikiText2Dataset(Dataset):
    """WikiText-2 dataset loaded from HuggingFace."""

    def __init__(self, split: str = "train", seq_len: int = 64, vocab_size: int = 50277):
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
        # Concatenate all text, then build character-level or word-level tokens
        # For simplicity, use character-level encoding
        all_text = "\n".join([row["text"] for row in ds if row["text"].strip()])
        self.seq_len = seq_len
        self.vocab_size = vocab_size

        # Build character vocab from data
        chars = sorted(set(all_text))
        self.char_to_idx = {c: i for i, c in enumerate(chars)}
        self.idx_to_char = {i: c for i, c in enumerate(chars)}
        self.actual_vocab_size = len(chars)
        self.data = torch.tensor(
            [self.char_to_idx[c] for c in all_text], dtype=torch.long
        )

    def __len__(self):
        return max(0, len(self.data) - self.seq_len - 1)

    def __getitem__(self, idx):
        chunk = self.data[idx : idx + self.seq_len + 1]
        return {
            "input_ids": chunk[:-1],
            "labels": chunk[1:],
        }

    def decode(self, indices):
        return "".join(self.idx_to_char.get(i.item(), "?") for i in indices)


def load_wikitext2(seq_len: int = 64):
    """Load WikiText-2 train/val/test splits."""
    train = WikiText2Dataset("train", seq_len)
    val = WikiText2Dataset("validation", seq_len)
    test = WikiText2Dataset("test", seq_len)
    return train, val, test


class TokenizedWikiText2(Dataset):
    """WikiText-2 tokenized with RWKV/GPT-NeoX BPE tokenizer (vocab=50277).

    This produces word-level perplexity comparable to published benchmarks,
    unlike the character-level WikiText2Dataset.
    """

    def __init__(self, split: str = "train", seq_len: int = 128, tokenizer=None):
        from datasets import load_dataset

        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
        all_text = "\n".join([row["text"] for row in ds if row["text"].strip()])

        if tokenizer is None:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("RWKV/rwkv-4-169m-pile")

        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.vocab_size = tokenizer.vocab_size  # 50277

        # Tokenize entire corpus
        tokens = tokenizer.encode(all_text)
        self.data = torch.tensor(tokens, dtype=torch.long)
        self.actual_vocab_size = self.vocab_size

    def __len__(self):
        return max(0, len(self.data) - self.seq_len - 1)

    def __getitem__(self, idx):
        chunk = self.data[idx : idx + self.seq_len + 1]
        return {
            "input_ids": chunk[:-1],
            "labels": chunk[1:],
        }


def load_tokenized_wikitext2(seq_len: int = 128, tokenizer=None):
    """Load WikiText-2 with BPE tokenizer (for use with pretrained RWKV teacher)."""
    train = TokenizedWikiText2("train", seq_len, tokenizer)
    val = TokenizedWikiText2("validation", seq_len, tokenizer)
    test = TokenizedWikiText2("test", seq_len, tokenizer)
    return train, val, test


class TokenizedWikiText103(Dataset):
    """WikiText-103 tokenized with RWKV/GPT-NeoX BPE tokenizer (vocab=50277).

    ~100M tokens — 40x more data than WikiText-2. Uses strided windowing
    to avoid creating too many overlapping samples.
    """

    def __init__(self, split: str = "train", seq_len: int = 256, tokenizer=None,
                 stride: Optional[int] = None):
        from datasets import load_dataset

        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split=split)

        if tokenizer is None:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("RWKV/rwkv-4-169m-pile")

        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.vocab_size = tokenizer.vocab_size
        self.stride = stride or seq_len  # non-overlapping by default

        # Tokenize in chunks to avoid massive string concatenation
        all_tokens = []
        chunk_texts = []
        chunk_size = 0
        CHUNK_LIMIT = 50_000_000  # ~50MB chunks
        for row in ds:
            text = row["text"]
            if not text.strip():
                continue
            chunk_texts.append(text)
            chunk_size += len(text)
            if chunk_size >= CHUNK_LIMIT:
                all_tokens.extend(tokenizer.encode("\n".join(chunk_texts)))
                chunk_texts = []
                chunk_size = 0
        if chunk_texts:
            all_tokens.extend(tokenizer.encode("\n".join(chunk_texts)))

        self.data = torch.tensor(all_tokens, dtype=torch.long)
        self.actual_vocab_size = self.vocab_size
        print(f"  TokenizedWikiText103({split}): {len(self.data):,} tokens, "
              f"stride={self.stride}, {len(self)} samples")

    def __len__(self):
        return max(0, (len(self.data) - self.seq_len - 1) // self.stride)

    def __getitem__(self, idx):
        start = idx * self.stride
        chunk = self.data[start : start + self.seq_len + 1]
        return {
            "input_ids": chunk[:-1],
            "labels": chunk[1:],
        }


class PennTreebankDataset(Dataset):
    """Character-level Penn Treebank dataset.

    Downloads PTB from HuggingFace (ptb-text-only/ptb_text_only) or falls back
    to a raw text download. Uses character-level tokenization (no external
    tokenizer needed).

    Interface matches WikiText2Dataset: returns dict with input_ids and labels,
    has actual_vocab_size property, works with create_dataloader().
    """

    # PTB raw text URLs (mirrored in multiple places)
    _URLS = {
        "train": "https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.train.txt",
        "validation": "https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.valid.txt",
        "test": "https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.test.txt",
    }

    def __init__(self, split: str = "train", seq_len: int = 64, vocab_size: int = 50277,
                 data_dir: Optional[str] = None):
        self.seq_len = seq_len
        self.vocab_size = vocab_size

        raw_text = self._load_split(split, data_dir)

        # Build character vocab from data
        chars = sorted(set(raw_text))
        self.char_to_idx = {c: i for i, c in enumerate(chars)}
        self.idx_to_char = {i: c for i, c in enumerate(chars)}
        self._actual_vocab_size = len(chars)
        self.data = torch.tensor(
            [self.char_to_idx[c] for c in raw_text], dtype=torch.long
        )

    def _load_split(self, split: str, data_dir: Optional[str]) -> str:
        """Load PTB split text, downloading if necessary."""
        import urllib.request

        if data_dir is None:
            data_dir = os.path.join(os.path.expanduser("~"), ".cache", "neuromorphic", "ptb")
        os.makedirs(data_dir, exist_ok=True)

        fname_map = {"train": "ptb.train.txt", "validation": "ptb.valid.txt", "test": "ptb.test.txt"}
        if split not in fname_map:
            raise ValueError(f"Invalid split '{split}'. Must be one of: train, validation, test")

        local_path = os.path.join(data_dir, fname_map[split])

        if not os.path.exists(local_path):
            url = self._URLS[split]
            print(f"Downloading PTB {split} split from {url}...")
            urllib.request.urlretrieve(url, local_path)

        with open(local_path, "r", encoding="utf-8") as f:
            text = f.read()

        return text

    @property
    def actual_vocab_size(self) -> int:
        return self._actual_vocab_size

    def __len__(self):
        return max(0, len(self.data) - self.seq_len - 1)

    def __getitem__(self, idx):
        chunk = self.data[idx : idx + self.seq_len + 1]
        return {
            "input_ids": chunk[:-1],
            "labels": chunk[1:],
        }

    def decode(self, indices):
        return "".join(self.idx_to_char.get(i.item(), "?") for i in indices)


class TokenizedPTB(Dataset):
    """Penn Treebank tokenized with BPE tokenizer (vocab=50277).

    Uses the same raw PTB data as PennTreebankDataset but with BPE tokenization
    for compatibility with pretrained teachers (RWKV, Mamba).
    """

    def __init__(self, split: str = "train", seq_len: int = 128, tokenizer=None):
        import urllib.request

        self.seq_len = seq_len

        if tokenizer is None:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("RWKV/rwkv-4-169m-pile")

        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size

        # Download PTB
        data_dir = os.path.join(os.path.expanduser("~"), ".cache", "neuromorphic", "ptb")
        os.makedirs(data_dir, exist_ok=True)
        fname_map = {"train": "ptb.train.txt", "validation": "ptb.valid.txt", "test": "ptb.test.txt"}
        urls = {
            "train": "https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.train.txt",
            "validation": "https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.valid.txt",
            "test": "https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.test.txt",
        }
        local_path = os.path.join(data_dir, fname_map[split])
        if not os.path.exists(local_path):
            print(f"Downloading PTB {split}...")
            urllib.request.urlretrieve(urls[split], local_path)

        with open(local_path, "r") as f:
            text = f.read()

        tokens = tokenizer.encode(text)
        self.data = torch.tensor(tokens, dtype=torch.long)
        self.actual_vocab_size = self.vocab_size

    def __len__(self):
        return max(0, len(self.data) - self.seq_len - 1)

    def __getitem__(self, idx):
        chunk = self.data[idx : idx + self.seq_len + 1]
        return {"input_ids": chunk[:-1], "labels": chunk[1:]}


class Enwik8Dataset(Dataset):
    """Byte-level enwik8 dataset (first 100MB of English Wikipedia XML dump).

    Standard split: first 90M bytes for train, next 5M for validation,
    next 5M for test. Uses byte-level tokenization (vocab size = 256 max,
    actual vocab size depends on bytes present in each split).

    Interface matches WikiText2Dataset: returns dict with input_ids and labels,
    has actual_vocab_size property, works with create_dataloader().
    """

    _URL = "http://mattmahoney.net/dc/enwik8.zip"

    def __init__(self, split: str = "train", seq_len: int = 64, vocab_size: int = 50277,
                 data_dir: Optional[str] = None):
        self.seq_len = seq_len
        self.vocab_size = vocab_size

        raw_bytes = self._load_split(split, data_dir)

        # Byte-level tokenization: each byte is its own token (0-255)
        # Use full 256 vocab to handle all possible byte values
        byte_values = list(raw_bytes)
        self._actual_vocab_size = 256  # Full byte range
        self.data = torch.tensor(byte_values, dtype=torch.long)

    def _load_split(self, split: str, data_dir: Optional[str]) -> bytes:
        """Load enwik8 split, downloading if necessary."""
        import urllib.request
        import zipfile

        if data_dir is None:
            data_dir = os.path.join(os.path.expanduser("~"), ".cache", "neuromorphic", "enwik8")
        os.makedirs(data_dir, exist_ok=True)

        raw_path = os.path.join(data_dir, "enwik8")

        if not os.path.exists(raw_path):
            zip_path = os.path.join(data_dir, "enwik8.zip")
            if not os.path.exists(zip_path):
                print(f"Downloading enwik8 from {self._URL}...")
                urllib.request.urlretrieve(self._URL, zip_path)
            print("Extracting enwik8...")
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extract("enwik8", data_dir)

        with open(raw_path, "rb") as f:
            all_data = f.read()

        # Standard split boundaries: 90M / 5M / 5M
        if split == "train":
            return all_data[:90_000_000]
        elif split == "validation":
            return all_data[90_000_000:95_000_000]
        elif split == "test":
            return all_data[95_000_000:100_000_000]
        else:
            raise ValueError(f"Invalid split '{split}'. Must be one of: train, validation, test")

    @property
    def actual_vocab_size(self) -> int:
        return self._actual_vocab_size

    def __len__(self):
        return max(0, len(self.data) - self.seq_len - 1)

    def __getitem__(self, idx):
        chunk = self.data[idx : idx + self.seq_len + 1]
        return {
            "input_ids": chunk[:-1],
            "labels": chunk[1:],
        }

    def decode(self, indices):
        return "".join(chr(i.item()) for i in indices)


def load_ptb(seq_len: int = 64, data_dir: Optional[str] = None):
    """Load Penn Treebank train/val/test splits (character-level)."""
    train = PennTreebankDataset("train", seq_len, data_dir=data_dir)
    val = PennTreebankDataset("validation", seq_len, data_dir=data_dir)
    test = PennTreebankDataset("test", seq_len, data_dir=data_dir)
    return train, val, test


def load_enwik8_dataset(seq_len: int = 64, data_dir: Optional[str] = None):
    """Load enwik8 train/val/test splits (byte-level)."""
    train = Enwik8Dataset("train", seq_len, data_dir=data_dir)
    val = Enwik8Dataset("validation", seq_len, data_dir=data_dir)
    test = Enwik8Dataset("test", seq_len, data_dir=data_dir)
    return train, val, test


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=True,
    )
