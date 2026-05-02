"""Pre-trained RWKV teacher model for knowledge distillation.

Downloads and wraps the RWKV-4-169M-Pile model from HuggingFace as a
teacher for distilling into SpikeGPT / SpikeRWKV-7 student models.

The model is:
- 169M parameters (fits comfortably on CPU, ~340MB)
- RWKV-4 architecture (same family as SpikeGPT)
- Pre-trained on The Pile dataset
- vocab_size=50277, n_embd=768, n_layers=12

Usage:
    teacher = PretrainedRWKVTeacher()
    teacher.load()
    output = teacher(input_ids, targets=targets, return_hidden_states=True)
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .spikegpt_wrapper import SpikeGPTOutput


# Default model and cache location
DEFAULT_MODEL_NAME = "RWKV/rwkv-4-169m-pile"
DEFAULT_CACHE_DIR = Path(__file__).parent.parent.parent / "models" / "pretrained"


class PretrainedRWKVTeacher(nn.Module):
    """Pre-trained RWKV-4-169M teacher for knowledge distillation.

    Wraps a HuggingFace RWKV model and provides the same interface as
    SpikeGPTOutput (logits, hidden_states, spike_rates, loss) so it
    can be used interchangeably in the KD pipeline.

    Attributes:
        model_name: HuggingFace model identifier.
        cache_dir: Local directory for caching downloaded weights.
        device: Device to run on (default: "cpu").
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        cache_dir: Optional[str] = None,
        device: str = "cpu",
    ):
        super().__init__()
        self.model_name = model_name
        self.cache_dir = str(cache_dir or DEFAULT_CACHE_DIR)
        self.device = device
        self._model = None
        self._tokenizer = None
        self._config = None

    def to(self, *args, **kwargs):
        result = super().to(*args, **kwargs)
        if self._model is not None:
            self._model = self._model.to(*args, **kwargs)
        # Update self.device for forward() input tensor placement
        if args:
            dev = args[0]
            if isinstance(dev, (str, torch.device)):
                self.device = str(dev)
        if "device" in kwargs:
            self.device = str(kwargs["device"])
        return result

    def load(self):
        """Download (if needed) and load the pre-trained model.

        This is called automatically on first forward pass, but can be
        called explicitly to control when the download/load happens.
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

        os.makedirs(self.cache_dir, exist_ok=True)

        print(f"[PretrainedRWKVTeacher] Loading {self.model_name}...")
        self._config = AutoConfig.from_pretrained(
            self.model_name, cache_dir=self.cache_dir
        )
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, cache_dir=self.cache_dir
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            dtype=torch.float32,
        ).to(self.device)
        self._model.eval()

        n_params = sum(p.numel() for p in self._model.parameters())
        print(
            f"[PretrainedRWKVTeacher] Loaded: {n_params / 1e6:.1f}M params, "
            f"hidden_size={self._config.hidden_size}, "
            f"layers={self._config.num_hidden_layers}, "
            f"vocab={self._config.vocab_size}"
        )

    def _ensure_loaded(self):
        """Lazy-load on first use."""
        if self._model is None:
            self.load()

    @property
    def model(self):
        self._ensure_loaded()
        return self._model

    @property
    def tokenizer(self):
        self._ensure_loaded()
        return self._tokenizer

    @property
    def config(self):
        self._ensure_loaded()
        return self._config

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        return_hidden_states: bool = False,
        return_spike_rates: bool = False,
    ) -> SpikeGPTOutput:
        """Forward pass matching SpikeGPTOutput interface.

        Args:
            input_ids: [batch, seq_len] token IDs (vocab_size=50277).
            targets: [batch, seq_len] target token IDs for loss computation.
            return_hidden_states: If True, return per-layer hidden states.
            return_spike_rates: Ignored (ANN teacher has no spikes), kept
                for interface compatibility.

        Returns:
            SpikeGPTOutput with logits, optional hidden_states, spike_rates=None, loss.
        """
        self._ensure_loaded()

        with torch.no_grad():
            outputs = self._model(
                input_ids=input_ids.to(self.device),
                output_hidden_states=return_hidden_states,
            )

        logits = outputs.logits.detach()

        # Extract hidden states if requested
        hidden_states = None
        if return_hidden_states and hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
            # HF returns (embedding_output, layer_1, ..., layer_N)
            # Skip the embedding output to match student convention
            hidden_states = [h.detach() for h in outputs.hidden_states[1:]]

        # Compute loss if targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1).to(logits.device),
            )

        return SpikeGPTOutput(
            logits=logits,
            hidden_states=hidden_states,
            spike_rates=None,  # ANN teacher has no spike rates
            loss=loss,
        )

    def get_hidden_dim(self) -> int:
        """Return hidden dimension (768 for the 169M model)."""
        self._ensure_loaded()
        return self._config.hidden_size

    def get_num_layers(self) -> int:
        """Return number of transformer layers (12 for the 169M model)."""
        self._ensure_loaded()
        return self._config.num_hidden_layers

    def get_vocab_size(self) -> int:
        """Return vocabulary size (50277 for the 169M model)."""
        self._ensure_loaded()
        return self._config.vocab_size

    def get_num_params(self) -> int:
        """Return total parameter count."""
        self._ensure_loaded()
        return sum(p.numel() for p in self._model.parameters())

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int = 40,
    ) -> torch.Tensor:
        """Generate text autoregressively.

        Args:
            input_ids: [batch, seq_len] prompt token IDs.
            max_new_tokens: Number of tokens to generate.
            temperature: Sampling temperature.
            top_k: Top-k filtering (0 = no filtering).

        Returns:
            [batch, seq_len + max_new_tokens] generated token IDs.
        """
        self._ensure_loaded()
        with torch.no_grad():
            generated = self._model.generate(
                input_ids.to(self.device),
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k if top_k > 0 else None,
                do_sample=True,
            )
        return generated


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import time

    print("=" * 60)
    print("PretrainedRWKVTeacher — self-test")
    print("=" * 60)

    # 1. Instantiate and load
    t0 = time.time()
    teacher = PretrainedRWKVTeacher()
    teacher.load()
    load_time = time.time() - t0
    print(f"\nLoad time: {load_time:.1f}s")

    # 2. Basic info
    print(f"Hidden dim:  {teacher.get_hidden_dim()}")
    print(f"Num layers:  {teacher.get_num_layers()}")
    print(f"Vocab size:  {teacher.get_vocab_size()}")
    print(f"Num params:  {teacher.get_num_params() / 1e6:.1f}M")

    # 3. Forward pass without hidden states
    print("\n--- Forward pass (no hidden states) ---")
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    t0 = time.time()
    out = teacher(input_ids)
    fwd_time = time.time() - t0
    print(f"logits shape:    {out.logits.shape}")
    print(f"hidden_states:   {out.hidden_states}")
    print(f"spike_rates:     {out.spike_rates}")
    print(f"loss:            {out.loss}")
    print(f"Forward time:    {fwd_time:.3f}s")
    assert out.logits.shape == (1, 8, 50277), f"Unexpected logits shape: {out.logits.shape}"
    assert out.hidden_states is None
    assert out.spike_rates is None
    assert out.loss is None

    # 4. Forward pass with hidden states and loss
    print("\n--- Forward pass (hidden states + loss) ---")
    targets = torch.tensor([[2, 3, 4, 5, 6, 7, 8, 9]])
    out = teacher(input_ids, targets=targets, return_hidden_states=True)
    print(f"logits shape:    {out.logits.shape}")
    print(f"hidden_states:   {len(out.hidden_states)} layers, each {out.hidden_states[0].shape}")
    print(f"loss:            {out.loss.item():.4f}")
    assert len(out.hidden_states) == 12, f"Expected 12 hidden state layers, got {len(out.hidden_states)}"
    assert out.hidden_states[0].shape == (1, 8, 768)
    assert out.loss is not None and out.loss.item() > 0

    # 5. Interface compatibility check (spike_rates always None for ANN teacher)
    print("\n--- Interface compatibility ---")
    out = teacher(input_ids, return_spike_rates=True)
    assert out.spike_rates is None, "ANN teacher should always return spike_rates=None"
    print("spike_rates=None for ANN teacher: OK")

    # 6. Forward pass with return type check
    assert isinstance(out, SpikeGPTOutput), f"Expected SpikeGPTOutput, got {type(out)}"
    print("Returns SpikeGPTOutput: OK")

    # 7. Generate test
    print("\n--- Generation test ---")
    prompt = torch.tensor([[50256]])  # <|endoftext|> token for GPT-NeoX tokenizer
    generated = teacher.generate(prompt, max_new_tokens=10, temperature=0.8)
    print(f"Generated shape: {generated.shape}")
    decoded = teacher.tokenizer.decode(generated[0].tolist())
    print(f"Generated text:  {decoded[:100]}...")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
