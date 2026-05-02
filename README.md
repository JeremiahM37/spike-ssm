# spike-ssm: Spiking State-Space Models with Learned Mixed Precision

Experiments with learned spike/continuous mixing for spiking state-space language models, evaluated on [SpikingSSMs](https://github.com/shenshuaijie/SDN) (S4D backbone, AAAI 2025) and a Mamba backbone.

**Note on prior work:** The core idea of mixing a continuous auxiliary path with a spiking path is similar to Smoothed Gradient Compensation (SGC) from [SpikingMamba](https://arxiv.org/abs/2510.04595) (Huang et al., TMLR 2026). The main difference is that SGC drops the continuous path at inference, while LeakyTernaryLIF keeps it with a learned per-layer weight. The S4D result (α→0.05, nearly fully continuous) suggests the spiking path may not be earning its place on that backbone. See [Limitations](#limitations) for honest assessment.

## Key Results

**Cross-architecture ablation** (128d, ~13M params, WikiText-2, 3K steps):

| Configuration | S4D PPL | S4D Δ | Mamba PPL | Mamba Δ |
|---|---|---|---|---|
| Binary LIF (baseline) | 102.3 | — | 189.1 | — |
| + Ternary LIF | 94.7 | -7.4% | 190.9 | +1.0% |
| + **LeakyTernaryLIF** | **87.8** | **-14.2%** | **110.3** | **-41.7%** |
| + Top-K 30% | 91.8 | -10.3% | 107.4 | -43.2% |
| + Top-K + TWN | 93.0 | -9.1% | 118.9 | -37.1% |

**Spiking penalty at matched parameters:** Only 9% quality cost vs continuous Mamba student (PPL 94.8 vs 87.2).

## Contributions

### 1. LeakyTernaryLIF

A neuron with learned per-layer mixing between ternary spikes and continuous SiLU:

```python
out = α * spike + (1 - α) * silu(x)
```

- `α` is learned per layer via gradient descent
- On Mamba: α → 0.90 (keeps ~90% spiking)
- On S4D: α → 0.05 (learns to go ~95% continuous)
- The mechanism **adapts to each backbone**, discovering the right precision automatically
- Ablation confirms it's not a skip connection: SiLU nonlinearity and learning both matter

### 2. Top-K Sparsity

Only the top K% of neurons (by |membrane potential|) fire each step:

- **Top-K 30% (70% sparsity) improves quality** — acts as a regularizer
- PPL 91.0 ± 6.7 with 70% sparsity vs 99.6 ± 13.0 dense (multi-seed)
- Without top-K, actual spike sparsity is only 1-2% (almost all neurons fire every step)
- Higher thresholds don't work — adaptive thresholds compensate

### 3. Ternary Weight Quantization (TWN)

Weights quantized to {-1, 0, +1} via QAT compose well with spiking activations:

- TWN + LeakyTernaryLIF: PPL 85.1 (+5.2% vs fp32 baseline)
- The model compensates by making L0 almost fully continuous (α → 0.14)
- Only QAT discovers this compensation; PTQ and gradual annealing cannot

### Extension: Per-Token Dynamic Gating

Replace fixed α with a content-dependent gate:

```python
gate = sigmoid(W_g · stop_grad(x) + b_g)
out = gate * spike + (1 - gate) * silu(x)
```

- Numbers and punctuation spike; content words stay continuous
- Binary gate approximation works (+5.6% cost) — fully neuromorphic compatible

## Project Structure

```
src/
  models/
    spike_mamba.py       # SpikeMamba: spiking Mamba with LeakyTernaryLIF, DynamicLeakyTernaryLIF
    spiking_s4d.py       # SpikingS4D: spiking S4D with all neuron variants
    snn_layers.py        # Base SNN layers (LIF, surrogate gradients)
    pretrained_teacher.py # RWKV teacher loading
  utils/
    data_loaders.py      # WikiText-2, PTB, enwik8 loaders
    metrics.py           # Evaluation metrics
  distillation/
    logit_matching.py    # KD loss functions
  hardware_sim/
    synth_estimate.py    # FPGA resource estimation (KV260)
    verilog/             # Basic LIF neuron + spike-driven linear RTL

experiments/
    spiking_s4d_ablation.py  # Main ablation on SpikingSSMs backbone
    mamba_ablation.py        # Main ablation on Mamba backbone
    force_sparsity.py        # Top-K sparsity experiments
    residual_ablation.py     # Skip connection ablation
    continuous_baseline.py   # Continuous (non-spiking) baseline
    ternary_weights.py       # TWN/TTQ experiments
    token_analysis.py        # Per-token gate analysis
    results/                 # All experiment results (JSON)
    figures/                 # Generated figures

tests/
    test_datasets.py         # Data loader tests
```

## Quick Start

### Requirements

```
python >= 3.11
torch >= 2.0
transformers
datasets
```

### Train SpikingS4D with LeakyTernaryLIF

```python
from src.models.spiking_s4d import SpikingS4DLM, SpikingS4DConfig

config = SpikingS4DConfig(
    d_model=128, n_layers=4, d_state=32,
    neuron_type="leaky_ternary",  # or "sltt", "ternary", "topk_leaky_ternary"
    init_alpha=0.95,
    top_k_frac=0.3,  # for topk_leaky_ternary
)
model = SpikingS4DLM(config)
```

### Train SpikeMamba with LeakyTernaryLIF

```python
from src.models.spike_mamba import SpikeMambaModel, SpikeMambaConfig

config = SpikeMambaConfig(
    n_layers=6, d_model=128, d_state=16, expand=2,
    vocab_size=50277,
    ternary=True, ternary_threshold=1.5,
    leaky_ternary=True, leaky_alpha_init=0.95,
    continuous_gate=True, soft_reset=True,
)
model = SpikeMambaModel(config)
```

### Run the cross-architecture ablation

```bash
# SpikingSSMs (S4D backbone)
python experiments/spiking_s4d_ablation.py

# Mamba backbone
python experiments/mamba_ablation.py
```

## Neuron Variants

| Neuron | Description | Use case |
|---|---|---|
| `sltt` | Binary LIF (SpikingSSMs baseline) | Baseline comparison |
| `ternary` | Ternary LIF {-1, 0, +1} | Better than binary |
| `leaky_ternary` | Learned α mixing (ours) | Best quality |
| `topk_leaky_ternary` | + Top-K sparsity (ours) | Best for hardware |

## Hardware Feasibility

LeakyTernaryLIF's continuous SiLU term appears to violate the accumulate-only paradigm, but several deployment strategies exist:

- **Loihi 2**: Graded spike payloads with piecewise-linear SiLU approximation in neuron microcode
- **BrainScaleS-2**: Native hybrid spiking/continuous neuron support
- **FPGA**: Included Verilog RTL implements basic LIF neuron and spike-driven linear layers for the KV260. The LeakyTernaryLIF blend would require additional DSP resources for the SiLU pathway.

The Verilog now includes LeakyTernaryLIF with piecewise-linear SiLU approximation, top-K selector, and ternary spike-driven linear layer. Testbench passes on Icarus Verilog 11.0.

## Limitations

- All experiments at 13M parameter scale on WikiText-2
- Scaling to 100M+ untested
- Energy projections (5-8x) use idealized operation costs — not validated on real hardware
- Multi-seed variance is non-negligible (± 6.7-13.0 PPL)
- α initialization affects final pattern
- The S4D result (α→0.05) suggests the model learns to bypass spiking entirely on that backbone, which undermines the neuromorphic efficiency narrative
- LeakyTernaryLIF is conceptually similar to SGC (SpikingMamba, Huang et al. 2026) — see note at top

## Attribution

This project adapts code from:

- **[state-spaces/mamba](https://github.com/state-spaces/mamba)** — Apache 2.0, Copyright 2023 Albert Gu, Tri Dao. Block structure adapted for SpikeMamba.
- **[state-spaces/s4](https://github.com/state-spaces/s4)** — Apache 2.0, Copyright 2022 Albert Gu. S4D diagonal kernel algorithm.
- **[shenshuaijie/SDN](https://github.com/shenshuaijie/SDN)** — MIT, Copyright 2024 Shuaijie Shen. SpikingSSMs layer structure, spiking neurons, surrogate gradients.

See [NOTICE](NOTICE) for details. All upstream licenses are permissive and permit derivative works with attribution.

## License

MIT — see [LICENSE](LICENSE).
