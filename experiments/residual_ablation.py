#!/usr/bin/env python3
"""Residual connection ablation: differentiate LeakyTernaryLIF from skip connections.

A reviewer will say "LeakyTernaryLIF is just a skip connection."
This ablation tests:
1. No bypass (pure ternary spikes) — baseline
2. Additive residual: spike + β*x (learned β, like standard skip)
3. Fixed interpolation: 0.95*spike + 0.05*silu(x) (no learning)
4. Learned interpolation: α*spike + (1-α)*silu(x) (LeakyTernaryLIF — ours)
5. Learned interpolation + SiLU (ours, as is)

If learned interpolation discovers structure that additive residual doesn't,
that's a clean rebuttal.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import json, math, time

from src.models.spike_mamba import (
    SpikeMambaConfig, SpikeMambaModel,
    LeakyTernaryLIF, TernaryLIF, _ContinuousPassthrough
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_float32_matmul_precision('high')

SEQ = 256
BATCH = 16
EVAL_BATCHES = 50
STEPS = 3000
LR = 2e-2

print(f"Device: {device}", flush=True)

# Load teacher
print("Loading teacher...", flush=True)
from transformers import MambaForCausalLM
teacher = MambaForCausalLM.from_pretrained("state-spaces/mamba-130m-hf").to(device)
teacher.eval()
for p in teacher.parameters():
    p.requires_grad = False

# Load tokenizer + data
print("Loading tokenizer...", flush=True)
from src.models.pretrained_teacher import PretrainedRWKVTeacher
_rt = PretrainedRWKVTeacher(); _rt.load()
tokenizer = _rt._tokenizer; del _rt
torch.cuda.empty_cache()

print("Loading data...", flush=True)
from src.utils.data_loaders import TokenizedWikiText2, create_dataloader
train_ds = TokenizedWikiText2("train", SEQ, tokenizer)
val_ds = TokenizedWikiText2("validation", SEQ, tokenizer)
train_loader = create_dataloader(train_ds, batch_size=BATCH, shuffle=True)
val_loader = create_dataloader(val_ds, batch_size=BATCH, shuffle=False)
print("Ready.\n", flush=True)


# ============================================================================
# Custom LIF variants for ablation
# ============================================================================

class AdditiveResidualLIF(nn.Module):
    """spike + β*x — standard additive skip connection."""
    def __init__(self, n_neurons, beta=0.9, base_threshold=1.5, soft_reset=True):
        super().__init__()
        self.lif = TernaryLIF(n_neurons, beta, base_threshold, soft_reset)
        # Learned scale for the residual
        self.beta_logit = nn.Parameter(torch.tensor(-3.0))  # init small (~0.05)
        self.threshold_offset = self.lif.threshold_offset
        self.register_buffer("threshold_trace", torch.zeros(n_neurons))

    @property
    def alpha(self):
        # Report effective "spike fraction" for comparison
        return 1.0 / (1.0 + torch.sigmoid(self.beta_logit).item())

    @property
    def effective_threshold(self):
        return self.lif.effective_threshold

    def forward(self, x, mem=None):
        spikes, mem = self.lif(x, mem)
        beta = torch.sigmoid(self.beta_logit)
        out = spikes + beta * x  # additive, NOT interpolation
        with torch.no_grad():
            self.threshold_trace = self.lif.threshold_trace
        return out, mem


class FixedAlphaLIF(nn.Module):
    """Fixed α*spike + (1-α)*silu(x) — no learning on α."""
    def __init__(self, n_neurons, beta=0.9, base_threshold=1.5, soft_reset=True,
                 fixed_alpha=0.95):
        super().__init__()
        self.lif = TernaryLIF(n_neurons, beta, base_threshold, soft_reset)
        self.fixed_alpha = fixed_alpha
        self.threshold_offset = self.lif.threshold_offset
        self.register_buffer("threshold_trace", torch.zeros(n_neurons))

    @property
    def alpha(self):
        return self.fixed_alpha

    @property
    def effective_threshold(self):
        return self.lif.effective_threshold

    def forward(self, x, mem=None):
        spikes, mem = self.lif(x, mem)
        out = self.fixed_alpha * spikes + (1 - self.fixed_alpha) * F.silu(x)
        with torch.no_grad():
            self.threshold_trace = self.lif.threshold_trace
        return out, mem


class AdditiveResidualPlainLIF(nn.Module):
    """spike + β*identity(x) — additive skip without SiLU nonlinearity."""
    def __init__(self, n_neurons, beta=0.9, base_threshold=1.5, soft_reset=True):
        super().__init__()
        self.lif = TernaryLIF(n_neurons, beta, base_threshold, soft_reset)
        self.beta_logit = nn.Parameter(torch.tensor(-3.0))
        self.threshold_offset = self.lif.threshold_offset
        self.register_buffer("threshold_trace", torch.zeros(n_neurons))

    @property
    def alpha(self):
        return 1.0 / (1.0 + torch.sigmoid(self.beta_logit).item())

    @property
    def effective_threshold(self):
        return self.lif.effective_threshold

    def forward(self, x, mem=None):
        spikes, mem = self.lif(x, mem)
        beta = torch.sigmoid(self.beta_logit)
        out = spikes + beta * x
        with torch.no_grad():
            self.threshold_trace = self.lif.threshold_trace
        return out, mem


def make_model_with_lif(lif_class, **lif_kwargs):
    """Create 128d/6L SpikeMamba and replace lif_out with custom LIF."""
    config = SpikeMambaConfig(
        n_layers=6, d_model=128, d_state=16, d_conv=4, expand=2,
        vocab_size=50277, ctx_len=SEQ,
        ternary=True, ternary_threshold=1.5,
        leaky_ternary=False,  # we'll replace manually
        continuous_gate=True, soft_reset=True,
        adaptive_threshold=True, surrogate="cauchy",
    )
    model = SpikeMambaModel(config).to(device)

    # Replace lif_out in each block
    for block in model.blocks:
        block.lif_out = lif_class(config.d_model, **lif_kwargs).to(device)

    return model


def get_alphas(model):
    alphas = []
    for block in model.blocks:
        lif = block.lif_out
        if hasattr(lif, 'alpha'):
            a = lif.alpha
            alphas.append(a.item() if torch.is_tensor(a) else a)
    return alphas


@torch.no_grad()
def eval_ppl(model):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for i, batch in enumerate(val_loader):
        if i >= EVAL_BATCHES: break
        ids = batch["input_ids"].to(device)
        out = model(ids)
        logits = out.logits if hasattr(out, 'logits') else out
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)),
                               ids[:, 1:].reshape(-1))
        total_loss += loss.item() * ids[:, 1:].numel()
        total_tokens += ids[:, 1:].numel()
    model.train()
    return math.exp(total_loss / total_tokens)


def train_kd(model, name, steps=STEPS):
    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"  Steps: {steps}")
    print(f"{'='*70}\n", flush=True)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params/1e6:.2f}M", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
    train_iter = iter(train_loader)
    model.train()
    history = []
    best_ppl = float('inf')

    for step in range(1, steps + 1):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        ids = batch["input_ids"].to(device)
        with torch.no_grad():
            t_logits = teacher(ids).logits[:, :-1]

        s_out = model(ids)
        s_logits = s_out.logits if hasattr(s_out, 'logits') else s_out
        s_logits = s_logits[:, :-1]
        V = min(s_logits.size(-1), t_logits.size(-1))

        kd = F.kl_div(F.log_softmax(s_logits[:,:,:V], -1),
                       F.softmax(t_logits[:,:,:V], -1), reduction='batchmean')
        ce = F.cross_entropy(s_logits[:,:,:V].reshape(-1, V), ids[:, 1:].reshape(-1))
        loss = 0.5 * kd + 0.5 * ce

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if step % 500 == 0:
            ppl = eval_ppl(model)
            alphas = get_alphas(model)
            best_ppl = min(best_ppl, ppl)
            entry = {'step': step, 'ppl': round(ppl, 1), 'best': round(best_ppl, 1),
                     'alphas': [round(a, 3) for a in alphas]}
            history.append(entry)
            alpha_str = ", ".join(f"{a:.3f}" for a in alphas) if alphas else "N/A"
            print(f"  Step {step:5d} | PPL {ppl:7.1f} | best {best_ppl:7.1f} | [{alpha_str}]",
                  flush=True)

    final_ppl = eval_ppl(model)
    return {
        'name': name,
        'final_ppl': round(final_ppl, 1),
        'best_ppl': round(best_ppl, 1),
        'final_alphas': [round(a, 3) for a in get_alphas(model)],
        'history': history,
    }


# ============================================================================
# Run ablation
# ============================================================================

results = {}
out_path = os.path.join(os.path.dirname(__file__), "residual_ablation_results.json")

def save():
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  [saved]", flush=True)

# 1. No bypass (pure ternary)
model = make_model_with_lif(TernaryLIF, beta=0.9, base_threshold=1.5, soft_reset=True)
results['no_bypass'] = train_kd(model, "No bypass (pure ternary spikes)")
del model; torch.cuda.empty_cache(); save()

# 2. Additive residual: spike + β*x
model = make_model_with_lif(AdditiveResidualLIF)
results['additive_residual'] = train_kd(model, "Additive residual: spike + β*x")
del model; torch.cuda.empty_cache(); save()

# 3. Fixed α=0.95 interpolation (no learning)
model = make_model_with_lif(FixedAlphaLIF, fixed_alpha=0.95)
results['fixed_alpha_095'] = train_kd(model, "Fixed α=0.95 interpolation")
del model; torch.cuda.empty_cache(); save()

# 4. Fixed α=0.80 interpolation
model = make_model_with_lif(FixedAlphaLIF, fixed_alpha=0.80)
results['fixed_alpha_080'] = train_kd(model, "Fixed α=0.80 interpolation")
del model; torch.cuda.empty_cache(); save()

# 5. Learned α (LeakyTernaryLIF — ours)
config = SpikeMambaConfig(
    n_layers=6, d_model=128, d_state=16, d_conv=4, expand=2,
    vocab_size=50277, ctx_len=SEQ,
    ternary=True, ternary_threshold=1.5,
    leaky_ternary=True, leaky_alpha_init=0.95,
    continuous_gate=True, soft_reset=True,
    adaptive_threshold=True, surrogate="cauchy",
)
model = SpikeMambaModel(config).to(device)
results['learned_alpha'] = train_kd(model, "Learned α (LeakyTernaryLIF — ours)")
del model; torch.cuda.empty_cache(); save()

# Summary
print("\n" + "="*70)
print("  RESIDUAL ABLATION SUMMARY")
print("="*70)
ref = results['no_bypass']['best_ppl']
for name, r in results.items():
    delta = (r['best_ppl'] / ref - 1) * 100
    alphas = r.get('final_alphas', [])
    a_str = ", ".join(f"{a:.3f}" for a in alphas) if alphas else "N/A"
    print(f"  {name:25s} | PPL {r['best_ppl']:7.1f} | vs pure {delta:+6.1f}% | [{a_str}]")
print(flush=True)
