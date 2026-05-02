#!/usr/bin/env python3
"""Force real spike sparsity through direct mechanisms.

The penalty approach fails because the model games it via α.
Instead, we directly control sparsity:
1. Very high thresholds (5.0, 8.0, 12.0)
2. Top-K sparsity: only top K% neurons fire per step
3. Best threshold + TWN
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import json, math

from src.models.spike_mamba import (
    SpikeMambaConfig, SpikeMambaModel, LeakyTernaryLIF, TernaryLIF
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_float32_matmul_precision('high')

SEQ = 256
BATCH = 16
EVAL_BATCHES = 50
STEPS = 3000
LR = 2e-2

print(f"Device: {device}", flush=True)

print("Loading teacher...", flush=True)
from transformers import MambaForCausalLM
teacher = MambaForCausalLM.from_pretrained("state-spaces/mamba-130m-hf").to(device)
teacher.eval()
for p in teacher.parameters():
    p.requires_grad = False

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
# Top-K Sparse LIF: only top K% of neurons fire each step
# ============================================================================
class TopKSparseTernaryLIF(nn.Module):
    """Ternary LIF with top-K sparsity enforcement.

    After computing membrane potentials, only the top K% neurons
    (by |membrane|) are allowed to fire. Rest are clamped to 0.
    This guarantees a specific sparsity level.
    """
    def __init__(self, n_neurons, beta=0.9, base_threshold=1.5,
                 soft_reset=True, top_k_frac=0.3):
        super().__init__()
        self.n_neurons = n_neurons
        self.beta = beta
        self.base_threshold = base_threshold
        self.soft_reset = soft_reset
        self.top_k_frac = top_k_frac  # fraction of neurons allowed to fire
        self.threshold_offset = nn.Parameter(torch.zeros(n_neurons))
        self.register_buffer("threshold_trace", torch.zeros(n_neurons))
        self.register_buffer("mem", torch.zeros(n_neurons))

    @property
    def effective_threshold(self):
        return self.base_threshold + torch.tanh(self.threshold_offset) * 0.5

    def forward(self, x, mem=None):
        if mem is None:
            mem = torch.zeros_like(x)

        # LIF dynamics
        mem = self.beta * mem + x
        threshold = self.effective_threshold

        # Compute who WOULD fire
        pos_fire = mem > threshold
        neg_fire = mem < -threshold

        # Top-K enforcement: only allow top K% by |membrane| to actually fire
        B, T, D = x.shape
        k = max(1, int(D * self.top_k_frac))

        # For each (batch, time) position, find top-K neurons by |membrane|
        abs_mem = mem.abs()  # [B, T, D]
        _, topk_idx = abs_mem.topk(k, dim=-1)  # [B, T, k]

        # Create mask: only top-K neurons can fire
        topk_mask = torch.zeros_like(mem, dtype=torch.bool)
        topk_mask.scatter_(-1, topk_idx, True)

        # Apply mask to firing decisions
        pos_fire = pos_fire & topk_mask
        neg_fire = neg_fire & topk_mask

        # Ternary spike with STE
        spikes = torch.zeros_like(mem)
        spikes[pos_fire] = 1.0
        spikes[neg_fire] = -1.0

        # Soft reset
        if self.soft_reset:
            mem = mem * (1.0 - spikes.abs())
        else:
            mem = mem - spikes * threshold

        with torch.no_grad():
            self.threshold_trace = spikes.abs().mean(dim=(0, 1))

        return spikes, mem


class TopKLeakyTernaryLIF(nn.Module):
    """LeakyTernaryLIF with top-K sparsity on the spike component."""
    def __init__(self, n_neurons, beta=0.9, base_threshold=1.5,
                 soft_reset=True, init_alpha=0.95, top_k_frac=0.3):
        super().__init__()
        self.lif = TopKSparseTernaryLIF(n_neurons, beta, base_threshold,
                                         soft_reset, top_k_frac)
        self.alpha_logit = nn.Parameter(
            torch.tensor(math.log(init_alpha / (1 - init_alpha))))
        self.register_buffer("threshold_trace", torch.zeros(n_neurons))
        self.threshold_offset = self.lif.threshold_offset

    @property
    def alpha(self):
        return torch.sigmoid(self.alpha_logit)

    @property
    def effective_threshold(self):
        return self.lif.effective_threshold

    def forward(self, x, mem=None):
        spikes, mem = self.lif(x, mem)
        a = self.alpha
        out = a * spikes + (1 - a) * F.silu(x)
        with torch.no_grad():
            self.threshold_trace = self.lif.threshold_trace
        return out, mem


def make_model(threshold=1.5, top_k_frac=None):
    """Create model, optionally with top-K sparsity."""
    config = SpikeMambaConfig(
        n_layers=6, d_model=128, d_state=16, d_conv=4, expand=2,
        vocab_size=50277, ctx_len=SEQ,
        ternary=True, ternary_threshold=threshold,
        leaky_ternary=True, leaky_alpha_init=0.95,
        continuous_gate=True, soft_reset=True,
        adaptive_threshold=True, surrogate="cauchy",
    )
    model = SpikeMambaModel(config).to(device)

    # Replace LIF with top-K variant if requested
    if top_k_frac is not None:
        for block in model.blocks:
            block.lif_out = TopKLeakyTernaryLIF(
                config.d_model, beta=config.spike_beta,
                base_threshold=threshold, soft_reset=True,
                init_alpha=0.95, top_k_frac=top_k_frac
            ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: thresh={threshold}, top_k={top_k_frac}, {n_params/1e6:.2f}M", flush=True)
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
def measure_sparsity(model, n_batches=10):
    model.eval()
    total, nonzero = 0, 0
    per_layer = {i: {'total': 0, 'nz': 0} for i in range(6)}

    def make_hook(idx):
        def hook(module, input, output):
            spikes = output[0] if isinstance(output, tuple) else output
            # For LeakyTernaryLIF, we need the raw spikes from the inner LIF
            x = input[0]
            if hasattr(module, 'lif'):
                with torch.no_grad():
                    raw_spikes, _ = module.lif(x)
                    n = raw_spikes.numel()
                    nz = (raw_spikes != 0).sum().item()
            else:
                n = spikes.numel()
                nz = (spikes != 0).sum().item()
            per_layer[idx]['total'] += n
            per_layer[idx]['nz'] += nz
        return hook

    hooks = []
    for i, block in enumerate(model.blocks):
        h = block.lif_out.register_forward_hook(make_hook(i))
        hooks.append(h)

    for i, batch in enumerate(val_loader):
        if i >= n_batches: break
        ids = batch["input_ids"].to(device)
        _ = model(ids)

    for h in hooks:
        h.remove()

    for idx in per_layer:
        total += per_layer[idx]['total']
        nonzero += per_layer[idx]['nz']

    fire_rate = nonzero / total if total > 0 else 0
    layer_rates = {f'L{i}': round(per_layer[i]['nz'] / max(per_layer[i]['total'], 1), 4)
                   for i in range(6)}
    model.train()
    return round(fire_rate, 4), layer_rates


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
    print(f"{'='*70}\n", flush=True)

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
        s_logits = (s_out.logits if hasattr(s_out, 'logits') else s_out)[:, :-1]
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
            fire_rate, layer_rates = measure_sparsity(model)
            best_ppl = min(best_ppl, ppl)
            sparsity = 1.0 - fire_rate

            history.append({
                'step': step, 'ppl': round(ppl, 1),
                'fire_rate': fire_rate, 'sparsity': round(sparsity, 4),
                'alphas': [round(a, 3) for a in alphas],
            })

            a_str = ", ".join(f"{a:.3f}" for a in alphas)
            lr_str = ", ".join(f"{v:.3f}" for v in layer_rates.values())
            print(f"  Step {step:5d} | PPL {ppl:7.1f} | best {best_ppl:7.1f} | "
                  f"fire={fire_rate:.3f} sparse={sparsity:.1%}", flush=True)
            print(f"           | α=[{a_str}] | fire/layer=[{lr_str}]", flush=True)

    final_ppl = eval_ppl(model)
    final_fr, final_lr = measure_sparsity(model)
    return {
        'name': name,
        'final_ppl': round(final_ppl, 1),
        'best_ppl': round(best_ppl, 1),
        'final_fire_rate': final_fr,
        'final_sparsity': round(1.0 - final_fr, 4),
        'final_alphas': [round(a, 3) for a in get_alphas(model)],
        'final_per_layer_fire_rate': final_lr,
        'history': history,
    }


# ============================================================================
results = {}
out_path = os.path.join(os.path.dirname(__file__), "force_sparsity_results.json")

def save():
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  [saved]", flush=True)

# 1. Baseline (threshold=1.5, no top-K)
model = make_model(threshold=1.5)
results['baseline'] = train_kd(model, "Baseline (thresh=1.5)")
del model; torch.cuda.empty_cache(); save()

# 2. High threshold 5.0
model = make_model(threshold=5.0)
results['thresh_5'] = train_kd(model, "Threshold=5.0")
del model; torch.cuda.empty_cache(); save()

# 3. High threshold 8.0
model = make_model(threshold=8.0)
results['thresh_8'] = train_kd(model, "Threshold=8.0")
del model; torch.cuda.empty_cache(); save()

# 4. Top-K 30% (guaranteed 70% sparsity)
model = make_model(threshold=1.5, top_k_frac=0.3)
results['topk_30'] = train_kd(model, "Top-K 30% (70% sparsity)")
del model; torch.cuda.empty_cache(); save()

# 5. Top-K 50% (guaranteed 50% sparsity)
model = make_model(threshold=1.5, top_k_frac=0.5)
results['topk_50'] = train_kd(model, "Top-K 50% (50% sparsity)")
del model; torch.cuda.empty_cache(); save()

# 6. Top-K 10% (guaranteed 90% sparsity)
model = make_model(threshold=1.5, top_k_frac=0.1)
results['topk_10'] = train_kd(model, "Top-K 10% (90% sparsity)")
del model; torch.cuda.empty_cache(); save()

# 7. Best threshold + TWN (if threshold works)
model = make_model(threshold=5.0)
sys.path.insert(0, os.path.dirname(__file__))
from ternary_weights import replace_linear_with_ternary
replace_linear_with_ternary(model, skip_embeddings=True)
results['thresh_5_twn'] = train_kd(model, "Threshold=5.0 + TWN")
del model; torch.cuda.empty_cache(); save()

# Summary
print("\n" + "="*70)
print("  FORCE SPARSITY SUMMARY")
print("="*70)
for name, r in results.items():
    fr = r.get('final_fire_rate', 0)
    sp = r.get('final_sparsity', 0)
    print(f"  {name:25s} | PPL {r['best_ppl']:7.1f} | fire={fr:.3f} sparse={sp:.1%}")
print(flush=True)
