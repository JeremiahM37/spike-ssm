#!/usr/bin/env python3
"""Same 5-step ablation on SpikeMamba backbone for comparison with SpikingS4D.

Matches spiking_s4d_ablation.py exactly: same steps, LR, KD setup, eval.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import json, math, time

from src.models.spike_mamba import SpikeMambaConfig, SpikeMambaModel, LeakyTernaryLIF, TernaryLIF

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

# Top-K LIF for Mamba (reused from force_sparsity)
class TopKSparseTernaryLIF(nn.Module):
    def __init__(self, n_neurons, beta=0.9, base_threshold=1.5,
                 soft_reset=True, top_k_frac=0.3):
        super().__init__()
        self.beta = beta
        self.base_threshold = base_threshold
        self.soft_reset = soft_reset
        self.top_k_frac = top_k_frac
        self.threshold_offset = nn.Parameter(torch.zeros(n_neurons))
        self.register_buffer("threshold_trace", torch.zeros(n_neurons))

    @property
    def effective_threshold(self):
        return self.base_threshold + torch.tanh(self.threshold_offset) * 0.5

    def forward(self, x, mem=None):
        if mem is None:
            mem = torch.zeros_like(x)
        mem = self.beta * mem + x
        threshold = self.effective_threshold
        pos_fire = mem > threshold
        neg_fire = mem < -threshold
        B, T, D = x.shape
        k = max(1, int(D * self.top_k_frac))
        _, topk_idx = mem.abs().topk(k, dim=-1)
        topk_mask = torch.zeros_like(mem, dtype=torch.bool)
        topk_mask.scatter_(-1, topk_idx, True)
        pos_fire = pos_fire & topk_mask
        neg_fire = neg_fire & topk_mask
        spikes = torch.zeros_like(mem)
        spikes[pos_fire] = 1.0
        spikes[neg_fire] = -1.0
        if self.soft_reset:
            mem = mem * (1.0 - spikes.abs())
        else:
            mem = mem - spikes * threshold
        with torch.no_grad():
            self.threshold_trace = spikes.abs().mean(dim=(0, 1))
        return spikes, mem


class TopKLeakyTernaryLIF(nn.Module):
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


def make_mamba(neuron_mode="leaky", top_k_frac=None):
    """Create SpikeMamba 128d/6L with specified neuron type."""
    # Binary LIF = ternary=False, leaky=False
    # Ternary LIF = ternary=True, leaky=False
    # LeakyTernary = ternary=True, leaky=True
    # TopK+Leaky = replace after creation
    config = SpikeMambaConfig(
        n_layers=6, d_model=128, d_state=16, d_conv=4, expand=2,
        vocab_size=50277, ctx_len=SEQ,
        ternary=(neuron_mode != "binary"),
        ternary_threshold=1.5,
        leaky_ternary=(neuron_mode == "leaky"),
        leaky_alpha_init=0.95,
        continuous_gate=True, soft_reset=True,
        adaptive_threshold=(neuron_mode == "binary"),  # binary uses adaptive, ternary uses fixed
        surrogate="cauchy",
    )
    model = SpikeMambaModel(config).to(device)

    if top_k_frac is not None:
        for block in model.blocks:
            block.lif_out = TopKLeakyTernaryLIF(
                config.d_model, beta=config.spike_beta,
                base_threshold=config.ternary_threshold, soft_reset=True,
                init_alpha=0.95, top_k_frac=top_k_frac
            ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Mamba {neuron_mode}, topk={top_k_frac}, {n_params/1e6:.2f}M params", flush=True)
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


@torch.no_grad()
def measure_sparsity(model, n_batches=5):
    model.eval()
    total, nonzero = 0, 0
    def hook_fn(module, input, output):
        nonlocal total, nonzero
        x = input[0]
        if hasattr(module, 'lif'):
            raw, _ = module.lif(x)
        else:
            raw = output[0] if isinstance(output, tuple) else output
        total += raw.numel()
        nonzero += (raw.abs() > 0.5).sum().item()
    hooks = [block.lif_out.register_forward_hook(hook_fn) for block in model.blocks]
    for i, batch in enumerate(val_loader):
        if i >= n_batches: break
        ids = batch["input_ids"].to(device)
        _ = model(ids)
    for h in hooks: h.remove()
    model.train()
    return round(nonzero / total, 4) if total > 0 else 0


def train_kd(model, name, steps=STEPS, use_twn=False):
    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"{'='*70}", flush=True)

    if use_twn:
        sys.path.insert(0, os.path.dirname(__file__))
        from ternary_weights import replace_linear_with_ternary
        n = replace_linear_with_ternary(model, skip_embeddings=True)
        print(f"  Applied TWN to {n} layers", flush=True)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {n_params/1e6:.2f}M", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
    train_iter = iter(train_loader)
    model.train()
    history = []
    best_ppl = float('inf')
    t0 = time.time()

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
            fire_rate = measure_sparsity(model)
            alphas = get_alphas(model)
            best_ppl = min(best_ppl, ppl)
            elapsed = time.time() - t0

            entry = {'step': step, 'ppl': round(ppl, 1), 'fire_rate': fire_rate,
                     'alphas': [round(a, 3) for a in alphas]}
            history.append(entry)

            a_str = ", ".join(f"{a:.3f}" for a in alphas) if alphas else "N/A"
            print(f"  Step {step:5d} | PPL {ppl:7.1f} | best {best_ppl:7.1f} | "
                  f"fire={fire_rate:.3f} | {elapsed:.0f}s | [{a_str}]", flush=True)

    final_ppl = eval_ppl(model)
    final_fr = measure_sparsity(model)
    return {
        'name': name,
        'final_ppl': round(final_ppl, 1),
        'best_ppl': round(best_ppl, 1),
        'final_fire_rate': final_fr,
        'final_sparsity': round(1.0 - final_fr, 4),
        'final_alphas': [round(a, 3) for a in get_alphas(model)],
        'n_params': sum(p.numel() for p in model.parameters()),
        'history': history,
    }


# ============================================================================
results = {}
out_path = os.path.join(os.path.dirname(__file__), "mamba_ablation_results.json")

def save():
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  [saved]", flush=True)

# 1. Binary LIF baseline (closest to SpikingSSM's SLTT on Mamba)
model = make_mamba("binary")
results['1_baseline_binary'] = train_kd(model, "Mamba + binary LIF (baseline)")
del model; torch.cuda.empty_cache(); save()

# 2. Ternary LIF
model = make_mamba("ternary")
results['2_ternary_lif'] = train_kd(model, "Mamba + ternary LIF")
del model; torch.cuda.empty_cache(); save()

# 3. LeakyTernaryLIF
model = make_mamba("leaky")
results['3_leaky_ternary'] = train_kd(model, "Mamba + LeakyTernaryLIF")
del model; torch.cuda.empty_cache(); save()

# 4. + Top-K 30%
model = make_mamba("leaky", top_k_frac=0.3)
results['4_topk_30'] = train_kd(model, "Mamba + LeakyTernary + Top-K 30%")
del model; torch.cuda.empty_cache(); save()

# 5. + TWN
model = make_mamba("leaky", top_k_frac=0.3)
results['5_topk_twn'] = train_kd(model, "Mamba + LeakyTernary + Top-K + TWN", use_twn=True)
del model; torch.cuda.empty_cache(); save()

# Summary
print("\n" + "="*70)
print("  MAMBA ABLATION SUMMARY")
print("="*70)
ref_ppl = results['1_baseline_binary']['best_ppl']
for name, r in sorted(results.items()):
    delta = (r['best_ppl'] / ref_ppl - 1) * 100
    fr = r.get('final_fire_rate', 0)
    a = r.get('final_alphas', [])
    a_str = ", ".join(f"{x:.2f}" for x in a) if a else "N/A"
    print(f"  {name:35s} | PPL {r['best_ppl']:7.1f} | Δ {delta:+6.1f}% | "
          f"fire={fr:.1%} | [{a_str}]")
print(flush=True)
