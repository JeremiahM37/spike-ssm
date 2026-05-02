#!/usr/bin/env python3
"""Ablation: SpikingSSMs baseline → +LeakyTernaryLIF → +Top-K → +TWN.

Building on Shen et al. (AAAI 2025) SpikingSSMs architecture.
Each row adds one modification, showing additive improvement.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import json, math, time

from src.models.spiking_s4d import SpikingS4DLM, SpikingS4DConfig

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


def apply_twn(model):
    """Replace nn.Linear layers (except head) with ternary weights."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    from ternary_weights import replace_linear_with_ternary
    n = replace_linear_with_ternary(model, skip_embeddings=True)
    print(f"  Applied TWN to {n} layers", flush=True)
    return n


@torch.no_grad()
def eval_ppl(model):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for i, batch in enumerate(val_loader):
        if i >= EVAL_BATCHES: break
        ids = batch["input_ids"].to(device)
        out = model(ids)
        logits = out.logits
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)),
                               ids[:, 1:].reshape(-1))
        total_loss += loss.item() * ids[:, 1:].numel()
        total_tokens += ids[:, 1:].numel()
    model.train()
    return math.exp(total_loss / total_tokens)


@torch.no_grad()
def measure_sparsity(model, n_batches=5):
    """Measure actual spike fire rate."""
    model.eval()
    total, nonzero = 0, 0

    def hook_fn(module, input, output):
        nonlocal total, nonzero
        # output from the neuron — check what fires
        if isinstance(output, torch.Tensor):
            spikes = output
        else:
            return
        total += spikes.numel()
        nonzero += (spikes.abs() > 0.5).sum().item()

    hooks = []
    for layer in model.body.layers:
        h = layer.neuron.register_forward_hook(hook_fn)
        hooks.append(h)

    for i, batch in enumerate(val_loader):
        if i >= n_batches: break
        ids = batch["input_ids"].to(device)
        _ = model(ids)

    for h in hooks:
        h.remove()
    model.train()
    fire_rate = nonzero / total if total > 0 else 0
    return round(fire_rate, 4)


def train_kd(model, name, steps=STEPS, use_twn=False):
    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"{'='*70}", flush=True)

    if use_twn:
        apply_twn(model)

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
        s_logits = s_out.logits[:, :-1]
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
            alphas = model.get_alphas()
            best_ppl = min(best_ppl, ppl)
            elapsed = time.time() - t0

            entry = {
                'step': step, 'ppl': round(ppl, 1),
                'fire_rate': fire_rate,
                'alphas': [round(a, 3) for a in alphas],
            }
            history.append(entry)

            a_str = ", ".join(f"{a:.3f}" for a in alphas) if alphas else "binary"
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
        'final_alphas': [round(a, 3) for a in model.get_alphas()],
        'n_params': sum(p.numel() for p in model.parameters()),
        'history': history,
    }


# ============================================================================
results = {}
out_path = os.path.join(os.path.dirname(__file__), "spiking_s4d_ablation_results.json")

def save():
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  [saved]", flush=True)

# --- Config ---
D_MODEL = 128
N_LAYERS = 4
D_STATE = 32

# 1. SpikingSSMs baseline: SLTT binary LIF (their approach)
cfg = SpikingS4DConfig(d_model=D_MODEL, n_layers=N_LAYERS, d_state=D_STATE,
                       neuron_type="sltt")
model = SpikingS4DLM(cfg).to(device)
results['1_baseline_sltt'] = train_kd(model, "SpikingSSM baseline (SLTT binary LIF)")
del model; torch.cuda.empty_cache(); save()

# 2. + Ternary LIF (upgrade from binary to ternary spikes)
cfg = SpikingS4DConfig(d_model=D_MODEL, n_layers=N_LAYERS, d_state=D_STATE,
                       neuron_type="ternary")
model = SpikingS4DLM(cfg).to(device)
results['2_ternary_lif'] = train_kd(model, "+ Ternary LIF")
del model; torch.cuda.empty_cache(); save()

# 3. + LeakyTernaryLIF (our contribution: learned alpha mixing)
cfg = SpikingS4DConfig(d_model=D_MODEL, n_layers=N_LAYERS, d_state=D_STATE,
                       neuron_type="leaky_ternary", init_alpha=0.95)
model = SpikingS4DLM(cfg).to(device)
results['3_leaky_ternary'] = train_kd(model, "+ LeakyTernaryLIF (learned alpha)")
del model; torch.cuda.empty_cache(); save()

# 4. + Top-K 30% sparsity
cfg = SpikingS4DConfig(d_model=D_MODEL, n_layers=N_LAYERS, d_state=D_STATE,
                       neuron_type="topk_leaky_ternary", init_alpha=0.95, top_k_frac=0.3)
model = SpikingS4DLM(cfg).to(device)
results['4_topk_30'] = train_kd(model, "+ Top-K 30% sparsity")
del model; torch.cuda.empty_cache(); save()

# 5. + TWN ternary weights (all our contributions)
cfg = SpikingS4DConfig(d_model=D_MODEL, n_layers=N_LAYERS, d_state=D_STATE,
                       neuron_type="topk_leaky_ternary", init_alpha=0.95, top_k_frac=0.3)
model = SpikingS4DLM(cfg).to(device)
results['5_topk_twn'] = train_kd(model, "+ TWN ternary weights (full stack)", use_twn=True)
del model; torch.cuda.empty_cache(); save()

# Summary
print("\n" + "="*70)
print("  SPIKINGS4D ABLATION: baseline → our contributions")
print("="*70)
ref_ppl = results['1_baseline_sltt']['best_ppl']
for name, r in sorted(results.items()):
    delta = (r['best_ppl'] / ref_ppl - 1) * 100
    fr = r.get('final_fire_rate', '?')
    sp = r.get('final_sparsity', '?')
    fr_str = f"{fr:.1%}" if isinstance(fr, float) else "?"
    sp_str = f"{sp:.1%}" if isinstance(sp, float) else "?"
    a = r.get('final_alphas', [])
    a_str = ", ".join(f"{x:.2f}" for x in a) if a else "binary"
    print(f"  {name:30s} | PPL {r['best_ppl']:7.1f} | Δ {delta:+6.1f}% | "
          f"fire={fr_str} sparse={sp_str} | [{a_str}]")
print(flush=True)
