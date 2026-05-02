#!/usr/bin/env python3
"""Critical missing baseline: continuous (non-spiking) Mamba student at matched params.

A 13.55M continuous Mamba student distilled from Mamba-130m, using the EXACT same
training setup as our spiking models. This isolates the spiking penalty:
  spiking_penalty = spiking_PPL / continuous_PPL

If continuous gets PPL ~50, the spiking penalty is 82.9/50 = 1.66x.
If continuous gets PPL ~80, then spiking barely hurts and our approach is validated.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import json, math, time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_float32_matmul_precision('high')

SEQ = 256
BATCH = 16
EVAL_BATCHES = 50
LR = 2e-2

print(f"Device: {device}", flush=True)

# Load teacher
print("Loading teacher...", flush=True)
from transformers import MambaForCausalLM, MambaConfig
teacher = MambaForCausalLM.from_pretrained("state-spaces/mamba-130m-hf").to(device)
teacher.eval()
for p in teacher.parameters():
    p.requires_grad = False
teacher_ppl_cache = None

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


def make_continuous_student(hidden_size=240, num_layers=4):
    """Create a continuous (non-spiking) Mamba student at matched param count.

    HF MambaConfig uses hidden_size/num_hidden_layers (not d_model/n_layer).
    d=240/4L = 13.61M params ≈ our spiking 13.55M.
    d=224/6L = 13.29M params (matched depth alternative).
    """
    config = MambaConfig(
        vocab_size=50280,  # match teacher
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
        state_size=16,
        conv_kernel=4,
        expand=2,
    )
    model = MambaForCausalLM(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Continuous student: {hidden_size}d/{num_layers}L, {n_params/1e6:.2f}M params", flush=True)
    return model


def make_spiking_student(d_model=128, n_layers=6):
    """Create a spiking SpikeMamba student for comparison."""
    from src.models.spike_mamba import SpikeMambaConfig, SpikeMambaModel
    config = SpikeMambaConfig(
        n_layers=n_layers, d_model=d_model, d_state=16, d_conv=4, expand=2,
        vocab_size=50277, ctx_len=SEQ,
        ternary=True, ternary_threshold=1.5,
        leaky_ternary=True, leaky_alpha_init=0.95,
        continuous_gate=True, soft_reset=True,
        adaptive_threshold=True, surrogate="cauchy",
    )
    model = SpikeMambaModel(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Spiking student: {d_model}d/{n_layers}L, {n_params/1e6:.2f}M params", flush=True)
    return model


@torch.no_grad()
def eval_ppl(model, is_hf=False):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for i, batch in enumerate(val_loader):
        if i >= EVAL_BATCHES: break
        ids = batch["input_ids"].to(device)
        if is_hf:
            out = model(ids)
            logits = out.logits
        else:
            out = model(ids)
            logits = out.logits if hasattr(out, 'logits') else out
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)),
                               ids[:, 1:].reshape(-1))
        total_loss += loss.item() * ids[:, 1:].numel()
        total_tokens += ids[:, 1:].numel()
    model.train()
    return math.exp(total_loss / total_tokens)


def train_kd(model, name, steps, is_hf=False):
    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"  Steps: {steps}, LR: {LR}, KD: T=1.0, alpha=0.5")
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

        if is_hf:
            s_logits = model(ids).logits[:, :-1]
        else:
            s_out = model(ids)
            s_logits = (s_out.logits if hasattr(s_out, 'logits') else s_out)[:, :-1]

        targets = ids[:, 1:]
        V = min(s_logits.size(-1), t_logits.size(-1))

        kd = F.kl_div(F.log_softmax(s_logits[:,:,:V], -1),
                       F.softmax(t_logits[:,:,:V], -1), reduction='batchmean')
        ce = F.cross_entropy(s_logits[:,:,:V].reshape(-1, V), targets.reshape(-1))
        loss = 0.5 * kd + 0.5 * ce

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if step % 100 == 0 and step < 500:
            print(f"  Step {step:5d} | loss {loss.item():.3f}", flush=True)

        if step % 500 == 0:
            ppl = eval_ppl(model, is_hf=is_hf)
            best_ppl = min(best_ppl, ppl)
            history.append({'step': step, 'ppl': round(ppl, 1), 'best': round(best_ppl, 1)})
            print(f"  Step {step:5d} | PPL {ppl:7.1f} | best {best_ppl:7.1f}", flush=True)

    final_ppl = eval_ppl(model, is_hf=is_hf)
    return {
        'name': name,
        'final_ppl': round(final_ppl, 1),
        'best_ppl': round(best_ppl, 1),
        'history': history,
    }


# ============================================================================
results = {}
out_path = os.path.join(os.path.dirname(__file__), "continuous_baseline_results.json")

def save():
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  [saved]", flush=True)

# 0. Teacher PPL (for reference)
print("Evaluating teacher PPL...", flush=True)
teacher_ppl = eval_ppl(teacher, is_hf=True)
print(f"  Teacher (Mamba-130m): PPL {teacher_ppl:.1f}\n", flush=True)
results['teacher'] = {'name': 'Mamba-130m teacher', 'ppl': round(teacher_ppl, 1)}

# 1. Continuous 240d/4L (13.61M, matched params) — 5K steps
model = make_continuous_student(240, 4)
results['continuous_240d4L_5k'] = train_kd(model, "Continuous 240d/4L 13.61M (5K)", 5000, is_hf=True)
del model; torch.cuda.empty_cache(); save()

# 2. Continuous 224d/6L (13.29M, matched params+depth) — 5K steps
model = make_continuous_student(224, 6)
results['continuous_224d6L_5k'] = train_kd(model, "Continuous 224d/6L 13.29M (5K)", 5000, is_hf=True)
del model; torch.cuda.empty_cache(); save()

# 3. Spiking student 128d/6L — 5K steps (rerun for fair comparison)
model = make_spiking_student(128, 6)
results['spiking_128d6L_5k'] = train_kd(model, "Spiking LeakyTernary 128d/6L (5K)", 5000, is_hf=False)
del model; torch.cuda.empty_cache(); save()

# Summary
print("\n" + "="*70)
print("  CONTINUOUS vs SPIKING BASELINE")
print("="*70)
print(f"  Teacher (Mamba-130m): PPL {teacher_ppl:.1f}")
for name, r in results.items():
    if name == 'teacher': continue
    ppl = r['best_ppl']
    gap = ppl / teacher_ppl
    print(f"  {name:30s} | PPL {ppl:7.1f} | {gap:.2f}x teacher gap")

cont_5k = results.get('continuous_240d4L_5k', {}).get('best_ppl', 0)
cont_6L = results.get('continuous_224d6L_5k', {}).get('best_ppl', 0)
spike_5k = results.get('spiking_128d6L_5k', {}).get('best_ppl', 0)
if cont_5k > 0 and spike_5k > 0:
    penalty = spike_5k / cont_5k
    print(f"\n  SPIKING PENALTY (vs 240d/4L): {penalty:.2f}x ({spike_5k:.1f} / {cont_5k:.1f})")
    print(f"  This means spiking costs {(penalty-1)*100:.0f}% quality vs continuous at matched params.")
if cont_6L > 0 and spike_5k > 0:
    penalty = spike_5k / cont_6L
    print(f"  SPIKING PENALTY (vs 224d/6L): {penalty:.2f}x ({spike_5k:.1f} / {cont_6L:.1f})")
print(flush=True)
