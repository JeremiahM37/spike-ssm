#!/usr/bin/env python3
"""Analyze which tokens trigger continuous vs spiking in the dynamic alpha model.

Run inference on validation data and record per-token gate values at each layer.
Then analyze: do rare words go continuous? Common words spike? Punctuation?
Is there structure in what the model decides needs continuous precision?
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import json, math
from collections import defaultdict, Counter

from src.models.spike_mamba import (
    SpikeMambaConfig, SpikeMambaModel, DynamicLeakyTernaryLIF
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_float32_matmul_precision('high')

SEQ = 256
BATCH = 16
N_BATCHES = 50  # analyze 50 batches = 800 sequences = 204K tokens

print(f"Device: {device}", flush=True)

# Load tokenizer
print("Loading tokenizer...", flush=True)
from src.models.pretrained_teacher import PretrainedRWKVTeacher
_rt = PretrainedRWKVTeacher(); _rt.load()
tokenizer = _rt._tokenizer; del _rt

# Load data
print("Loading data...", flush=True)
from src.utils.data_loaders import TokenizedWikiText2, create_dataloader
val_ds = TokenizedWikiText2("validation", SEQ, tokenizer)
val_loader = create_dataloader(val_ds, batch_size=BATCH, shuffle=False)
print("Ready.\n", flush=True)

# Create dynamic alpha model and load weights if available
config = SpikeMambaConfig(
    n_layers=6, d_model=128, d_state=16, d_conv=4, expand=2,
    vocab_size=50277, ctx_len=SEQ,
    ternary=True, ternary_threshold=1.5,
    dynamic_alpha=True, dynamic_alpha_init_bias=3.0,
    continuous_gate=True, soft_reset=True,
    adaptive_threshold=True, surrogate="cauchy",
)
model = SpikeMambaModel(config).to(device)

# Try to load trained checkpoint
ckpt_path = os.path.join(os.path.dirname(__file__), "dynamic_alpha_checkpoints", "best.pt")
if not os.path.exists(ckpt_path):
    # Train briefly for analysis (we need trained gate weights)
    print("No checkpoint found. Training dynamic alpha model for 3K steps...", flush=True)

    # Load teacher
    from transformers import MambaForCausalLM
    teacher = MambaForCausalLM.from_pretrained("state-spaces/mamba-130m-hf").to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    train_ds = TokenizedWikiText2("train", SEQ, tokenizer)
    train_loader = create_dataloader(train_ds, batch_size=BATCH, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-2, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3000)
    train_iter = iter(train_loader)
    model.train()

    for step in range(1, 3001):
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
        ce = F.cross_entropy(s_logits[:,:,:V].reshape(-1,V), ids[:,1:].reshape(-1))
        loss = 0.5 * kd + 0.5 * ce

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if step % 500 == 0:
            print(f"  Train step {step}, loss={loss.item():.3f}", flush=True)

    del teacher, optimizer, scheduler, train_loader, train_ds
    torch.cuda.empty_cache()

    # Save checkpoint
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save(model.state_dict(), ckpt_path)
    print(f"  Saved checkpoint to {ckpt_path}", flush=True)
else:
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    print(f"Loaded checkpoint from {ckpt_path}", flush=True)


# ============================================================================
# Collect per-token gate values
# ============================================================================
print("\n" + "="*70)
print("  Collecting per-token gate values...")
print("="*70 + "\n", flush=True)

# Hook to capture gate values
gate_values = {i: [] for i in range(6)}  # per layer

def make_hook(layer_idx):
    def hook_fn(module, input, output):
        x = input[0]  # input to the LIF
        with torch.no_grad():
            gate = torch.sigmoid(module.gate_proj(x))  # [B, T, 1]
            gate_values[layer_idx].append(gate.squeeze(-1).cpu())  # [B, T]
    return hook_fn

hooks = []
for i, block in enumerate(model.blocks):
    if isinstance(block.lif_out, DynamicLeakyTernaryLIF):
        h = block.lif_out.register_forward_hook(make_hook(i))
        hooks.append(h)

# Run inference
model.eval()
all_token_ids = []
all_positions = []

with torch.no_grad():
    for batch_idx, batch in enumerate(val_loader):
        if batch_idx >= N_BATCHES:
            break
        ids = batch["input_ids"].to(device)
        _ = model(ids)
        all_token_ids.append(ids.cpu())
        if batch_idx % 10 == 0:
            print(f"  Batch {batch_idx}/{N_BATCHES}", flush=True)

# Remove hooks
for h in hooks:
    h.remove()

print(f"\n  Collected gate values for {len(all_token_ids)} batches", flush=True)

# ============================================================================
# Analysis
# ============================================================================
print("\n" + "="*70)
print("  ANALYSIS")
print("="*70 + "\n", flush=True)

# Concatenate
token_ids = torch.cat(all_token_ids, dim=0)  # [N, T]
N, T = token_ids.shape
total_tokens = N * T

for layer_idx in gate_values:
    gate_values[layer_idx] = torch.cat(gate_values[layer_idx], dim=0)  # [N, T]

print(f"Total tokens analyzed: {total_tokens:,}\n", flush=True)

# 1. Per-layer gate distribution
print("--- Per-layer gate statistics ---")
for layer_idx in range(6):
    g = gate_values[layer_idx]
    print(f"  L{layer_idx}: mean={g.mean():.4f}, std={g.std():.4f}, "
          f"min={g.min():.4f}, max={g.max():.4f}, "
          f"frac>0.9={float((g>0.9).float().mean()):.3f}, "
          f"frac<0.1={float((g<0.1).float().mean()):.3f}", flush=True)

# 2. Per-token-type analysis
print("\n--- Gate values by token frequency ---")

# Count token frequencies
token_freq = Counter()
for row in token_ids:
    for tid in row.tolist():
        token_freq[tid] += 1

# Sort tokens by frequency
sorted_tokens = sorted(token_freq.items(), key=lambda x: -x[1])
total = sum(v for _, v in sorted_tokens)

# Decode tokens for display
def decode_token(tid):
    try:
        t = tokenizer.decode([tid])
        return repr(t)
    except:
        return f"<{tid}>"

# Bin tokens into frequency quintiles
freq_bins = {"top-1%": [], "top-10%": [], "top-50%": [], "bottom-50%": [], "bottom-10%": []}
cumsum = 0
for tid, count in sorted_tokens:
    frac = cumsum / total
    if frac < 0.01:
        freq_bins["top-1%"].append(tid)
    elif frac < 0.10:
        freq_bins["top-10%"].append(tid)
    elif frac < 0.50:
        freq_bins["top-50%"].append(tid)
    elif frac < 0.90:
        freq_bins["bottom-50%"].append(tid)
    else:
        freq_bins["bottom-10%"].append(tid)
    cumsum += count

for bin_name, tids in freq_bins.items():
    if not tids:
        continue
    tid_set = set(tids)
    # Find all positions where these tokens appear
    mask = torch.zeros(N, T, dtype=torch.bool)
    for i in range(N):
        for j in range(T):
            if token_ids[i, j].item() in tid_set:
                mask[i, j] = True
    n_tok = mask.sum().item()
    if n_tok == 0:
        continue

    gates_per_layer = []
    for layer_idx in range(6):
        g = gate_values[layer_idx][mask].mean().item()
        gates_per_layer.append(g)
    gate_str = ", ".join(f"{g:.4f}" for g in gates_per_layer)
    print(f"  {bin_name:15s} ({len(tids):5d} types, {n_tok:7d} tokens): gates=[{gate_str}]", flush=True)

# 3. Specific token categories
print("\n--- Gate values by token type ---")

categories = {
    "punctuation": [],
    "numbers": [],
    "short_words": [],  # 1-3 chars
    "long_words": [],   # 8+ chars
    "whitespace": [],
    "capitalized": [],
    "the/a/an": [],
}

for tid in range(min(50277, tokenizer.vocab_size if hasattr(tokenizer, 'vocab_size') else 50277)):
    try:
        text = tokenizer.decode([tid])
    except:
        continue
    stripped = text.strip()
    if not stripped:
        categories["whitespace"].append(tid)
    elif stripped in ".,;:!?-()[]{}\"'":
        categories["punctuation"].append(tid)
    elif stripped.isdigit():
        categories["numbers"].append(tid)
    elif len(stripped) <= 3 and stripped.isalpha():
        categories["short_words"].append(tid)
    elif len(stripped) >= 8 and stripped.isalpha():
        categories["long_words"].append(tid)
    if stripped and stripped[0].isupper():
        categories["capitalized"].append(tid)
    if stripped.lower() in ("the", "a", "an"):
        categories["the/a/an"].append(tid)

for cat_name, tids in categories.items():
    if not tids:
        continue
    tid_set = set(tids)
    mask = torch.zeros(N, T, dtype=torch.bool)
    for i in range(N):
        for j in range(T):
            if token_ids[i, j].item() in tid_set:
                mask[i, j] = True
    n_tok = mask.sum().item()
    if n_tok == 0:
        continue

    gates_per_layer = []
    for layer_idx in range(6):
        g = gate_values[layer_idx][mask].mean().item()
        gates_per_layer.append(g)
    gate_str = ", ".join(f"{g:.4f}" for g in gates_per_layer)
    print(f"  {cat_name:15s} ({n_tok:7d} tokens): gates=[{gate_str}]", flush=True)

# 4. Position analysis
print("\n--- Gate values by position in sequence ---")
for pos_range in [(0, 16), (16, 64), (64, 128), (128, 192), (192, 256)]:
    s, e = pos_range
    gates_per_layer = []
    for layer_idx in range(6):
        g = gate_values[layer_idx][:, s:e].mean().item()
        gates_per_layer.append(g)
    gate_str = ", ".join(f"{g:.4f}" for g in gates_per_layer)
    print(f"  pos [{s:3d}-{e:3d}]: gates=[{gate_str}]", flush=True)

# 5. Top most-continuous and most-spiking tokens
print("\n--- Most continuous tokens (lowest gate, L2) ---")
# Use L2 since it's the most dynamic layer
l2_gates = gate_values[2]  # [N, T]
# Average gate per token type
token_gate_avg = defaultdict(list)
for i in range(N):
    for j in range(T):
        tid = token_ids[i, j].item()
        token_gate_avg[tid].append(l2_gates[i, j].item())

# Get mean gate per token
token_mean_gate = {tid: sum(vals)/len(vals) for tid, vals in token_gate_avg.items() if len(vals) >= 5}

sorted_by_gate = sorted(token_mean_gate.items(), key=lambda x: x[1])

print("  Most continuous (gate→0, want continuous precision):")
for tid, gate in sorted_by_gate[:15]:
    text = decode_token(tid)
    freq = token_freq[tid]
    print(f"    gate={gate:.4f}  freq={freq:5d}  token={text}", flush=True)

print("\n  Most spiking (gate→1, comfortable with spikes):")
for tid, gate in sorted_by_gate[-15:]:
    text = decode_token(tid)
    freq = token_freq[tid]
    print(f"    gate={gate:.4f}  freq={freq:5d}  token={text}", flush=True)

# Save full results
results = {
    'total_tokens': total_tokens,
    'per_layer_stats': {},
    'freq_bin_gates': {},
    'category_gates': {},
    'position_gates': {},
    'top_continuous_tokens': [],
    'top_spiking_tokens': [],
}

for layer_idx in range(6):
    g = gate_values[layer_idx]
    results['per_layer_stats'][f'L{layer_idx}'] = {
        'mean': round(g.mean().item(), 4),
        'std': round(g.std().item(), 4),
        'frac_gt_0.9': round(float((g>0.9).float().mean()), 4),
        'frac_lt_0.1': round(float((g<0.1).float().mean()), 4),
    }

for tid, gate in sorted_by_gate[:30]:
    results['top_continuous_tokens'].append({
        'token': decode_token(tid), 'gate': round(gate, 4), 'freq': token_freq[tid]
    })
for tid, gate in sorted_by_gate[-30:]:
    results['top_spiking_tokens'].append({
        'token': decode_token(tid), 'gate': round(gate, 4), 'freq': token_freq[tid]
    })

out_path = os.path.join(os.path.dirname(__file__), "token_analysis_results.json")
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\nSaved to {out_path}", flush=True)
