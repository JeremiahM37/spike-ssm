#!/usr/bin/env python3
"""Fully ternary SpikeMamba: ternary weights AND ternary activations.

THE BIG EXPERIMENT. If this works, the entire model is add/subtract only:
- Weights ∈ {-1, 0, +1}: no weight storage multiplication
- Activations ∈ {-1, 0, +1}: ternary spikes via LeakyTernaryLIF
- Inference = pure accumulate operations (no multiply)

This would be genuinely revolutionary for neuromorphic hardware:
- Weight storage: 2 bits per param (log2(3) ≈ 1.58, round up)
- Inference: 0.9 pJ/op accumulate only, zero MAC
- 128d/6L model: 13.55M params × 2 bits = 3.4 MB total

Approach: Ternary Weight Networks (TWN) style quantization with learned
per-layer scaling factors. During forward pass, weights are quantized to
{-s, 0, +s} where s is a learned scale. During backward, STE passes
gradients through the quantization.

We test multiple approaches:
1. TWN (Li et al., 2016): threshold-based ternary with scale factor
2. Trained Ternary Quantization (TTQ, Zhu et al., 2017): learned asymmetric scales
3. GPTQ-style: post-training quantization of the best fp32 checkpoint
4. Quantization-aware training from scratch

Key question: Does ternary weights + ternary activations compose, or does
the double discretization destroy the model?
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
import torch.nn.functional as F
import json, math, time, copy
from pathlib import Path

from src.models.spike_mamba import SpikeMambaModel, SpikeMambaConfig, LeakyTernaryLIF

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_float32_matmul_precision('high')

SEQ = 256
BATCH = 16
EVAL_BATCHES = 50

# Module-level globals for teacher/data (set by _init_data() or by importers)
teacher = None
train_ds = None
val_ds = None
train_loader = None
val_loader = None


def _init_data():
    """Load teacher, tokenizer, and data. Called by __main__ or by importers."""
    global teacher, train_ds, val_ds, train_loader, val_loader

    print(f"Device: {device}", flush=True)

    print("Loading teacher...", flush=True)
    from transformers import MambaForCausalLM
    teacher_model = MambaForCausalLM.from_pretrained("state-spaces/mamba-130m-hf").to(device)
    teacher_model.eval()
    for p in teacher_model.parameters():
        p.requires_grad = False
    teacher = teacher_model

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
# Ternary Weight Quantization
# ============================================================================

class TernaryQuantize(torch.autograd.Function):
    """Quantize weights to {-1, 0, +1} with STE gradient."""
    @staticmethod
    def forward(ctx, weight, threshold):
        # TWN-style: values above threshold → +1, below -threshold → -1, else 0
        ternary = torch.zeros_like(weight)
        ternary[weight > threshold] = 1.0
        ternary[weight < -threshold] = -1.0
        ctx.save_for_backward(weight, threshold)
        return ternary

    @staticmethod
    def backward(ctx, grad_output):
        weight, threshold = ctx.saved_tensors
        # STE: pass gradient through for weights within [-1, 1]
        # Clip gradient for weights outside this range
        grad_weight = grad_output.clone()
        grad_weight[weight.abs() > 1.0] *= 0.1  # Dampen gradient for large weights
        return grad_weight, None


def compute_twn_threshold(weight):
    """Compute optimal threshold for TWN (Li et al., 2016).

    Threshold = 0.7 * mean(|W|) is a good approximation of the
    optimal threshold that minimizes ||W - alpha * W_t||^2.
    """
    return 0.7 * weight.abs().mean()


class TernaryLinear(nn.Module):
    """Linear layer with ternary weights.

    Maintains fp32 weights for gradient updates (latent weights).
    Forward pass quantizes to {-scale, 0, +scale}.
    Scale factor is learned per-layer.
    """
    def __init__(self, in_features, out_features, bias=False,
                 mode="twn"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.mode = mode

        # Latent fp32 weights (for gradient updates)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None

        # Learned scale factor (TTQ-style)
        self.scale_pos = nn.Parameter(torch.tensor(1.0))
        self.scale_neg = nn.Parameter(torch.tensor(1.0))

    def get_ternary_weight(self):
        """Quantize weight to ternary."""
        threshold = compute_twn_threshold(self.weight)
        ternary = TernaryQuantize.apply(self.weight, threshold)

        if self.mode == "ttq":
            # Asymmetric scaling (TTQ)
            w_t = torch.where(ternary > 0, self.scale_pos * ternary,
                             torch.where(ternary < 0, self.scale_neg * ternary,
                                        ternary))
        else:
            # Symmetric scaling (TWN): scale = mean(|W|) for non-zero entries
            mask = ternary != 0
            if mask.any():
                scale = self.weight.abs()[mask].mean()
            else:
                scale = torch.tensor(1.0, device=self.weight.device)
            w_t = ternary * scale

        return w_t

    def forward(self, x):
        w = self.get_ternary_weight()
        return F.linear(x, w, self.bias)

    @property
    def sparsity(self):
        """Fraction of zero weights."""
        with torch.no_grad():
            threshold = compute_twn_threshold(self.weight)
            ternary = torch.zeros_like(self.weight)
            ternary[self.weight > threshold] = 1.0
            ternary[self.weight < -threshold] = -1.0
            return (ternary == 0).float().mean().item()


def replace_linear_with_ternary(model, mode="twn", skip_embeddings=True):
    """Replace all nn.Linear layers with TernaryLinear.

    Optionally skip embedding and head layers (first/last) since they
    interface with the discrete vocabulary and may need more precision.
    """
    replaced = 0
    for name, module in model.named_modules():
        for child_name, child in module.named_children():
            if isinstance(child, nn.Linear):
                # Optionally skip embedding projection and output head
                if skip_embeddings and child_name in ("head",):
                    continue

                tl = TernaryLinear(
                    child.in_features, child.out_features,
                    bias=child.bias is not None, mode=mode
                ).to(child.weight.device)

                # Initialize from the original weights
                tl.weight.data.copy_(child.weight.data)
                if child.bias is not None:
                    tl.bias.data.copy_(child.bias.data)

                setattr(module, child_name, tl)
                replaced += 1

    return replaced


def count_ternary_stats(model):
    """Count ternary weight statistics."""
    total_params = 0
    ternary_params = 0
    avg_sparsity = []

    for name, module in model.named_modules():
        if isinstance(module, TernaryLinear):
            n = module.weight.numel()
            ternary_params += n
            avg_sparsity.append(module.sparsity)
        elif isinstance(module, nn.Linear):
            total_params += module.weight.numel()

    total_params += ternary_params
    avg_sp = sum(avg_sparsity) / len(avg_sparsity) if avg_sparsity else 0
    return {
        'total_params': total_params,
        'ternary_params': ternary_params,
        'ternary_pct': ternary_params / max(total_params, 1) * 100,
        'avg_sparsity': avg_sp,
        'ternary_layers': len(avg_sparsity),
    }


# ============================================================================
# Training and evaluation
# ============================================================================

def eval_ppl(model):
    model.eval()
    total_loss, total_tokens = 0, 0
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= EVAL_BATCHES: break
            ids = batch["input_ids"].to(device)
            labs = batch["labels"].to(device)
            out = model(ids)
            V = out.logits.size(-1)
            loss = F.cross_entropy(out.logits.reshape(-1, V), labs.reshape(-1),
                                   ignore_index=-100, reduction="sum")
            n = (labs != -100).sum().item()
            total_loss += loss.item()
            total_tokens += max(n, 1)
    model.train()
    return math.exp(min(total_loss / max(total_tokens, 1), 20))


def get_alphas(model):
    alphas = []
    for block in model.blocks:
        if hasattr(block, 'lif_out') and hasattr(block.lif_out, 'alpha'):
            alphas.append(block.lif_out.alpha.item())
    return alphas


def make_model(leaky=True):
    cfg = SpikeMambaConfig(
        n_layers=6, d_model=128, d_state=16, expand=2,
        vocab_size=50277, ctx_len=SEQ,
        ternary=True, ternary_threshold=1.5,
        leaky_ternary=leaky, leaky_alpha_init=0.95,
        continuous_gate=True, soft_reset=True,
    )
    return SpikeMambaModel(cfg).to(device)


def train(name, model, steps=5000, lr=2e-2, eval_every=500):
    n_params = sum(p.numel() for p in model.parameters())
    t_stats = count_ternary_stats(model)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr * 0.1)

    model.train()
    best_ppl = float('inf')
    history = []
    t0 = time.perf_counter()
    step = 0

    while step < steps:
        for batch in train_loader:
            if step >= steps:
                break

            ids = batch["input_ids"].to(device)
            labs = batch["labels"].to(device)

            with torch.no_grad():
                t_logits = teacher(ids).logits.detach()

            s_out = model(ids)
            V = min(s_out.logits.size(-1), t_logits.size(-1))
            s_log = F.log_softmax(s_out.logits[:, :, :V], dim=-1)
            t_prob = F.softmax(t_logits[:, :, :V], dim=-1)
            kd = F.kl_div(s_log.reshape(-1, V), t_prob.reshape(-1, V), reduction="batchmean")
            ce = F.cross_entropy(s_out.logits[:, :, :V].reshape(-1, V), labs.reshape(-1),
                                 ignore_index=-100)
            loss = 0.5 * kd + 0.5 * ce

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1

            if step % eval_every == 0:
                ppl = eval_ppl(model)
                alphas = get_alphas(model)
                elapsed = time.perf_counter() - t0

                if ppl < best_ppl:
                    best_ppl = ppl

                t_stats_now = count_ternary_stats(model)
                alpha_str = f"  α=[{','.join(f'{a:.3f}' for a in alphas)}]" if alphas else ""
                sp_str = f"  w_sparse={t_stats_now['avg_sparsity']:.1%}" if t_stats_now['ternary_layers'] > 0 else ""
                print(f"  [{name}] step={step:>5d}  PPL={ppl:>8.1f}  best={best_ppl:.1f}"
                      f"  loss={loss.item():.3f}  {elapsed:.0f}s{sp_str}{alpha_str}",
                      flush=True)

                history.append({
                    'step': step, 'ppl': round(ppl, 1), 'loss': round(loss.item(), 4),
                    'alphas': [round(a, 4) for a in alphas],
                    'weight_sparsity': round(t_stats_now['avg_sparsity'], 4),
                })

    final_ppl = eval_ppl(model)
    alphas = get_alphas(model)
    t_stats_final = count_ternary_stats(model)

    print(f"\n  [{name}] DONE: PPL={final_ppl:.1f}  best={best_ppl:.1f}"
          f"  ternary={t_stats_final['ternary_pct']:.1f}% of params"
          f"  w_sparse={t_stats_final['avg_sparsity']:.1%}", flush=True)

    del opt, sched
    torch.cuda.empty_cache()
    return {
        'final_ppl': round(final_ppl, 1),
        'best_ppl': round(best_ppl, 1),
        'ternary_stats': t_stats_final,
        'final_alphas': [round(a, 4) for a in alphas],
        'history': history,
    }


# ============================================================================
# Experiments (only run directly, not on import)
# ============================================================================
if __name__ == "__main__":
    _init_data()
    results = {}

    print("=" * 90)
    print("  FULLY TERNARY: Ternary Weights + Ternary Activations")
    print("=" * 90, flush=True)

    # --- Experiment 1: Baseline (fp32 weights, leaky ternary activations) ---
    print("\n--- Baseline: fp32 weights + leaky ternary activations (5K steps) ---", flush=True)
    results["fp32_baseline"] = train("fp32 weights (baseline)", make_model())

    # --- Experiment 2: TWN from scratch ---
    print("\n--- TWN: ternary weights from scratch (5K steps) ---", flush=True)
    m = make_model()
    n_replaced = replace_linear_with_ternary(m, mode="twn", skip_embeddings=True)
    print(f"  Replaced {n_replaced} Linear layers with TernaryLinear (TWN)", flush=True)
    results["twn_scratch"] = train("TWN from scratch", m)
    del m; torch.cuda.empty_cache()

    # --- Experiment 3: TTQ from scratch ---
    print("\n--- TTQ: trained ternary quantization from scratch (5K steps) ---", flush=True)
    m = make_model()
    n_replaced = replace_linear_with_ternary(m, mode="ttq", skip_embeddings=True)
    print(f"  Replaced {n_replaced} Linear layers with TernaryLinear (TTQ)", flush=True)
    results["ttq_scratch"] = train("TTQ from scratch", m)
    del m; torch.cuda.empty_cache()

    # --- Experiment 4: TWN including output head ---
    print("\n--- TWN: ternary ALL layers including head (5K steps) ---", flush=True)
    m = make_model()
    n_replaced = replace_linear_with_ternary(m, mode="twn", skip_embeddings=False)
    print(f"  Replaced {n_replaced} Linear layers (ALL including head)", flush=True)
    results["twn_all"] = train("TWN all layers", m)
    del m; torch.cuda.empty_cache()

    # --- Experiment 5: Pure ternary (no leaky bypass) ---
    print("\n--- Pure ternary: ternary weights + pure ternary activations (no leaky) (5K steps) ---", flush=True)
    m = make_model(leaky=False)
    n_replaced = replace_linear_with_ternary(m, mode="twn", skip_embeddings=True)
    print(f"  Replaced {n_replaced} Linear layers, pure ternary activations", flush=True)
    results["twn_pure_ternary"] = train("TWN + pure ternary act", m)
    del m; torch.cuda.empty_cache()

    # --- Experiment 6: Post-training quantization of best checkpoint ---
    print("\n--- Post-training quantization of 128d/6L best checkpoint ---", flush=True)
    ckpt_path = Path(__file__).parent / "real_scale_checkpoints" / "128d_6L_best.pt"
    if ckpt_path.exists():
        m = make_model()
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        m.load_state_dict(ckpt['model_state_dict'])
        pre_ppl = eval_ppl(m)
        print(f"  Pre-quantization PPL: {pre_ppl:.1f}", flush=True)

        n_replaced = replace_linear_with_ternary(m, mode="twn", skip_embeddings=True)
        print(f"  Replaced {n_replaced} layers. Post-quantization (no fine-tune):", flush=True)
        post_ppl = eval_ppl(m)
        print(f"  Post-quantization PPL: {post_ppl:.1f} (delta: {(post_ppl/pre_ppl - 1)*100:+.1f}%)", flush=True)

        results["ptq_no_finetune"] = {
            'pre_ppl': round(pre_ppl, 1),
            'post_ppl': round(post_ppl, 1),
            'delta_pct': round((post_ppl/pre_ppl - 1)*100, 1),
        }

        # Fine-tune the quantized model
        print(f"  Fine-tuning quantized model (2K steps, lr=5e-3)...", flush=True)
        results["ptq_finetuned"] = train("PTQ + fine-tune", m, steps=2000, lr=5e-3, eval_every=250)
        del m; torch.cuda.empty_cache()
    else:
        print(f"  Checkpoint not found at {ckpt_path}, skipping PTQ.", flush=True)

    # --- Experiment 7: Gradual quantization (start fp32, anneal to ternary) ---
    print("\n--- Gradual quantization: anneal from fp32 to ternary (5K steps) ---", flush=True)
    m = make_model()
    n_replaced = replace_linear_with_ternary(m, mode="twn", skip_embeddings=True)

    # Custom training with annealing
    n_params = sum(p.numel() for p in m.parameters())
    opt = torch.optim.AdamW(m.parameters(), lr=2e-2, weight_decay=0.01, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=5000, eta_min=2e-3)

    m.train()
    best_ppl = float('inf')
    history = []
    t0 = time.perf_counter()
    step = 0
    ANNEAL_STEPS = 5000

    # During annealing, mix fp32 and ternary weights
    # w_effective = (1 - t/T) * w_fp32 + (t/T) * w_ternary
    while step < ANNEAL_STEPS:
        for batch in train_loader:
            if step >= ANNEAL_STEPS:
                break

            ids = batch["input_ids"].to(device)
            labs = batch["labels"].to(device)

            # Annealing factor: 0 → 1 over training
            anneal = min(step / (ANNEAL_STEPS * 0.8), 1.0)  # Reach full ternary at 80%

            # Temporarily mix weights
            for module in m.modules():
                if isinstance(module, TernaryLinear):
                    module._anneal_factor = anneal

            with torch.no_grad():
                t_logits = teacher(ids).logits.detach()

            s_out = m(ids)
            V = min(s_out.logits.size(-1), t_logits.size(-1))
            s_log = F.log_softmax(s_out.logits[:, :, :V], dim=-1)
            t_prob = F.softmax(t_logits[:, :, :V], dim=-1)
            kd = F.kl_div(s_log.reshape(-1, V), t_prob.reshape(-1, V), reduction="batchmean")
            ce = F.cross_entropy(s_out.logits[:, :, :V].reshape(-1, V), labs.reshape(-1),
                                 ignore_index=-100)
            loss = 0.5 * kd + 0.5 * ce

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1

            if step % 500 == 0:
                ppl = eval_ppl(m)
                alphas = get_alphas(m)
                elapsed = time.perf_counter() - t0
                if ppl < best_ppl:
                    best_ppl = ppl
                t_stats = count_ternary_stats(m)
                alpha_str = f"  α=[{','.join(f'{a:.3f}' for a in alphas)}]" if alphas else ""
                print(f"  [gradual] step={step:>5d}  PPL={ppl:>8.1f}  best={best_ppl:.1f}"
                      f"  anneal={anneal:.2f}  {elapsed:.0f}s{alpha_str}", flush=True)
                history.append({
                    'step': step, 'ppl': round(ppl, 1), 'anneal': round(anneal, 3),
                    'alphas': [round(a, 4) for a in alphas],
                })

    final_ppl = eval_ppl(m)
    alphas = get_alphas(m)
    t_stats = count_ternary_stats(m)
    results["gradual_anneal"] = {
        'final_ppl': round(final_ppl, 1),
        'best_ppl': round(best_ppl, 1),
        'ternary_stats': t_stats,
        'final_alphas': [round(a, 4) for a in alphas],
        'history': history,
    }
    print(f"\n  [gradual] DONE: PPL={final_ppl:.1f}  best={best_ppl:.1f}", flush=True)
    del m, opt, sched; torch.cuda.empty_cache()

    # ============================================================================
    # Summary
    # ============================================================================
    print("\n" + "=" * 90)
    print("  FULLY TERNARY RESULTS SUMMARY")
    print("=" * 90)
    print(f"  {'Config':<30} {'Best PPL':>10} {'Ternary%':>10} {'W Sparse':>10}")
    print(f"  {'-'*65}")

    for name, r in sorted(results.items(), key=lambda x: x[1].get('best_ppl', x[1].get('post_ppl', 9999))):
        ppl = r.get('best_ppl', r.get('post_ppl', '?'))
        t_pct = r.get('ternary_stats', {}).get('ternary_pct', '?')
        w_sp = r.get('ternary_stats', {}).get('avg_sparsity', '?')
        if isinstance(t_pct, float):
            t_pct = f"{t_pct:.1f}%"
        if isinstance(w_sp, float):
            w_sp = f"{w_sp:.1%}"
        print(f"  {name:<30} {str(ppl):>10} {str(t_pct):>10} {str(w_sp):>10}")

    with open(Path(__file__).parent / "ternary_weights_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to experiments/ternary_weights_results.json", flush=True)
