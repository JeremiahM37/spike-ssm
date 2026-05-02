#!/usr/bin/env python3
"""FPGA synthesis resource estimation for SpikeGPT on KV260.

Estimates LUT, BRAM, DSP, and FF usage for each Verilog module and
computes total resource utilization as percentage of KV260 capacity.

Target: AMD/Xilinx Kria KV260 (XCK26)
  - 117,120 LUTs (CLBs)
  - 234,240 Flip-Flops
  - 144 DSP48E2 slices
  - 5.3 MB BRAM (144 BRAM36K = 5,184 Kb)
  - 4 GB DDR4
"""

import argparse
import json
import math
from dataclasses import dataclass, field

# =============================================================================
# KV260 XCK26 resource limits
# =============================================================================
KV260_LUTS = 117_120
KV260_FFS = 234_240
KV260_DSP48E2 = 144
KV260_BRAM36K = 144  # 36Kb each = 5,184 Kb = 648 KB
KV260_BRAM_BYTES = 144 * 36 * 1024 // 8  # 648 KB
KV260_DDR4_BYTES = 4 * 1024 * 1024 * 1024  # 4 GB


@dataclass
class ModuleResources:
    """Resource usage for one Verilog module."""
    name: str
    luts: int = 0
    ffs: int = 0
    dsps: int = 0
    bram36k: int = 0
    bram_bytes: int = 0
    description: str = ""


@dataclass
class SynthEstimate:
    """Complete synthesis estimation for the design."""
    modules: list = field(default_factory=list)
    n_embd: int = 64
    hidden_sz: int = 256
    n_blocks: int = 12
    vocab_size: int = 50277
    weight_bits: int = 8
    data_bits: int = 16
    accum_bits: int = 24

    @property
    def total_luts(self):
        return sum(m.luts for m in self.modules)

    @property
    def total_ffs(self):
        return sum(m.ffs for m in self.modules)

    @property
    def total_dsps(self):
        return sum(m.dsps for m in self.modules)

    @property
    def total_bram36k(self):
        return sum(m.bram36k for m in self.modules)

    @property
    def total_bram_bytes(self):
        return sum(m.bram_bytes for m in self.modules)


def estimate_lif_neuron(data_width=16, beta_width=8) -> ModuleResources:
    """Estimate resources for lif_neuron.v."""
    m = ModuleResources(name="lif_neuron", description="Leaky Integrate-and-Fire neuron")

    # Membrane register: data_width FFs
    m.ffs += data_width  # membrane_r
    m.ffs += data_width  # o_membrane
    m.ffs += 1           # o_spike
    m.ffs += 4           # refrac_cnt

    # Leak multiply: data_width * beta_width -> needs DSP or LUT multiplier
    # On XCK26, 16x8 multiply fits in 1 DSP48E2
    m.dsps += 1  # leak_product

    # Saturation logic, comparison, mux: ~30 LUTs
    m.luts += 30  # Comparators, saturation, mux
    m.luts += data_width  # Threshold comparison
    m.luts += 10  # Reset mux and control

    return m


def estimate_spike_linear(in_features, out_features, weight_width=8,
                           accum_width=16) -> ModuleResources:
    """Estimate resources for spike_linear.v."""
    m = ModuleResources(
        name="spike_linear",
        description=f"Spike-driven linear ({in_features}->{out_features})"
    )

    # Accumulator bank: out_features * accum_width FFs
    m.ffs += out_features * accum_width

    # Spike register: in_features bits
    m.ffs += in_features

    # State machine + counters: ~30 FFs
    m.ffs += 30

    # Weight BRAM: in_features * out_features * weight_width bits
    weight_bits = in_features * out_features * weight_width
    m.bram36k = math.ceil(weight_bits / (36 * 1024))
    m.bram_bytes = in_features * out_features * weight_width // 8

    # LUTs for state machine, address generation, muxing
    m.luts += 50  # State machine
    m.luts += 20  # Address computation
    m.luts += out_features  # Accumulator write-enable decode
    m.luts += 15  # Spike scanning logic

    # Accumulation: weight sign-extension + addition
    # Spike-gated: no DSP needed (conditional add, not multiply)
    # Uses LUT-based adders
    m.luts += accum_width * 2  # Sign-extend + add

    return m


def estimate_rwkv_time_mix(n_embd, data_width=16, weight_width=8,
                            accum_width=24) -> ModuleResources:
    """Estimate resources for rwkv_time_mix.v."""
    m = ModuleResources(
        name="rwkv_time_mix",
        description=f"RWKV time-mixing (WKV kernel, dim={n_embd})"
    )

    # Internal buffers: xk, xv, xr, k, v, r, wkv, gated = 8 * n_embd * data_width
    buffer_ffs = 8 * n_embd * data_width
    m.ffs += buffer_ffs

    # WKV recurrent state: aa, bb (accum_width) + pp (data_width) per channel
    m.ffs += n_embd * (2 * accum_width + data_width)

    # MAC accumulator + counters
    m.ffs += accum_width + 40  # mac_accum + counters + state

    # 4 linear layers (K, V, R, O): each n_embd * n_embd weights
    # Stored in shared external BRAM
    total_weight_bits = 4 * n_embd * n_embd * weight_width
    m.bram36k = math.ceil(total_weight_bits / (36 * 1024))
    m.bram_bytes = total_weight_bits // 8

    # DSP for MAC operations (time-multiplexed, 1 MAC unit)
    m.dsps += 1  # Single MAC unit for linear layers

    # DSP for fp_mul operations (time mixing interpolation)
    m.dsps += 3  # 3 parallel mixes (xk, xv, xr)

    # LUTs
    m.luts += 100  # State machine (10 states)
    m.luts += 50   # Address generation for 4 weight banks
    m.luts += 40   # fp_mul function (3 instances)
    m.luts += 60   # fp_sigmoid piecewise linear
    m.luts += 80   # WKV recurrence logic
    m.luts += n_embd  # Channel muxing

    return m


def estimate_rwkv_channel_mix(n_embd, hidden_sz, data_width=16,
                               weight_width=8, accum_width=24) -> ModuleResources:
    """Estimate resources for rwkv_channel_mix.v."""
    m = ModuleResources(
        name="rwkv_channel_mix",
        description=f"RWKV channel-mixing (FFN, dim={n_embd}, hidden={hidden_sz})"
    )

    # Buffers: xk, xr (n_embd), k (hidden_sz), kv, r (n_embd)
    m.ffs += 2 * n_embd * data_width       # xk, xr
    m.ffs += hidden_sz * data_width          # k_buf
    m.ffs += 2 * n_embd * data_width        # kv_buf, r_buf

    # MAC + counters + state
    m.ffs += accum_width + 50

    # Weights: key (n_embd * hidden_sz) + value (hidden_sz * n_embd) +
    #          receptance (n_embd * n_embd)
    key_bits = n_embd * hidden_sz * weight_width
    value_bits = hidden_sz * n_embd * weight_width
    recep_bits = n_embd * n_embd * weight_width
    total_bits = key_bits + value_bits + recep_bits
    m.bram36k = math.ceil(total_bits / (36 * 1024))
    m.bram_bytes = total_bits // 8

    # DSP for MAC (1 time-multiplexed unit)
    m.dsps += 1

    # DSP for fp_mul (2 for mixing) + sigmoid + relu_squared
    m.dsps += 2  # time_mix interpolation
    m.dsps += 1  # relu_squared (x*x)

    # LUTs
    m.luts += 80   # State machine (8 states)
    m.luts += 40   # Address generation (3 weight banks)
    m.luts += 40   # fp_mul, fp_sigmoid
    m.luts += 30   # relu_squared (comparator + multiply routing)
    m.luts += 20   # Gating logic
    m.luts += hidden_sz // 4  # Hidden dimension muxing

    return m


def estimate_spikegpt_block(n_embd, hidden_sz, data_width=16,
                             weight_width=8, accum_width=24) -> ModuleResources:
    """Estimate resources for spikegpt_block.v (excluding sub-modules)."""
    m = ModuleResources(
        name="spikegpt_block",
        description=f"SpikeGPT block controller (dim={n_embd})"
    )

    # Input/output/layernorm buffers: x_buf, x_prev, ln_buf, sub_out
    m.ffs += 4 * n_embd * data_width

    # LayerNorm accumulator
    m.ffs += accum_width  # ln_sum
    m.ffs += data_width   # ln_mean

    # Counters and state
    m.ffs += 40

    # Block controller LUTs (state machine, muxing)
    m.luts += 60   # State machine (8 states)
    m.luts += 50   # Data routing muxes
    m.luts += 30   # LayerNorm mean subtraction
    m.luts += 20   # Residual addition

    return m


def estimate_spikegpt_top(n_embd, data_width=16) -> ModuleResources:
    """Estimate resources for spikegpt_top.v (excluding block)."""
    m = ModuleResources(
        name="spikegpt_top",
        description="Top-level controller + AXI interfaces"
    )

    # Embedding buffer
    m.ffs += n_embd * data_width

    # Block output buffer
    m.ffs += n_embd * data_width

    # Control registers
    m.ffs += 32 * 8  # 8 x 32-bit registers

    # AXI-Lite slave logic
    m.ffs += 100  # State, address, data registers
    m.luts += 120  # AXI-Lite protocol handling + register decode

    # AXI-Stream logic
    m.ffs += 50
    m.luts += 60   # AXIS handshaking

    # Top state machine
    m.ffs += 40
    m.luts += 80   # State machine (7 states) + token management

    # Spike counters
    m.ffs += 64   # 2 x 32-bit counters

    return m


def estimate_mf_hls(n_embd=64, hidden_sz=256, weight_width=8) -> ModuleResources:
    """Estimate resources for MF perturbation HLS IP."""
    m = ModuleResources(
        name="mf_perturbation_hls",
        description="MF perturbation learning (HLS IP)"
    )

    # HLS typically generates ~2-3x more resources than hand-coded RTL
    hls_overhead = 2.5

    # Weight buffer (local copy): out * in * weight_bits
    weight_bits = hidden_sz * n_embd * weight_width
    m.bram36k = math.ceil(weight_bits / (36 * 1024))
    m.bram_bytes = weight_bits // 8

    # Spike rate buffers
    m.ffs += int(hidden_sz * 16 * hls_overhead)

    # MAC units for forward pass (HLS will unroll partially)
    m.dsps += 8  # Unroll factor of 8

    # LIF neuron logic (in HLS)
    m.luts += int(200 * hls_overhead)

    # Goodness computation
    m.luts += int(100 * hls_overhead)

    # Perturbation and update logic
    m.luts += int(150 * hls_overhead)

    # AXI interface logic (auto-generated by HLS)
    m.luts += 400
    m.ffs += 300

    return m


def compute_weight_memory(n_embd, hidden_sz, n_blocks, vocab_size,
                           weight_bits=8):
    """Compute total weight memory requirements."""
    # Per block:
    #   Time-mix: 4 linear layers * n_embd * n_embd = 4 * n^2
    #   Channel-mix: key(n*4n) + value(4n*n) + receptance(n*n) = 9 * n^2
    #   Time-mix params: 3 * n (mix_k, mix_v, mix_r) + 2 * n (decay, first) = 5n
    #   LayerNorm: 2 * 2 * n (2 LN with weight + bias) = 4n

    per_block_weights = (4 + 9) * n_embd * n_embd  # 13 * n^2 parameters
    per_block_params = 5 * n_embd + 4 * n_embd      # 9n parameters
    per_block_total = per_block_weights + per_block_params

    all_blocks = per_block_total * n_blocks

    # Embedding + output head
    embedding = vocab_size * n_embd
    output_head = n_embd * vocab_size  # Shared with embedding typically
    final_ln = 2 * n_embd

    total_params = all_blocks + embedding + output_head + final_ln
    total_bytes = total_params * weight_bits // 8

    return {
        "per_block_params": per_block_total,
        "all_blocks_params": all_blocks,
        "embedding_params": embedding,
        "head_params": output_head,
        "total_params": total_params,
        "total_bytes": total_bytes,
        "total_mb": total_bytes / (1024 * 1024),
    }


def run_estimation(n_embd=64, hidden_sz=256, n_blocks=12,
                    vocab_size=50277, weight_bits=8, data_bits=16,
                    accum_bits=24):
    """Run full synthesis estimation."""
    est = SynthEstimate(
        n_embd=n_embd,
        hidden_sz=hidden_sz,
        n_blocks=n_blocks,
        vocab_size=vocab_size,
        weight_bits=weight_bits,
        data_bits=data_bits,
        accum_bits=accum_bits,
    )

    # Estimate each module
    # Note: single block instance, time-multiplexed across n_blocks
    lif1 = estimate_lif_neuron(data_bits)
    lif1.name = "lif_neuron_att"
    lif1.description = "LIF neuron (attention, 1 instance)"

    lif2 = estimate_lif_neuron(data_bits)
    lif2.name = "lif_neuron_ffn"
    lif2.description = "LIF neuron (FFN, 1 instance)"

    spike_lin = estimate_spike_linear(n_embd, n_embd, weight_bits, data_bits)

    time_mix = estimate_rwkv_time_mix(n_embd, data_bits, weight_bits, accum_bits)
    chan_mix = estimate_rwkv_channel_mix(n_embd, hidden_sz, data_bits, weight_bits, accum_bits)
    block_ctrl = estimate_spikegpt_block(n_embd, hidden_sz, data_bits, weight_bits, accum_bits)
    top = estimate_spikegpt_top(n_embd, data_bits)
    mf = estimate_mf_hls(n_embd, hidden_sz, weight_bits)

    est.modules = [lif1, lif2, spike_lin, time_mix, chan_mix, block_ctrl, top, mf]

    return est


def print_report(est: SynthEstimate):
    """Print formatted resource utilization report."""
    print("=" * 80)
    print("  SpikeGPT FPGA Synthesis Resource Estimation")
    print("  Target: AMD/Xilinx Kria KV260 (XCK26)")
    print("=" * 80)

    print(f"\n  Design Parameters:")
    print(f"    Embedding dim:    {est.n_embd}")
    print(f"    Hidden dim (FFN): {est.hidden_sz}")
    print(f"    Blocks:           {est.n_blocks} (time-multiplexed on 1 HW block)")
    print(f"    Vocab size:       {est.vocab_size}")
    print(f"    Weight precision: INT{est.weight_bits}")
    print(f"    Data precision:   INT{est.data_bits}")
    print(f"    Accum precision:  INT{est.accum_bits}")

    # Weight memory analysis
    wmem = compute_weight_memory(
        est.n_embd, est.hidden_sz, est.n_blocks,
        est.vocab_size, est.weight_bits
    )
    print(f"\n  Weight Memory:")
    print(f"    Per block:        {wmem['per_block_params']:,} params")
    print(f"    All blocks:       {wmem['all_blocks_params']:,} params")
    print(f"    Embedding:        {wmem['embedding_params']:,} params")
    print(f"    Output head:      {wmem['head_params']:,} params")
    print(f"    Total params:     {wmem['total_params']:,}")
    print(f"    Total memory:     {wmem['total_mb']:.2f} MB")

    fits_bram = wmem["total_bytes"] <= KV260_BRAM_BYTES
    fits_ddr = wmem["total_bytes"] <= KV260_DDR4_BYTES
    print(f"    Fits in BRAM:     {'YES' if fits_bram else 'NO'} "
          f"({wmem['total_bytes']//1024} KB / {KV260_BRAM_BYTES//1024} KB)")
    print(f"    Fits in DDR4:     {'YES' if fits_ddr else 'NO'} "
          f"({wmem['total_mb']:.1f} MB / {KV260_DDR4_BYTES//(1024**3)} GB)")

    # Per-module breakdown
    print(f"\n  {'Module':<30s} {'LUTs':>8s} {'FFs':>8s} {'DSPs':>6s} {'BRAM36K':>8s}")
    print(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*6} {'-'*8}")

    for m in est.modules:
        print(f"  {m.name:<30s} {m.luts:>8,d} {m.ffs:>8,d} {m.dsps:>6d} {m.bram36k:>8d}")

    print(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*6} {'-'*8}")
    print(f"  {'TOTAL':<30s} {est.total_luts:>8,d} {est.total_ffs:>8,d} "
          f"{est.total_dsps:>6d} {est.total_bram36k:>8d}")

    # Utilization percentages
    lut_pct = est.total_luts / KV260_LUTS * 100
    ff_pct = est.total_ffs / KV260_FFS * 100
    dsp_pct = est.total_dsps / KV260_DSP48E2 * 100
    bram_pct = est.total_bram36k / KV260_BRAM36K * 100

    print(f"\n  Resource Utilization (% of KV260 capacity):")
    print(f"    LUTs:     {est.total_luts:>8,d} / {KV260_LUTS:>8,d}  = {lut_pct:5.1f}%"
          f"  {'OK' if lut_pct < 80 else 'WARNING' if lut_pct < 100 else 'OVERFLOW'}")
    print(f"    FFs:      {est.total_ffs:>8,d} / {KV260_FFS:>8,d}  = {ff_pct:5.1f}%"
          f"  {'OK' if ff_pct < 80 else 'WARNING' if ff_pct < 100 else 'OVERFLOW'}")
    print(f"    DSP48E2:  {est.total_dsps:>8d} / {KV260_DSP48E2:>8d}  = {dsp_pct:5.1f}%"
          f"  {'OK' if dsp_pct < 80 else 'WARNING' if dsp_pct < 100 else 'OVERFLOW'}")
    print(f"    BRAM36K:  {est.total_bram36k:>8d} / {KV260_BRAM36K:>8d}  = {bram_pct:5.1f}%"
          f"  {'OK' if bram_pct < 80 else 'WARNING' if bram_pct < 100 else 'OVERFLOW'}")

    # Performance estimates
    clock_mhz = 100
    # Cycles per token per block:
    #   LayerNorm: 2 * N_EMBD cycles (mean + subtract)
    #   TimeMix: N_EMBD (mix) + 4 * N_EMBD * (N_EMBD+1) (4 linears) + N_EMBD (WKV) + N_EMBD (gate) + N_EMBD (out linear)
    #   LIF: N_EMBD cycles
    #   ChannelMix: similar
    tm_linear_cycles = 4 * est.n_embd * (est.n_embd + 1)  # 4 linear layers
    cm_linear_cycles = (est.n_embd * (est.hidden_sz + 1) +  # key
                        est.hidden_sz +                      # relu_sq
                        est.hidden_sz * (est.n_embd + 1) +  # value (HIDDEN->N_EMBD input)
                        est.n_embd * (est.n_embd + 1))       # receptance
    ln_cycles = 2 * (2 * est.n_embd + 1)  # 2 layernorms
    lif_cycles = 2 * est.n_embd
    block_cycles = ln_cycles + tm_linear_cycles + cm_linear_cycles + lif_cycles + est.n_embd * 3

    total_cycles = block_cycles * est.n_blocks
    latency_us = total_cycles / clock_mhz
    tokens_per_sec = 1_000_000 / latency_us if latency_us > 0 else 0

    print(f"\n  Performance Estimates ({clock_mhz} MHz):")
    print(f"    Cycles per block: {block_cycles:,d}")
    print(f"    Cycles per token: {total_cycles:,d} ({est.n_blocks} blocks)")
    print(f"    Latency/token:    {latency_us:,.1f} us ({latency_us/1000:.2f} ms)")
    print(f"    Throughput:       {tokens_per_sec:,.0f} tokens/sec")

    # Energy estimate
    # Dynamic power at 100 MHz, 0.85V (XCK26):
    # ~0.5 mW per active LUT, ~0.1 mW per FF, ~5 mW per DSP, ~3 mW per BRAM36K
    power_lut_mw = est.total_luts * 0.5 / 1000  # mW -> W... actually keep in mW
    power_ff_mw = est.total_ffs * 0.1 / 1000
    power_dsp_mw = est.total_dsps * 5
    power_bram_mw = est.total_bram36k * 3
    total_power_mw = (est.total_luts * 0.5 + est.total_ffs * 0.1 +
                      est.total_dsps * 5000 / 1000 * est.total_dsps +  # DSP
                      est.total_bram36k * 3)
    # Simplified: rough estimate
    total_power_w = (est.total_luts * 0.0001 + est.total_ffs * 0.00002 +
                     est.total_dsps * 0.005 + est.total_bram36k * 0.003)

    energy_per_token_uj = total_power_w * latency_us  # W * us = uJ

    print(f"\n  Power Estimates (rough):")
    print(f"    Estimated PL power: {total_power_w*1000:.0f} mW ({total_power_w:.2f} W)")
    print(f"    Energy per token:   {energy_per_token_uj:.1f} uJ")

    # Comparison with GPU
    gpu_power_w = 300  # Typical GPU
    gpu_latency_us = 1000  # 1ms per token on GPU
    gpu_energy_uj = gpu_power_w * gpu_latency_us
    if energy_per_token_uj > 0:
        print(f"    vs GPU (300W, 1ms): {gpu_energy_uj/energy_per_token_uj:.0f}x more efficient")

    print(f"\n{'=' * 80}")

    # Warnings
    warnings = []
    if lut_pct > 100:
        warnings.append(f"LUT overflow: {lut_pct:.1f}% - reduce N_EMBD or use more time-multiplexing")
    if ff_pct > 100:
        warnings.append(f"FF overflow: {ff_pct:.1f}% - reduce buffer sizes")
    if dsp_pct > 100:
        warnings.append(f"DSP overflow: {dsp_pct:.1f}% - use LUT-based multipliers instead")
    if bram_pct > 100:
        warnings.append(f"BRAM overflow: {bram_pct:.1f}% - store weights in DDR4, stream per block")

    if warnings:
        print("\n  WARNINGS:")
        for w in warnings:
            print(f"    - {w}")
    else:
        print("\n  All resources within KV260 capacity.")

    return {
        "lut_pct": lut_pct,
        "ff_pct": ff_pct,
        "dsp_pct": dsp_pct,
        "bram_pct": bram_pct,
        "tokens_per_sec": tokens_per_sec,
        "power_w": total_power_w,
        "energy_per_token_uj": energy_per_token_uj,
    }


def main():
    parser = argparse.ArgumentParser(
        description="FPGA synthesis resource estimation for SpikeGPT on KV260"
    )
    parser.add_argument("--n-embd", type=int, default=64,
                        help="Embedding dimension (default: 64)")
    parser.add_argument("--hidden-sz", type=int, default=256,
                        help="FFN hidden dimension (default: 256 = 4*n_embd)")
    parser.add_argument("--n-blocks", type=int, default=12,
                        help="Number of transformer blocks (default: 12)")
    parser.add_argument("--vocab-size", type=int, default=50277,
                        help="Vocabulary size (default: 50277)")
    parser.add_argument("--weight-bits", type=int, default=8,
                        help="Weight precision in bits (default: 8)")
    parser.add_argument("--json", action="store_true",
                        help="Output results as JSON")
    parser.add_argument("--sweep", action="store_true",
                        help="Sweep common configurations")
    args = parser.parse_args()

    if args.sweep:
        print("Configuration Sweep:")
        print(f"{'n_embd':>8s} {'hidden':>8s} {'blocks':>8s} {'LUT%':>8s} "
              f"{'FF%':>8s} {'DSP%':>8s} {'BRAM%':>8s} {'tok/s':>10s}")
        print("-" * 76)
        configs = [
            (32, 128, 6),
            (64, 256, 6),
            (64, 256, 12),
            (128, 512, 6),
            (128, 512, 12),
            (256, 1024, 6),
            (256, 1024, 12),
            (768, 3072, 12),  # Full SpikeGPT-45M
        ]
        for n_embd, hidden_sz, n_blocks in configs:
            est = run_estimation(n_embd, hidden_sz, n_blocks, args.vocab_size)
            lut_pct = est.total_luts / KV260_LUTS * 100
            ff_pct = est.total_ffs / KV260_FFS * 100
            dsp_pct = est.total_dsps / KV260_DSP48E2 * 100
            bram_pct = est.total_bram36k / KV260_BRAM36K * 100

            clock_mhz = 100
            tm_cycles = 4 * n_embd * (n_embd + 1)
            cm_cycles = (n_embd * (hidden_sz + 1) + hidden_sz +
                         hidden_sz * (n_embd + 1) + n_embd * (n_embd + 1))
            block_cycles = 2 * (2 * n_embd + 1) + tm_cycles + cm_cycles + 2 * n_embd + 3 * n_embd
            total_cycles = block_cycles * n_blocks
            tok_s = clock_mhz * 1_000_000 / total_cycles if total_cycles > 0 else 0

            flag = ""
            if lut_pct > 100 or ff_pct > 100 or dsp_pct > 100 or bram_pct > 100:
                flag = " *OVERFLOW*"

            print(f"{n_embd:>8d} {hidden_sz:>8d} {n_blocks:>8d} {lut_pct:>7.1f}% "
                  f"{ff_pct:>7.1f}% {dsp_pct:>7.1f}% {bram_pct:>7.1f}% "
                  f"{tok_s:>9,.0f}{flag}")
        return

    est = run_estimation(
        args.n_embd, args.hidden_sz, args.n_blocks,
        args.vocab_size, args.weight_bits
    )
    results = print_report(est)

    if args.json:
        print("\nJSON output:")
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
