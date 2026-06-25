# Hardware RTL Verification

**Scope: functional RTL simulation only.** The Verilog in this directory is
verified by **behavioral simulation under [Icarus Verilog](https://github.com/steveicarus/iverilog)**.
It has **not** been synthesized, placed-and-routed, or run on real silicon /
an FPGA board. The KV260 resource numbers reported elsewhere come from the
analytical estimator `../synth_estimate.py`, **not** from Vivado synthesis or
place-and-route (Vivado is not part of this flow).

## How to reproduce

```bash
cd src/hardware_sim/verilog
./run_tests.sh        # or:  make test
```

Requires `iverilog` + `vvp` (Icarus Verilog 11.x or newer;
`apt-get install iverilog`). The runner compiles and runs every testbench and
exits non-zero if any testbench reports a failure, error, or timeout. Build
artifacts land in `build/` (git-ignored).

## Testbenches

| Testbench | DUT(s) exercised | Checks |
|---|---|---|
| `tb_lif_neuron.v` | `lif_neuron.v` (hard + soft reset) | reset state, sub-threshold integration, leak decay, spike firing, hard/soft reset, inhibition, enable gating, spike train |
| `tb_spike_linear.v` | `spike_linear.v` | zero/single/multi/all/sparse spike accumulation vs golden weight matrix, busy/done timing, back-to-back ops, negative weights |
| `tb_leaky_ternary_lif.v` | `leaky_ternary_lif.v` | ternary +1/-1/0 firing, SiLU-blended output range, top-K gating block/unblock |

Each testbench self-checks with `$display` assertions and prints
`ALL TESTS PASSED` only when its internal error count is zero.

## Sample run log

Captured on this machine. Toolchain: **Icarus Verilog 12.0 (stable)**. The GPL
license banner printed by `vvp -V` is elided for brevity; nothing else is
trimmed.

```
==============================================================
Icarus Verilog version 12.0 (stable) ()
Icarus Verilog runtime version 12.0 (stable) ()
==============================================================

########## tb_lif_neuron ##########
VCD info: dumpfile tb_lif_neuron.vcd opened for output.
Test 1: Reset state
  membrane=0 spike=0
Test 2: Sub-threshold integration
  After 1 cycle: membrane=100 spike=0
Test 3: Accumulation with leak (beta=0.5)
  After 2 cycles: membrane=150 spike=0
  After 3 cycles: membrane=175 spike=0
Test 4: Spike generation
  Large current: membrane=287 spike=1
  membrane=200 spike=0
  membrane=300 spike=1
  SPIKE DETECTED at membrane=300
  membrane=200 spike=0
  membrane=300 spike=1
  SPIKE DETECTED at membrane=300
  membrane=200 spike=0
Test 5: Hard reset verification
  After spike (hard reset): membrane=100
Test 6: Negative current (inhibition)
  Negative current: membrane=-50 spike=0
Test 7: Enable gating
  With en=0, large current: membrane=0 (should be 0)
Test 8: Soft reset behavior
  Hard: membrane=300 spike=1 | Soft: membrane=300 spike=1
  Hard: membrane=300 spike=1 | Soft: membrane=322 spike=1
Test 9: Rapid spike train with constant high input
  Spikes in 20 cycles (high input): 20

========================================
  ALL TESTS PASSED
========================================

testbench/tb_lif_neuron.v:278: $finish called at 576000 (1ps)
>>> tb_lif_neuron: PASS

########## tb_spike_linear ##########
VCD info: dumpfile tb_spike_linear.vcd opened for output.
  Weight matrix initialized:
    in[0]:    1    2    3    4
    in[1]:   11   12   13   14
    in[2]:   21   22   23   24
    in[3]:   31   32   33   34
    in[4]:   41   42   43   44
    in[5]:   51   52   53   54
    in[6]:   61   62   63   64
    in[7]:   71   72   73   74

Test 1: All-zero spikes
  OK: out[0] = 0
  OK: out[1] = 0
  OK: out[2] = 0
  OK: out[3] = 0

Test 2: Single spike at input[0]
  OK: out[0] = 1
  OK: out[1] = 2
  OK: out[2] = 3
  OK: out[3] = 4

Test 3: Single spike at input[2]
  OK: out[0] = 21
  OK: out[1] = 22
  OK: out[2] = 23
  OK: out[3] = 24

Test 4: Two spikes at input[0] and input[1]
  OK: out[0] = 12
  OK: out[1] = 14
  OK: out[2] = 16
  OK: out[3] = 18

Test 5: All-ones spikes
  OK: out[0] = 288
  OK: out[1] = 296
  OK: out[2] = 304
  OK: out[3] = 312

Test 6: Alternating spikes (0,2,4,6)
  OK: out[0] = 124
  OK: out[1] = 128
  OK: out[2] = 132
  OK: out[3] = 136

Test 7: Busy/done signal timing
  Busy asserted correctly
  Done asserted, busy=0

Test 8: Back-to-back operations
  First operation:
  OK: out[0] = 224
  OK: out[1] = 228
  OK: out[2] = 232
  OK: out[3] = 236
  Second operation:
  OK: out[0] = 64
  OK: out[1] = 68
  OK: out[2] = 72
  OK: out[3] = 76

Test 9: Negative weights
  With negative weights:
    out[0] = -10
    out[1] = -20
    out[2] = 30
    out[3] = -40

========================================
  ALL TESTS PASSED
========================================

testbench/tb_spike_linear.v:311: $finish called at 3656000 (1ps)
>>> tb_spike_linear: PASS

########## tb_leaky_ternary_lif ##########
VCD info: dumpfile tb_leaky_ternary_lif.vcd opened for output.
PASS test 1: spike=00 output=3 membrane=50
PASS test 2: spike=00 output=3 membrane=94
PASS test 3: spike=01 output=275 membrane=596
After sustained positive: spike=01 membrane=400 output=259
After sustained negative: spike=11 membrane=-500 output=-237
Top-K blocked: spike=00 (should be 00) output=55
PASS: top-K correctly blocked spike
Top-K re-enabled: spike=01 output=273
Small input, no spike: output=8 (should be small, SiLU only)

=== RESULTS: 4 passed, 0 failed ===
ALL TESTS PASSED
testbench/tb_leaky_ternary_lif.v:156: $finish called at 290000 (1ps)
>>> tb_leaky_ternary_lif: PASS

==============================================================
RESULT: ALL TESTBENCHES PASSED
==============================================================
```

## Notes on fidelity to the trained model

- The RTL `leaky_ternary_lif` uses a **fixed compile-time α constant**
  (`ALPHA_NUM = 218/256 = 0.85`), frozen for inference. This is *not* the
  learned per-layer α (or the per-token dynamic gate) used by the PyTorch
  model during training — on hardware the spike/continuous blend ratio is a
  baked-in constant.
- `ternary_lif_neuron.v`, `ternary_spike_linear.v`, and `topk_selector.v` are
  also provided but are exercised indirectly / not yet covered by a dedicated
  self-checking testbench.
