#!/usr/bin/env bash
# =============================================================================
# run_tests.sh — Compile + run every Verilog testbench under Icarus Verilog.
#
# Functional simulation ONLY (Icarus Verilog / vvp). This does NOT synthesize,
# place-and-route, or run on a KV260 board — it verifies RTL behavior in sim.
#
# Exits non-zero if any testbench fails, errors, or times out, so it is safe to
# wire into CI. Each testbench self-checks with $display assertions and prints
# "ALL TESTS PASSED" only when its internal error count is zero.
#
# Usage:  ./run_tests.sh
# Requires: iverilog + vvp (Icarus Verilog 11.x or newer).
# =============================================================================
set -u

# Resolve directory of this script so it works from any cwd.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

BUILD="${HERE}/build"
mkdir -p "$BUILD"

if ! command -v iverilog >/dev/null 2>&1; then
    echo "ERROR: iverilog not found. Install Icarus Verilog (e.g. 'apt-get install iverilog')." >&2
    exit 127
fi

echo "=============================================================="
iverilog -V | head -1
vvp -V | head -1
echo "=============================================================="

# testbench  ->  source files it needs (space separated, paths relative to HERE)
declare -A TESTS=(
    [tb_lif_neuron]="lif_neuron.v testbench/tb_lif_neuron.v"
    [tb_spike_linear]="spike_linear.v testbench/tb_spike_linear.v"
    [tb_leaky_ternary_lif]="leaky_ternary_lif.v testbench/tb_leaky_ternary_lif.v"
)

# Deterministic order.
ORDER=(tb_lif_neuron tb_spike_linear tb_leaky_ternary_lif)

overall=0
for tb in "${ORDER[@]}"; do
    echo
    echo "########## ${tb} ##########"
    out="${BUILD}/${tb}.vvp"

    if ! iverilog -g2012 -o "$out" ${TESTS[$tb]}; then
        echo ">>> ${tb}: COMPILE FAILED"
        overall=1
        continue
    fi

    log="${BUILD}/${tb}.log"
    ( cd "$BUILD" && vvp "$out" ) | tee "$log"

    # A run is good only if it announced PASS and never printed a failure
    # marker. Markers are matched case-sensitively (uppercase) so the lowercase
    # "0 failed" in a results summary line is not a false positive.
    if grep -qE "ALL TESTS PASSED" "$log" \
       && ! grep -qE "FAIL|ERRORS detected|SOME TESTS FAILED|TIMEOUT" "$log"; then
        echo ">>> ${tb}: PASS"
    else
        echo ">>> ${tb}: FAIL"
        overall=1
    fi
done

echo
echo "=============================================================="
if [ "$overall" -eq 0 ]; then
    echo "RESULT: ALL TESTBENCHES PASSED"
else
    echo "RESULT: ONE OR MORE TESTBENCHES FAILED"
fi
echo "=============================================================="
exit "$overall"
