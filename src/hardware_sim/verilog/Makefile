# =============================================================================
# Makefile — Icarus Verilog functional-simulation test runner.
#
# Functional simulation ONLY (Icarus Verilog / vvp). Does NOT synthesize,
# place-and-route, or run on a KV260 board.
#
#   make test    # compile + run every testbench, non-zero exit on failure
#   make clean   # remove build artifacts
# =============================================================================

IVERILOG ?= iverilog
VVP      ?= vvp
IFLAGS   ?= -g2012
BUILD    := build

TESTS := tb_lif_neuron tb_spike_linear tb_leaky_ternary_lif

tb_lif_neuron_SRCS        := lif_neuron.v testbench/tb_lif_neuron.v
tb_spike_linear_SRCS      := spike_linear.v testbench/tb_spike_linear.v
tb_leaky_ternary_lif_SRCS := leaky_ternary_lif.v testbench/tb_leaky_ternary_lif.v

.PHONY: test clean version
test:
	@./run_tests.sh

version:
	@$(IVERILOG) -V | head -1
	@$(VVP) -V | head -1

clean:
	rm -rf $(BUILD) *.vcd testbench/*.vcd
