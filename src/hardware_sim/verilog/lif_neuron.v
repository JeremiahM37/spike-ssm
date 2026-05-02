// =============================================================================
// lif_neuron.v — Leaky Integrate-and-Fire Neuron (Fixed-Point, Parameterized)
// =============================================================================
// Target: AMD/Xilinx Kria KV260 (XCK26) @ 100 MHz
// Fixed-point: INT16 membrane potential, INT8 weights, 1-bit spikes
//
// Dynamics per clock cycle (one timestep):
//   U[t] = (beta * U[t-1]) >> BETA_SHIFT + input_current
//   spike = (U[t] >= THRESHOLD)
//   U[t] = spike ? reset_val : U[t]
//
// Reset modes:
//   RESET_MODE=0: Hard reset (U -> 0 after spike)
//   RESET_MODE=1: Soft reset (U -> U - threshold after spike)
// =============================================================================

module lif_neuron #(
    parameter DATA_WIDTH     = 16,           // Membrane potential width (INT16)
    parameter BETA_WIDTH     = 8,            // Leak factor width
    parameter BETA_VALUE     = 8'h80,        // beta=0.5 in Q0.8 (128/256)
    parameter BETA_SHIFT     = 8,            // Right-shift after beta multiply
    parameter THRESHOLD      = 16'sd256,     // Firing threshold (1.0 in Q8.8)
    parameter RESET_MODE     = 0,            // 0=hard, 1=soft (subtract)
    parameter REFRACTORY_LEN = 0             // Refractory period in cycles
)(
    input  wire                        clk,
    input  wire                        rst_n,
    input  wire                        en,           // Enable (valid input)
    input  wire signed [DATA_WIDTH-1:0] i_current,   // Input current (weighted sum)
    output reg                         o_spike,      // Output spike (1-bit)
    output reg  signed [DATA_WIDTH-1:0] o_membrane   // Membrane potential (debug/readback)
);

    // Internal signals
    reg  signed [DATA_WIDTH-1:0]   membrane_r;
    wire signed [2*DATA_WIDTH-1:0] leak_product;
    wire signed [DATA_WIDTH-1:0]   leaked_membrane;
    wire signed [DATA_WIDTH-1:0]   new_membrane;
    wire                           fire;

    // Refractory counter
    reg [$clog2(REFRACTORY_LEN+1):0] refrac_cnt;

    // Leak: membrane * beta (fixed-point multiply + shift)
    assign leak_product   = membrane_r * $signed({{(DATA_WIDTH-BETA_WIDTH){1'b0}}, BETA_VALUE});
    assign leaked_membrane = leak_product[DATA_WIDTH-1+BETA_SHIFT:BETA_SHIFT]; // Q-shift

    // Integrate: leaked membrane + input current
    // Saturating addition to prevent overflow
    wire signed [DATA_WIDTH:0] sum_extended;
    assign sum_extended = {leaked_membrane[DATA_WIDTH-1], leaked_membrane} +
                          {i_current[DATA_WIDTH-1], i_current};

    // Saturation logic
    wire overflow_pos = (~sum_extended[DATA_WIDTH] & sum_extended[DATA_WIDTH-1] &
                         ~leaked_membrane[DATA_WIDTH-1] & ~i_current[DATA_WIDTH-1]);
    wire overflow_neg = (sum_extended[DATA_WIDTH] & ~sum_extended[DATA_WIDTH-1] &
                         leaked_membrane[DATA_WIDTH-1] & i_current[DATA_WIDTH-1]);

    assign new_membrane = overflow_pos ? {1'b0, {(DATA_WIDTH-1){1'b1}}} :  // Max positive
                          overflow_neg ? {1'b1, {(DATA_WIDTH-1){1'b0}}} :  // Max negative
                          sum_extended[DATA_WIDTH-1:0];

    // Fire: compare against threshold
    assign fire = (new_membrane >= THRESHOLD);

    // Sequential logic
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            membrane_r <= {DATA_WIDTH{1'b0}};
            o_spike    <= 1'b0;
            o_membrane <= {DATA_WIDTH{1'b0}};
            refrac_cnt <= 0;
        end else if (en) begin
            if (REFRACTORY_LEN > 0 && refrac_cnt > 0) begin
                // In refractory period — no integration, no spike
                o_spike    <= 1'b0;
                o_membrane <= membrane_r;
                refrac_cnt <= refrac_cnt - 1;
            end else if (fire) begin
                o_spike <= 1'b1;
                if (RESET_MODE == 0) begin
                    // Hard reset
                    membrane_r <= {DATA_WIDTH{1'b0}};
                end else begin
                    // Soft reset (subtract threshold)
                    membrane_r <= new_membrane - THRESHOLD;
                end
                o_membrane <= new_membrane;
                refrac_cnt <= REFRACTORY_LEN;
            end else begin
                o_spike    <= 1'b0;
                membrane_r <= new_membrane;
                o_membrane <= new_membrane;
            end
        end
    end

endmodule
