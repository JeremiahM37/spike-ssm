// =============================================================================
// ternary_lif_neuron.v — Ternary Leaky Integrate-and-Fire Neuron
// =============================================================================
// Target: AMD/Xilinx Kria KV260 (XCK26) @ 100 MHz
// Fixed-point: INT16 membrane, 2-bit ternary spikes {-1, 0, +1}
//
// Extends lif_neuron.v with:
//   - Ternary output: fires +1 if U >= +threshold, -1 if U <= -threshold
//   - Soft reset: U = U * (1 - |spike|) = 0 when spike, U when no spike
//
// Output encoding: o_spike = 2'b01 (+1), 2'b11 (-1), 2'b00 (no spike)
// =============================================================================

module ternary_lif_neuron #(
    parameter DATA_WIDTH = 16,
    parameter BETA_WIDTH = 8,
    parameter BETA_VALUE = 8'd230,       // beta=0.9 in Q0.8 (230/256)
    parameter BETA_SHIFT = 8,
    parameter THRESHOLD  = 16'sd384      // 1.5 in Q8.8
)(
    input  wire                         clk,
    input  wire                         rst_n,
    input  wire                         en,
    input  wire signed [DATA_WIDTH-1:0] i_current,
    output reg  [1:0]                   o_spike,       // 2-bit ternary
    output reg  signed [DATA_WIDTH-1:0] o_membrane,
    output wire signed [DATA_WIDTH-1:0] o_membrane_abs // for top-K selection
);

    reg  signed [DATA_WIDTH-1:0]   membrane_r;
    wire signed [2*DATA_WIDTH-1:0] leak_product;
    wire signed [DATA_WIDTH-1:0]   leaked_membrane;

    // Leak
    assign leak_product    = membrane_r * $signed({{(DATA_WIDTH-BETA_WIDTH){1'b0}}, BETA_VALUE});
    assign leaked_membrane = leak_product[DATA_WIDTH-1+BETA_SHIFT:BETA_SHIFT];

    // Integrate with saturation
    wire signed [DATA_WIDTH:0] sum_ext;
    assign sum_ext = {leaked_membrane[DATA_WIDTH-1], leaked_membrane} +
                     {i_current[DATA_WIDTH-1], i_current};

    wire signed [DATA_WIDTH-1:0] new_membrane;
    wire overflow_pos = (~sum_ext[DATA_WIDTH] & sum_ext[DATA_WIDTH-1]);
    wire overflow_neg = (sum_ext[DATA_WIDTH] & ~sum_ext[DATA_WIDTH-1]);
    assign new_membrane = overflow_pos ? {1'b0, {(DATA_WIDTH-1){1'b1}}} :
                          overflow_neg ? {1'b1, {(DATA_WIDTH-1){1'b0}}} :
                          sum_ext[DATA_WIDTH-1:0];

    // Ternary fire
    wire fire_pos = (new_membrane >= THRESHOLD);
    wire fire_neg = (new_membrane <= -THRESHOLD);

    // Absolute membrane for top-K ranking
    assign o_membrane_abs = new_membrane[DATA_WIDTH-1] ? -new_membrane : new_membrane;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            membrane_r <= 0;
            o_spike    <= 2'b00;
            o_membrane <= 0;
        end else if (en) begin
            if (fire_pos) begin
                o_spike    <= 2'b01;    // +1
                membrane_r <= 0;        // soft reset
                o_membrane <= new_membrane;
            end else if (fire_neg) begin
                o_spike    <= 2'b11;    // -1
                membrane_r <= 0;
                o_membrane <= new_membrane;
            end else begin
                o_spike    <= 2'b00;    // no spike
                membrane_r <= new_membrane;
                o_membrane <= new_membrane;
            end
        end
    end

endmodule
