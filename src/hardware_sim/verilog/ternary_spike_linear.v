// =============================================================================
// ternary_spike_linear.v — Ternary Spike-Driven Linear Layer
// =============================================================================
// Target: AMD/Xilinx Kria KV260 (XCK26) @ 100 MHz
//
// Extends spike_linear.v for ternary spikes {-1, 0, +1}:
//   spike = +1: accumulate  +weight[i][j]
//   spike = -1: accumulate  -weight[i][j]
//   spike =  0: skip (event-driven, no operation)
//
// Still zero multiplications — only addition and subtraction.
// Ternary encoding: 2 bits per spike (00=none, 01=+1, 11=-1)
//
// With top-K 30% sparsity: 70% of neurons are 0 → 70% of operations skipped.
// =============================================================================

module ternary_spike_linear #(
    parameter IN_FEATURES  = 128,
    parameter OUT_FEATURES = 128,
    parameter WEIGHT_WIDTH = 8,
    parameter ACCUM_WIDTH  = 16,
    parameter ADDR_WIDTH   = $clog2(IN_FEATURES * OUT_FEATURES)
)(
    input  wire                          clk,
    input  wire                          rst_n,
    input  wire                          i_start,
    output reg                           o_done,
    output reg                           o_busy,

    // Ternary spike input: 2 bits per neuron
    input  wire [2*IN_FEATURES-1:0]      i_spikes,    // packed: {spike[N-1], ..., spike[0]}

    // Weight BRAM interface
    output reg  [ADDR_WIDTH-1:0]         o_weight_addr,
    output reg                           o_weight_rd_en,
    input  wire signed [WEIGHT_WIDTH-1:0] i_weight_data,

    // Output
    output reg                           o_valid,
    output reg  [$clog2(OUT_FEATURES)-1:0] o_out_idx,
    output reg  signed [ACCUM_WIDTH-1:0] o_out_data
);

    localparam S_IDLE   = 3'd0;
    localparam S_SCAN   = 3'd1;
    localparam S_ACCUM  = 3'd2;
    localparam S_OUTPUT = 3'd3;
    localparam S_DONE   = 3'd4;

    reg [2:0] state;
    reg [$clog2(IN_FEATURES):0]  in_idx;
    reg [$clog2(OUT_FEATURES):0] out_idx;
    reg signed [ACCUM_WIDTH-1:0] accum [0:OUT_FEATURES-1];

    // Current spike value for active input
    reg [1:0] current_spike;
    wire spike_is_pos = (current_spike == 2'b01);
    wire spike_is_neg = (current_spike == 2'b11);

    // Pipeline registers for BRAM latency
    reg                           rd_valid_d1, rd_valid_d2;
    reg [$clog2(OUT_FEATURES):0]  out_idx_d1, out_idx_d2;
    reg                           negate_d1, negate_d2;  // negate weight for -1 spike

    integer k;

    // Extract 2-bit spike for neuron in_idx
    wire [1:0] spike_at_idx = i_spikes[2*in_idx[($clog2(IN_FEATURES)-1):0] +: 2];

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state         <= S_IDLE;
            o_done        <= 0;
            o_busy        <= 0;
            o_valid       <= 0;
            o_weight_rd_en <= 0;
            in_idx        <= 0;
            out_idx       <= 0;
            rd_valid_d1   <= 0;
            rd_valid_d2   <= 0;
            negate_d1     <= 0;
            negate_d2     <= 0;
            current_spike <= 0;
            o_out_idx     <= 0;
            o_out_data    <= 0;
            o_weight_addr <= 0;
            for (k = 0; k < OUT_FEATURES; k = k + 1)
                accum[k] <= 0;
        end else begin
            o_done  <= 0;
            o_valid <= 0;
            rd_valid_d2 <= rd_valid_d1;
            out_idx_d2  <= out_idx_d1;
            negate_d2   <= negate_d1;
            rd_valid_d1 <= 0;

            case (state)
                S_IDLE: begin
                    if (i_start) begin
                        state   <= S_SCAN;
                        o_busy  <= 1;
                        in_idx  <= 0;
                        out_idx <= 0;
                        for (k = 0; k < OUT_FEATURES; k = k + 1)
                            accum[k] <= 0;
                    end
                end

                S_SCAN: begin
                    if (in_idx >= IN_FEATURES) begin
                        state   <= S_OUTPUT;
                        out_idx <= 0;
                    end else if (spike_at_idx != 2'b00) begin
                        // Active spike (+1 or -1)
                        current_spike <= spike_at_idx;
                        state   <= S_ACCUM;
                        out_idx <= 0;
                        o_weight_addr  <= in_idx * OUT_FEATURES;
                        o_weight_rd_en <= 1;
                    end else begin
                        // No spike — skip entirely (event-driven)
                        in_idx <= in_idx + 1;
                    end
                end

                S_ACCUM: begin
                    // Accumulate: add or subtract weight based on spike sign
                    if (rd_valid_d2) begin
                        if (negate_d2)
                            accum[out_idx_d2] <= accum[out_idx_d2] -
                                {{(ACCUM_WIDTH-WEIGHT_WIDTH){i_weight_data[WEIGHT_WIDTH-1]}}, i_weight_data};
                        else
                            accum[out_idx_d2] <= accum[out_idx_d2] +
                                {{(ACCUM_WIDTH-WEIGHT_WIDTH){i_weight_data[WEIGHT_WIDTH-1]}}, i_weight_data};
                    end

                    if (out_idx < OUT_FEATURES) begin
                        o_weight_addr  <= in_idx * OUT_FEATURES + out_idx;
                        o_weight_rd_en <= 1;
                        rd_valid_d1    <= 1;
                        out_idx_d1     <= out_idx;
                        negate_d1      <= spike_is_neg;  // -1 spike → subtract
                        out_idx        <= out_idx + 1;
                    end else if (rd_valid_d1 || rd_valid_d2) begin
                        o_weight_rd_en <= 0;
                    end else begin
                        in_idx <= in_idx + 1;
                        state  <= S_SCAN;
                    end
                end

                S_OUTPUT: begin
                    if (out_idx < OUT_FEATURES) begin
                        o_valid    <= 1;
                        o_out_idx  <= out_idx;
                        o_out_data <= accum[out_idx];
                        out_idx    <= out_idx + 1;
                    end else begin
                        state <= S_DONE;
                    end
                end

                S_DONE: begin
                    o_done <= 1;
                    o_busy <= 0;
                    state  <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
