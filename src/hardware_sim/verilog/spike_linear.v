// =============================================================================
// spike_linear.v — Sparse Spike-Driven Linear Layer
// =============================================================================
// Target: AMD/Xilinx Kria KV260 (XCK26) @ 100 MHz
//
// Neuromorphic advantage: binary spikes mean no multiplications needed.
// When spike[j]=1, accumulate weight[i][j] into output[i].
// When spike[j]=0, skip entirely (event-driven, energy-efficient).
//
// Weights stored in BRAM, addressed by (output_neuron, input_neuron).
// Processes one input neuron per clock cycle (spike-gated).
// =============================================================================

module spike_linear #(
    parameter IN_FEATURES   = 64,            // Number of input neurons
    parameter OUT_FEATURES  = 64,            // Number of output neurons
    parameter WEIGHT_WIDTH  = 8,             // INT8 weights
    parameter ACCUM_WIDTH   = 16,            // INT16 accumulator
    parameter ADDR_WIDTH    = $clog2(IN_FEATURES * OUT_FEATURES)
)(
    input  wire                          clk,
    input  wire                          rst_n,

    // Control
    input  wire                          i_start,       // Pulse to begin computation
    output reg                           o_done,        // Pulse when complete
    output reg                           o_busy,        // High during computation

    // Spike input vector (one bit per input neuron)
    input  wire [IN_FEATURES-1:0]        i_spikes,

    // Weight memory interface (external BRAM)
    output reg  [ADDR_WIDTH-1:0]         o_weight_addr,
    output reg                           o_weight_rd_en,
    input  wire signed [WEIGHT_WIDTH-1:0] i_weight_data,

    // Output: accumulated results (active-low CE for downstream)
    output reg                           o_valid,
    output reg  [$clog2(OUT_FEATURES)-1:0] o_out_idx,
    output reg  signed [ACCUM_WIDTH-1:0] o_out_data
);

    // ---------------------------------------------------------------------------
    // State machine
    // ---------------------------------------------------------------------------
    localparam S_IDLE     = 3'd0;
    localparam S_SCAN     = 3'd1;    // Scan input spikes for active ones
    localparam S_ACCUM    = 3'd2;    // Accumulate weights for active input
    localparam S_OUTPUT   = 3'd3;    // Stream out results
    localparam S_DONE     = 3'd4;

    reg [2:0] state;

    // Counters (extra bit to detect overflow past max)
    reg [$clog2(IN_FEATURES):0]  in_idx;
    reg [$clog2(OUT_FEATURES):0] out_idx;

    // Accumulator bank (one per output neuron)
    reg signed [ACCUM_WIDTH-1:0] accum [0:OUT_FEATURES-1];

    // Registered spike vector
    reg [IN_FEATURES-1:0] spike_reg;

    // Pipeline: weight read takes 2 cycle latency (DUT reg + BRAM reg)
    reg                           rd_valid_d1, rd_valid_d2;
    reg [$clog2(OUT_FEATURES):0] out_idx_d1, out_idx_d2;
    // Track if we need one more drain cycle
    reg                           drain_pending;

    integer k;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state         <= S_IDLE;
            o_done        <= 1'b0;
            o_busy        <= 1'b0;
            o_valid       <= 1'b0;
            o_weight_rd_en <= 1'b0;
            in_idx        <= 0;
            out_idx       <= 0;
            rd_valid_d1   <= 1'b0;
            rd_valid_d2   <= 1'b0;
            out_idx_d1    <= 0;
            out_idx_d2    <= 0;
            drain_pending <= 1'b0;
            spike_reg     <= 0;
            o_out_idx     <= 0;
            o_out_data    <= 0;
            o_weight_addr <= 0;
            for (k = 0; k < OUT_FEATURES; k = k + 1)
                accum[k] <= 0;
        end else begin
            o_done  <= 1'b0;
            o_valid <= 1'b0;
            // Pipeline shift: d1 -> d2
            rd_valid_d2 <= rd_valid_d1;
            out_idx_d2  <= out_idx_d1;
            rd_valid_d1 <= 1'b0;

            case (state)
                S_IDLE: begin
                    if (i_start) begin
                        state     <= S_SCAN;
                        o_busy    <= 1'b1;
                        spike_reg <= i_spikes;
                        in_idx    <= 0;
                        out_idx   <= 0;
                        for (k = 0; k < OUT_FEATURES; k = k + 1)
                            accum[k] <= 0;
                    end
                end

                S_SCAN: begin
                    // Find next active spike (event-driven skip)
                    if (in_idx >= IN_FEATURES) begin
                        // All inputs scanned, output results
                        state   <= S_OUTPUT;
                        out_idx <= 0;
                    end else if (spike_reg[in_idx]) begin
                        // Active spike found — accumulate all output weights
                        state   <= S_ACCUM;
                        out_idx <= 0;
                        // Start first weight read
                        o_weight_addr  <= in_idx * OUT_FEATURES;
                        o_weight_rd_en <= 1'b1;
                    end else begin
                        // No spike at this input — skip entirely
                        in_idx <= in_idx + 1;
                    end
                end

                S_ACCUM: begin
                    // Pipeline: 2-cycle latency (DUT reg -> BRAM reg -> data)
                    // Cycle N:   DUT sets o_weight_addr for out_idx
                    // Cycle N+1: BRAM captures addr, DUT records rd_valid_d1
                    // Cycle N+2: BRAM outputs data, DUT accumulates via rd_valid_d2

                    // Accumulate from 2 cycles ago
                    if (rd_valid_d2) begin
                        accum[out_idx_d2] <= accum[out_idx_d2] + {{(ACCUM_WIDTH-WEIGHT_WIDTH){i_weight_data[WEIGHT_WIDTH-1]}}, i_weight_data};
                    end

                    if (out_idx < OUT_FEATURES) begin
                        // Issue next weight read
                        o_weight_addr  <= in_idx * OUT_FEATURES + out_idx;
                        o_weight_rd_en <= 1'b1;
                        rd_valid_d1    <= 1'b1;
                        out_idx_d1     <= out_idx;
                        out_idx        <= out_idx + 1;
                    end else if (rd_valid_d1 || rd_valid_d2) begin
                        // All reads issued; drain pipeline
                        o_weight_rd_en <= 1'b0;
                    end else begin
                        // Pipeline fully drained — move to next input
                        in_idx <= in_idx + 1;
                        state  <= S_SCAN;
                    end
                end

                S_OUTPUT: begin
                    if (out_idx < OUT_FEATURES) begin
                        o_valid    <= 1'b1;
                        o_out_idx  <= out_idx;
                        o_out_data <= accum[out_idx];
                        out_idx    <= out_idx + 1;
                    end else begin
                        state  <= S_DONE;
                    end
                end

                S_DONE: begin
                    o_done <= 1'b1;
                    o_busy <= 1'b0;
                    state  <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
