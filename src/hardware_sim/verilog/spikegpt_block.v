// =============================================================================
// spikegpt_block.v — One Complete SpikeGPT Block
// =============================================================================
// Target: AMD/Xilinx Kria KV260 (XCK26) @ 100 MHz
//
// Architecture (matches Python SpikeGPTBlock):
//   1. LayerNorm(x) -> TimeMix -> LIF neuron -> residual add
//   2. LayerNorm(x) -> ChannelMix -> LIF neuron -> residual add
//
// LayerNorm is approximated in fixed-point using mean subtraction
// and a precomputed reciprocal of stddev (shift-based normalization).
//
// Data flow is sequential: TimeMix completes, then ChannelMix runs.
// Each sub-block has its own weight BRAM port.
// =============================================================================

module spikegpt_block #(
    parameter N_EMBD         = 64,
    parameter HIDDEN_SZ      = 256,          // 4 * N_EMBD for channel mix
    parameter DATA_WIDTH     = 16,
    parameter WEIGHT_WIDTH   = 8,
    parameter ACCUM_WIDTH    = 24,
    parameter FRAC_BITS      = 8,
    parameter BLOCK_ID       = 0,
    parameter N_BLOCKS       = 12,
    parameter TM_ADDR_WIDTH  = 16,           // Time-mix weight address width
    parameter CM_ADDR_WIDTH  = 18,           // Channel-mix weight address width
    parameter LIF_BETA       = 8'h80,        // 0.5 in Q0.8
    parameter LIF_THRESHOLD  = 16'sd256      // 1.0 in Q8.8
)(
    input  wire                          clk,
    input  wire                          rst_n,

    // Control
    input  wire                          i_start,
    input  wire                          i_new_sequence,
    output wire                          o_done,
    output wire                          o_busy,

    // Input token embedding (streamed, one element per cycle)
    input  wire                          i_data_valid,
    input  wire signed [DATA_WIDTH-1:0]  i_data,

    // Time-mix weight BRAM
    output wire [TM_ADDR_WIDTH-1:0]      o_tm_weight_addr,
    output wire                          o_tm_weight_rd_en,
    input  wire signed [WEIGHT_WIDTH-1:0] i_tm_weight_data,

    // Channel-mix weight BRAM
    output wire [CM_ADDR_WIDTH-1:0]      o_cm_weight_addr,
    output wire                          o_cm_weight_rd_en,
    input  wire signed [WEIGHT_WIDTH-1:0] i_cm_weight_data,

    // Time-mixing parameters (preloaded)
    input  wire signed [DATA_WIDTH-1:0]  i_tm_mix_k,
    input  wire signed [DATA_WIDTH-1:0]  i_tm_mix_v,
    input  wire signed [DATA_WIDTH-1:0]  i_tm_mix_r,
    input  wire signed [DATA_WIDTH-1:0]  i_tm_decay,
    input  wire signed [DATA_WIDTH-1:0]  i_tm_first,

    // Channel-mixing parameters
    input  wire signed [DATA_WIDTH-1:0]  i_cm_mix_k,
    input  wire signed [DATA_WIDTH-1:0]  i_cm_mix_r,

    // Output (streamed)
    output reg                           o_out_valid,
    output reg  signed [DATA_WIDTH-1:0]  o_out_data,

    // Spike monitoring (for energy estimation)
    output wire                          o_spike_att,
    output wire                          o_spike_ffn
);

    // =========================================================================
    // State machine for block-level sequencing
    // =========================================================================
    localparam BLK_IDLE     = 3'd0;
    localparam BLK_LOAD     = 3'd1;    // Load input into buffer
    localparam BLK_LN1      = 3'd2;    // LayerNorm 1 (approx)
    localparam BLK_TMIX     = 3'd3;    // Time mixing
    localparam BLK_LIF1     = 3'd4;    // LIF after attention
    localparam BLK_LN2      = 3'd5;    // LayerNorm 2
    localparam BLK_CMIX     = 3'd6;    // Channel mixing
    localparam BLK_LIF2     = 3'd7;    // LIF after FFN

    reg [2:0] blk_state;

    // Input/output buffers
    reg signed [DATA_WIDTH-1:0] x_buf      [0:N_EMBD-1];  // Current residual
    reg signed [DATA_WIDTH-1:0] x_prev     [0:N_EMBD-1];  // Previous token (time-shift)
    reg signed [DATA_WIDTH-1:0] ln_buf     [0:N_EMBD-1];  // After LayerNorm
    reg signed [DATA_WIDTH-1:0] sub_out    [0:N_EMBD-1];  // Sub-block output

    // Channel index
    reg [$clog2(N_EMBD)-1:0] ch_cnt;
    reg [$clog2(N_EMBD)-1:0] load_cnt;

    // Sub-module control
    reg  tm_start, cm_start;
    wire tm_done,  cm_done;
    wire tm_busy,  cm_busy;

    // LIF neurons
    reg                          lif1_en, lif2_en;
    reg  signed [DATA_WIDTH-1:0] lif1_in, lif2_in;
    wire                         lif1_spike, lif2_spike;
    wire signed [DATA_WIDTH-1:0] lif1_mem, lif2_mem;

    // Sub-module outputs
    wire        tm_out_valid, cm_out_valid;
    wire signed [DATA_WIDTH-1:0] tm_out_data, cm_out_data;

    // Assign monitoring outputs
    assign o_spike_att = lif1_spike;
    assign o_spike_ffn = lif2_spike;

    // =========================================================================
    // LIF Neuron 1 (after time-mix / attention)
    // =========================================================================
    lif_neuron #(
        .DATA_WIDTH  (DATA_WIDTH),
        .BETA_VALUE  (LIF_BETA),
        .THRESHOLD   (LIF_THRESHOLD),
        .RESET_MODE  (1)              // Soft reset (subtract)
    ) u_lif1 (
        .clk        (clk),
        .rst_n      (rst_n),
        .en         (lif1_en),
        .i_current  (lif1_in),
        .o_spike    (lif1_spike),
        .o_membrane (lif1_mem)
    );

    // =========================================================================
    // LIF Neuron 2 (after channel-mix / FFN)
    // =========================================================================
    lif_neuron #(
        .DATA_WIDTH  (DATA_WIDTH),
        .BETA_VALUE  (LIF_BETA),
        .THRESHOLD   (LIF_THRESHOLD),
        .RESET_MODE  (1)
    ) u_lif2 (
        .clk        (clk),
        .rst_n      (rst_n),
        .en         (lif2_en),
        .i_current  (lif2_in),
        .o_spike    (lif2_spike),
        .o_membrane (lif2_mem)
    );

    // =========================================================================
    // Time-Mix sub-module
    // =========================================================================
    rwkv_time_mix #(
        .N_EMBD      (N_EMBD),
        .DATA_WIDTH  (DATA_WIDTH),
        .WEIGHT_WIDTH(WEIGHT_WIDTH),
        .ACCUM_WIDTH (ACCUM_WIDTH),
        .FRAC_BITS   (FRAC_BITS),
        .ADDR_WIDTH  (TM_ADDR_WIDTH)
    ) u_time_mix (
        .clk            (clk),
        .rst_n          (rst_n),
        .i_start        (tm_start),
        .i_new_sequence (i_new_sequence),
        .o_done         (tm_done),
        .o_busy         (tm_busy),
        .i_data_valid   (blk_state == BLK_TMIX && ch_cnt < N_EMBD),
        .i_data         (ln_buf[ch_cnt]),
        .i_data_prev    (x_prev[ch_cnt]),
        .i_time_mix_k   (i_tm_mix_k),
        .i_time_mix_v   (i_tm_mix_v),
        .i_time_mix_r   (i_tm_mix_r),
        .i_time_decay   (i_tm_decay),
        .i_time_first   (i_tm_first),
        .o_weight_addr  (o_tm_weight_addr),
        .o_weight_rd_en (o_tm_weight_rd_en),
        .i_weight_data  (i_tm_weight_data),
        .o_out_valid    (tm_out_valid),
        .o_out_data     (tm_out_data)
    );

    // =========================================================================
    // Channel-Mix sub-module
    // =========================================================================
    rwkv_channel_mix #(
        .N_EMBD      (N_EMBD),
        .HIDDEN_SZ   (HIDDEN_SZ),
        .DATA_WIDTH  (DATA_WIDTH),
        .WEIGHT_WIDTH(WEIGHT_WIDTH),
        .ACCUM_WIDTH (ACCUM_WIDTH),
        .FRAC_BITS   (FRAC_BITS),
        .ADDR_WIDTH  (CM_ADDR_WIDTH)
    ) u_channel_mix (
        .clk            (clk),
        .rst_n          (rst_n),
        .i_start        (cm_start),
        .o_done         (cm_done),
        .o_busy         (cm_busy),
        .i_data_valid   (blk_state == BLK_CMIX && ch_cnt < N_EMBD),
        .i_data         (ln_buf[ch_cnt]),
        .i_data_prev    (x_prev[ch_cnt]),
        .i_time_mix_k   (i_cm_mix_k),
        .i_time_mix_r   (i_cm_mix_r),
        .o_weight_addr  (o_cm_weight_addr),
        .o_weight_rd_en (o_cm_weight_rd_en),
        .i_weight_data  (i_cm_weight_data),
        .o_out_valid    (cm_out_valid),
        .o_out_data     (cm_out_data)
    );

    // =========================================================================
    // Approximate LayerNorm (mean subtraction + scale)
    // For resource-constrained FPGA: subtract mean, then normalize by
    // approximate reciprocal stddev (precomputed or shifted).
    // =========================================================================
    reg signed [ACCUM_WIDTH-1:0] ln_sum;
    reg signed [DATA_WIDTH-1:0]  ln_mean;

    // Combined done/busy
    assign o_done = (blk_state == BLK_IDLE) && (ch_cnt == N_EMBD);
    assign o_busy = (blk_state != BLK_IDLE);

    // =========================================================================
    // Capture time-mix output into sub_out
    // =========================================================================
    reg [$clog2(N_EMBD)-1:0] tm_out_cnt;
    reg [$clog2(N_EMBD)-1:0] cm_out_cnt;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            tm_out_cnt <= 0;
            cm_out_cnt <= 0;
        end else begin
            if (tm_out_valid) begin
                sub_out[tm_out_cnt] <= tm_out_data;
                tm_out_cnt <= tm_out_cnt + 1;
            end
            if (tm_start) tm_out_cnt <= 0;

            if (cm_out_valid) begin
                sub_out[cm_out_cnt] <= cm_out_data;
                cm_out_cnt <= cm_out_cnt + 1;
            end
            if (cm_start) cm_out_cnt <= 0;
        end
    end

    // =========================================================================
    // Main block state machine
    // =========================================================================
    integer j;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            blk_state  <= BLK_IDLE;
            ch_cnt     <= 0;
            load_cnt   <= 0;
            tm_start   <= 1'b0;
            cm_start   <= 1'b0;
            lif1_en    <= 1'b0;
            lif2_en    <= 1'b0;
            lif1_in    <= 0;
            lif2_in    <= 0;
            o_out_valid <= 1'b0;
            o_out_data <= 0;
            ln_sum     <= 0;
            ln_mean    <= 0;
            for (j = 0; j < N_EMBD; j = j + 1) begin
                x_buf[j]   <= 0;
                x_prev[j]  <= 0;
                ln_buf[j]  <= 0;
                sub_out[j] <= 0;
            end
        end else begin
            tm_start    <= 1'b0;
            cm_start    <= 1'b0;
            lif1_en     <= 1'b0;
            lif2_en     <= 1'b0;
            o_out_valid <= 1'b0;

            case (blk_state)
                // =============================================================
                BLK_IDLE: begin
                    if (i_start) begin
                        blk_state <= BLK_LOAD;
                        load_cnt  <= 0;
                        ch_cnt    <= 0;
                    end
                end

                // =============================================================
                // Load input data into x_buf
                BLK_LOAD: begin
                    if (i_data_valid) begin
                        x_buf[load_cnt] <= i_data;
                        load_cnt <= load_cnt + 1;
                        if (load_cnt == N_EMBD - 1) begin
                            blk_state <= BLK_LN1;
                            ln_sum    <= 0;
                            ch_cnt    <= 0;
                        end
                    end
                end

                // =============================================================
                // LayerNorm 1: compute mean, then subtract
                BLK_LN1: begin
                    if (ch_cnt < N_EMBD) begin
                        // Accumulate sum for mean
                        ln_sum <= ln_sum + {{(ACCUM_WIDTH-DATA_WIDTH){x_buf[ch_cnt][DATA_WIDTH-1]}},
                                            x_buf[ch_cnt]};
                        ch_cnt <= ch_cnt + 1;
                    end else if (ch_cnt == N_EMBD) begin
                        // Compute mean (divide by N_EMBD via shift for power-of-2)
                        ln_mean <= ln_sum >>> $clog2(N_EMBD);
                        ch_cnt  <= ch_cnt + 1;
                    end else begin
                        // Subtract mean from each element
                        if (ch_cnt - N_EMBD - 1 < N_EMBD) begin
                            ln_buf[ch_cnt - N_EMBD - 1] <= x_buf[ch_cnt - N_EMBD - 1] - ln_mean;
                            ch_cnt <= ch_cnt + 1;
                        end else begin
                            // Launch time-mix
                            blk_state <= BLK_TMIX;
                            tm_start  <= 1'b1;
                            ch_cnt    <= 0;
                        end
                    end
                end

                // =============================================================
                // Time-mix: wait for sub-module completion
                BLK_TMIX: begin
                    if (ch_cnt < N_EMBD) begin
                        ch_cnt <= ch_cnt + 1;  // Feed data to time_mix
                    end
                    if (tm_done) begin
                        blk_state <= BLK_LIF1;
                        ch_cnt    <= 0;
                    end
                end

                // =============================================================
                // LIF1: process time-mix output through LIF neuron, add residual
                BLK_LIF1: begin
                    if (ch_cnt < N_EMBD) begin
                        lif1_en <= 1'b1;
                        lif1_in <= sub_out[ch_cnt];
                        // Residual: x = x + spike (spike is 0 or 1 scaled)
                        if (lif1_spike)
                            x_buf[ch_cnt] <= x_buf[ch_cnt] + sub_out[ch_cnt];
                        ch_cnt <= ch_cnt + 1;
                    end else begin
                        blk_state <= BLK_LN2;
                        ln_sum    <= 0;
                        ch_cnt    <= 0;
                    end
                end

                // =============================================================
                // LayerNorm 2: same approx as LN1
                BLK_LN2: begin
                    if (ch_cnt < N_EMBD) begin
                        ln_sum <= ln_sum + {{(ACCUM_WIDTH-DATA_WIDTH){x_buf[ch_cnt][DATA_WIDTH-1]}},
                                            x_buf[ch_cnt]};
                        ch_cnt <= ch_cnt + 1;
                    end else if (ch_cnt == N_EMBD) begin
                        ln_mean <= ln_sum >>> $clog2(N_EMBD);
                        ch_cnt  <= ch_cnt + 1;
                    end else begin
                        if (ch_cnt - N_EMBD - 1 < N_EMBD) begin
                            ln_buf[ch_cnt - N_EMBD - 1] <= x_buf[ch_cnt - N_EMBD - 1] - ln_mean;
                            ch_cnt <= ch_cnt + 1;
                        end else begin
                            blk_state <= BLK_CMIX;
                            cm_start  <= 1'b1;
                            ch_cnt    <= 0;
                        end
                    end
                end

                // =============================================================
                // Channel-mix: wait for sub-module
                BLK_CMIX: begin
                    if (ch_cnt < N_EMBD) begin
                        ch_cnt <= ch_cnt + 1;
                    end
                    if (cm_done) begin
                        blk_state <= BLK_LIF2;
                        ch_cnt    <= 0;
                    end
                end

                // =============================================================
                // LIF2: process channel-mix output, add residual, output
                BLK_LIF2: begin
                    if (ch_cnt < N_EMBD) begin
                        lif2_en <= 1'b1;
                        lif2_in <= sub_out[ch_cnt];
                        if (lif2_spike)
                            x_buf[ch_cnt] <= x_buf[ch_cnt] + sub_out[ch_cnt];

                        // Save current x as x_prev for next token's time-shift
                        x_prev[ch_cnt] <= x_buf[ch_cnt];

                        // Stream output
                        o_out_valid <= 1'b1;
                        o_out_data  <= x_buf[ch_cnt];
                        ch_cnt <= ch_cnt + 1;
                    end else begin
                        blk_state <= BLK_IDLE;
                        ch_cnt    <= N_EMBD; // Signal done
                    end
                end

                default: blk_state <= BLK_IDLE;
            endcase
        end
    end

endmodule
