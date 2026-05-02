// =============================================================================
// rwkv_channel_mix.v — RWKV Channel-Mixing (FFN with Spiking Activation)
// =============================================================================
// Target: AMD/Xilinx Kria KV260 (XCK26) @ 100 MHz
//
// Implements SpikeGPT's channel mixing (FFN):
//   xk = x * time_mix_k + x_prev * (1 - time_mix_k)
//   xr = x * time_mix_r + x_prev * (1 - time_mix_r)
//   k  = W_key @ xk                           [n_embd -> 4*n_embd]
//   k  = relu(k)^2                             (SpikeGPT's squared ReLU)
//   kv = W_value @ k                           [4*n_embd -> n_embd]
//   r  = sigmoid(W_receptance @ xr)            [n_embd -> n_embd]
//   out = r * kv
//
// Fixed-point: INT8 weights, INT16 activations, INT24 accumulators.
// Hidden FFN dimension = 4 * N_EMBD.
// =============================================================================

module rwkv_channel_mix #(
    parameter N_EMBD       = 64,
    parameter HIDDEN_SZ    = 256,            // 4 * N_EMBD
    parameter DATA_WIDTH   = 16,
    parameter WEIGHT_WIDTH = 8,
    parameter ACCUM_WIDTH  = 24,
    parameter FRAC_BITS    = 8,
    parameter ADDR_WIDTH   = 18              // Weight BRAM address width
)(
    input  wire                          clk,
    input  wire                          rst_n,

    // Control
    input  wire                          i_start,
    output reg                           o_done,
    output reg                           o_busy,

    // Token input (one element at a time)
    input  wire                          i_data_valid,
    input  wire signed [DATA_WIDTH-1:0]  i_data,
    input  wire signed [DATA_WIDTH-1:0]  i_data_prev,

    // Time-mixing parameters
    input  wire signed [DATA_WIDTH-1:0]  i_time_mix_k,
    input  wire signed [DATA_WIDTH-1:0]  i_time_mix_r,

    // Weight BRAM interface
    output reg  [ADDR_WIDTH-1:0]         o_weight_addr,
    output reg                           o_weight_rd_en,
    input  wire signed [WEIGHT_WIDTH-1:0] i_weight_data,

    // Output (streamed)
    output reg                           o_out_valid,
    output reg  signed [DATA_WIDTH-1:0]  o_out_data
);

    // -------------------------------------------------------------------------
    // State machine
    // -------------------------------------------------------------------------
    localparam S_IDLE       = 4'd0;
    localparam S_MIX        = 4'd1;
    localparam S_LINEAR_K   = 4'd2;   // key linear: n_embd -> hidden_sz
    localparam S_RELU_SQ    = 4'd3;   // squared ReLU activation
    localparam S_LINEAR_V   = 4'd4;   // value linear: hidden_sz -> n_embd
    localparam S_LINEAR_R   = 4'd5;   // receptance linear + sigmoid
    localparam S_GATE       = 4'd6;   // r * kv
    localparam S_DONE       = 4'd7;

    reg [3:0] state;

    // Channel counter
    reg [$clog2(N_EMBD)-1:0]    ch_idx;
    reg [$clog2(HIDDEN_SZ)-1:0] hid_idx;

    // Buffers
    reg signed [DATA_WIDTH-1:0] xk_buf   [0:N_EMBD-1];
    reg signed [DATA_WIDTH-1:0] xr_buf   [0:N_EMBD-1];
    reg signed [DATA_WIDTH-1:0] k_buf    [0:HIDDEN_SZ-1];  // After key linear
    reg signed [DATA_WIDTH-1:0] kv_buf   [0:N_EMBD-1];     // After value linear
    reg signed [DATA_WIDTH-1:0] r_buf    [0:N_EMBD-1];     // Receptance (sigmoided)

    // MAC
    reg signed [ACCUM_WIDTH-1:0] mac_accum;
    reg [$clog2(HIDDEN_SZ)-1:0]  mac_idx;
    reg [$clog2(HIDDEN_SZ)-1:0]  out_ch;

    // Fixed-point multiply
    function signed [DATA_WIDTH-1:0] fp_mul;
        input signed [DATA_WIDTH-1:0] a;
        input signed [DATA_WIDTH-1:0] b;
        reg signed [2*DATA_WIDTH-1:0] product;
        begin
            product = a * b;
            fp_mul  = product[DATA_WIDTH-1+FRAC_BITS:FRAC_BITS];
        end
    endfunction

    // Sigmoid approximation
    function signed [DATA_WIDTH-1:0] fp_sigmoid;
        input signed [DATA_WIDTH-1:0] x;
        reg signed [DATA_WIDTH-1:0] half;
        begin
            half = (1 << FRAC_BITS) >> 1;
            if (x < -((4) << FRAC_BITS))
                fp_sigmoid = 0;
            else if (x > ((4) << FRAC_BITS))
                fp_sigmoid = (1 << FRAC_BITS);
            else
                fp_sigmoid = half + (x >>> 3);
        end
    endfunction

    // Squared ReLU: max(0, x)^2
    function signed [DATA_WIDTH-1:0] relu_squared;
        input signed [DATA_WIDTH-1:0] x;
        reg signed [2*DATA_WIDTH-1:0] sq;
        begin
            if (x <= 0)
                relu_squared = 0;
            else begin
                sq = x * x;
                // Saturate to DATA_WIDTH
                if (sq[2*DATA_WIDTH-1:DATA_WIDTH+FRAC_BITS] != 0)
                    relu_squared = {1'b0, {(DATA_WIDTH-1){1'b1}}}; // Saturate
                else
                    relu_squared = sq[DATA_WIDTH-1+FRAC_BITS:FRAC_BITS];
            end
        end
    endfunction

    integer j;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state          <= S_IDLE;
            o_done         <= 1'b0;
            o_busy         <= 1'b0;
            o_out_valid    <= 1'b0;
            o_weight_rd_en <= 1'b0;
            ch_idx         <= 0;
            hid_idx        <= 0;
            mac_accum      <= 0;
            mac_idx        <= 0;
            out_ch         <= 0;
            o_weight_addr  <= 0;
            o_out_data     <= 0;
        end else begin
            o_done      <= 1'b0;
            o_out_valid <= 1'b0;

            case (state)
                // =============================================================
                S_IDLE: begin
                    if (i_start) begin
                        state  <= S_MIX;
                        o_busy <= 1'b1;
                        ch_idx <= 0;
                    end
                end

                // =============================================================
                // Time-mixing: xk, xr interpolation
                S_MIX: begin
                    if (i_data_valid) begin
                        xk_buf[ch_idx] <= fp_mul(i_data, i_time_mix_k) +
                                          fp_mul(i_data_prev, (1 << FRAC_BITS) - i_time_mix_k);
                        xr_buf[ch_idx] <= fp_mul(i_data, i_time_mix_r) +
                                          fp_mul(i_data_prev, (1 << FRAC_BITS) - i_time_mix_r);

                        if (ch_idx == N_EMBD - 1) begin
                            state     <= S_LINEAR_K;
                            out_ch    <= 0;
                            mac_idx   <= 0;
                            mac_accum <= 0;
                        end else begin
                            ch_idx <= ch_idx + 1;
                        end
                    end
                end

                // =============================================================
                // Key linear: k[h] = sum_j W_key[h][j] * xk[j], h in [0, HIDDEN_SZ)
                // Weight base address: 0
                S_LINEAR_K: begin
                    o_weight_addr  <= out_ch * N_EMBD + mac_idx;
                    o_weight_rd_en <= 1'b1;

                    if (mac_idx > 0) begin
                        mac_accum <= mac_accum +
                            $signed(i_weight_data) * $signed(xk_buf[mac_idx - 1]);
                    end

                    if (mac_idx == N_EMBD) begin
                        mac_accum <= mac_accum +
                            $signed(i_weight_data) * $signed(xk_buf[N_EMBD - 1]);
                        k_buf[out_ch] <= mac_accum[ACCUM_WIDTH-1:ACCUM_WIDTH-DATA_WIDTH];
                        o_weight_rd_en <= 1'b0;

                        if (out_ch == HIDDEN_SZ - 1) begin
                            state   <= S_RELU_SQ;
                            hid_idx <= 0;
                        end else begin
                            out_ch    <= out_ch + 1;
                            mac_idx   <= 0;
                            mac_accum <= 0;
                        end
                    end else begin
                        mac_idx <= mac_idx + 1;
                    end
                end

                // =============================================================
                // Squared ReLU activation on k_buf
                S_RELU_SQ: begin
                    k_buf[hid_idx] <= relu_squared(k_buf[hid_idx]);

                    if (hid_idx == HIDDEN_SZ - 1) begin
                        state     <= S_LINEAR_V;
                        out_ch    <= 0;
                        mac_idx   <= 0;
                        mac_accum <= 0;
                    end else begin
                        hid_idx <= hid_idx + 1;
                    end
                end

                // =============================================================
                // Value linear: kv[n] = sum_h W_value[n][h] * k_activated[h]
                // Weight base: HIDDEN_SZ * N_EMBD
                S_LINEAR_V: begin
                    o_weight_addr  <= (HIDDEN_SZ * N_EMBD) + out_ch * HIDDEN_SZ + mac_idx;
                    o_weight_rd_en <= 1'b1;

                    if (mac_idx > 0) begin
                        mac_accum <= mac_accum +
                            $signed(i_weight_data) * $signed(k_buf[mac_idx - 1]);
                    end

                    if (mac_idx == HIDDEN_SZ) begin
                        mac_accum <= mac_accum +
                            $signed(i_weight_data) * $signed(k_buf[HIDDEN_SZ - 1]);
                        kv_buf[out_ch] <= mac_accum[ACCUM_WIDTH-1:ACCUM_WIDTH-DATA_WIDTH];
                        o_weight_rd_en <= 1'b0;

                        if (out_ch == N_EMBD - 1) begin
                            state     <= S_LINEAR_R;
                            out_ch    <= 0;
                            mac_idx   <= 0;
                            mac_accum <= 0;
                        end else begin
                            out_ch    <= out_ch + 1;
                            mac_idx   <= 0;
                            mac_accum <= 0;
                        end
                    end else begin
                        mac_idx <= mac_idx + 1;
                    end
                end

                // =============================================================
                // Receptance linear + sigmoid: r[n] = sigmoid(sum_j W_r[n][j] * xr[j])
                // Weight base: HIDDEN_SZ*N_EMBD + N_EMBD*HIDDEN_SZ
                S_LINEAR_R: begin
                    o_weight_addr  <= (HIDDEN_SZ * N_EMBD + N_EMBD * HIDDEN_SZ) +
                                      out_ch * N_EMBD + mac_idx;
                    o_weight_rd_en <= 1'b1;

                    if (mac_idx > 0) begin
                        mac_accum <= mac_accum +
                            $signed(i_weight_data) * $signed(xr_buf[mac_idx - 1]);
                    end

                    if (mac_idx == N_EMBD) begin
                        mac_accum <= mac_accum +
                            $signed(i_weight_data) * $signed(xr_buf[N_EMBD - 1]);
                        r_buf[out_ch] <= fp_sigmoid(mac_accum[ACCUM_WIDTH-1:ACCUM_WIDTH-DATA_WIDTH]);
                        o_weight_rd_en <= 1'b0;

                        if (out_ch == N_EMBD - 1) begin
                            state  <= S_GATE;
                            ch_idx <= 0;
                        end else begin
                            out_ch    <= out_ch + 1;
                            mac_idx   <= 0;
                            mac_accum <= 0;
                        end
                    end else begin
                        mac_idx <= mac_idx + 1;
                    end
                end

                // =============================================================
                // Gating: out = r * kv
                S_GATE: begin
                    o_out_valid <= 1'b1;
                    o_out_data  <= fp_mul(r_buf[ch_idx], kv_buf[ch_idx]);

                    if (ch_idx == N_EMBD - 1) begin
                        state <= S_DONE;
                    end else begin
                        ch_idx <= ch_idx + 1;
                    end
                end

                // =============================================================
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
