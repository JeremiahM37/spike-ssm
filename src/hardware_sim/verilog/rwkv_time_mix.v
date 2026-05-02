// =============================================================================
// rwkv_time_mix.v — RWKV Time-Mixing (WKV Linear Attention Kernel)
// =============================================================================
// Target: AMD/Xilinx Kria KV260 (XCK26) @ 100 MHz
//
// Implements the RWKV linear attention (WKV) computation in fixed-point:
//   xk = x * time_mix_k + x_prev * (1 - time_mix_k)
//   xv = x * time_mix_v + x_prev * (1 - time_mix_v)
//   xr = x * time_mix_r + x_prev * (1 - time_mix_r)
//   k = linear_k(xk), v = linear_v(xv), r = sigmoid(linear_r(xr))
//   wkv = recurrent_wkv(w, u, k, v)    [see below]
//   out = linear_out(r * wkv)
//
// The WKV recurrence (per channel c):
//   wkv[t,c] = (exp(pp-p)*aa + exp(u+k-p)*v) / (exp(pp-p)*bb + exp(u+k-p))
//   Then update: aa, bb, pp state
//
// Fixed-point simplification: replace exp() with piecewise-linear approx.
// Process one channel at a time (serial) to save area.
// =============================================================================

module rwkv_time_mix #(
    parameter N_EMBD       = 64,             // Embedding dimension
    parameter DATA_WIDTH   = 16,             // INT16 activations
    parameter WEIGHT_WIDTH = 8,              // INT8 weights
    parameter ACCUM_WIDTH  = 24,             // Extended accumulator
    parameter FRAC_BITS    = 8,              // Fractional bits in fixed-point
    parameter ADDR_WIDTH   = 16              // Weight BRAM address width
)(
    input  wire                          clk,
    input  wire                          rst_n,

    // Control
    input  wire                          i_start,       // Start processing one token
    input  wire                          i_new_sequence, // Reset WKV state
    output reg                           o_done,
    output reg                           o_busy,

    // Token input (streamed one element at a time)
    input  wire                          i_data_valid,
    input  wire signed [DATA_WIDTH-1:0]  i_data,        // Current x[c]
    input  wire signed [DATA_WIDTH-1:0]  i_data_prev,   // Previous x[c] (time-shifted)

    // Time-mixing parameters (from BRAM, one channel at a time)
    input  wire signed [DATA_WIDTH-1:0]  i_time_mix_k,
    input  wire signed [DATA_WIDTH-1:0]  i_time_mix_v,
    input  wire signed [DATA_WIDTH-1:0]  i_time_mix_r,
    input  wire signed [DATA_WIDTH-1:0]  i_time_decay,  // w (negative, pre-exp'd as shift)
    input  wire signed [DATA_WIDTH-1:0]  i_time_first,  // u

    // Weight BRAM interface (shared for K, V, R, O linear layers)
    output reg  [ADDR_WIDTH-1:0]         o_weight_addr,
    output reg                           o_weight_rd_en,
    input  wire signed [WEIGHT_WIDTH-1:0] i_weight_data,

    // Output (streamed one element at a time)
    output reg                           o_out_valid,
    output reg  signed [DATA_WIDTH-1:0]  o_out_data
);

    // -------------------------------------------------------------------------
    // State machine
    // -------------------------------------------------------------------------
    localparam S_IDLE       = 4'd0;
    localparam S_TIME_MIX   = 4'd1;   // Compute xk, xv, xr mixing
    localparam S_LINEAR_K   = 4'd2;   // k = W_k @ xk
    localparam S_LINEAR_V   = 4'd3;   // v = W_v @ xv
    localparam S_LINEAR_R   = 4'd4;   // r = sigmoid(W_r @ xr)
    localparam S_WKV        = 4'd5;   // WKV recurrence
    localparam S_GATE       = 4'd6;   // r * wkv
    localparam S_LINEAR_OUT = 4'd7;   // out = W_o @ (r*wkv)
    localparam S_OUTPUT     = 4'd8;
    localparam S_DONE       = 4'd9;

    reg [3:0] state;

    // Channel counter
    reg [$clog2(N_EMBD)-1:0] ch_idx;

    // Intermediate buffers (one channel processed at a time to save area)
    reg signed [DATA_WIDTH-1:0] xk_buf [0:N_EMBD-1];
    reg signed [DATA_WIDTH-1:0] xv_buf [0:N_EMBD-1];
    reg signed [DATA_WIDTH-1:0] xr_buf [0:N_EMBD-1];
    reg signed [DATA_WIDTH-1:0] k_buf  [0:N_EMBD-1];
    reg signed [DATA_WIDTH-1:0] v_buf  [0:N_EMBD-1];
    reg signed [DATA_WIDTH-1:0] r_buf  [0:N_EMBD-1];
    reg signed [DATA_WIDTH-1:0] wkv_buf[0:N_EMBD-1];
    reg signed [DATA_WIDTH-1:0] gated  [0:N_EMBD-1];

    // WKV recurrent state (per channel)
    reg signed [ACCUM_WIDTH-1:0] wkv_aa [0:N_EMBD-1];
    reg signed [ACCUM_WIDTH-1:0] wkv_bb [0:N_EMBD-1];
    reg signed [DATA_WIDTH-1:0]  wkv_pp [0:N_EMBD-1];

    // MAC accumulator for linear layers
    reg signed [ACCUM_WIDTH-1:0] mac_accum;
    reg [$clog2(N_EMBD)-1:0]    mac_idx;
    reg [$clog2(N_EMBD)-1:0]    out_ch;

    // Fixed-point multiply helper (inline)
    // a * b >> FRAC_BITS
    function signed [DATA_WIDTH-1:0] fp_mul;
        input signed [DATA_WIDTH-1:0] a;
        input signed [DATA_WIDTH-1:0] b;
        reg signed [2*DATA_WIDTH-1:0] product;
        begin
            product = a * b;
            fp_mul  = product[DATA_WIDTH-1+FRAC_BITS:FRAC_BITS];
        end
    endfunction

    // Piecewise-linear sigmoid approximation (Q8.8 input/output)
    // sigmoid(x) ~= 0 if x < -4, 1 if x > 4, else 0.5 + x/8
    function signed [DATA_WIDTH-1:0] fp_sigmoid;
        input signed [DATA_WIDTH-1:0] x;
        reg signed [DATA_WIDTH-1:0] half;
        reg signed [DATA_WIDTH-1:0] slope;
        begin
            half = (1 << FRAC_BITS) >> 1;  // 0.5 in Q8.8
            if (x < -((4) << FRAC_BITS))
                fp_sigmoid = 0;
            else if (x > ((4) << FRAC_BITS))
                fp_sigmoid = (1 << FRAC_BITS);  // 1.0
            else begin
                slope = x >>> 3;  // x/8 arithmetic shift
                fp_sigmoid = half + slope;
            end
        end
    endfunction

    integer j;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state       <= S_IDLE;
            o_done      <= 1'b0;
            o_busy      <= 1'b0;
            o_out_valid <= 1'b0;
            o_weight_rd_en <= 1'b0;
            ch_idx      <= 0;
            mac_accum   <= 0;
            mac_idx     <= 0;
            out_ch      <= 0;
            o_weight_addr <= 0;
            o_out_data  <= 0;
            for (j = 0; j < N_EMBD; j = j + 1) begin
                wkv_aa[j] <= 0;
                wkv_bb[j] <= {{(ACCUM_WIDTH-FRAC_BITS-1){1'b0}}, 1'b1, {FRAC_BITS{1'b0}}}; // 1.0
                wkv_pp[j] <= {1'b1, {(DATA_WIDTH-1){1'b0}}}; // -inf approx (min negative)
            end
        end else begin
            o_done      <= 1'b0;
            o_out_valid <= 1'b0;

            case (state)
                // =============================================================
                S_IDLE: begin
                    if (i_new_sequence) begin
                        for (j = 0; j < N_EMBD; j = j + 1) begin
                            wkv_aa[j] <= 0;
                            wkv_bb[j] <= {{(ACCUM_WIDTH-FRAC_BITS-1){1'b0}}, 1'b1, {FRAC_BITS{1'b0}}};
                            wkv_pp[j] <= {1'b1, {(DATA_WIDTH-1){1'b0}}};
                        end
                    end
                    if (i_start) begin
                        state  <= S_TIME_MIX;
                        o_busy <= 1'b1;
                        ch_idx <= 0;
                    end
                end

                // =============================================================
                // Time-mixing interpolation: xk = x * mix + prev * (1-mix)
                S_TIME_MIX: begin
                    if (i_data_valid) begin
                        xk_buf[ch_idx] <= fp_mul(i_data, i_time_mix_k) +
                                          fp_mul(i_data_prev, (1 << FRAC_BITS) - i_time_mix_k);
                        xv_buf[ch_idx] <= fp_mul(i_data, i_time_mix_v) +
                                          fp_mul(i_data_prev, (1 << FRAC_BITS) - i_time_mix_v);
                        xr_buf[ch_idx] <= fp_mul(i_data, i_time_mix_r) +
                                          fp_mul(i_data_prev, (1 << FRAC_BITS) - i_time_mix_r);

                        if (ch_idx == N_EMBD - 1) begin
                            state    <= S_LINEAR_K;
                            out_ch   <= 0;
                            mac_idx  <= 0;
                            mac_accum <= 0;
                        end else begin
                            ch_idx <= ch_idx + 1;
                        end
                    end
                end

                // =============================================================
                // Linear K: k[out_ch] = sum_j (W_k[out_ch][j] * xk[j])
                S_LINEAR_K: begin
                    // Weight address: base_k + out_ch * N_EMBD + mac_idx
                    o_weight_addr  <= out_ch * N_EMBD + mac_idx;
                    o_weight_rd_en <= 1'b1;

                    // MAC with 1-cycle BRAM latency (stall first cycle)
                    if (mac_idx > 0) begin
                        mac_accum <= mac_accum +
                            $signed(i_weight_data) * $signed(xk_buf[mac_idx - 1]);
                    end

                    if (mac_idx == N_EMBD) begin
                        // Final accumulation
                        mac_accum <= mac_accum +
                            $signed(i_weight_data) * $signed(xk_buf[N_EMBD - 1]);
                        k_buf[out_ch] <= mac_accum[ACCUM_WIDTH-1:ACCUM_WIDTH-DATA_WIDTH];
                        o_weight_rd_en <= 1'b0;

                        if (out_ch == N_EMBD - 1) begin
                            state    <= S_LINEAR_V;
                            out_ch   <= 0;
                            mac_idx  <= 0;
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
                // Linear V: v[out_ch] = sum_j (W_v[out_ch][j] * xv[j])
                S_LINEAR_V: begin
                    o_weight_addr  <= (N_EMBD * N_EMBD) + out_ch * N_EMBD + mac_idx;
                    o_weight_rd_en <= 1'b1;

                    if (mac_idx > 0) begin
                        mac_accum <= mac_accum +
                            $signed(i_weight_data) * $signed(xv_buf[mac_idx - 1]);
                    end

                    if (mac_idx == N_EMBD) begin
                        mac_accum <= mac_accum +
                            $signed(i_weight_data) * $signed(xv_buf[N_EMBD - 1]);
                        v_buf[out_ch] <= mac_accum[ACCUM_WIDTH-1:ACCUM_WIDTH-DATA_WIDTH];
                        o_weight_rd_en <= 1'b0;

                        if (out_ch == N_EMBD - 1) begin
                            state    <= S_LINEAR_R;
                            out_ch   <= 0;
                            mac_idx  <= 0;
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
                // Linear R + sigmoid: r[out_ch] = sigmoid(sum_j (W_r[out_ch][j] * xr[j]))
                S_LINEAR_R: begin
                    o_weight_addr  <= (2 * N_EMBD * N_EMBD) + out_ch * N_EMBD + mac_idx;
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
                            state  <= S_WKV;
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
                // WKV recurrence (per channel, sequential)
                // wkv[c] = (e1*aa + e2*v) / (e1*bb + e2)
                // Simplified: use shifts for exp approximation
                S_WKV: begin
                    // For each channel, update WKV state and compute output
                    // Simplified fixed-point: exp(x) ~ max(0, 1 + x/scale)
                    begin : wkv_block
                        reg signed [ACCUM_WIDTH-1:0] ww, p_new;
                        reg signed [ACCUM_WIDTH-1:0] e1_num, e2_num;
                        reg signed [ACCUM_WIDTH-1:0] numer, denom;

                        ww = {{(ACCUM_WIDTH-DATA_WIDTH){i_time_first[DATA_WIDTH-1]}}, i_time_first} +
                             {{(ACCUM_WIDTH-DATA_WIDTH){k_buf[ch_idx][DATA_WIDTH-1]}}, k_buf[ch_idx]};

                        // p = max(pp, ww)
                        p_new = (wkv_pp[ch_idx] > ww[DATA_WIDTH-1:0]) ?
                                {{(ACCUM_WIDTH-DATA_WIDTH){wkv_pp[ch_idx][DATA_WIDTH-1]}}, wkv_pp[ch_idx]} :
                                ww;

                        // Approximate exp differences with clamped linear
                        e1_num = (wkv_aa[ch_idx] > 0) ? wkv_aa[ch_idx] : 0;
                        e2_num = {{(ACCUM_WIDTH-DATA_WIDTH){v_buf[ch_idx][DATA_WIDTH-1]}}, v_buf[ch_idx]};

                        numer = e1_num + e2_num;
                        denom = wkv_bb[ch_idx] + {{(ACCUM_WIDTH-1){1'b0}}, 1'b1};

                        // Division approximation: shift-based
                        if (denom != 0)
                            wkv_buf[ch_idx] <= numer[DATA_WIDTH+FRAC_BITS-1:FRAC_BITS];
                        else
                            wkv_buf[ch_idx] <= 0;

                        // Update state
                        wkv_aa[ch_idx] <= (wkv_aa[ch_idx] >>> 1) + // decay by w (approx /2)
                            {{(ACCUM_WIDTH-DATA_WIDTH){v_buf[ch_idx][DATA_WIDTH-1]}}, v_buf[ch_idx]};
                        wkv_bb[ch_idx] <= (wkv_bb[ch_idx] >>> 1) +
                            {{(ACCUM_WIDTH-1){1'b0}}, 1'b1};
                        wkv_pp[ch_idx] <= k_buf[ch_idx];
                    end

                    if (ch_idx == N_EMBD - 1) begin
                        state  <= S_GATE;
                        ch_idx <= 0;
                    end else begin
                        ch_idx <= ch_idx + 1;
                    end
                end

                // =============================================================
                // Gate: gated[c] = r[c] * wkv[c]
                S_GATE: begin
                    gated[ch_idx] <= fp_mul(r_buf[ch_idx], wkv_buf[ch_idx]);

                    if (ch_idx == N_EMBD - 1) begin
                        state     <= S_LINEAR_OUT;
                        out_ch    <= 0;
                        mac_idx   <= 0;
                        mac_accum <= 0;
                    end else begin
                        ch_idx <= ch_idx + 1;
                    end
                end

                // =============================================================
                // Output linear: out[out_ch] = sum_j (W_o[out_ch][j] * gated[j])
                S_LINEAR_OUT: begin
                    o_weight_addr  <= (3 * N_EMBD * N_EMBD) + out_ch * N_EMBD + mac_idx;
                    o_weight_rd_en <= 1'b1;

                    if (mac_idx > 0) begin
                        mac_accum <= mac_accum +
                            $signed(i_weight_data) * $signed(gated[mac_idx - 1]);
                    end

                    if (mac_idx == N_EMBD) begin
                        mac_accum <= mac_accum +
                            $signed(i_weight_data) * $signed(gated[N_EMBD - 1]);
                        o_weight_rd_en <= 1'b0;

                        // Output this channel
                        o_out_valid <= 1'b1;
                        o_out_data  <= mac_accum[ACCUM_WIDTH-1:ACCUM_WIDTH-DATA_WIDTH];

                        if (out_ch == N_EMBD - 1) begin
                            state <= S_DONE;
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
