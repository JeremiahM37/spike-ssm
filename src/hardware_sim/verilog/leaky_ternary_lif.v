// =============================================================================
// leaky_ternary_lif.v — LeakyTernaryLIF Neuron with SiLU Bypass
// =============================================================================
// Target: AMD/Xilinx Kria KV260 (XCK26) @ 100 MHz
//
// Implements: out = alpha * spike + (1 - alpha) * silu_approx(x)
//
// The alpha value is fixed at compile time (learned during training, frozen
// at inference). This means the continuous/spike blend ratio is a hardware
// constant, enabling efficient implementation:
//
//   Spike path:  ternary {-1, 0, +1} * alpha  → 3-entry LUT (no multiply)
//   SiLU path:   piecewise-linear approximation → 1 multiply + add
//   Blend:       alpha * spike_val + (1-alpha) * silu_val → pre-scaled add
//
// SiLU approximation (3-segment piecewise linear):
//   x < -4:   silu(x) = 0
//   -4 <= x <= 4:  silu(x) = (x + 4) * x / 8  (quadratic approx)
//   x > 4:    silu(x) = x
//
// For FPGA: the quadratic term uses 1 DSP slice per neuron group
// (time-multiplexed across the 128 neurons in a layer).
// =============================================================================

module leaky_ternary_lif #(
    parameter DATA_WIDTH  = 16,           // Q8.8 fixed point
    parameter BETA_WIDTH  = 8,
    parameter BETA_VALUE  = 8'd230,       // beta=0.9
    parameter BETA_SHIFT  = 8,
    parameter THRESHOLD   = 16'sd384,     // 1.5 in Q8.8
    parameter ALPHA_NUM   = 8'd218,       // alpha=0.85 in Q0.8 (218/256)
    parameter ALPHA_SHIFT = 8,
    parameter TOPK_EN     = 0,            // 1 = enable top-K gating
    parameter SILU_CLAMP  = 16'sd1024     // |x| > 4.0 in Q8.8
)(
    input  wire                         clk,
    input  wire                         rst_n,
    input  wire                         en,
    input  wire signed [DATA_WIDTH-1:0] i_current,     // input x
    input  wire                         i_topk_allow,  // from topk_selector
    output reg  signed [DATA_WIDTH-1:0] o_output,      // blended output
    output reg  [1:0]                   o_spike_raw,   // raw ternary spike (for debug)
    output reg  signed [DATA_WIDTH-1:0] o_membrane
);

    // --- Ternary LIF ---
    reg  signed [DATA_WIDTH-1:0]   membrane_r;
    wire signed [2*DATA_WIDTH-1:0] leak_product;
    wire signed [DATA_WIDTH-1:0]   leaked_membrane;

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

    // Ternary firing decision
    wire fire_pos = (new_membrane >= THRESHOLD);
    wire fire_neg = (new_membrane <= -THRESHOLD);

    // Top-K gating: only fire if top-K selector allows
    wire gated_fire_pos = TOPK_EN ? (fire_pos & i_topk_allow) : fire_pos;
    wire gated_fire_neg = TOPK_EN ? (fire_neg & i_topk_allow) : fire_neg;

    // Spike value: +alpha, -alpha, or 0
    // Since alpha is a compile-time constant, these are just constants
    wire signed [DATA_WIDTH-1:0] spike_scaled;
    assign spike_scaled = gated_fire_pos ?  $signed({{(DATA_WIDTH-ALPHA_SHIFT){1'b0}}, ALPHA_NUM}) :
                          gated_fire_neg ? -$signed({{(DATA_WIDTH-ALPHA_SHIFT){1'b0}}, ALPHA_NUM}) :
                          {DATA_WIDTH{1'b0}};

    // --- SiLU approximation (piecewise linear) ---
    // silu(x) ≈ x * sigmoid(x)
    // Simplified 3-segment:
    //   x < -CLAMP:           0
    //   -CLAMP <= x <= CLAMP: x * (x + CLAMP) / (2*CLAMP)  [linear ramp approx]
    //   x > CLAMP:            x

    wire signed [DATA_WIDTH-1:0] x_in = i_current;
    wire x_neg_clamp = (x_in < -SILU_CLAMP);
    wire x_pos_clamp = (x_in > SILU_CLAMP);

    // Linear ramp: silu ≈ x * (x + 4) / 8 in the middle region
    // In fixed point: (x * (x + CLAMP)) >> (CLAMP_SHIFT+1)
    wire signed [DATA_WIDTH-1:0] x_plus_clamp = x_in + SILU_CLAMP;
    wire signed [2*DATA_WIDTH-1:0] silu_product = x_in * x_plus_clamp;
    // Shift right by 11 for Q8.8 with CLAMP=1024 (4.0): 1024*2 = 2048 = 2^11
    wire signed [DATA_WIDTH-1:0] silu_mid = silu_product[2*DATA_WIDTH-1:11];

    wire signed [DATA_WIDTH-1:0] silu_approx;
    assign silu_approx = x_neg_clamp ? {DATA_WIDTH{1'b0}} :
                         x_pos_clamp ? x_in :
                         silu_mid;

    // Scale by (1-alpha): (256-ALPHA_NUM)/256
    localparam [ALPHA_SHIFT-1:0] ONE_MINUS_ALPHA = (1 << ALPHA_SHIFT) - ALPHA_NUM;
    wire signed [2*DATA_WIDTH-1:0] silu_scaled_wide;
    assign silu_scaled_wide = silu_approx * $signed({{(DATA_WIDTH-ALPHA_SHIFT){1'b0}}, ONE_MINUS_ALPHA});
    wire signed [DATA_WIDTH-1:0] silu_scaled = silu_scaled_wide[DATA_WIDTH-1+ALPHA_SHIFT:ALPHA_SHIFT];

    // --- Blend: alpha*spike + (1-alpha)*silu(x) ---
    wire signed [DATA_WIDTH:0] blend_ext;
    assign blend_ext = {spike_scaled[DATA_WIDTH-1], spike_scaled} +
                       {silu_scaled[DATA_WIDTH-1], silu_scaled};
    wire signed [DATA_WIDTH-1:0] blended;
    assign blended = blend_ext[DATA_WIDTH-1:0]; // truncate (no overflow expected)

    // --- Sequential logic ---
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            membrane_r <= 0;
            o_output   <= 0;
            o_spike_raw <= 2'b00;
            o_membrane <= 0;
        end else if (en) begin
            // Output the blended value
            o_output <= blended;

            // Raw spike for monitoring
            o_spike_raw <= gated_fire_pos ? 2'b01 :
                           gated_fire_neg ? 2'b11 : 2'b00;

            // Membrane update (soft reset on fire)
            if (gated_fire_pos || gated_fire_neg) begin
                membrane_r <= 0;
            end else begin
                membrane_r <= new_membrane;
            end
            o_membrane <= new_membrane;
        end
    end

endmodule
