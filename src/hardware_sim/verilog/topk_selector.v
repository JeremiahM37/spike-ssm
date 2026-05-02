// =============================================================================
// topk_selector.v — Top-K Neuron Selector for Spike Sparsity
// =============================================================================
// Target: AMD/Xilinx Kria KV260 (XCK26) @ 100 MHz
//
// Given N neurons' |membrane potential| values, outputs a bitmask of which
// K neurons are allowed to fire (the K with highest |membrane|).
//
// Implementation: iterative bubble-sort-style selection over K passes.
// Each pass finds the max unfired neuron and marks it. O(K*N) cycles.
//
// For N=128, K=38 (30%): 38*128 = 4,864 cycles = 48.6 us at 100 MHz.
// This runs in parallel with the next timestep's integration.
// =============================================================================

module topk_selector #(
    parameter N_NEURONS = 128,
    parameter DATA_WIDTH = 16,
    parameter K = 38                     // top 30% of 128
)(
    input  wire                          clk,
    input  wire                          rst_n,
    input  wire                          i_start,
    input  wire [DATA_WIDTH-1:0]         i_abs_membrane [0:N_NEURONS-1],
    output reg  [N_NEURONS-1:0]          o_fire_mask,   // 1 = allowed to fire
    output reg                           o_done
);

    localparam IDX_WIDTH = $clog2(N_NEURONS);

    reg [2:0] state;
    localparam S_IDLE = 0, S_FIND_MAX = 1, S_MARK = 2, S_DONE = 3;

    reg [IDX_WIDTH:0]     scan_idx;
    reg [IDX_WIDTH:0]     k_count;
    reg [DATA_WIDTH-1:0]  best_val;
    reg [IDX_WIDTH-1:0]   best_idx;
    reg [N_NEURONS-1:0]   selected;      // already selected mask

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state      <= S_IDLE;
            o_fire_mask <= 0;
            o_done     <= 0;
            k_count    <= 0;
            scan_idx   <= 0;
            best_val   <= 0;
            best_idx   <= 0;
            selected   <= 0;
        end else begin
            o_done <= 0;

            case (state)
                S_IDLE: begin
                    if (i_start) begin
                        state    <= S_FIND_MAX;
                        k_count  <= 0;
                        scan_idx <= 0;
                        best_val <= 0;
                        best_idx <= 0;
                        selected <= 0;
                        o_fire_mask <= 0;
                    end
                end

                S_FIND_MAX: begin
                    if (scan_idx < N_NEURONS) begin
                        // Check if this neuron is better and not already selected
                        if (!selected[scan_idx] &&
                            i_abs_membrane[scan_idx] > best_val) begin
                            best_val <= i_abs_membrane[scan_idx];
                            best_idx <= scan_idx[IDX_WIDTH-1:0];
                        end
                        scan_idx <= scan_idx + 1;
                    end else begin
                        state <= S_MARK;
                    end
                end

                S_MARK: begin
                    // Mark the best neuron as selected
                    selected[best_idx]    <= 1'b1;
                    o_fire_mask[best_idx] <= 1'b1;
                    k_count <= k_count + 1;

                    if (k_count + 1 >= K) begin
                        state <= S_DONE;
                    end else begin
                        // Reset for next pass
                        state    <= S_FIND_MAX;
                        scan_idx <= 0;
                        best_val <= 0;
                        best_idx <= 0;
                    end
                end

                S_DONE: begin
                    o_done <= 1;
                    state  <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
