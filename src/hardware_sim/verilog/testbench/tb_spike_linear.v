// =============================================================================
// tb_spike_linear.v — Testbench for Spike-Driven Linear Layer
// =============================================================================
// Verifies:
//   1. All-zero spikes produce zero output
//   2. Single active spike accumulates correct weight column
//   3. Multiple active spikes accumulate correctly
//   4. All-ones spikes produce full sum
//   5. Done/busy signaling works correctly
// =============================================================================

`timescale 1ns / 1ps

module tb_spike_linear;

    // Parameters
    parameter IN_FEATURES   = 8;
    parameter OUT_FEATURES  = 4;
    parameter WEIGHT_WIDTH  = 8;
    parameter ACCUM_WIDTH   = 16;
    parameter ADDR_WIDTH    = $clog2(IN_FEATURES * OUT_FEATURES);
    parameter CLK_PERIOD    = 10;

    // Signals
    reg                          clk;
    reg                          rst_n;
    reg                          i_start;
    wire                         o_done;
    wire                         o_busy;
    reg  [IN_FEATURES-1:0]       i_spikes;
    wire [ADDR_WIDTH-1:0]        o_weight_addr;
    wire                         o_weight_rd_en;
    reg  signed [WEIGHT_WIDTH-1:0] i_weight_data;
    wire                         o_valid;
    wire [$clog2(OUT_FEATURES)-1:0] o_out_idx;
    wire signed [ACCUM_WIDTH-1:0] o_out_data;

    // Weight memory (8 inputs x 4 outputs = 32 entries)
    reg signed [WEIGHT_WIDTH-1:0] weight_mem [0:IN_FEATURES*OUT_FEATURES-1];

    // Output capture
    reg signed [ACCUM_WIDTH-1:0] captured_out [0:OUT_FEATURES-1];
    reg [$clog2(OUT_FEATURES):0] capture_cnt;

    // Test tracking
    integer test_num;
    integer errors;
    integer i, j;

    // =========================================================================
    // DUT
    // =========================================================================
    spike_linear #(
        .IN_FEATURES  (IN_FEATURES),
        .OUT_FEATURES (OUT_FEATURES),
        .WEIGHT_WIDTH (WEIGHT_WIDTH),
        .ACCUM_WIDTH  (ACCUM_WIDTH)
    ) dut (
        .clk           (clk),
        .rst_n         (rst_n),
        .i_start       (i_start),
        .o_done        (o_done),
        .o_busy        (o_busy),
        .i_spikes      (i_spikes),
        .o_weight_addr (o_weight_addr),
        .o_weight_rd_en(o_weight_rd_en),
        .i_weight_data (i_weight_data),
        .o_valid       (o_valid),
        .o_out_idx     (o_out_idx),
        .o_out_data    (o_out_data)
    );

    // =========================================================================
    // Clock
    // =========================================================================
    initial clk = 0;
    always #(CLK_PERIOD/2) clk = ~clk;

    // =========================================================================
    // Weight BRAM model (1-cycle read latency)
    // =========================================================================
    always @(posedge clk) begin
        if (o_weight_rd_en)
            i_weight_data <= weight_mem[o_weight_addr];
    end

    // =========================================================================
    // Output capture
    // =========================================================================
    always @(posedge clk) begin
        if (o_valid) begin
            captured_out[o_out_idx] <= o_out_data;
            capture_cnt <= capture_cnt + 1;
        end
    end

    // =========================================================================
    // Task: Initialize weights with known pattern
    // weight[in][out] = in * 10 + out + 1
    // =========================================================================
    task init_weights;
        integer ii, oo;
        begin
            for (ii = 0; ii < IN_FEATURES; ii = ii + 1) begin
                for (oo = 0; oo < OUT_FEATURES; oo = oo + 1) begin
                    weight_mem[ii * OUT_FEATURES + oo] = ii * 10 + oo + 1;
                end
            end
            $display("  Weight matrix initialized:");
            for (ii = 0; ii < IN_FEATURES; ii = ii + 1) begin
                $write("    in[%0d]: ", ii);
                for (oo = 0; oo < OUT_FEATURES; oo = oo + 1) begin
                    $write("%4d ", weight_mem[ii * OUT_FEATURES + oo]);
                end
                $write("\n");
            end
        end
    endtask

    // =========================================================================
    // Task: Run one computation and wait for done
    // =========================================================================
    task run_and_wait;
        input [IN_FEATURES-1:0] spikes;
        begin
            @(posedge clk);
            i_spikes = spikes;
            i_start  = 1'b1;
            capture_cnt = 0;
            @(posedge clk);
            i_start = 1'b0;

            // Wait for done
            wait(o_done);
            @(posedge clk);
            #1;
        end
    endtask

    // =========================================================================
    // Task: Check output against expected values
    // =========================================================================
    task check_output;
        input [IN_FEATURES-1:0] spikes;
        integer oo, ii;
        reg signed [ACCUM_WIDTH-1:0] expected;
        begin
            for (oo = 0; oo < OUT_FEATURES; oo = oo + 1) begin
                expected = 0;
                for (ii = 0; ii < IN_FEATURES; ii = ii + 1) begin
                    if (spikes[ii])
                        expected = expected + weight_mem[ii * OUT_FEATURES + oo];
                end
                if (captured_out[oo] !== expected) begin
                    $display("  FAIL: out[%0d] = %0d, expected %0d",
                             oo, captured_out[oo], expected);
                    errors = errors + 1;
                end else begin
                    $display("  OK: out[%0d] = %0d", oo, captured_out[oo]);
                end
            end
        end
    endtask

    // =========================================================================
    // Main test sequence
    // =========================================================================
    initial begin
        $dumpfile("tb_spike_linear.vcd");
        $dumpvars(0, tb_spike_linear);

        // Initialize
        rst_n     = 0;
        i_start   = 0;
        i_spikes  = 0;
        i_weight_data = 0;
        test_num  = 0;
        errors    = 0;
        capture_cnt = 0;

        for (i = 0; i < OUT_FEATURES; i = i + 1)
            captured_out[i] = 0;

        // Reset
        #(CLK_PERIOD * 5);
        rst_n = 1;
        #(CLK_PERIOD * 2);

        // Initialize weight memory
        init_weights();
        #(CLK_PERIOD * 2);

        // -----------------------------------------------------------------
        // Test 1: All-zero spikes (no accumulation)
        // -----------------------------------------------------------------
        test_num = 1;
        $display("\nTest %0d: All-zero spikes", test_num);
        run_and_wait({IN_FEATURES{1'b0}});
        check_output({IN_FEATURES{1'b0}});

        // -----------------------------------------------------------------
        // Test 2: Single spike at input 0
        // -----------------------------------------------------------------
        test_num = 2;
        $display("\nTest %0d: Single spike at input[0]", test_num);
        run_and_wait({{(IN_FEATURES-1){1'b0}}, 1'b1});
        // Expected: out[j] = weight[0][j] = {1, 2, 3, 4}
        check_output({{(IN_FEATURES-1){1'b0}}, 1'b1});

        // -----------------------------------------------------------------
        // Test 3: Single spike at input 2
        // -----------------------------------------------------------------
        test_num = 3;
        $display("\nTest %0d: Single spike at input[2]", test_num);
        run_and_wait(8'b00000100);
        // Expected: out[j] = weight[2][j] = {21, 22, 23, 24}
        check_output(8'b00000100);

        // -----------------------------------------------------------------
        // Test 4: Two spikes (input 0 and input 1)
        // -----------------------------------------------------------------
        test_num = 4;
        $display("\nTest %0d: Two spikes at input[0] and input[1]", test_num);
        run_and_wait(8'b00000011);
        // Expected: out[j] = weight[0][j] + weight[1][j]
        check_output(8'b00000011);

        // -----------------------------------------------------------------
        // Test 5: All-ones spikes
        // -----------------------------------------------------------------
        test_num = 5;
        $display("\nTest %0d: All-ones spikes", test_num);
        run_and_wait({IN_FEATURES{1'b1}});
        check_output({IN_FEATURES{1'b1}});

        // -----------------------------------------------------------------
        // Test 6: Sparse pattern (every other input)
        // -----------------------------------------------------------------
        test_num = 6;
        $display("\nTest %0d: Alternating spikes (0,2,4,6)", test_num);
        run_and_wait(8'b01010101);
        check_output(8'b01010101);

        // -----------------------------------------------------------------
        // Test 7: Busy/done signaling
        // -----------------------------------------------------------------
        test_num = 7;
        $display("\nTest %0d: Busy/done signal timing", test_num);
        @(posedge clk);
        if (o_busy !== 1'b0) begin
            $display("  FAIL: busy should be 0 in idle");
            errors = errors + 1;
        end

        i_spikes = 8'b00000001;
        i_start  = 1'b1;
        @(posedge clk);
        i_start = 1'b0;
        @(posedge clk); #1;

        if (o_busy !== 1'b1) begin
            $display("  FAIL: busy should be 1 after start");
            errors = errors + 1;
        end
        $display("  Busy asserted correctly");

        wait(o_done);
        @(posedge clk); #1;
        $display("  Done asserted, busy=%b", o_busy);

        // -----------------------------------------------------------------
        // Test 8: Back-to-back operations
        // -----------------------------------------------------------------
        test_num = 8;
        $display("\nTest %0d: Back-to-back operations", test_num);
        run_and_wait(8'b11110000);
        $display("  First operation:");
        check_output(8'b11110000);

        run_and_wait(8'b00001111);
        $display("  Second operation:");
        check_output(8'b00001111);

        // -----------------------------------------------------------------
        // Test 9: Negative weights
        // -----------------------------------------------------------------
        test_num = 9;
        $display("\nTest %0d: Negative weights", test_num);
        // Set some negative weights
        weight_mem[0] = -8'sd10;
        weight_mem[1] = -8'sd20;
        weight_mem[2] = 8'sd30;
        weight_mem[3] = -8'sd40;
        run_and_wait(8'b00000001);
        $display("  With negative weights:");
        for (i = 0; i < OUT_FEATURES; i = i + 1) begin
            $display("    out[%0d] = %0d", i, captured_out[i]);
        end

        // -----------------------------------------------------------------
        // Summary
        // -----------------------------------------------------------------
        #(CLK_PERIOD * 5);
        $display("\n========================================");
        if (errors == 0)
            $display("  ALL TESTS PASSED");
        else
            $display("  %0d ERRORS detected", errors);
        $display("========================================\n");

        $finish;
    end

    // Timeout
    initial begin
        #(CLK_PERIOD * 100000);
        $display("TIMEOUT: Simulation exceeded maximum time");
        $finish;
    end

endmodule
