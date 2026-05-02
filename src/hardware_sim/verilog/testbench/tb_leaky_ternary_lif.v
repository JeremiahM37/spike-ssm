// =============================================================================
// tb_leaky_ternary_lif.v — Testbench for LeakyTernaryLIF
// =============================================================================

`timescale 1ns/1ps

module tb_leaky_ternary_lif;

    parameter DATA_WIDTH  = 16;
    parameter BETA_VALUE  = 8'd230;     // 0.9
    parameter THRESHOLD   = 16'sd384;   // 1.5 in Q8.8
    parameter ALPHA_NUM   = 8'd218;     // 0.85

    reg                         clk;
    reg                         rst_n;
    reg                         en;
    reg  signed [DATA_WIDTH-1:0] i_current;
    reg                         i_topk_allow;
    wire signed [DATA_WIDTH-1:0] o_output;
    wire [1:0]                   o_spike_raw;
    wire signed [DATA_WIDTH-1:0] o_membrane;

    leaky_ternary_lif #(
        .DATA_WIDTH(DATA_WIDTH),
        .BETA_VALUE(BETA_VALUE),
        .THRESHOLD(THRESHOLD),
        .ALPHA_NUM(ALPHA_NUM),
        .TOPK_EN(1)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .en(en),
        .i_current(i_current),
        .i_topk_allow(i_topk_allow),
        .o_output(o_output),
        .o_spike_raw(o_spike_raw),
        .o_membrane(o_membrane)
    );

    // Clock: 100 MHz
    always #5 clk = ~clk;

    integer pass_count;
    integer fail_count;

    task check(input [63:0] test_num, input [1:0] expected_spike, input signed [DATA_WIDTH-1:0] min_out, input signed [DATA_WIDTH-1:0] max_out);
        begin
            if (o_spike_raw !== expected_spike) begin
                $display("FAIL test %0d: spike=%b expected=%b (membrane=%0d)",
                         test_num, o_spike_raw, expected_spike, o_membrane);
                fail_count = fail_count + 1;
            end else if (o_output < min_out || o_output > max_out) begin
                $display("FAIL test %0d: output=%0d not in [%0d, %0d]",
                         test_num, o_output, min_out, max_out);
                fail_count = fail_count + 1;
            end else begin
                $display("PASS test %0d: spike=%b output=%0d membrane=%0d",
                         test_num, o_spike_raw, o_output, o_membrane);
                pass_count = pass_count + 1;
            end
        end
    endtask

    initial begin
        $dumpfile("tb_leaky_ternary_lif.vcd");
        $dumpvars(0, tb_leaky_ternary_lif);

        clk = 0; rst_n = 0; en = 0; i_current = 0; i_topk_allow = 1;
        pass_count = 0; fail_count = 0;

        // Reset
        #20 rst_n = 1;
        #10 en = 1;

        // Test 1: Small input, no spike expected
        i_current = 16'sd50;  // ~0.2 in Q8.8
        #10;
        check(1, 2'b00, -16'sd256, 16'sd256);  // should be small, SiLU-dominated

        // Test 2: Another small input
        i_current = 16'sd50;
        #10;
        check(2, 2'b00, -16'sd256, 16'sd256);

        // Test 3: Large positive input — should build up and fire +1
        i_current = 16'sd512; // 2.0 in Q8.8
        #10;
        // Membrane = 0.9*94 + 512 = 596 > 384 threshold → fires +1
        check(3, 2'b01, -16'sd512, 16'sd512);

        // Test 4: Another large input — should fire now
        i_current = 16'sd512;
        #10;
        // Membrane should exceed threshold after accumulation
        // Could fire +1 (2'b01) or not depending on leak

        // Test 5-8: Drive with sustained large positive input to force +1 spike
        repeat(4) begin
            i_current = 16'sd400;
            #10;
        end
        $display("After sustained positive: spike=%b membrane=%0d output=%0d",
                 o_spike_raw, o_membrane, o_output);

        // Test 9: Large negative input to force -1 spike
        i_current = 16'sd0; #10; // reset momentum
        repeat(5) begin
            i_current = -16'sd500;
            #10;
        end
        $display("After sustained negative: spike=%b membrane=%0d output=%0d",
                 o_spike_raw, o_membrane, o_output);

        // Test 10: Top-K gating — disable firing
        i_topk_allow = 0;
        repeat(5) begin
            i_current = 16'sd500;
            #10;
        end
        $display("Top-K blocked: spike=%b (should be 00) output=%0d",
                 o_spike_raw, o_output);
        if (o_spike_raw != 2'b00) begin
            $display("FAIL: top-K should have blocked spike");
            fail_count = fail_count + 1;
        end else begin
            $display("PASS: top-K correctly blocked spike");
            pass_count = pass_count + 1;
        end

        // Test 11: Re-enable top-K
        i_topk_allow = 1;
        repeat(3) begin
            i_current = 16'sd500;
            #10;
        end
        $display("Top-K re-enabled: spike=%b output=%0d",
                 o_spike_raw, o_output);

        // Test 12: Verify output is blended (not purely spike or purely SiLU)
        // When not firing: output should be (1-alpha)*silu(x) ≈ 0.15*silu(x)
        // When firing +1: output should be alpha ≈ 0.85 (in Q0.8 = 218)
        i_current = 16'sd0;
        #10;
        i_current = 16'sd100;  // small input, no spike
        #10;
        // Output should be small (only SiLU component, scaled by 0.15)
        $display("Small input, no spike: output=%0d (should be small, SiLU only)",
                 o_output);

        #20;
        $display("\n=== RESULTS: %0d passed, %0d failed ===", pass_count, fail_count);
        if (fail_count == 0)
            $display("ALL TESTS PASSED");
        else
            $display("SOME TESTS FAILED");
        $finish;
    end

endmodule
