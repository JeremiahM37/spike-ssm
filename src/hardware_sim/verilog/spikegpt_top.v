// =============================================================================
// spikegpt_top.v — Top-Level SpikeGPT Accelerator with AXI-Stream Interface
// =============================================================================
// Target: AMD/Xilinx Kria KV260 (XCK26) @ 100 MHz
//
// Top-level architecture:
//   - AXI4-Stream slave for input tokens (from PS or DMA)
//   - AXI4-Lite slave for control/status registers
//   - N_BLOCKS SpikeGPT blocks (time-multiplexed on single block HW)
//   - Weight BRAM banks (loaded from DDR4 via DMA before inference)
//   - AXI4-Stream master for output logits
//
// Resource strategy:
//   - Single block instance, time-multiplexed across N_BLOCKS layers
//   - Weights for current block loaded from DDR4 into on-chip BRAM
//   - Embedding table stays in DDR4, fetched per token
//
// Register map (AXI4-Lite, base + offset):
//   0x00: CTRL    [0]=start, [1]=new_seq, [2]=soft_reset
//   0x04: STATUS  [0]=busy, [1]=done, [7:4]=current_block
//   0x08: N_TOKENS (number of tokens to process)
//   0x0C: CONFIG  [3:0]=n_blocks, [7:4]=reserved
//   0x10: SPIKE_CNT_ATT (spike count from attention LIF)
//   0x14: SPIKE_CNT_FFN (spike count from FFN LIF)
//   0x18: CYCLE_CNT (total cycles for last inference)
//   0x1C: ERROR   [0]=overflow, [1]=timeout
// =============================================================================

module spikegpt_top #(
    parameter N_EMBD         = 64,
    parameter HIDDEN_SZ      = 256,          // 4 * N_EMBD
    parameter N_BLOCKS       = 12,           // Number of SpikeGPT blocks
    parameter VOCAB_SIZE     = 50277,
    parameter DATA_WIDTH     = 16,
    parameter WEIGHT_WIDTH   = 8,
    parameter ACCUM_WIDTH    = 24,
    parameter FRAC_BITS      = 8,
    parameter TM_ADDR_WIDTH  = 16,
    parameter CM_ADDR_WIDTH  = 18,
    parameter AXI_DATA_WIDTH = 32,           // AXI data bus width
    parameter AXI_ADDR_WIDTH = 16,
    parameter AXIL_ADDR_WIDTH = 8            // AXI-Lite address width
)(
    // Clock and reset
    input  wire                           aclk,
    input  wire                           aresetn,

    // =========================================================================
    // AXI4-Stream Slave (Input tokens)
    // =========================================================================
    input  wire [AXI_DATA_WIDTH-1:0]      s_axis_tdata,
    input  wire                           s_axis_tvalid,
    output reg                            s_axis_tready,
    input  wire                           s_axis_tlast,

    // =========================================================================
    // AXI4-Stream Master (Output logits / embeddings)
    // =========================================================================
    output reg  [AXI_DATA_WIDTH-1:0]      m_axis_tdata,
    output reg                            m_axis_tvalid,
    input  wire                           m_axis_tready,
    output reg                            m_axis_tlast,

    // =========================================================================
    // AXI4-Lite Slave (Control registers)
    // =========================================================================
    // Write address
    input  wire [AXIL_ADDR_WIDTH-1:0]     s_axil_awaddr,
    input  wire                           s_axil_awvalid,
    output reg                            s_axil_awready,
    // Write data
    input  wire [31:0]                    s_axil_wdata,
    input  wire [3:0]                     s_axil_wstrb,
    input  wire                           s_axil_wvalid,
    output reg                            s_axil_wready,
    // Write response
    output reg  [1:0]                     s_axil_bresp,
    output reg                            s_axil_bvalid,
    input  wire                           s_axil_bready,
    // Read address
    input  wire [AXIL_ADDR_WIDTH-1:0]     s_axil_araddr,
    input  wire                           s_axil_arvalid,
    output reg                            s_axil_arready,
    // Read data
    output reg  [31:0]                    s_axil_rdata,
    output reg  [1:0]                     s_axil_rresp,
    output reg                            s_axil_rvalid,
    input  wire                           s_axil_rready,

    // =========================================================================
    // Weight BRAM interface (to external BRAM or DDR controller)
    // =========================================================================
    // Time-mix weights
    output wire [TM_ADDR_WIDTH-1:0]       o_tm_bram_addr,
    output wire                           o_tm_bram_en,
    input  wire signed [WEIGHT_WIDTH-1:0] i_tm_bram_data,
    // Channel-mix weights
    output wire [CM_ADDR_WIDTH-1:0]       o_cm_bram_addr,
    output wire                           o_cm_bram_en,
    input  wire signed [WEIGHT_WIDTH-1:0] i_cm_bram_data
);

    // =========================================================================
    // Control Registers
    // =========================================================================
    reg        ctrl_start;
    reg        ctrl_new_seq;
    reg        ctrl_soft_reset;
    reg [15:0] ctrl_n_tokens;
    reg [31:0] reg_spike_cnt_att;
    reg [31:0] reg_spike_cnt_ffn;
    reg [31:0] reg_cycle_cnt;
    reg [1:0]  reg_error;

    // =========================================================================
    // Top-level state machine
    // =========================================================================
    localparam TOP_IDLE      = 4'd0;
    localparam TOP_LOAD_TOK  = 4'd1;   // Receive token from AXI-Stream
    localparam TOP_EMBED     = 4'd2;   // Lookup embedding (simplified)
    localparam TOP_BLOCK     = 4'd3;   // Process through SpikeGPT block
    localparam TOP_NEXT_BLK  = 4'd4;   // Advance to next block
    localparam TOP_OUTPUT    = 4'd5;   // Stream output via AXI-Stream master
    localparam TOP_DONE      = 4'd6;

    reg [3:0]  top_state;
    reg [3:0]  current_block;
    reg [15:0] token_count;
    reg [15:0] current_token_idx;
    reg [31:0] cycle_counter;

    // Token buffer
    reg [15:0] token_id;

    // Embedding buffer (simplified: use input directly as fixed-point)
    reg signed [DATA_WIDTH-1:0] embed_buf [0:N_EMBD-1];

    // Block I/O
    reg                          blk_start;
    reg                          blk_new_seq;
    wire                         blk_done;
    wire                         blk_busy;
    reg                          blk_data_valid;
    reg  signed [DATA_WIDTH-1:0] blk_data_in;
    wire                         blk_out_valid;
    wire signed [DATA_WIDTH-1:0] blk_out_data;
    wire                         blk_spike_att;
    wire                         blk_spike_ffn;

    // Block output buffer
    reg signed [DATA_WIDTH-1:0] block_out_buf [0:N_EMBD-1];
    reg [$clog2(N_EMBD)-1:0]   out_cnt;

    // Time-mix parameters (simplified: constant for now, loadable via AXI-Lite)
    wire signed [DATA_WIDTH-1:0] tm_mix_k = 16'sh0080;  // ~0.5
    wire signed [DATA_WIDTH-1:0] tm_mix_v = 16'sh0080;
    wire signed [DATA_WIDTH-1:0] tm_mix_r = 16'sh0080;
    wire signed [DATA_WIDTH-1:0] tm_decay = -16'sh0010;  // Small negative
    wire signed [DATA_WIDTH-1:0] tm_first = -16'sh0080;
    wire signed [DATA_WIDTH-1:0] cm_mix_k = 16'sh0080;
    wire signed [DATA_WIDTH-1:0] cm_mix_r = 16'sh0080;

    // =========================================================================
    // SpikeGPT Block Instance (time-multiplexed)
    // =========================================================================
    spikegpt_block #(
        .N_EMBD        (N_EMBD),
        .HIDDEN_SZ     (HIDDEN_SZ),
        .DATA_WIDTH    (DATA_WIDTH),
        .WEIGHT_WIDTH  (WEIGHT_WIDTH),
        .ACCUM_WIDTH   (ACCUM_WIDTH),
        .FRAC_BITS     (FRAC_BITS),
        .N_BLOCKS      (N_BLOCKS),
        .TM_ADDR_WIDTH (TM_ADDR_WIDTH),
        .CM_ADDR_WIDTH (CM_ADDR_WIDTH)
    ) u_block (
        .clk              (aclk),
        .rst_n            (aresetn & ~ctrl_soft_reset),
        .i_start          (blk_start),
        .i_new_sequence   (blk_new_seq),
        .o_done           (blk_done),
        .o_busy           (blk_busy),
        .i_data_valid     (blk_data_valid),
        .i_data           (blk_data_in),
        .o_tm_weight_addr (o_tm_bram_addr),
        .o_tm_weight_rd_en(o_tm_bram_en),
        .i_tm_weight_data (i_tm_bram_data),
        .o_cm_weight_addr (o_cm_bram_addr),
        .o_cm_weight_rd_en(o_cm_bram_en),
        .i_cm_weight_data (i_cm_bram_data),
        .i_tm_mix_k       (tm_mix_k),
        .i_tm_mix_v       (tm_mix_v),
        .i_tm_mix_r       (tm_mix_r),
        .i_tm_decay       (tm_decay),
        .i_tm_first       (tm_first),
        .i_cm_mix_k       (cm_mix_k),
        .i_cm_mix_r       (cm_mix_r),
        .o_out_valid      (blk_out_valid),
        .o_out_data       (blk_out_data),
        .o_spike_att      (blk_spike_att),
        .o_spike_ffn      (blk_spike_ffn)
    );

    // Capture block outputs
    reg [$clog2(N_EMBD)-1:0] blk_out_idx;
    always @(posedge aclk or negedge aresetn) begin
        if (!aresetn) begin
            blk_out_idx <= 0;
        end else begin
            if (blk_start) blk_out_idx <= 0;
            if (blk_out_valid) begin
                block_out_buf[blk_out_idx] <= blk_out_data;
                blk_out_idx <= blk_out_idx + 1;
            end
        end
    end

    // =========================================================================
    // AXI-Lite Write Logic
    // =========================================================================
    reg [AXIL_ADDR_WIDTH-1:0] wr_addr_reg;

    always @(posedge aclk or negedge aresetn) begin
        if (!aresetn) begin
            s_axil_awready <= 1'b0;
            s_axil_wready  <= 1'b0;
            s_axil_bvalid  <= 1'b0;
            s_axil_bresp   <= 2'b00;
            ctrl_start     <= 1'b0;
            ctrl_new_seq   <= 1'b0;
            ctrl_soft_reset <= 1'b0;
            ctrl_n_tokens  <= 16'd1;
            wr_addr_reg    <= 0;
        end else begin
            ctrl_start <= 1'b0;  // Auto-clear

            // Address phase
            if (s_axil_awvalid && !s_axil_awready) begin
                s_axil_awready <= 1'b1;
                wr_addr_reg    <= s_axil_awaddr;
            end else begin
                s_axil_awready <= 1'b0;
            end

            // Data phase
            if (s_axil_wvalid && !s_axil_wready) begin
                s_axil_wready <= 1'b1;
                case (wr_addr_reg[7:0])
                    8'h00: begin // CTRL
                        ctrl_start      <= s_axil_wdata[0];
                        ctrl_new_seq    <= s_axil_wdata[1];
                        ctrl_soft_reset <= s_axil_wdata[2];
                    end
                    8'h08: ctrl_n_tokens <= s_axil_wdata[15:0];
                    default: ;
                endcase
            end else begin
                s_axil_wready <= 1'b0;
            end

            // Response
            if (s_axil_wready && !s_axil_bvalid) begin
                s_axil_bvalid <= 1'b1;
                s_axil_bresp  <= 2'b00;
            end else if (s_axil_bvalid && s_axil_bready) begin
                s_axil_bvalid <= 1'b0;
            end
        end
    end

    // =========================================================================
    // AXI-Lite Read Logic
    // =========================================================================
    always @(posedge aclk or negedge aresetn) begin
        if (!aresetn) begin
            s_axil_arready <= 1'b0;
            s_axil_rvalid  <= 1'b0;
            s_axil_rdata   <= 32'h0;
            s_axil_rresp   <= 2'b00;
        end else begin
            if (s_axil_arvalid && !s_axil_arready) begin
                s_axil_arready <= 1'b1;
                s_axil_rvalid  <= 1'b1;
                s_axil_rresp   <= 2'b00;
                case (s_axil_araddr[7:0])
                    8'h00: s_axil_rdata <= {29'b0, ctrl_soft_reset, ctrl_new_seq, ctrl_start};
                    8'h04: s_axil_rdata <= {24'b0, current_block, 2'b0,
                                            (top_state == TOP_DONE), blk_busy};
                    8'h08: s_axil_rdata <= {16'b0, ctrl_n_tokens};
                    8'h0C: s_axil_rdata <= {28'b0, N_BLOCKS[3:0]};
                    8'h10: s_axil_rdata <= reg_spike_cnt_att;
                    8'h14: s_axil_rdata <= reg_spike_cnt_ffn;
                    8'h18: s_axil_rdata <= reg_cycle_cnt;
                    8'h1C: s_axil_rdata <= {30'b0, reg_error};
                    default: s_axil_rdata <= 32'hDEADBEEF;
                endcase
            end else begin
                s_axil_arready <= 1'b0;
                if (s_axil_rvalid && s_axil_rready)
                    s_axil_rvalid <= 1'b0;
            end
        end
    end

    // =========================================================================
    // Top-Level Inference State Machine
    // =========================================================================
    reg [$clog2(N_EMBD)-1:0] feed_cnt;

    integer j;

    always @(posedge aclk or negedge aresetn) begin
        if (!aresetn || ctrl_soft_reset) begin
            top_state         <= TOP_IDLE;
            current_block     <= 0;
            token_count       <= 0;
            current_token_idx <= 0;
            cycle_counter     <= 0;
            reg_spike_cnt_att <= 0;
            reg_spike_cnt_ffn <= 0;
            reg_cycle_cnt     <= 0;
            reg_error         <= 0;
            s_axis_tready     <= 1'b0;
            m_axis_tvalid     <= 1'b0;
            m_axis_tdata      <= 0;
            m_axis_tlast      <= 1'b0;
            blk_start         <= 1'b0;
            blk_new_seq       <= 1'b0;
            blk_data_valid    <= 1'b0;
            blk_data_in       <= 0;
            token_id          <= 0;
            feed_cnt          <= 0;
            out_cnt           <= 0;
            for (j = 0; j < N_EMBD; j = j + 1)
                embed_buf[j] <= 0;
        end else begin
            blk_start      <= 1'b0;
            blk_new_seq    <= 1'b0;
            blk_data_valid <= 1'b0;
            m_axis_tvalid  <= 1'b0;
            m_axis_tlast   <= 1'b0;

            // Cycle counter
            if (top_state != TOP_IDLE && top_state != TOP_DONE)
                cycle_counter <= cycle_counter + 1;

            // Spike counting
            if (blk_spike_att) reg_spike_cnt_att <= reg_spike_cnt_att + 1;
            if (blk_spike_ffn) reg_spike_cnt_ffn <= reg_spike_cnt_ffn + 1;

            case (top_state)
                // =============================================================
                TOP_IDLE: begin
                    if (ctrl_start) begin
                        top_state         <= TOP_LOAD_TOK;
                        s_axis_tready     <= 1'b1;
                        token_count       <= ctrl_n_tokens;
                        current_token_idx <= 0;
                        current_block     <= 0;
                        cycle_counter     <= 0;
                        reg_spike_cnt_att <= 0;
                        reg_spike_cnt_ffn <= 0;
                        reg_error         <= 0;
                        if (ctrl_new_seq)
                            blk_new_seq <= 1'b1;
                    end
                end

                // =============================================================
                // Load token ID from AXI-Stream
                TOP_LOAD_TOK: begin
                    if (s_axis_tvalid && s_axis_tready) begin
                        token_id      <= s_axis_tdata[15:0];
                        s_axis_tready <= 1'b0;
                        top_state     <= TOP_EMBED;
                        feed_cnt      <= 0;
                    end
                end

                // =============================================================
                // Embedding lookup (simplified: generate pseudo-embedding)
                // In real design, this would fetch from DDR4 via AXI
                TOP_EMBED: begin
                    // Simple embedding: distribute token_id bits across channels
                    // Real implementation would use BRAM lookup table
                    if (feed_cnt < N_EMBD) begin
                        embed_buf[feed_cnt] <= $signed({token_id[7:0], 8'h00}) >>>
                                               (feed_cnt & 4'hF);
                        feed_cnt <= feed_cnt + 1;
                    end else begin
                        top_state     <= TOP_BLOCK;
                        current_block <= 0;
                        feed_cnt      <= 0;
                        blk_start     <= 1'b1;
                    end
                end

                // =============================================================
                // Process through SpikeGPT block
                TOP_BLOCK: begin
                    // Feed embedding data to block
                    if (feed_cnt < N_EMBD) begin
                        blk_data_valid <= 1'b1;
                        if (current_block == 0)
                            blk_data_in <= embed_buf[feed_cnt];
                        else
                            blk_data_in <= block_out_buf[feed_cnt];
                        feed_cnt <= feed_cnt + 1;
                    end

                    // Wait for block completion
                    if (blk_done) begin
                        top_state <= TOP_NEXT_BLK;
                    end
                end

                // =============================================================
                // Advance to next block or output
                TOP_NEXT_BLK: begin
                    if (current_block < N_BLOCKS - 1) begin
                        current_block <= current_block + 1;
                        top_state     <= TOP_BLOCK;
                        feed_cnt      <= 0;
                        blk_start     <= 1'b1;
                    end else begin
                        // All blocks done, output result
                        top_state <= TOP_OUTPUT;
                        out_cnt   <= 0;
                    end
                end

                // =============================================================
                // Stream output via AXI-Stream Master
                TOP_OUTPUT: begin
                    if (m_axis_tready || !m_axis_tvalid) begin
                        if (out_cnt < N_EMBD) begin
                            m_axis_tvalid <= 1'b1;
                            m_axis_tdata  <= {{(AXI_DATA_WIDTH-DATA_WIDTH){block_out_buf[out_cnt][DATA_WIDTH-1]}},
                                              block_out_buf[out_cnt]};
                            m_axis_tlast  <= (out_cnt == N_EMBD - 1);
                            out_cnt <= out_cnt + 1;
                        end else begin
                            // Check if more tokens
                            current_token_idx <= current_token_idx + 1;
                            if (current_token_idx + 1 < token_count) begin
                                top_state     <= TOP_LOAD_TOK;
                                s_axis_tready <= 1'b1;
                                current_block <= 0;
                            end else begin
                                top_state     <= TOP_DONE;
                                reg_cycle_cnt <= cycle_counter;
                            end
                        end
                    end
                end

                // =============================================================
                TOP_DONE: begin
                    // Stay here until next start
                    top_state <= TOP_IDLE;
                end

                default: top_state <= TOP_IDLE;
            endcase
        end
    end

endmodule
