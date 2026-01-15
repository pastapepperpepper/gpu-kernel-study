import triton
import triton.language as tl


@triton.jit
def matrix_transpose_kernel(
    input,
    output,
    rows,
    cols,
    stride_ir,
    stride_ic,
    stride_or,
    stride_oc,
    BLOCK_SIZE: tl.constexpr = 32,
):
    # 2D Grid PID
    pid_row = tl.program_id(0)
    pid_col = tl.program_id(1)

    # Input Offsets
    offs_row = pid_row * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_col = pid_col * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # Input Pointer & Masking
    input_ptrs = input + (offs_row[:, None] * stride_ir + offs_col[None, :] * stride_ic)
    input_mask = (offs_row[:, None] < rows) & (offs_col[None, :] < cols)

    # Memory Load
    tile = tl.load(input_ptrs, mask=input_mask, other=0.0)

    # Compute
    tile_t = tl.trans(tile)

    # Output Offsets
    offs_out_row = offs_col
    offs_out_col = offs_row

    # Output Pointer & Masking
    output_ptrs = output + (offs_out_row[:, None] * stride_or + offs_out_col[None, :] * stride_oc)
    output_mask = (offs_out_row[:, None] < cols) & (offs_out_col[None, :] < rows)

    # Store
    tl.store(output_ptrs, tile_t, mask=output_mask)
