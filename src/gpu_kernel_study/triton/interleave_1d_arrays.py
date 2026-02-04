import torch
import triton
import triton.language as tl


@triton.jit
def interleave_1d_arrays_kernel(A_ptr, B_ptr, output_ptr, N, BLOCK_SIZE: tl.constexpr):
    # Check block id
    pid = tl.program_id(0)

    # Offset calculation
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Masking
    mask = offsets < N

    # Load
    val_a = tl.load(A_ptr + offsets, mask=mask, other=0.0)
    val_b = tl.load(B_ptr + offsets, mask=mask, other=0.0)

    # Store
    tl.store(output_ptr + (offsets * 2), val_a, mask=mask)
    tl.store(output_ptr + (offsets * 2) + 1, val_b, mask=mask)


# A, B, output are tensors on the GPU
def solve(A: torch.Tensor, B: torch.Tensor, output: torch.Tensor, N: int):
    BLOCK_SIZE = 256

    def grid(meta):
        return (triton.cdiv(N, meta["BLOCK_SIZE"]),)

    interleave_1d_arrays_kernel[grid](A, B, output, N, BLOCK_SIZE=BLOCK_SIZE)
