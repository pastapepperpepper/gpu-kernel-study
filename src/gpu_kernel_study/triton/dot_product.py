import torch
import triton
import triton.language as tl


@triton.jit
def dot_product_kernel(a_ptr, b_ptr, result_ptr, n, BLOCK_SIZE: tl.constexpr):
    # Check block id
    pid = tl.program_id(axis=0)

    # Offset calculation
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Masking
    mask = offsets < n

    # Load
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0)

    # Compute
    partial_sum = tl.sum(a * b)

    # Store: Global accumulation
    tl.atomic_add(result_ptr, partial_sum)


# a, b, result are tensors on the GPU
def solve(a: torch.Tensor, b: torch.Tensor, result: torch.Tensor, n: int):
    # Init result tensor (because atomic_add is used, the initial value must be 0)
    result.zero_()

    # Compute Grid
    BLOCK_SIZE = 1024
    n_blocks = triton.cdiv(n, BLOCK_SIZE)
    grid = (n_blocks,)

    # Execute kernel
    dot_product_kernel[grid](a, b, result, n, BLOCK_SIZE=BLOCK_SIZE)
