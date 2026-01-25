import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(input, output, N, BLOCK_SIZE: tl.constexpr):
    # -----------------------------------------------------------
    # Pass 1: Get Max & Sum
    # -----------------------------------------------------------

    # Init max, sum
    max_prev = -float("inf")
    sum_prev = 0.0

    for i in range(0, N, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N

        # Data Load (to SRAM)
        val = tl.load(input + offsets, mask=mask, other=-float("inf"))

        # Max of current chunk
        max_curr = tl.max(val)

        max_new = tl.maximum(max_prev, max_curr)
        sum_prev = sum_prev * tl.exp(max_prev - max_new) + tl.sum(tl.exp(val - max_new))
        max_prev = max_new

    # Update total max, sum (of N)
    total_max = max_prev
    total_sum = sum_prev

    # -----------------------------------------------------------
    # Pass 2: Normalize
    # -----------------------------------------------------------
    for i in range(0, N, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N

        val = tl.load(input + offsets, mask=mask, other=-float("inf"))

        numerator = tl.exp(val - total_max)
        result = numerator / total_sum

        tl.store(output + offsets, result, mask=mask)


# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    grid = (1,)
    BLOCK_SIZE = 1024
    softmax_kernel[grid](input, output, N, BLOCK_SIZE=BLOCK_SIZE)
