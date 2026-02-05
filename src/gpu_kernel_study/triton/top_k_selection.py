import torch
import triton
import triton.language as tl


@triton.jit
def top_k_selection_in_block_kernel(input_ptr, output_ptr, n_elements, k, BLOCK_SIZE: tl.constexpr):
    # Grid pid & offsets
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Masking
    mask = offsets < n_elements

    # Load
    val = tl.load(input_ptr + offsets, mask=mask, other=-float("inf"))

    # Compute: local sort
    sorted_val = tl.sort(val, descending=True)

    # Store
    # Store offsets of a block
    output_block_start = pid * k
    output_offsets = output_block_start + tl.arange(0, BLOCK_SIZE)
    # K elements masking
    store_mask = tl.arange(0, BLOCK_SIZE) < k
    # Store top k in a block
    tl.store(output_ptr + output_offsets, sorted_val, mask=store_mask)


# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int, k: int):
    BLOCK_SIZE = 1024
    grid_size = triton.cdiv(N, BLOCK_SIZE)

    # Alloc temp buffer
    # Total size: block_num * k
    temp_output = torch.empty((grid_size * k,), device=input.device, dtype=input.dtype)

    grid = (grid_size,)
    top_k_selection_in_block_kernel[grid](input, temp_output, N, k, BLOCK_SIZE=BLOCK_SIZE)

    # Select results (reduce)
    final_vals, _ = torch.topk(temp_output, k)

    # Store final output
    output[:] = final_vals
