import torch
import triton
import triton.language as tl


@triton.jit
def conv1d_kernel(input, kernel, output, input_size, kernel_size, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Mask: check valid output index
    output_len = input_size - kernel_size + 1
    mask = offsets < output_len

    # Initialize accumulator
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    for i in range(kernel_size):
        # Load: kernel unit data from kernel start address, and loop in kernel_size
        kernel_value = tl.load(kernel + i)
        # Load: load input unit data at a distance (offset) from my output start address
        input_value = tl.load(input + offsets + i, mask=mask, other=0.0)

        # Compute
        acc += input_value * kernel_value

    # Store
    tl.store(output + offsets, acc, mask=mask)


# input, kernel, output are tensors on the GPU
def solve(
    input: torch.Tensor,
    kernel: torch.Tensor,
    output: torch.Tensor,
    input_size: int,
    kernel_size: int,
):
    BLOCK_SIZE = 1024
    n_blocks = triton.cdiv(input_size - kernel_size + 1, BLOCK_SIZE)
    grid = (n_blocks,)

    conv1d_kernel[grid](input, kernel, output, input_size, kernel_size, BLOCK_SIZE)
