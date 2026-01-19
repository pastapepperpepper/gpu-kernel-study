import triton
import triton.language as tl


@triton.jit
def relu_kernel(input, output, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(input + offsets, mask=mask, other=0.0)
    result = tl.maximum(0.0, x)
    tl.store(output + offsets, result, mask=mask)
