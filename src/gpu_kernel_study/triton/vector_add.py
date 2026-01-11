import torch
import triton
import triton.language as tl


@triton.jit  # triton.jit decorator is used to compile the function into a triton kernel
def vector_add_kernel(a_ptr, b_ptr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    c = a + b
    tl.store(c_ptr + offsets, c, mask=mask)


def vector_add_triton(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Vector addition using Triton kernel."""
    assert a.shape == b.shape, "Tensors must have the same shape"
    n_elements = a.numel()
    output = torch.empty_like(a)

    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    vector_add_kernel[grid](a, b, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)

    return output
