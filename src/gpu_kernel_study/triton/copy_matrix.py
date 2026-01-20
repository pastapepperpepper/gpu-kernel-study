import triton
import triton.language as tl


@triton.jit
def copy_matrix_kernel(a, b, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N * N

    x = tl.load(a + offsets, mask=mask, other=0.0)
    tl.store(b + offsets, x, mask=mask)
