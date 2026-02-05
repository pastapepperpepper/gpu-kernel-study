import torch
import triton
import triton.language as tl


@triton.jit
def bmm_kernel(
    # Pointers
    a_ptr,
    b_ptr,
    c_ptr,
    # Matrix dimensions
    M,
    N,
    K,
    # Strides
    stride_a_batch,
    stride_am,
    stride_ak,
    stride_b_batch,
    stride_bk,
    stride_bn,
    stride_c_batch,
    stride_cm,
    stride_cn,
    # Meta-parameters
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # 3D grid pid
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_b = tl.program_id(2)

    # Batch offsets
    a_ptr += pid_b * stride_a_batch
    b_ptr += pid_b * stride_b_batch
    c_ptr += pid_b * stride_c_batch

    # Initial offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Input pointer
    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    # Result initialization
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Main loop
    for k in range(0, K, BLOCK_K):
        # K masking
        k_mask = (k + offs_k) < K

        # M, N masking
        a_mask = (offs_m[:, None] < M) & (k_mask[None, :])
        b_mask = (k_mask[:, None]) & (offs_n[None, :] < N)

        # Load
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Compute
        accumulator += tl.dot(a, b)

        # Move to next block (Advance pointers)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # 7. Store
    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    tl.store(c_ptrs, accumulator, mask=c_mask)


# a, b, c are tensors on the GPU
def solve(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, BATCH: int, M: int, N: int, K: int):
    # 3D reshape input tensor
    a = a.view(BATCH, M, K)
    b = b.view(BATCH, K, N)
    c = c.view(BATCH, M, N)

    # Block sizes
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32

    # 3D grid
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N), BATCH)

    bmm_kernel[grid](
        # Pointers
        a,
        b,
        c,
        # Dimensions
        M,
        N,
        K,
        # Strides (Batch stride, Row stride, Col stride)
        # stride(0) -> Batch stride, stride(1) -> Row stride, stride(2) -> Col stride
        a.stride(0),
        a.stride(1),
        a.stride(2),
        b.stride(0),
        b.stride(1),
        b.stride(2),
        c.stride(0),
        c.stride(1),
        c.stride(2),
        # Meta-parameters
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
