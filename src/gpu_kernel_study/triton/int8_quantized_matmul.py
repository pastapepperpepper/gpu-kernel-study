import torch
import triton
import triton.language as tl


@triton.jit
def int8_quantized_matmul_kernel(
    # Pointers
    a_ptr,
    b_ptr,
    c_ptr,
    # Matrix dimensions
    M,
    N,
    K,
    # Strides
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    # Quantization parameters
    scale_A,
    scale_B,
    scale_C,
    zero_point_A,
    zero_point_B,
    zero_point_C,
    # Meta-parameters
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # 2D grid pid
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets
    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Input pointer
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # Result initialization
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Main loop
    for k in range(0, K, BLOCK_K):
        # k masking (common for A, B)
        k_mask = (k + offs_k) < K

        # a, b masking
        a_mask = (offs_am[:, None] < M) & (k_mask[None, :])
        b_mask = (k_mask[:, None]) & (offs_bn[None, :] < N)

        # Padding with Zero Point
        # Fill in the out-of-range with zp so that it becomes 0 when calculating (val - zp) later
        a = tl.load(a_ptrs, mask=a_mask, other=zero_point_A)
        b = tl.load(b_ptrs, mask=b_mask, other=zero_point_B)

        # Compute
        # 1. Subtract zp
        a_curr = (a.to(tl.int32) - zero_point_A).to(tl.float32)
        b_curr = (b.to(tl.int32) - zero_point_B).to(tl.float32)
        # 2. MAC
        accumulator += tl.dot(a_curr, b_curr)

        # Move to next block
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Output Processing
    # 3. Calculate effective scale value
    effective_scale = (scale_A * scale_B) / scale_C
    # 4. int32 -> float32 -> Scaling
    c = accumulator.to(tl.float32) * effective_scale
    # 5. Round & Shift (libdevice round is not supported in LeetGPU, so use floor(c+0.5))
    c = tl.floor(c + 0.5)
    c = c + zero_point_C
    # 6. Clamp & Cast
    c = tl.clamp(c, -128.0, 127.0)
    c = c.to(tl.int8)

    # Store
    c_ptrs = c_ptr + (offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn)
    c_mask = (offs_am[:, None] < M) & (offs_bn[None, :] < N)

    tl.store(c_ptrs, c, mask=c_mask)


# a, b, c are tensors on the GPU
def solve(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    M: int,
    N: int,
    K: int,
    scale_A: float,
    scale_B: float,
    scale_C: float,
    zero_point_A: int,
    zero_point_B: int,
    zero_point_C: int,
):
    a = a.view(M, K)
    b = b.view(K, N)
    c = c.view(M, N)

    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    int8_quantized_matmul_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        scale_A,
        scale_B,
        scale_C,
        zero_point_A,
        zero_point_B,
        zero_point_C,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
