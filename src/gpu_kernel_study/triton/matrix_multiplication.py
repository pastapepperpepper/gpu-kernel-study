import triton
import triton.language as tl


@triton.jit
def matrix_multiplication_kernel(
    a,
    b,
    c,
    M,
    N,
    K,
    stride_am,
    stride_an,
    stride_bn,
    stride_bk,
    stride_cm,
    stride_ck,
    BLOCK_SIZE_M: tl.constexpr = 32,
    BLOCK_SIZE_N: tl.constexpr = 32,
    BLOCK_SIZE_K: tl.constexpr = 32,
):
    # 2D Grid Pid of matrix c
    pid_row = tl.program_id(0)
    pid_col = tl.program_id(1)

    # Offset
    offs_m = pid_row * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_k = pid_col * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    offs_n = tl.arange(0, BLOCK_SIZE_N)

    # Input Pointer & Masking
    # Row=offs_m, Col=0~BLOCK_N
    a_ptrs = a + (offs_m[:, None] * stride_am + offs_n[None, :] * stride_an)
    # Row=0~BLOCK_N, Col=offs_k
    b_ptrs = b + (offs_n[:, None] * stride_bn + offs_k[None, :] * stride_bk)

    # Result initialization
    result = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)

    # Main loop
    for n in range(0, tl.cdiv(N, BLOCK_SIZE_N)):
        # Get start position of block ((n+1)th tile * block size)
        current_n_start = n * BLOCK_SIZE_N

        # Masking
        a_mask = (offs_m[:, None] < M) & ((current_n_start + offs_n[None, :]) < N)
        b_mask = ((current_n_start + offs_n[:, None]) < N) & (offs_k[None, :] < K)

        # Load
        a_tile = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b_tile = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Compute
        result = tl.dot(a_tile, b_tile, result)

        # Move to next N block
        a_ptrs += BLOCK_SIZE_N * stride_an
        b_ptrs += BLOCK_SIZE_N * stride_bn

    # Store
    c_ptrs = c + (stride_cm * offs_m[:, None] + stride_ck * offs_k[None, :])
    c_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)

    tl.store(c_ptrs, result, mask=c_mask)
