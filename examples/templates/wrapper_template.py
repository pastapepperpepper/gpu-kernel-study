"""Template for Unified Python Wrapper.

This file serves as a boilerplate for the unified interface that handles
both CUDA and Triton backends.
Copy this file to `src/gpu_kernel_study/kernels/your_kernel_name.py`.
"""

import torch
from typing import Literal

# -----------------------------------------------------------------------------
# Backend Import Handling
# -----------------------------------------------------------------------------

# Try to import CUDA extension
try:
    from .. import cuda_ops

    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False

# Try to import Triton implementation
try:
    from ..triton import my_kernel as my_kernel_triton

    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


# -----------------------------------------------------------------------------
# Main Wrapper Function
# -----------------------------------------------------------------------------


def my_kernel(
    input_tensor: torch.Tensor,
    backend: Literal["cuda", "triton"] = "cuda",
) -> torch.Tensor:
    """
    Unified interface for My Kernel with multiple backends.

    Args:
        input_tensor: Input tensor (must be on CUDA)
        backend: 'cuda' or 'triton'

    Returns:
        Processed tensor

    Raises:
        ValueError: If backend is unavailable or invalid
    """
    if backend == "cuda":
        if not CUDA_AVAILABLE:
            raise ValueError("CUDA backend not available. Build with: pip install -e .")
        # Call the C++ extension function defined in csrc
        return cuda_ops.my_kernel(input_tensor)

    elif backend == "triton":
        if not TRITON_AVAILABLE:
            raise ValueError(
                "Triton backend not available. Install with: pip install triton"
            )
        # Call the Triton wrapper function
        return my_kernel_triton.my_kernel(input_tensor)

    else:
        raise ValueError(f"Unknown backend: {backend}. Use 'cuda' or 'triton'.")


# -----------------------------------------------------------------------------
# Benchmark Function
# -----------------------------------------------------------------------------


def benchmark(
    input_tensor: torch.Tensor,
    num_runs: int = 100,
    warmup: int = 10,
) -> dict:
    """
    Benchmark CUDA vs Triton performance.

    Args:
        input_tensor: Test input
        num_runs: Number of measurement runs
        warmup: Number of warmup runs

    Returns:
        Dictionary containing timing results and speedup
    """
    import time

    if not CUDA_AVAILABLE or not TRITON_AVAILABLE:
        raise RuntimeError("Both backends needed for benchmarking")

    # Warmup phase
    for _ in range(warmup):
        _ = my_kernel(input_tensor, backend="cuda")
        _ = my_kernel(input_tensor, backend="triton")
    torch.cuda.synchronize()

    # Measure CUDA
    start = time.time()
    for _ in range(num_runs):
        _ = my_kernel(input_tensor, backend="cuda")
    torch.cuda.synchronize()
    cuda_time = (time.time() - start) / num_runs

    # Measure Triton
    start = time.time()
    for _ in range(num_runs):
        _ = my_kernel(input_tensor, backend="triton")
    torch.cuda.synchronize()
    triton_time = (time.time() - start) / num_runs

    # Calculate speedup
    speedup = cuda_time / triton_time if triton_time > 0 else float("inf")

    print(f"CUDA: {cuda_time*1000:.3f} ms")
    print(f"Triton: {triton_time*1000:.3f} ms")
    print(f"Speedup (CUDA/Triton): {speedup:.2f}x")

    return {
        "cuda_ms": cuda_time * 1000,
        "triton_ms": triton_time * 1000,
        "speedup": speedup,
    }
