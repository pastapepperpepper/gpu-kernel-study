"""Vector addition kernel with CUDA and Triton backends."""

import torch
from typing import Literal

try:
    from .. import cuda_ops  # built CUDA extension

    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False

try:
    from ..triton import vector_add_triton

    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


def vector_add(
    a: torch.Tensor,
    b: torch.Tensor,
    backend: Literal["cuda", "triton"] = "cuda",
) -> torch.Tensor:
    """
    Vector addition with multiple backends.

    Args:
        a: First tensor (must be on CUDA device)
        b: Second tensor (must be on CUDA device)
        backend: Backend to use ('cuda' or 'triton')

    Returns:
        Result tensor: a + b

    Raises:
        ValueError: If backend is not available or invalid
        RuntimeError: If CUDA kernel launch fails
    """
    if backend == "cuda":
        if not CUDA_AVAILABLE:
            raise ValueError(
                "CUDA backend not available. Build the extension with: pip install -e ."
            )
        return cuda_ops.vector_add(a, b)
    elif backend == "triton":
        if not TRITON_AVAILABLE:
            raise ValueError(
                "Triton backend not available. Install triton: pip install triton"
            )
        return vector_add_triton(a, b)
    else:
        raise ValueError(f"Unknown backend: {backend}. Use 'cuda' or 'triton'.")


def benchmark(
    a: torch.Tensor,
    b: torch.Tensor,
    num_runs: int = 100,
    warmup: int = 10,
) -> dict:
    """
    Benchmark CUDA vs Triton performance.

    Args:
        a: First tensor
        b: Second tensor
        num_runs: Number of benchmark runs
        warmup: Number of warmup runs

    Returns:
        Dictionary with benchmark results
    """
    import time

    if not CUDA_AVAILABLE or not TRITON_AVAILABLE:
        raise RuntimeError("Both CUDA and Triton backends must be available")

    # Warmup
    for _ in range(warmup):
        _ = vector_add(a, b, backend="cuda")
        _ = vector_add(a, b, backend="triton")
    torch.cuda.synchronize()

    # CUDA benchmark
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_runs):
        result_cuda = vector_add(a, b, backend="cuda")
    torch.cuda.synchronize()
    cuda_time = (time.time() - start) / num_runs

    # Triton benchmark
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_runs):
        result_triton = vector_add(a, b, backend="triton")
    torch.cuda.synchronize()
    triton_time = (time.time() - start) / num_runs

    # Verify correctness
    if not torch.allclose(result_cuda, result_triton, rtol=1e-5):
        raise RuntimeError("CUDA and Triton results don't match!")

    speedup = cuda_time / triton_time if triton_time > 0 else float("inf")

    results = {
        "cuda_time_ms": cuda_time * 1000,
        "triton_time_ms": triton_time * 1000,
        "speedup": speedup,
        "faster": "cuda" if speedup > 1.0 else "triton",
    }

    print("\n=== Vector Addition Benchmark ===")
    print(f"Tensor size: {a.numel()} elements")
    print(f"CUDA time: {results['cuda_time_ms']:.3f} ms")
    print(f"Triton time: {results['triton_time_ms']:.3f} ms")
    print(f"Speedup: {speedup:.2f}x ({results['faster']} is faster)")

    return results
