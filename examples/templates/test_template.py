"""Template for Kernel Tests.

This file serves as a boilerplate for testing your kernels.
Copy this file to `tests/test_your_kernel_name.py`.
"""

import pytest
import torch

# Try to import the kernel wrapper
try:
    from gpu_kernel_study.kernels import my_kernel

    KERNELS_AVAILABLE = True
except ImportError:
    KERNELS_AVAILABLE = False

# Skip all tests if kernels are not installed
pytestmark = pytest.mark.skipif(not KERNELS_AVAILABLE, reason="Kernels not available")


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    size = 1024
    x = torch.randn(size, device="cuda", dtype=torch.float32)
    # Expected result (using PyTorch built-in for reference)
    expected = x.clone()  # Replace with actual expected logic
    return x, expected


def test_my_kernel_cuda(sample_data):
    """Test CUDA backend correctness."""
    x, expected = sample_data
    try:
        result = my_kernel.my_kernel(x, backend="cuda")
        assert torch.allclose(result, expected, rtol=1e-5)
    except ValueError:
        pytest.skip("CUDA backend not available")


def test_my_kernel_triton(sample_data):
    """Test Triton backend correctness."""
    x, expected = sample_data
    try:
        result = my_kernel.my_kernel(x, backend="triton")
        assert torch.allclose(result, expected, rtol=1e-5)
    except ValueError:
        pytest.skip("Triton backend not available")


def test_backends_match(sample_data):
    """Ensure CUDA and Triton produce identical results."""
    x, _ = sample_data
    try:
        result_cuda = my_kernel.my_kernel(x, backend="cuda")
        result_triton = my_kernel.my_kernel(x, backend="triton")
        assert torch.allclose(result_cuda, result_triton, rtol=1e-5)
    except ValueError:
        pytest.skip("One or both backends not available")
