"""Tests for vector addition kernel."""

import pytest
import torch

try:
    from gpu_kernel_study.kernels import vector_add

    KERNELS_AVAILABLE = True
except ImportError:
    KERNELS_AVAILABLE = False

pytestmark = pytest.mark.skipif(not KERNELS_AVAILABLE, reason="Kernels not available")


@pytest.fixture
def sample_tensors():
    """Create sample tensors for testing."""
    size = 1024
    a = torch.ones(size, device="cuda", dtype=torch.float32)
    b = torch.full((size,), 2.0, device="cuda", dtype=torch.float32)
    expected = torch.full((size,), 3.0, device="cuda", dtype=torch.float32)
    return a, b, expected


def test_vector_add_cuda(sample_tensors):
    """Test CUDA backend."""
    a, b, expected = sample_tensors
    try:
        result = vector_add(a, b, backend="cuda")
        assert torch.allclose(result, expected, rtol=1e-5)
    except ValueError:
        pytest.skip("CUDA backend not available")


def test_vector_add_triton(sample_tensors):
    """Test Triton backend."""
    a, b, expected = sample_tensors
    try:
        result = vector_add(a, b, backend="triton")
        assert torch.allclose(result, expected, rtol=1e-5)
    except ValueError:
        pytest.skip("Triton backend not available")


def test_cuda_triton_match(sample_tensors):
    """Test that CUDA and Triton produce the same results."""
    a, b, _ = sample_tensors
    try:
        result_cuda = vector_add(a, b, backend="cuda")
        result_triton = vector_add(a, b, backend="triton")
        assert torch.allclose(result_cuda, result_triton, rtol=1e-5)
    except ValueError:
        pytest.skip("One or both backends not available")


def test_invalid_backend(sample_tensors):
    """Test invalid backend raises error."""
    a, b, _ = sample_tensors
    with pytest.raises(ValueError, match="Unknown backend"):
        vector_add(a, b, backend="invalid")
