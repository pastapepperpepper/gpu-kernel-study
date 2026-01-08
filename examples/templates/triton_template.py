"""Template for Triton Kernel.

This file serves as a boilerplate for adding new Triton kernels.
Copy this file to `src/gpu_kernel_study/triton/your_kernel_name.py`.
"""

import torch
import triton
import triton.language as tl


# -----------------------------------------------------------------------------
# Triton Kernel Implementation
# -----------------------------------------------------------------------------


@triton.jit
def my_kernel_triton(
    x_ptr,  # Pointer to input data
    y_ptr,  # Pointer to output data
    n_elements,  # Total number of elements
    BLOCK_SIZE: tl.constexpr,  # Block size (must be constexpr)
):
    """
    Triton kernel function.

    Args:
        x_ptr: Pointer to input tensor
        y_ptr: Pointer to output tensor
        n_elements: Size of the tensors
        BLOCK_SIZE: Number of elements processed by each program instance
    """
    # 1. Program ID - Corresponds to blockIdx in CUDA
    pid = tl.program_id(axis=0)

    # 2. Block Start Index
    block_start = pid * BLOCK_SIZE

    # 3. Offsets for this block
    # tl.arange generates a range of indices [0, 1, ..., BLOCK_SIZE-1]
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # 4. Create Mask to handle boundary conditions
    mask = offsets < n_elements

    # 5. Load Data
    # Load data from memory, using mask to avoid out-of-bounds access
    x = tl.load(x_ptr + offsets, mask=mask)

    # 6. Compute
    # TODO: Implement your kernel logic here
    # Example: y = x * 2.0
    y = x

    # 7. Store Result
    tl.store(y_ptr + offsets, y, mask=mask)


# -----------------------------------------------------------------------------
# Python Wrapper for Triton Kernel
# -----------------------------------------------------------------------------


def my_kernel(x: torch.Tensor) -> torch.Tensor:
    """
    Python wrapper to launch the Triton kernel.

    Args:
        x: Input tensor (must be on CUDA)

    Returns:
        Output tensor
    """
    # 1. Input Validation
    assert x.is_cuda, "Input must be on CUDA device"

    # 2. Output Allocation
    n_elements = x.numel()
    y = torch.empty_like(x)

    # 3. Kernel Configuration
    BLOCK_SIZE = 1024

    # 4. Grid Configuration
    # The grid function defines how many program instances to launch
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    # 5. Launch Kernel
    my_kernel_triton[grid](x, y, n_elements, BLOCK_SIZE=BLOCK_SIZE)

    return y
