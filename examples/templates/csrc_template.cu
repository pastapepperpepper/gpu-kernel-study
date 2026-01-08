"""Template for CUDA Kernel and Python Binding.

This file serves as a boilerplate for adding new CUDA kernels.
Copy this file to `csrc/your_kernel_name.cu` and modify as needed.
"""

#include <torch/extension.h>
#include <cuda_runtime.h>

// ----------------------------------------------------------------------------
// CUDA Kernel Implementation
// ----------------------------------------------------------------------------

/**
 * CUDA Kernel Function.
 *
 * Args:
 *     input: Input data pointer
 *     output: Output data pointer
 *     n: Size of the data
 *
 * Note: Use __global__ to define a kernel function callable from host.
 */
__global__ void my_kernel(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // TODO: Implement your kernel logic here
        // Example: output[idx] = input[idx] * 2.0f;
        output[idx] = input[idx];
    }
}

// ----------------------------------------------------------------------------
// Python Binding Function
// ----------------------------------------------------------------------------

/**
 * C++ function callable from Python.
 * This function handles memory management, error checking, and kernel launch.
 *
 * Args:
 *     input_tensor: PyTorch tensor input
 *
 * Returns:
 *     PyTorch tensor output
 */
torch::Tensor my_kernel_cuda(torch::Tensor input_tensor) {
    // 1. Input Validation
    TORCH_CHECK(input_tensor.device().is_cuda(), "Input tensor must be on CUDA device");
    TORCH_CHECK(input_tensor.is_contiguous(), "Input tensor must be contiguous");

    // 2. Output Tensor Allocation
    auto output_tensor = torch::empty_like(input_tensor);
    int n = input_tensor.numel();

    // 3. Kernel Launch Configuration
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    // 4. Launch Kernel
    // AT_DISPATCH_FLOATING_TYPES handles float/double dispatching if needed
    my_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        input_tensor.data_ptr<float>(),
        output_tensor.data_ptr<float>(),
        n
    );

    // 5. Error Checking
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("CUDA kernel launch failed: ") + cudaGetErrorString(err)
        );
    }

    return output_tensor;
}

// ----------------------------------------------------------------------------
// Python Module Registration
// ----------------------------------------------------------------------------

// This macro registers the C++ function as a Python extension module.
// The name "my_kernel" will be used in Python as `cuda_ops.my_kernel`.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("my_kernel", &my_kernel_cuda, "My custom CUDA kernel");
}
