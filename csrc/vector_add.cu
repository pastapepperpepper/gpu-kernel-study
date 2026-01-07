#include <torch/extension.h>
#include <cuda_runtime.h>

// CUDA kernel
__global__ void vector_add_kernel(
    const float* a, const float* b, float* c, int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// Python binding
torch::Tensor vector_add_cuda(torch::Tensor a, torch::Tensor b) {
    // input validation
    TORCH_CHECK(a.device().is_cuda(), "Tensor a must be on CUDA device");
    TORCH_CHECK(b.device().is_cuda(), "Tensor b must be on CUDA device");
    TORCH_CHECK(a.dtype() == torch::kFloat32, "Tensor a must be float32");
    TORCH_CHECK(b.dtype() == torch::kFloat32, "Tensor b must be float32");
    TORCH_CHECK(a.sizes() == b.sizes(), "Tensors must have the same shape");

    int n = a.numel();
    auto c = torch::empty_like(a);

    // get memory pointers
    float* d_a = a.data_ptr<float>();
    float* d_b = b.data_ptr<float>();
    float* d_c = c.data_ptr<float>();

    // run kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    vector_add_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);

    // error check
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("CUDA kernel launch failed: ") + cudaGetErrorString(err)
        );
    }
    cudaDeviceSynchronize();

    return c;
}

// Python module registration
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("vector_add", &vector_add_cuda, "Vector addition (CUDA)");
}
