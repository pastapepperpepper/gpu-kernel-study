#include <cuda_runtime.h>
__global__ void dot_product_naive_kernel(const float* A, const float* B, float* result, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Grid-stride loop (for very large N)
    float local_value = 0.0f;
    for (int i = idx; i < N; i+= stride) {
        local_value += A[i] * B[i];
    }

    // Atomic add to global memory (prevent race condition)
    atomicAdd(result, local_value);
}

// A, B, result are device pointers
extern "C" void solve(const float* A, const float* B, float* result, int N) {
    // Init result variable
    cudaMemset(result, 0, sizeof(float));

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    dot_product_naive_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, result, N);

    cudaDeviceSynchronize();
}
