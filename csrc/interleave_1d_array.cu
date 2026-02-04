#include <cuda_runtime.h>
#include <vector_types.h>

__global__ void interleave_array_1d_kernel(const float* A, const float* B, float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float2* output_as_vector = reinterpret_cast<float2*>(output);

    if (idx < N) {
        // Read separate
        float val_a = A[idx];
        float val_b = B[idx];
        // Combine to vector & Store
        output_as_vector[idx] = make_float2(val_a, val_b);
    }
}

// A, B, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    interleave_array_1d_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, output, N);
    cudaDeviceSynchronize();
}
