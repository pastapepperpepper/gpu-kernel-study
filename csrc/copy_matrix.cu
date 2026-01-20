#include <torch/extension.h>
#include <cuda_runtime.h>

// Use float4 type to move 16 byte data at once
__global__ void copy_matrix_kernel(const float* A, float* B, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int vectorLimit = (N * N) / 4;

    // Vectorized (use float4)
    if (idx < vectorLimit) {
        reinterpret_cast<float4*>(B)[idx] = reinterpret_cast<const float4*>(A)[idx];
    }

    // Scalar (remainder)
    if (idx == 0) {
        for (int i = 0; i < (N * N); i++) {
            B[i] = A[i];
        }
    }
}
