#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    if (col < K && row < M) {
        float elementResult = 0.0f;
        for (int num = 0; num < N; num++) {
            elementResult += A[row * N + num] * B[num * N + col];
        }
        C[row * K + col] = elementResult;
    }
}
