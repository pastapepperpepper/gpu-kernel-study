#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void relu_kernel(const float* input, float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        if (input[idx] < 0) {
            output[idx] = 0.0f;
        }
        else {
            output[idx] = input[idx];
        }
    }
}
