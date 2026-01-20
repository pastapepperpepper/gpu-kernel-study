#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void leaky_relu_kernel(const float* input, float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float alpha = 0.01f;
    if (idx < N) {
        float x = input[idx];
        output [idx] = fmaxf(x, x * alpha);
    }
}
