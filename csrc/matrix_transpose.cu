#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void matrix_transpose_kernel(const float* input, float* output, int rows, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < cols && row < rows){
        int inputIdx = row * cols + col;
        int outputIdx = col * rows + row;

        output[outputIdx] = input[inputIdx];
    }
}
