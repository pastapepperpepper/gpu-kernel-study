#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void convolution_1d_kernel(const float* input, const float* kernel, float* output,
                                      int input_size, int kernel_size, int output_size) {
    // 1. Dynamic shared memory declaration
    // Shared memory size will be set in runner
    extern __shared__ float s_mem[];

    // 2. Divide one common storage (s_mem) into two areas
    float* s_input = s_mem;
    float* s_kernel = &s_mem[blockDim.x + kernel_size - 1];

    // 3. Index setup
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int block_start_idx = bid * blockDim.x;

    // -------------------------------------------------------
    // Global Memory -> Shared Memory
    // -------------------------------------------------------

    // 4. Kernel loading
    // Copy the kernel value in global memory to s_kernel
    // Kernel size can be larger than block size, so use for loop
    for (int i = tid; i < kernel_size; i += blockDim.x) {
        s_kernel[i] = kernel[i];
    }

    // 5. Input Loading (Tile + Halo)
    // Total inputs required: block size + any extra bits (Apron/Halo)
    int tile_size = blockDim.x + kernel_size - 1;

    for (int i = tid; i < tile_size; i += blockDim.x) {
        int global_idx = block_start_idx + i;

        if (global_idx < input_size) {
            s_input[i] = input[global_idx];
        } else {
            s_input[i] = 0.0f; // Zero padding
        }
    }

    // 6. Barrier
    __syncthreads();

    // -------------------------------------------------------
    // Compute using Shared Memory
    // -------------------------------------------------------

    // 7. Global output idx declaration
    int global_output_idx = block_start_idx + tid;
    // 8. Masking
    if (global_output_idx < output_size) {
        // 9. Calculate
        float sum = 0.0f;
        // Calculate using only shared memory (s_input, s_kernel) without accessing global memory
        for (int j = 0; j < kernel_size; ++j) {
            sum += s_input[tid + j] * s_kernel[j];
        }
        output[global_output_idx] = sum;
    }
}

// input, kernel, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, const float* kernel, float* output, int input_size,
                      int kernel_size) {
    int output_size = input_size - kernel_size + 1;
    int threadsPerBlock = 256;
    int blocksPerGrid = (output_size + threadsPerBlock - 1) / threadsPerBlock;

    // Calculate shared memory size
    size_t tile_mem_size = (threadsPerBlock + kernel_size - 1) * sizeof(float);
    size_t kernel_mem_size = kernel_size * sizeof(float);
    size_t total_shared_mem = tile_mem_size + kernel_mem_size;

    convolution_1d_kernel<<<blocksPerGrid, threadsPerBlock, total_shared_mem>>>(input, kernel, output, input_size,
                                                              kernel_size, output_size);
    cudaDeviceSynchronize();
}
