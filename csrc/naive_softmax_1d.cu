#include <cuda_runtime.h>
#include <float.h>
#include <math.h>
#include <stdio.h>

// Float atomic max helper (for solve race condition)
__device__ void atomicMaxFloat(float* address, float val) {
    int* address_as_int = (int*)address;    // Faking a float address as an int address
    int old = *address_as_int, assumed;
    do {
        assumed = old;
        if (__int_as_float(assumed) >= val) break;
        // Atomic compare and swap (CAS)
        // If assumed value is still in memory, swap value
        old = atomicCAS(address_as_int, assumed, __float_as_int(val));
    } while (assumed != old);
}

// ----------------------------------------------------------------
// [Kernel 1] Find Max Kernel
// ----------------------------------------------------------------
__global__ void find_max_kernel(const float* input, float* d_max, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Init Local Max
    float local_max = -FLT_MAX;

    // Grid-Stride Loop
    // Even if N is greater than the number of threads,
    // the threads jump by the stride and process until the end
    for (int i = idx; i < N; i += stride) {
        // CUDA function fmaxf (float max)
        local_max = fmaxf(local_max, input[i]);
    }

    // After loop, update local_max to global memory by using atomicMaxFloat
    atomicMaxFloat(d_max, local_max);
}

// ----------------------------------------------------------------
// [Kernel 2] Compute Sum Kernel
// ----------------------------------------------------------------
__global__ void compute_sum_kernel(const float* input, const float* d_max, float* d_sum, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Read global max value
    float max_val = *d_max;
    float local_sum = 0.0f;

    // Loop: sum of exp(x - max)
    for (int i = idx; i < N; i += stride) {
        local_sum += expf(input[i] - max_val);
    }

    // Atomic add to global memory
    atomicAdd(d_sum, local_sum);
}

// ----------------------------------------------------------------
// [Kernel 3] Normalize Kernel
// ----------------------------------------------------------------
__global__ void softmax_kernel(const float* input, float* output, const float* d_max, const float* d_sum, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Read global value
    float max_val = *d_max;
    float sum_val = *d_sum;

    for (int i = idx; i < N; i += stride) {
        // Safe softmax: e^(x - max) / sum
        output[i] = expf(input[i] - max_val) / sum_val;
    }
}

// ----------------------------------------------------------------
// Solve Function
// ----------------------------------------------------------------
extern "C" void solve(const float* input, float* output, int N) {
    // 1. Device memory alloc for store intermediate result
    float *d_max, *d_sum;
    cudaMalloc(&d_max, sizeof(float));
    cudaMalloc(&d_sum, sizeof(float));

    // 2. Init value (host -> device)
    float h_min = -FLT_MAX;
    float h_zero = 0.0f;
    cudaMemcpy(d_max, &h_min, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sum, &h_zero, sizeof(float), cudaMemcpyHostToDevice);

    // 3. Grid/Block
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    if (blocksPerGrid > 256) blocksPerGrid = 256;

    // 4. 3-pass kernel execution
    // Find max value -> Sum -> Divide
    find_max_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, d_max, N);
    compute_sum_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, d_max, d_sum, N);
    softmax_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, d_max, d_sum, N);

    // 5. sync & free memory
    cudaDeviceSynchronize();
    cudaFree(d_max);
    cudaFree(d_sum);
}
