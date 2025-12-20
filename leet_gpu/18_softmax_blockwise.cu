#include <cuda_runtime.h>
#include <float.h>

__global__ void block_max_kernel(const float* input, float* block_max, int N) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    sdata[tid] = (i < N) ? input[i] : -FLT_MAX;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }

    if (tid == 0)
        block_max[blockIdx.x] = sdata[0];
}

__global__ void block_sum_kernel(const float* input, const float* max_val,
                                 float* block_sum, int N) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    sdata[tid] = (i < N) ? __expf(input[i] - *max_val) : 0.0f;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0)
        block_sum[blockIdx.x] = sdata[0];
}

__global__ void softmax_kernel(
    const float* input,
    const float* max_val,
    const float* sum,
    float* output,
    int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        output[i] = __expf(input[i] - *max_val) / *sum;
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    float *block_max, *block_sum;
    float *d_max, *d_sum;

    cudaMalloc(&block_max, blocks * sizeof(float));
    cudaMalloc(&block_sum, blocks * sizeof(float));
    cudaMalloc(&d_max, sizeof(float));
    cudaMalloc(&d_sum, sizeof(float));

    // compute block max
    block_max_kernel<<<blocks, threads>>>(input, block_max, N);
    cudaDeviceSynchronize();

    // reduce block max on CPU-style single block
    block_max_kernel<<<1, threads>>>(block_max, d_max, blocks);
    cudaDeviceSynchronize();

    // compute block sum
    block_sum_kernel<<<blocks, threads>>>(input, d_max, block_sum, N);
    cudaDeviceSynchronize();

    // reduce block sum
    block_sum_kernel<<<1, threads>>>(block_sum, d_max, d_sum, blocks);
    cudaDeviceSynchronize();

    // normalize
    softmax_kernel<<<blocks, threads>>>(input, d_max, d_sum, output, N);
    cudaDeviceSynchronize();

    cudaFree(block_max);
    cudaFree(block_sum);
    cudaFree(d_max);
    cudaFree(d_sum);
}
