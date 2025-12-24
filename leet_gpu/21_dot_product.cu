#include <cuda_runtime.h>

__global__ void dot_partial(
    const float* A,
    const float* B,
    float* partial,
    int N
) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;

    float sum = 0.0f;

    for (int i = idx; i < N; i += stride) {
        sum += A[i] * B[i];
    }

    sdata[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial[blockIdx.x] = sdata[0];
    }
}

__global__ void reduce_final(float* data, int N) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;

    if (tid < N)
        sdata[tid] = data[tid];
    else
        sdata[tid] = 0.0f;

    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0)
        data[0] = sdata[0];
}


// A, B, result are device pointers
extern "C" void solve(const float* A, const float* B, float* result, int N) {
    int threads = 256;
    int blocks = min(1024, (N + threads - 1) / threads);

    float* d_partial;
    cudaMalloc(&d_partial, blocks * sizeof(float));

    dot_partial<<<blocks, threads, threads * sizeof(float)>>>(
        A, B, d_partial, N
    );

    reduce_final<<<1, 1024, 1024 * sizeof(float)>>>(
        d_partial, blocks
    );

    cudaMemcpy(result, d_partial, sizeof(float), cudaMemcpyDeviceToDevice);

    cudaFree(d_partial);
}
