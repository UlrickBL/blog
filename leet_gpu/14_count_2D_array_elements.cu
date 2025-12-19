#include <cuda_runtime.h>

__global__ void count_2d_equal_kernel(const int* input, int* output, int N, int M, int K) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx<N*M){
        int value = input[idx];
        if (value==K){
            atomicAdd(output,1);
        }
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const int* input, int* output, int N, int M, int K) {
    int threads = 256;
    int blocks = (N * M + threads - 1) / threads;

    count_2d_equal_kernel<<<blocks, threads>>>(input, output, N, M, K);
    cudaDeviceSynchronize();
}