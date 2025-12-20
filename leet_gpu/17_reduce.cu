#include <cuda_runtime.h>

__global__ void reduce(const float* input, float* output, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        atomicAdd(output, input[i]);
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {  
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    //cudaMemset(output, 0, sizeof(float));

    reduce<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}