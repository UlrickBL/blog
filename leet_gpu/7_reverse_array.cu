#include <cuda_runtime.h>

__global__ void reverse_array(float* input, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int middle = N/2; // if we are in the real middle it should stay like this
    int value_reverse = N - idx -1;
    if (idx<middle){
        int value = input[idx];
        int opposite = input[value_reverse];
        input[idx] = opposite;
        input[value_reverse] = value;
    }
}

// input is device pointer
extern "C" void solve(float* input, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    reverse_array<<<blocksPerGrid, threadsPerBlock>>>(input, N);
    cudaDeviceSynchronize();
}