#include <cuda_runtime.h>

__device__ float atomicMaxFloat(float* addr, float value) {
    int* addr_as_i = (int*)addr;
    int old = *addr_as_i, assumed;

    do {
        assumed = old;
        if (__int_as_float(assumed) >= value) break;
        old = atomicCAS(addr_as_i, assumed, __float_as_int(value));
    } while (assumed != old);

    return __int_as_float(old);
}

__global__ void max_kernel(const float* input, float* max, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i<N){
        atomicMaxFloat(max,input[i]);
    }
}

__global__ void sum_kernel(const float* input, const float* max, float* sum, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i<N){
        atomicAdd(sum,__expf(input[i]-*max));
    }
}
__global__ void softmax_kernel(const float* input, const float* max, const float* sum,float* output, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i<N){
        output[i] = __expf(input[i]-*max)/ *sum;
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    float* max;
    float* sum;

    cudaMalloc(&max, sizeof(float));
    cudaMalloc(&sum, sizeof(float));

    float neg_inf = -INFINITY;
    cudaMemcpy(max, &neg_inf, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(sum, 0, sizeof(float));

    max_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, max, N);
    cudaDeviceSynchronize();

    sum_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, max, sum, N);
    cudaDeviceSynchronize();

    softmax_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, max, sum,output, N);
    cudaDeviceSynchronize();
}