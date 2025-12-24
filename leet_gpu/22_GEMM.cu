#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cuda_runtime.h>

__global__ void matrix_multiplication_kernel(const half* A, const half* B, half* C, int M, int N, int K,float alpha, float beta) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; i++) {
            float a = __half2float(A[row * K + i]);
            float b = __half2float(B[i * N + col]);
            sum += a * b;
        }
        float old = __half2float(C[row * N + col]);
        float result = alpha * sum + beta * old;
        C[row * N + col] = __float2half(result);
    }
}

// A, B, and C are device pointers
extern "C" void solve(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K,alpha,beta);
    cudaDeviceSynchronize();
}
