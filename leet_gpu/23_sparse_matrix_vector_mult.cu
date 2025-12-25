#include <cuda_runtime.h>

#include <cuda_runtime.h>

__global__ void matrix_vector_multiplication_kernel(const float* A, const float* x, float* y, int M, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M) {
        float sum = 0;
        for (int i = 0; i < N; i++) {
            sum += A[row * N + i] * x[i];
        }
        y[row] = sum;
    }
}

// A, x, y are device pointers
extern "C" void solve(const float* A, const float* x, float* y, int M, int N, int nnz) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (M + threadsPerBlock - 1) / threadsPerBlock;

    matrix_vector_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, x, y, M, N);
    cudaDeviceSynchronize();
} 