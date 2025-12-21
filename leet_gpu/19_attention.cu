#include <cuda_runtime.h>
#include <math.h>

__global__ void matrix_transpose_kernel(
    const float* input,
    float* output,
    int rows, int cols
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < rows && col < cols)
        output[col * rows + row] = input[row * cols + col];
}

__global__ void matrix_multiplication_kernel(
    const float* A,
    const float* B,
    float* C,
    int M, int N, int K
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < K) {
        float sum = 0.0f;
        for (int i = 0; i < N; i++)
            sum += A[row * N + i] * B[i * K + col];
        C[row * K + col] = sum;
    }
}

__global__ void scale_kernel(float* data, int size, float scale) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
        data[i] *= scale;
}

__global__ void softmax_rowwise_kernel(
    const float* scores,
    float* probs,
    int N
) {
    int row = blockIdx.x;
    extern __shared__ float buf[];

    float max_val = -INFINITY;
    for (int i = threadIdx.x; i < N; i += blockDim.x)
        max_val = fmaxf(max_val, scores[row * N + i]);

    buf[threadIdx.x] = max_val;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            buf[threadIdx.x] = fmaxf(buf[threadIdx.x], buf[threadIdx.x + s]);
        __syncthreads();
    }
    max_val = buf[0];

    float sum = 0.0f;
    for (int i = threadIdx.x; i < N; i += blockDim.x)
        sum += expf(scores[row * N + i] - max_val);

    buf[threadIdx.x] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            buf[threadIdx.x] += buf[threadIdx.x + s];
        __syncthreads();
    }
    sum = buf[0];

    for (int i = threadIdx.x; i < N; i += blockDim.x)
        probs[row * N + i] = expf(scores[row * N + i] - max_val) / sum;
}

extern "C" void solve(
    const float* Q,
    const float* K,
    const float* V,
    float* output,
    int M, int N, int d
) {
    float* K_T;
    cudaMalloc(&K_T, sizeof(float) * d * N);

    dim3 block2D(16, 16);
    dim3 gridKT((d + 15) / 16, (N + 15) / 16);
    matrix_transpose_kernel<<<gridKT, block2D>>>(K, K_T, N, d);

    float* scores;
    cudaMalloc(&scores, sizeof(float) * M * N);

    dim3 gridScores((N + 15) / 16, (M + 15) / 16);
    matrix_multiplication_kernel<<<gridScores, block2D>>>(Q, K_T, scores, M, d, N);

    scale_kernel<<<(M * N + 255) / 256, 256>>>(
        scores, M * N, 1.0f / sqrtf((float)d)
    );

    float* probs;
    cudaMalloc(&probs, sizeof(float) * M * N);

    int threads = 256;
    softmax_rowwise_kernel<<<M, threads, threads * sizeof(float)>>>(
        scores, probs, N
    );

    dim3 gridOut((d + 15) / 16, (M + 15) / 16);
    matrix_multiplication_kernel<<<gridOut, block2D>>>(
        probs, V, output, M, N, d
    );

    cudaFree(K_T);
    cudaFree(scores);
    cudaFree(probs);
}
