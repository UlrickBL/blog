#include <cuda_runtime.h>

__global__ void convolution_2d_kernel(const float* input, const float* kernel, float* output,
           int input_rows, int input_cols, int kernel_rows, int kernel_cols,int output_cols,int output_rows) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if ((col<(input_cols-kernel_cols+1))&(row<(input_rows-kernel_rows+1))){
        float kernel_applied = 0;
        for (int m=0;m<kernel_rows;m++){
            for (int n=0;n<kernel_cols;n++){
                kernel_applied+=input[(row + m) * input_cols + (col + n)]*kernel[m*kernel_cols+n];
            }
        }
        output[row * output_cols + col]=kernel_applied;
    }

}

// input, kernel, output are device pointers
extern "C" void solve(const float* input, const float* kernel, float* output,
           int input_rows, int input_cols, int kernel_rows, int kernel_cols) {
    
    int output_rows = input_rows - kernel_rows + 1;
    int output_cols = input_cols - kernel_cols + 1;

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((output_cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (output_rows + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    convolution_2d_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, kernel, output, input_rows, input_cols, kernel_rows,kernel_cols,output_cols,output_rows);
    cudaDeviceSynchronize();
}