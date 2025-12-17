# 1 : Vector add

__global__ : run on the GPU, called by the CPU
const float* A, const float* B : pointers on the GPU memory, const = read only

threadIdx.x : index inside the block
blockIdx.x : block index
blockDim.x : thread per block
global index = block number * threads per block + local thread index

int threadsPerBlock = 256; : 8 wraps of 32 threads

vector_add<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N); : Kernel launch - CPU: launch a GPU kernel with this many threads

# 2 : Matrix multiplication

int col = blockIdx.x * blockDim.x + threadIdx.x : x is for columns in cuda
int row = blockIdx.y * blockDim.y + threadIdx.y : y is for rows in cuda
C[row * K + col] += A[row * N + i] * B[i * K + col] : always need to flatten indexes even in 2D
float sum = 0 : better to use it if C is not initialized and to avoid accessing C too much

# 2 : transpose matrix
/