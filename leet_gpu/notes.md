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

# 3 : transpose matrix
/

# 4 : vector add
Be carefull between 1D launch and 2D or 3D launch, I should always prioritize 1D if I can I guess 

# 5 : color inversion
Same logic as python for the maths

# 6 : 1D convolution
/

# 7 : reverse array
Need to be carefull with indexes

# 8 : ReLu
Be carefull with typing, it is not like python, compare floats with floats (so 0.0 not 0)

# 9 : Leaky Relu
/

# 10 : Rainbow table
/

# 11 : Matrix copy
Always go for 1D

# 12 : model inference
output.copy_ if allocated outside of a function

# 13 : count array elements
atomicAdd allows to modify an elements in each thread by adding the value. Other atomic operations :
atomicAdd(addr, val);   // add
atomicSub(addr, val);   // subtract
atomicExch(addr, val);  // exchange (swap)
atomicMin(addr, val);
atomicMax(addr, val);

# 14 : count 2D array elements
/

# 15 : SiLu
- exp() should be used for double precision, although should be overloaded for single
- expf() should be used for single precision (float)
- __expf() is the fast-math version, the performance is faster with some loss of precision (dependent on the input value, see the guide for more details).

# 16 : SwiGLU
Not sure about their definition

# 17 : reduce
cudaMemset(output, 0, sizeof(float)) : to initialize in memory if needed
float* means pointers
if we say in the function "const float*" we cannot modify it

# 18 : Softmax
atomicAdd work only for int, so I found something on forum that uses atomicCAS to do it 
Had to separate in 3 kernels the calculus
Be carefull with pointers (need to modify the memory) and value (use the value)
cudaMalloc : to put a value in GPU memory
cudaMemcpy
cudaMemset
& for adress

# 18 : Softmax blockwise
To optimize, each block has the thread values (256) in shared memory so the goal is to compute the max using this block wise

__shared__ float sdata[256]; : shared memory
__syncthreads() : make sure all threads are finished writing
for (int s = blockDim.x / 2; s > 0; s >>= 1) : tree reduction

# 19 : Attention
Reuses previous kernels but had to ask help or Mr. GPT for the softmax cause I was lost

Softmax optimized with tiling with one row per block + store the buffer in shared memory with extern __shared__ float buf[];
for (int i = threadIdx.x; i < N; i += blockDim.x) : each thread take all the threshIdx.x and jump from bockDim.x
The buffer contains all threads max and we need to take the max of all of them (reduction tree)

# 20 : 2D convolution
Easier to create output cols and rows in solve so we use it for block per grid and idx verification too