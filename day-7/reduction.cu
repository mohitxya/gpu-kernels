#include <cuda_runtime.h>

#define BLOCK_SIZE 256

__global__
void reduce_kernel(const float *input,
                   float *output,
                   int N)
{
    __shared__ float sdata[BLOCK_SIZE];
    // each block gets its own copy. 

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * (BLOCK_SIZE * 2) + tid;
    // Every thread - two elements. 

    float sum = 0.0f;

    if (idx < N)
        sum = input[idx];

    if (idx + BLOCK_SIZE < N)
        sum += input[idx + BLOCK_SIZE];

    sdata[tid] = sum;
    __syncthreads();

    for (unsigned int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
            sdata[tid] += sdata[tid + stride];

        __syncthreads();
    }

    if (tid == 0)
        output[blockIdx.x] = sdata[0];
}

// input, output are device pointers
extern "C"
void solve(const float* input, float* output, int N)
{
    if (N == 1)
    {
        cudaMemcpy(output, input, sizeof(float), cudaMemcpyDeviceToDevice);
        return;
    }

    int blocks = (N + BLOCK_SIZE * 2 - 1) / (BLOCK_SIZE * 2);

    float *buf1, *buf2;

    cudaMalloc(&buf1, blocks * sizeof(float));
    cudaMalloc(&buf2, blocks * sizeof(float));

    const float *curr_in = input;
    float *curr_out = buf1;
    int curr_N = N;

    while (true)
    {
        blocks = (curr_N + BLOCK_SIZE * 2 - 1) / (BLOCK_SIZE * 2);

        reduce_kernel<<<blocks, BLOCK_SIZE>>>(curr_in,
                                              curr_out,
                                              curr_N);

        if (blocks == 1)
            break;

        curr_N = blocks;
        curr_in = curr_out;
        curr_out = (curr_out == buf1) ? buf2 : buf1;
    }

    cudaMemcpy(output,
               curr_out,
               sizeof(float),
               cudaMemcpyDeviceToDevice);

    cudaFree(buf1);
    cudaFree(buf2);
}