#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>

#define BLOCK_SIZE 256

__global__
void reduce_max(const float *input, float *output, int N)
{
    __shared__ float sdata[BLOCK_SIZE]; 

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x*BLOCK_SIZE*2+tid;

    float local_max = -1e30f;

    if(idx < N)
        local_max = input[idx];
    if(idx + BLOCK_SIZE < N)
        local_max = fmaxf(local_max, input[idx + BLOCK_SIZE]);
    
    sdata[tid] = local_max; 
    __syncthreads();

    for(unsigned int stride = BLOCK_SIZE/2; stride>0; stride>>=1)
    {
        if(tid < stride)
        {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid+stride]);

        }
        __syncthreads();
    }

    if(tid == 0) output[blockIdx.x] = sdata[0];
}

__global__
void reduce_sum(const float *input,
                float *output,
                int N)
{
    __shared__ float sdata[BLOCK_SIZE];

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * BLOCK_SIZE * 2 + tid;

    float sum = 0.0f;

    if (idx < N)
        sum = input[idx];

    if (idx + BLOCK_SIZE < N)
        sum += input[idx + BLOCK_SIZE];

    sdata[tid] = sum;
    __syncthreads();

    for (unsigned int stride = BLOCK_SIZE / 2;
         stride > 0;
         stride >>= 1)
    {
        if (tid < stride)
            sdata[tid] += sdata[tid + stride];

        __syncthreads();
    }

    if (tid == 0)
        output[blockIdx.x] = sdata[0];
}

__global__
void exp_kernel(const float *input,
                float *output,
                const float *max_val,
                int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N)
        output[idx] = expf(input[idx] - max_val[0]);
}

__global__
void normalize_kernel(float *output,
                      const float *sum,
                      int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N)
        output[idx] /= sum[0];
}
void reduceMax(const float *input,
               float *result,
               int N)
{
    int blocks = (N + BLOCK_SIZE * 2 - 1)
               / (BLOCK_SIZE * 2);

    float *buf1, *buf2;

    cudaMalloc(&buf1, blocks * sizeof(float));
    cudaMalloc(&buf2, blocks * sizeof(float));

    const float *curr_in = input;
    float *curr_out = buf1;

    int curr_N = N;

    while (true)
    {
        blocks = (curr_N + BLOCK_SIZE * 2 - 1)
               / (BLOCK_SIZE * 2);

        reduce_max<<<blocks, BLOCK_SIZE>>>(
            curr_in,
            curr_out,
            curr_N);

        if (blocks == 1)
            break;

        curr_N = blocks;
        curr_in = curr_out;
        curr_out =
            (curr_out == buf1) ? buf2 : buf1;
    }

    cudaMemcpy(result,
               curr_out,
               sizeof(float),
               cudaMemcpyDeviceToDevice);

    cudaFree(buf1);
    cudaFree(buf2);
}

//---------------------------------------------
// Generic sum reduction
//---------------------------------------------
void reduceSum(const float *input,
               float *result,
               int N)
{
    int blocks = (N + BLOCK_SIZE * 2 - 1)
               / (BLOCK_SIZE * 2);

    float *buf1, *buf2;

    cudaMalloc(&buf1, blocks * sizeof(float));
    cudaMalloc(&buf2, blocks * sizeof(float));

    const float *curr_in = input;
    float *curr_out = buf1;

    int curr_N = N;

    while (true)
    {
        blocks = (curr_N + BLOCK_SIZE * 2 - 1)
               / (BLOCK_SIZE * 2);

        reduce_sum<<<blocks, BLOCK_SIZE>>>(
            curr_in,
            curr_out,
            curr_N);

        if (blocks == 1)
            break;

        curr_N = blocks;
        curr_in = curr_out;
        curr_out =
            (curr_out == buf1) ? buf2 : buf1;
    }

    cudaMemcpy(result,
               curr_out,
               sizeof(float),
               cudaMemcpyDeviceToDevice);

    cudaFree(buf1);
    cudaFree(buf2);
}

//---------------------------------------------
// Solve
//---------------------------------------------
extern "C"
void solve(const float *input,
           float *output,
           int N)
{
    float *d_max;
    float *d_sum;

    cudaMalloc(&d_max, sizeof(float));
    cudaMalloc(&d_sum, sizeof(float));

    reduceMax(input, d_max, N);

    int blocks =
        (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    exp_kernel<<<blocks, BLOCK_SIZE>>>(
        input,
        output,
        d_max,
        N);

    reduceSum(output, d_sum, N);

    normalize_kernel<<<blocks, BLOCK_SIZE>>>(
        output,
        d_sum,
        N);

    cudaFree(d_max);
    cudaFree(d_sum);
}

//---------------------------------------------
// Test program
//---------------------------------------------
int main()
{
    const int N = 8;

    float h_input[N] =
    {
        1,2,3,4,5,6,7,8
    };

    float *d_input;
    float *d_output;

    cudaMalloc(&d_input,
               N * sizeof(float));

    cudaMalloc(&d_output,
               N * sizeof(float));

    cudaMemcpy(d_input,
               h_input,
               N * sizeof(float),
               cudaMemcpyHostToDevice);

    solve(d_input,
          d_output,
          N);

    float h_output[N];

    cudaMemcpy(h_output,
               d_output,
               N * sizeof(float),
               cudaMemcpyDeviceToHost);

    printf("Softmax:\n");

    float check = 0.0f;

    for(int i=0;i<N;i++)
    {
        printf("%f\n", h_output[i]);
        check += h_output[i];
    }

    printf("Sum = %f\n", check);

    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}