#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define BLOCK_DIM 256

__global__ void SegmentedSumReductionKernel(const float *input,
                                            float *output,
                                            int N)
{
    __shared__ float input_s[BLOCK_DIM];

    unsigned int segment = 2 * blockDim.x * blockIdx.x;
    unsigned int i = segment + threadIdx.x;
    unsigned int t = threadIdx.x;

    // Each thread loads up to two elements
    float sum = 0.0f;

    if (i < N)
        sum += input[i];

    if (i + BLOCK_DIM < N)
        sum += input[i + BLOCK_DIM];

    input_s[t] = sum;

    __syncthreads();

    // Shared-memory reduction
    for (unsigned int stride = blockDim.x / 2; stride >= 1; stride /= 2)
    {
        if (t < stride)
        {
            input_s[t] += input_s[t + stride];
        }

        __syncthreads();
    }

    // One partial sum per block
    if (t == 0)
    {
        atomicAdd(output, input_s[0]);
    }
}

int main()
{
    const int N = 1000;

    float *h_input = (float *)malloc(N * sizeof(float));

    for (int i = 0; i < N; i++)
        h_input[i] = 1.0f;

    float cpu_sum = 0.0f;
    for (int i = 0; i < N; i++)
        cpu_sum += h_input[i];

    float *d_input;
    float *d_output;

    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, sizeof(float));

    cudaMemcpy(d_input,
               h_input,
               N * sizeof(float),
               cudaMemcpyHostToDevice);

    cudaMemset(d_output, 0, sizeof(float));

    int blocks = (N + (2 * BLOCK_DIM - 1)) / (2 * BLOCK_DIM);

    SegmentedSumReductionKernel<<<blocks, BLOCK_DIM>>>(d_input,
                                                       d_output,
                                                       N);

    cudaDeviceSynchronize();

    float gpu_sum;

    cudaMemcpy(&gpu_sum,
               d_output,
               sizeof(float),
               cudaMemcpyDeviceToHost);

    printf("CPU Sum : %.0f\n", cpu_sum);
    printf("GPU Sum : %.0f\n", gpu_sum);

    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);

    return 0;
}