#include <stdio.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 8

__global__ void ConvergentSumReductionKernel(float *input, float *output)
{
    unsigned int i = threadIdx.x;

    // Initial stride is half the number of input elements.
    // Input size = 2 * BLOCK_SIZE
    for (unsigned int stride = blockDim.x; stride >= 1; stride /= 2)
    {
        if (threadIdx.x < stride)
        {
            input[i] += input[i + stride];
        }

        __syncthreads();
    }

    if (threadIdx.x == 0)
    {
        *output = input[0];
    }
}

int main()
{
    const int N = 2 * BLOCK_SIZE;

    float h_input[N];
    float h_output;

    // Initialize input: 1, 2, ..., 16
    for (int i = 0; i < N; i++)
    {
        h_input[i] = (float)(i + 1);
    }

    float *d_input, *d_output;

    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, sizeof(float));

    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);

    // One block with BLOCK_SIZE threads
    ConvergentSumReductionKernel<<<1, BLOCK_SIZE>>>(d_input, d_output);

    cudaDeviceSynchronize();

    cudaMemcpy(&h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost);

    printf("Input:\n");
    for (int i = 0; i < N; i++)
    {
        printf("%.0f ", h_input[i]);
    }
    printf("\n");

    printf("Sum = %.0f\n", h_output);

    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}