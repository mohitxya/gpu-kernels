#include <stdio.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 8

__global__ void SimpleSumReductionKernel(float *input, float *output)
{
    unsigned int i = 2 * threadIdx.x;

    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2)
    {
        if (threadIdx.x % (2 * stride) == 0)
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
    // Number of input elements = 2 * BLOCK_SIZE
    const int N = 2 * BLOCK_SIZE;

    float h_input[N];
    float h_output;

    // Initialize input: 1,2,3,...,16
    for (int i = 0; i < N; i++)
        h_input[i] = i + 1;

    float *d_input, *d_output;

    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, sizeof(float));

    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);

    SimpleSumReductionKernel<<<1, BLOCK_SIZE>>>(d_input, d_output);

    cudaDeviceSynchronize();

    cudaMemcpy(&h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost);

    printf("Input:\n");
    for (int i = 0; i < N; i++)
        printf("%.0f ", h_input[i]);
    printf("\n");

    printf("Sum = %.0f\n", h_output);

    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}