#include <stdio.h>
#include <cuda_runtime.h>

#define BLOCK_DIM 8

__global__ void SharedMemorySumReductionKernel(float *input, float *output)
{
    __shared__ float input_s[BLOCK_DIM];

    unsigned int t = threadIdx.x;

    // Load two elements per thread into shared memory
    input_s[t] = input[t] + input[t + BLOCK_DIM];

    __syncthreads();

    // Reduce in shared memory
    for (unsigned int stride = blockDim.x / 2; stride >= 1; stride /= 2)
    {
        if (t < stride)
        {
            input_s[t] += input_s[t + stride];
        }

        __syncthreads();
    }

    if (t == 0)
    {
        *output = input_s[0];
    }
}

int main()
{
    const int N = 2 * BLOCK_DIM;

    float h_input[N];
    float h_output;

    // Initialize input: 1,2,...,16
    for (int i = 0; i < N; i++)
    {
        h_input[i] = (float)(i + 1);
    }

    float *d_input, *d_output;

    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, sizeof(float));

    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);

    SharedMemorySumReductionKernel<<<1, BLOCK_DIM>>>(d_input, d_output);

    cudaDeviceSynchronize();

    cudaMemcpy(&h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost);

    printf("Input:\n");
    for (int i = 0; i < N; i++)
    {
        printf("%.0f ", h_input[i]);
    }

    printf("\n\nSum = %.0f\n", h_output);

    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}