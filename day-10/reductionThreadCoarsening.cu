#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define BLOCK_DIM     1024
#define COARSE_FACTOR 2

__global__
void CoarsenedSumReductionKernel(float *input, float *output)
{
    __shared__ float input_s[BLOCK_DIM];

    unsigned int segment = COARSE_FACTOR * 2 * blockDim.x * blockIdx.x;
    unsigned int i = segment + threadIdx.x;
    unsigned int t = threadIdx.x;

    float sum = input[i];
    for (unsigned int tile = 1; tile < COARSE_FACTOR * 2; ++tile)
    {
        sum += input[i + tile * BLOCK_DIM];
    }
    input_s[t] = sum;

    for (unsigned int stride = blockDim.x / 2; stride >= 1; stride /= 2)
    {
        __syncthreads();
        if (t < stride)
        {
            input_s[t] += input_s[t + stride];
        }
    }

    if (t == 0)
    {
        atomicAdd(output, input_s[0]);
    }
}

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                                \
            exit(EXIT_FAILURE);                                             \
        }                                                                    \
    } while (0)

int main()
{
    const int numBlocks = 4;
    // Each block consumes COARSE_FACTOR*2*BLOCK_DIM elements.
    const long long elementsPerBlock = (long long)COARSE_FACTOR * 2 * BLOCK_DIM;
    const long long N = elementsPerBlock * numBlocks;

    size_t bytes = N * sizeof(float);
    float *h_input = (float *)malloc(bytes);
    for (long long idx = 0; idx < N; ++idx) {
        h_input[idx] = 1.0f; // sum should equal N
    }

    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, bytes));
    CUDA_CHECK(cudaMalloc(&d_output, sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_output, 0, sizeof(float)));

    dim3 gridDim(numBlocks);
    dim3 blockDim(BLOCK_DIM);
    CoarsenedSumReductionKernel<<<gridDim, blockDim>>>(d_input, d_output);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    float h_output = 0.0f;
    CUDA_CHECK(cudaMemcpy(&h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost));

    printf("N elements       : %lld\n", N);
    printf("GPU sum          : %f\n", h_output);
    printf("Expected sum     : %f\n", (float)N);
    printf("Match            : %s\n", (h_output == (float)N) ? "YES" : "NO (float rounding? check)");

    free(h_input);
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    return 0;
}