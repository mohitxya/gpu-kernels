#include <iostream>
#include <cuda_runtime.h>
#include <cmath>

__global__
void swiglu_kernel(const float* input, float* output, int halfN)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= halfN)
        return;

    float x1 = input[idx];
    float x2 = input[idx + halfN];

    // SiLU(x) = x * sigmoid(x)
    float silu = x1 / (1.0f + expf(-x1));

    output[idx] = silu * x2;
}

int main()
{
    // Example input: [x1 | x2]
    float h_input[] = {1.0f, 2.0f, 3.0f, 4.0f};

    const int N = sizeof(h_input) / sizeof(float);
    const int halfN = N / 2;

    float h_output[halfN];

    float *d_input, *d_output;

    // Allocate device memory
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, halfN * sizeof(float));

    // Copy input to device
    cudaMemcpy(d_input,
               h_input,
               N * sizeof(float),
               cudaMemcpyHostToDevice);

    // Launch kernel
    const int threadsPerBlock = 256;
    const int blocksPerGrid =
        (halfN + threadsPerBlock - 1) / threadsPerBlock;

    swiglu_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_input,
        d_output,
        halfN);

    cudaDeviceSynchronize();

    // Copy result back
    cudaMemcpy(h_output,
               d_output,
               halfN * sizeof(float),
               cudaMemcpyDeviceToHost);

    // Print input
    std::cout << "Input : ";
    for (int i = 0; i < N; i++)
        std::cout << h_input[i] << " ";

    std::cout << "\nOutput: ";
    for (int i = 0; i < halfN; i++)
        std::cout << h_output[i] << " ";

    std::cout << std::endl;

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}