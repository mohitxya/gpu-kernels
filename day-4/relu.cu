#include <iostream>
#include <cuda_runtime.h>

__global__
void relu_kernel(const float* __restrict__ input,
                 float* __restrict__ output,
                 int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= N)
        return;

    float x = input[idx];
    output[idx] = fmaxf(x, 0.0f);
}

int main()
{
    // Host input
    float h_input[] = {-2.5f, 3.1f, -1.0f, 0.0f, 4.8f, -7.2f, 5.5f};
    const int N = sizeof(h_input) / sizeof(float);

    float h_output[N];

    // Device pointers
    float *d_input, *d_output;

    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, N * sizeof(float));

    // Copy input to GPU
    cudaMemcpy(d_input,
               h_input,
               N * sizeof(float),
               cudaMemcpyHostToDevice);

    // Launch kernel
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    relu_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_input,
        d_output,
        N);

    cudaDeviceSynchronize();

    // Copy result back
    cudaMemcpy(h_output,
               d_output,
               N * sizeof(float),
               cudaMemcpyDeviceToHost);

    // Print results
    std::cout << "Input : ";
    for (int i = 0; i < N; i++)
        std::cout << h_input[i] << " ";

    std::cout << "\nOutput: ";
    for (int i = 0; i < N; i++)
        std::cout << h_output[i] << " ";

    std::cout << std::endl;

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}