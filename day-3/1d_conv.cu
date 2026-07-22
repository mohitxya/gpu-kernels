#include <iostream>
#include <cuda_runtime.h>

__global__
void convolution_1d_kernel(const float* input,
                           const float* kernel,
                           float* output,
                           int input_size,
                           int kernel_size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    int output_size = input_size - kernel_size + 1;

    if (i >= output_size)
        return;

    float sum = 0.0f;

    for (int j = 0; j < kernel_size; j++)
    {
        int idx = i + j;
        sum += input[idx] * kernel[j];
    }

    output[i] = sum;
}

int main()
{
    // Example
    float h_input[]  = {1, 2, 3, 4, 5};
    float h_kernel[] = {1, 0, -1};

    int input_size  = 5;
    int kernel_size = 3;
    int output_size = input_size - kernel_size + 1;

    float h_output[3];

    float *d_input, *d_kernel, *d_output;

    cudaMalloc(&d_input,  input_size * sizeof(float));
    cudaMalloc(&d_kernel, kernel_size * sizeof(float));
    cudaMalloc(&d_output, output_size * sizeof(float));

    cudaMemcpy(d_input,
               h_input,
               input_size * sizeof(float),
               cudaMemcpyHostToDevice);

    cudaMemcpy(d_kernel,
               h_kernel,
               kernel_size * sizeof(float),
               cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid =
        (output_size + threadsPerBlock - 1) / threadsPerBlock;

    convolution_1d_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_input,
        d_kernel,
        d_output,
        input_size,
        kernel_size);

    cudaDeviceSynchronize();

    cudaMemcpy(h_output,
               d_output,
               output_size * sizeof(float),
               cudaMemcpyDeviceToHost);

    std::cout << "Output:\n";

    for (int i = 0; i < output_size; i++)
        std::cout << h_output[i] << " ";

    std::cout << std::endl;

    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);

    return 0;
}