#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>
#include <iostream>

#define BLOCK_SIZE 16

__global__
void tiledMatrixMulKernel(const float *A,
                          const float *B,
                          float *C,
                          int N)
{
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockIdx.y * BLOCK_SIZE + ty;
    int col = blockIdx.x * BLOCK_SIZE + tx;

    float sum = 0.0f;

    // Loop over tiles
    for (int tile = 0; tile < (N + BLOCK_SIZE - 1) / BLOCK_SIZE; tile++)
    {
        // Global indices of tile elements
        int tiledColA = tile * BLOCK_SIZE + tx;
        int tiledRowB = tile * BLOCK_SIZE + ty;

        // Load tile of A
        if (row < N && tiledColA < N)
            As[ty][tx] = A[row * N + tiledColA];
        else
            As[ty][tx] = 0.0f;

        // Load tile of B
        if (tiledRowB < N && col < N)
            Bs[ty][tx] = B[tiledRowB * N + col];
        else
            Bs[ty][tx] = 0.0f;

        __syncthreads();

        // Multiply the two tiles
        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; k++)
            sum += As[ty][k] * Bs[k][tx];

        __syncthreads();
    }

    if (row < N && col < N)
        C[row * N + col] = sum;
}

void matrixMul(float *A, float *B, float *C, int N)
{
    nvtxRangePush("Matrix Multiplication");

    float *d_A, *d_B, *d_C;
    int size = N * N * sizeof(float);

    nvtxRangePush("Memory Allocation");
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);
    nvtxRangePop();

    nvtxRangePush("Memory Copy H2D");
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
    nvtxRangePop();

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
                   (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    nvtxRangePush("Kernel Execution");

    tiledMatrixMulKernel<<<numBlocks, threadsPerBlock>>>(
        d_A,
        d_B,
        d_C,
        N);

    cudaDeviceSynchronize();

    nvtxRangePop();

    nvtxRangePush("Memory Copy D2H");
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);
    nvtxRangePop();

    nvtxRangePush("Memory Deallocation");
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    nvtxRangePop();

    nvtxRangePop();
}

int main()
{
    const int N = 1024;

    float *A = new float[N * N];
    float *B = new float[N * N];
    float *C = new float[N * N];

    // Initialize matrices
    for (int i = 0; i < N * N; i++)
    {
        A[i] = 1.0f;
        B[i] = 1.0f;
    }

    matrixMul(A, B, C, N);

    std::cout << C[0] << std::endl;

    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}