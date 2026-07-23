#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 16

// C = A(M x K) * B(K x N)
__global__
void tiled_matmul(const float *A,
                  const float *B,
                  float *C,
                  int M, int K, int N)
{
    __shared__ float As[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];
  
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockIdx.y * TILE_WIDTH + ty;
    int col = blockIdx.x * TILE_WIDTH + tx;

    float sum = 0.0f;

    int numTiles = (K + TILE_WIDTH - 1) / TILE_WIDTH;

    for (int t = 0; t < numTiles; t++)
    {
        int aCol = t * TILE_WIDTH + tx;
        int bRow = t * TILE_WIDTH + ty;

        if (row < M && aCol < K)
            As[ty][tx] = A[row * K + aCol];
        else
            As[ty][tx] = 0.0f;

        if (bRow < K && col < N)
        {
            // Column-major indexing: B(col,row) = B[col*K + row]
            Bs[tx][ty] = B[col * K + bRow];
        }
        else
        {
            Bs[tx][ty] = 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; k++)
            sum += As[ty][k] * Bs[tx][k];

        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = sum;
}

// Returns GPU kernel time in milliseconds
float matmul_gpu(const float *A_h,
                  const float *B_h,
                  float *C_h,
                  int M, int K, int N)
{
    float *A_d, *B_d, *C_d;

    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);

    cudaMalloc(&A_d, sizeA);
    cudaMalloc(&B_d, sizeB);
    cudaMalloc(&C_d, sizeC);

    cudaMemcpy(A_d, A_h, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, sizeB, cudaMemcpyHostToDevice);

    dim3 threads(TILE_WIDTH, TILE_WIDTH);
    dim3 blocks((N + TILE_WIDTH - 1) / TILE_WIDTH,
                (M + TILE_WIDTH - 1) / TILE_WIDTH);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    tiled_matmul<<<blocks, threads>>>(A_d, B_d, C_d, M, K, N);
    cudaEventRecord(stop);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));

    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    cudaMemcpy(C_h, C_d, sizeC, cudaMemcpyDeviceToHost);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);

    return ms;
}

// Naive CPU reference: A row-major (M x K), B column-major (K x N), C row-major (M x N)
// Returns CPU time in milliseconds
float matmul_cpu(const float *A, const float *B, float *C, int M, int K, int N)
{
    clock_t start = clock();

    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
        {
            float acc = 0.0f;
            for (int k = 0; k < K; k++)
                acc += A[i * K + k] * B[j * K + k]; // B[j*K+k] = B(col=j,row=k)
            C[i * N + j] = acc;
        }

    clock_t end = clock();
    return 1000.0f * (float)(end - start) / CLOCKS_PER_SEC;
}

int main()
{
    const int M = 1024;
    const int K = 1024;
    const int N = 1024;

    float *A = (float*)malloc(M * K * sizeof(float));
    float *B = (float*)malloc(K * N * sizeof(float)); // column-major storage
    float *C_gpu = (float*)malloc(M * N * sizeof(float));
    float *C_cpu = (float*)malloc(M * N * sizeof(float));

    srand(42);
    for (int i = 0; i < M * K; i++) A[i] = (float)(rand() % 10);
    for (int i = 0; i < K * N; i++) B[i] = (float)(rand() % 10);

    float gpu_ms = matmul_gpu(A, B, C_gpu, M, K, N);
    float cpu_ms = matmul_cpu(A, B, C_cpu, M, K, N);

    float max_err = 0.0f;
    for (int i = 0; i < M * N; i++)
    {
        float err = fabsf(C_gpu[i] - C_cpu[i]);
        if (err > max_err) max_err = err;
    }

    printf("Matrix size: %dx%dx%d\n", M, K, N);
    printf("GPU time:    %8.3f ms\n", gpu_ms);
    printf("CPU time:    %8.3f ms\n", cpu_ms);
    printf("Speedup:     %8.2fx\n", cpu_ms / gpu_ms);
    printf("Max abs error vs CPU: %f -> %s\n",
           max_err, max_err < 1e-1f ? "PASS" : "FAIL");

    free(A); free(B); free(C_gpu); free(C_cpu);
    return 0;
}