#include <stdio.h>
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

        // Load one tile of A
        if (row < M && aCol < K)
            As[ty][tx] = A[row * K + aCol];
        else
            As[ty][tx] = 0.0f;

        // Load one tile of B
        if (bRow < K && col < N)
            Bs[ty][tx] = B[bRow * N + col];
        else
            Bs[ty][tx] = 0.0f;

        __syncthreads();

        // Compute partial dot product
        for (int k = 0; k < TILE_WIDTH; k++)
            sum += As[ty][k] * Bs[k][tx];

        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = sum;
}

void matmul(const float *A_h,
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

    tiled_matmul<<<blocks, threads>>>(A_d, B_d, C_d, M, K, N);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));

    cudaDeviceSynchronize();

    cudaMemcpy(C_h, C_d, sizeC, cudaMemcpyDeviceToHost);

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

void printMatrix(const float *A, int rows, int cols)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
            printf("%6.1f", A[i * cols + j]);

        printf("\n");
    }
}

int main()
{
    const int M = 2;
    const int K = 3;
    const int N = 2;

    // A is 2 x 3
    float A[M * K] = {
        1, 2, 3,
        4, 5, 6
    };

    // B is 3 x 2
    float B[K * N] = {
        7,  8,
        9, 10,
        11,12
    };

    // C is 2 x 2
    float C[M * N];

    matmul(A, B, C, M, K, N);

    printf("Matrix A:\n");
    printMatrix(A, M, K);

    printf("\nMatrix B:\n");
    printMatrix(B, K, N);

    printf("\nMatrix C = A x B:\n");
    printMatrix(C, M, N);

    return 0;
}