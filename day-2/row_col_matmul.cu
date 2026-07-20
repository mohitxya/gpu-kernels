#include <stdio.h>
#include <cuda_runtime.h>
#include <cstring>
// 1. Each thread calculates one row. 
// 2. Each thread calculates one col. 
__global__
void  row_kernel(float *A, float*B, float*C, int M, int K, int N)
{
    // we'll multiply a given row by all columns. 

    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M)
    {
        // Compute every column in this row
        for (int col = 0; col < N; col++) {

            float sum = 0;

            // Dot product
            for (int k = 0; k < K; k++) {
                sum += A[row * K + k] * B[k * N + col];
            }

            C[row*N+col] = sum;
    }
    }
}
void row(const float *A_h, const float *B_h, float *C_h, int M, int K, int N)
{
    // we'll use one thread per row. 

    float *A_d, *B_d, *C_d; 

    size_t sizeA = M * K * sizeof(float); 
    size_t sizeB = K * N * sizeof(float); 
    size_t sizeC = M * N * sizeof(float); 

    cudaMalloc(&A_d, sizeA); 
    cudaMalloc(&B_d, sizeB); 
    cudaMalloc(&C_d, sizeC); 

    cudaMemcpy(A_d, A_h, sizeA, cudaMemcpyHostToDevice); 
    cudaMemcpy(B_d, B_h, sizeB, cudaMemcpyHostToDevice); 

    dim3 threads(16); 
    dim3 blocks((M+threads.x-1)/threads.x);

    // number of blocks in the grid, number of threads in each block. 
    row_kernel<<<blocks, threads>>> (A_d, B_d, C_d, M, K, N);

    cudaDeviceSynchronize(); 

    cudaMemcpy(C_h, C_d, sizeC, cudaMemcpyDeviceToHost);

    cudaFree(A_d); 
    cudaFree(B_d); 
    cudaFree(C_d); 
    
}


__global__
void  col_kernel(float *A, float*B, float*C, int M, int K, int N)
{

    int col = blockIdx.x * blockDim.x + threadIdx.x; 

    if (col < N)
    {
        for(int row=0; row<M; row++)
        {
            float sum = 0.0; 
            for(int k=0; k<K; k++)
            {
                sum += A[row*K+k] * B[k*N+col];
            }
            C[row*N+col]=sum;
        }
    }
}
void col(const float *A_h, const float *B_h, float *C_h, int M, int K, int N)
{
    // We'll use one thread per column
    float *A_d, *B_d, *C_d; 

    size_t sizeA = M * K * sizeof(float); 
    size_t sizeB = K * N * sizeof(float); 
    size_t sizeC = M * N * sizeof(float); 

    cudaMalloc(&A_d, sizeA); 
    cudaMalloc(&B_d, sizeB); 
    cudaMalloc(&C_d, sizeC); 

    cudaMemcpy(A_d, A_h, sizeA, cudaMemcpyHostToDevice); 
    cudaMemcpy(B_d, B_h, sizeB, cudaMemcpyHostToDevice); 

    dim3 threads(16); 
    dim3 blocks((N+threads.x -1)/threads.x);

    // number of blocks in the grid, number of threads in each block. 
    col_kernel<<<blocks, threads>>> (A_d, B_d, C_d, M, K, N);

    cudaDeviceSynchronize(); 

    cudaMemcpy(C_h, C_d, sizeC, cudaMemcpyDeviceToHost);

    cudaFree(A_d); 
    cudaFree(B_d); 
    cudaFree(C_d); 
    
} 

void printMatrix(const float *A, int rows, int cols)
{
    for(int i=0; i<rows; i++)
    {
        for(int j=0; j<cols; j++)
        {
            printf("%6.1f", A[i*cols + j]);
        }
        printf("\n");
    }
}

int main()
{
    const int M = 2;
    const int K = 3;
    const int N = 2;

    float A[M * K] = {
        1, 2, 3,
        4, 5, 6
    };

    float B[K * N] = {
        7, 8,
        9,10,
        11,12
    };

    float C[M * N];

    row(A, B, C, M, K, N);

    printf("Matrix A:\n");
    printMatrix(A, M, K);

    printf("\nMatrix B:\n");
    printMatrix(B, K, N);

    printf("\nMatrix C = A x B:\n");
    printMatrix(C, M, N);

    memset(C, 0, sizeof(C));

    col(A, B, C, M, K, N);

    printf("Matrix A:\n");
    printMatrix(A, M, K);

    printf("\nMatrix B:\n");
    printMatrix(B, K, N);

    printf("\nMatrix C = A x B:\n");
    printMatrix(C, M, N);

    return 0;
}