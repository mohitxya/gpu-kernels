#include <stdio.h>
#include <cuda_runtime.h>

// each thread is responsible for calculating one p element. 

__global__
void  matMulKernel(float *A, float*B, float*C, int M, int K, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x; 

    if (row < M && col < N)
    {
        float sum = 0.0f;

        for (int i = 0; i < K; i++)
        {
            sum += A[row * K + i] * B[i * N + col];
        }

        C[row * N + col] = sum;
    }
}
void matMul(const float *A_h, const float *B_h, float *C_h, int M, int K, int N)
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

    dim3 threads(16,16); 
    dim3 blocks((N+threads.x -1)/threads.x, (M+threads.y-1)/threads.y);

    // number of blocks in the grid, number of threads in each block. 
    matMulKernel<<<blocks, threads>>> (A_d, B_d, C_d, M, K, N);

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

    matMul(A, B, C, M, K, N);

    printf("Matrix A:\n");
    printMatrix(A, M, K);

    printf("\nMatrix B:\n");
    printMatrix(B, K, N);

    printf("\nMatrix C = A x B:\n");
    printMatrix(C, M, N);

    return 0;
}