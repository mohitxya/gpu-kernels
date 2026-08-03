#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define BLOCKSIZE 32

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t err = call;                                                  \
    if (err != cudaSuccess) {                                                \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,       \
              cudaGetErrorString(err));                                      \
      exit(EXIT_FAILURE);                                                    \
    }                                                                        \
  } while (0)

__global__ void sgemm_global_mem_coalesce(int M, int N, int K, float alpha,
                                           const float *A, const float *B,
                                           float beta, float *C) {
  const int cRow = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
  const int cCol = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

  if (cRow < M && cCol < N) {
    float tmp = 0.0f;
    for (int i = 0; i < K; ++i) {
      tmp += A[cRow * K + i] * B[i * N + cCol];
    }
    C[cRow * N + cCol] = alpha * tmp + beta * C[cRow * N + cCol];
  }
}

void init_matrix(float *mat, int rows, int cols) {
  for (int i = 0; i < rows * cols; ++i) {
    mat[i] = static_cast<float>(rand() % 5);
  }
}

int main() {
  const int M = 4096;
  const int N = 4096;
  const int K = 4096;
  const float alpha = 1.0f;
  const float beta  = 0.0f;

  size_t sizeA = (size_t)M * K * sizeof(float);
  size_t sizeB = (size_t)K * N * sizeof(float);
  size_t sizeC = (size_t)M * N * sizeof(float);

  float *h_A = (float *)malloc(sizeA);
  float *h_B = (float *)malloc(sizeB);
  float *h_C = (float *)malloc(sizeC);

  if (!h_A || !h_B || !h_C) {
    fprintf(stderr, "Host memory allocation failed\n");
    return EXIT_FAILURE;
  }

  srand(42);
  init_matrix(h_A, M, K);
  init_matrix(h_B, K, N);
  for (int i = 0; i < M * N; ++i) h_C[i] = 0.0f;

  float *d_A, *d_B, *d_C;
  CUDA_CHECK(cudaMalloc((void **)&d_A, sizeA));
  CUDA_CHECK(cudaMalloc((void **)&d_B, sizeB));
  CUDA_CHECK(cudaMalloc((void **)&d_C, sizeC));

  CUDA_CHECK(cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_C, h_C, sizeC, cudaMemcpyHostToDevice));

  dim3 gridDim((M + BLOCKSIZE - 1) / BLOCKSIZE,
               (N + BLOCKSIZE - 1) / BLOCKSIZE);
  dim3 blockDim(BLOCKSIZE * BLOCKSIZE);

  printf("Matrix size: M=%d, N=%d, K=%d\n", M, N, K);
  printf("Grid dim: (%d, %d), Block dim: (%d)\n", gridDim.x, gridDim.y,
         blockDim.x);

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  sgemm_global_mem_coalesce<<<gridDim, blockDim>>>(M, N, K, alpha, d_A, d_B,
                                                    beta, d_C);
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(d_C, h_C, sizeC, cudaMemcpyHostToDevice));

  const int numRuns = 10;
  float totalMs = 0.0f;

  for (int r = 0; r < numRuns; ++r) {
    CUDA_CHECK(cudaMemcpy(d_C, h_C, sizeC, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaEventRecord(start));
    sgemm_global_mem_coalesce<<<gridDim, blockDim>>>(M, N, K, alpha, d_A, d_B,
                                                      beta, d_C);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    totalMs += ms;
    printf("Run %d: %.3f ms\n", r, ms);
  }

  float avgMs = totalMs / numRuns;
  double flops = 2.0 * (double)M * N * K;
  double gflops = (flops / (avgMs / 1000.0)) / 1e9;

  printf("\nAverage kernel time: %.3f ms\n", avgMs);
  printf("Performance: %.2f GFLOPS\n", gflops);

  CUDA_CHECK(cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost));

  float ref = 0.0f;
  for (int i = 0; i < K; ++i) ref += h_A[0 * K + i] * h_B[i * N + 0];
  ref = alpha * ref + beta * 0.0f;
  printf("\nSanity check C[0][0]: GPU=%.2f, CPU=%.2f\n", h_C[0], ref);

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B));
  CUDA_CHECK(cudaFree(d_C));
  free(h_A);
  free(h_B);
  free(h_C);

  return 0;
}