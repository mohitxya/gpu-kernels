#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

// ---------------------------------------------------------------------------
// Error checking helper
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                      \
  do {                                                                        \
    cudaError_t err__ = (call);                                               \
    if (err__ != cudaSuccess) {                                               \
      fprintf(stderr, "[CUDA ERROR] %s:%d: %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(err__));                                     \
      exit(EXIT_FAILURE);                                                     \
    }                                                                         \
  } while (0)

#define CUBLAS_CHECK(call)                                                    \
  do {                                                                        \
    cublasStatus_t st__ = (call);                                             \
    if (st__ != CUBLAS_STATUS_SUCCESS) {                                      \
      fprintf(stderr, "[CUBLAS ERROR] %s:%d: status %d\n", __FILE__,          \
              __LINE__, (int)st__);                                           \
      exit(EXIT_FAILURE);                                                     \
    }                                                                         \
  } while (0)

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

__global__ void sgemm_naive(int M, int N, int K, float alpha, const float *A,
                            const float *B, float beta, float *C) {
  const uint row = blockIdx.y * blockDim.y + threadIdx.y;
  const uint col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < M && col < N) {
    float acc = 0.0f;
    for (uint k = 0; k < (uint)K; ++k) {
      acc += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = alpha * acc + beta * C[row * N + col];
  }
}

template <const int BM, const int BN, const int BK, const int TM>
__global__ void __launch_bounds__((BM * BN) / TM, 1)
    sgemm_1d_blocktiling(int M, int N, int K, float alpha, const float *A,
                         const float *B, float beta, float *C) {
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  const int threadCol = threadIdx.x % BN;
  const int threadRow = threadIdx.x / BN;

  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  A += cRow * BM * K;
  B += cCol * BN;
  C += cRow * BM * N + cCol * BN;

  const uint innerColA = threadIdx.x % BK;
  const uint innerRowA = threadIdx.x / BK;
  const uint innerColB = threadIdx.x % BN;
  const uint innerRowB = threadIdx.x / BN;

  float threadResults[TM] = {0.0f};

  for (uint bkIdx = 0; bkIdx < (uint)K; bkIdx += BK) {
    As[innerRowA * BK + innerColA] = A[innerRowA * K + innerColA];
    Bs[innerRowB * BN + innerColB] = B[innerRowB * N + innerColB];
    __syncthreads();

    A += BK;
    B += BK * N;

    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
      float tmpB = Bs[dotIdx * BN + threadCol];
      for (uint resIdx = 0; resIdx < TM; ++resIdx) {
        threadResults[resIdx] +=
            As[(threadRow * TM + resIdx) * BK + dotIdx] * tmpB;
      }
    }
    __syncthreads();
  }

  for (uint resIdx = 0; resIdx < TM; ++resIdx) {
    uint outRow = threadRow * TM + resIdx;
    C[outRow * N + threadCol] =
        alpha * threadResults[resIdx] + beta * C[outRow * N + threadCol];
  }
}

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void __launch_bounds__((BM * BN) / (TM * TN), 1)
    sgemm2DBlocktiling(int M, int N, int K, float alpha, const float *A,
                       const float *B, float beta, float *C) {
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  const uint totalResultsBlocktile = BM * BN;
  const uint numThreadsBlocktile = totalResultsBlocktile / (TM * TN);

  assert(numThreadsBlocktile == blockDim.x);

  const int threadCol = threadIdx.x % (BN / TN);
  const int threadRow = threadIdx.x / (BN / TN);

  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  A += cRow * BM * K;
  B += cCol * BN;
  C += cRow * BM * N + cCol * BN;

  const uint innerRowA = threadIdx.x / BK;
  const uint innerColA = threadIdx.x % BK;
  const uint strideA = numThreadsBlocktile / BK;
  const uint innerRowB = threadIdx.x / BN;
  const uint innerColB = threadIdx.x % BN;
  const uint strideB = numThreadsBlocktile / BN;

  float threadResults[TM * TN] = {0.0};
  float regM[TM] = {0.0};
  float regN[TN] = {0.0};

  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    for (uint loadOffset = 0; loadOffset < BM; loadOffset += strideA) {
      As[(innerRowA + loadOffset) * BK + innerColA] =
          A[(innerRowA + loadOffset) * K + innerColA];
    }
    for (uint loadOffset = 0; loadOffset < BK; loadOffset += strideB) {
      Bs[(innerRowB + loadOffset) * BN + innerColB] =
          B[(innerRowB + loadOffset) * N + innerColB];
    }
    __syncthreads();

    A += BK; 
    B += BK * N; 

    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
      for (uint i = 0; i < TM; ++i) {
        regM[i] = As[(threadRow * TM + i) * BK + dotIdx];
      }
      for (uint i = 0; i < TN; ++i) {
        regN[i] = Bs[dotIdx * BN + threadCol * TN + i];
      }
      for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
        for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
          threadResults[resIdxM * TN + resIdxN] +=
              regM[resIdxM] * regN[resIdxN];
        }
      }
    }
    __syncthreads();
  }

  for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
    for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
      C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN] =
          alpha * threadResults[resIdxM * TN + resIdxN] +
          beta * C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN];
    }
  }
}

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void __launch_bounds__((BM * BN) / (TM * TN), 1)
    sgemmVectorize(int M, int N, int K, float alpha, const float * __restrict__ A,
                   const float * __restrict__ B, float beta, float * __restrict__ C) {
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  const uint numThreads = (BM * BN) / (TM * TN);
  const int threadCol = threadIdx.x % (BN / TN);
  const int threadRow = threadIdx.x / (BN / TN);

  __shared__ float As[BK][BM]; // Transposed layout for bank-conflict free accesses
  __shared__ float Bs[BK][BN];

  A += cRow * BM * K;
  B += cCol * BN;
  C += cRow * BM * N + cCol * BN;

  const uint innerRowA = threadIdx.x / (BK / 4);
  const uint innerColA = threadIdx.x % (BK / 4);
  const uint strideA = numThreads / (BK / 4);

  const uint innerRowB = threadIdx.x / (BN / 4);
  const uint innerColB = threadIdx.x % (BN / 4);
  const uint strideB = numThreads / (BN / 4);

  float threadResults[TM * TN] = {0.0f};
  float regM[TM] = {0.0f};
  float regN[TN] = {0.0f};

  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    for (uint loadOffset = 0; loadOffset < BM; loadOffset += strideA) {
      float4 tmp = reinterpret_cast<const float4*>(&A[(innerRowA + loadOffset) * K + innerColA * 4])[0];
      As[innerColA * 4 + 0][innerRowA + loadOffset] = tmp.x;
      As[innerColA * 4 + 1][innerRowA + loadOffset] = tmp.y;
      As[innerColA * 4 + 2][innerRowA + loadOffset] = tmp.z;
      As[innerColA * 4 + 3][innerRowA + loadOffset] = tmp.w;
    }

    for (uint loadOffset = 0; loadOffset < BK; loadOffset += strideB) {
      reinterpret_cast<float4*>(&Bs[innerRowB + loadOffset][innerColB * 4])[0] =
          reinterpret_cast<const float4*>(&B[(innerRowB + loadOffset) * N + innerColB * 4])[0];
    }
    __syncthreads();

    A += BK;
    B += BK * N;

    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
      for (uint i = 0; i < TM; ++i) {
        regM[i] = As[dotIdx][threadRow * TM + i];
      }
      for (uint i = 0; i < TN; ++i) {
        regN[i] = Bs[dotIdx][threadCol * TN + i];
      }
      for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
        for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
          threadResults[resIdxM * TN + resIdxN] += regM[resIdxM] * regN[resIdxN];
        }
      }
    }
    __syncthreads();
  }

  for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
    for (uint resIdxN = 0; resIdxN < TN; resIdxN += 4) {
      float4 oldC = reinterpret_cast<const float4*>(&C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN])[0];
      float4 val;
      val.x = alpha * threadResults[resIdxM * TN + resIdxN + 0] + beta * oldC.x;
      val.y = alpha * threadResults[resIdxM * TN + resIdxN + 1] + beta * oldC.y;
      val.z = alpha * threadResults[resIdxM * TN + resIdxN + 2] + beta * oldC.z;
      val.w = alpha * threadResults[resIdxM * TN + resIdxN + 3] + beta * oldC.w;
      reinterpret_cast<float4*>(&C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN])[0] = val;
    }
  }
}

// ---------------------------------------------------------------------------
// Host Launchers
// ---------------------------------------------------------------------------

void run_sgemm_naive(int M, int N, int K, float alpha, const float *A,
                      const float *B, float beta, float *C) {
  dim3 blockDim(32, 32);
  dim3 gridDim(CEIL_DIV(N, 32), CEIL_DIV(M, 32));
  sgemm_naive<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

void run_sgemm_1d_blocktiling(int M, int N, int K, float alpha,
                               const float *A, const float *B, float beta,
                               float *C) {
  const int BM = 64;
  const int BN = 64;
  const int BK = 8;
  const int TM = 8;

  dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
  dim3 blockDim((BM * BN) / TM);

  sgemm_1d_blocktiling<BM, BN, BK, TM>
      <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

void run_sgemm_2d_blocktiling(int M, int N, int K, float alpha, float *A, float *B,
                           float beta, float *C) {
  const uint BK = 8;
  const uint TM = 8;
  const uint TN = 8;
  const uint BM = 128;
  const uint BN = 128;
  dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
  dim3 blockDim((BM * BN) / (TM * TN));
  sgemm2DBlocktiling<BM, BN, BK, TM, TN>
      <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

void run_sgemm_vectorize(int M, int N, int K, float alpha, float *A, float *B,
                         float beta, float *C) {
  // Optimal autotuned configuration
  const uint BM = 64;
  const uint BN = 128;
  const uint BK = 8;
  const uint TM = 8;
  const uint TN = 8;

  dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
  dim3 blockDim((BM * BN) / (TM * TN));
  sgemmVectorize<BM, BN, BK, TM, TN>
      <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

void run_cublas(cublasHandle_t handle, int M, int N, int K, float alpha,
                 const float *A, const float *B, float beta, float *C) {
  CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                            B, N, A, K, &beta, C, N));
}

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

void randomize_matrix(float *mat, int n) {
  for (int i = 0; i < n; ++i) {
    float v = (float)(rand() % 5) + 0.01f * (float)(rand() % 5);
    mat[i] = (rand() % 2 == 0) ? v : -v;
  }
}

bool verify_matrix(const float *ref, const float *out, int n,
                    double tol = 1e-1) {
  for (int i = 0; i < n; ++i) {
    double diff = std::fabs((double)ref[i] - (double)out[i]);
    if (std::isnan(diff) || diff > tol) {
      fprintf(stderr,
              "Mismatch at %d: ref=%f got=%f diff=%f\n", i, ref[i], out[i],
              diff);
      return false;
    }
  }
  return true;
}

double now_seconds() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(void)
{
    int M = 4096;
    int N = 4096;
    int K = 4096;

    printf("M=%d N=%d K=%d\n", M, N, K);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    size_t sizeA = (size_t)M * K * sizeof(float);
    size_t sizeB = (size_t)K * N * sizeof(float);
    size_t sizeC = (size_t)M * N * sizeof(float);

    float *hA = (float *)malloc(sizeA);
    float *hB = (float *)malloc(sizeB);
    float *hC = (float *)malloc(sizeC);
    float *hC_ref = (float *)malloc(sizeC);

    if (!hA || !hB || !hC || !hC_ref) {
        fprintf(stderr, "Host memory allocation failed\n");
        return EXIT_FAILURE;
    }

    srand(42);
    randomize_matrix(hA, M * K);
    randomize_matrix(hB, K * N);

    memset(hC, 0, sizeC);
    memset(hC_ref, 0, sizeC);

    float *dA, *dB, *dC, *dC_ref;

    CUDA_CHECK(cudaMalloc(&dA, sizeA));
    CUDA_CHECK(cudaMalloc(&dB, sizeB));
    CUDA_CHECK(cudaMalloc(&dC, sizeC));
    CUDA_CHECK(cudaMalloc(&dC_ref, sizeC));

    CUDA_CHECK(cudaMemcpy(dA, hA, sizeA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB, sizeB, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dC, hC, sizeC, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dC_ref, hC_ref, sizeC, cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    // Correctness Verification
    run_cublas(handle, M, N, K, alpha, dA, dB, beta, dC_ref);
    run_sgemm_vectorize(M, N, K, alpha, dA, dB, beta, dC);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(hC, dC, sizeC, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hC_ref, dC_ref, sizeC, cudaMemcpyDeviceToHost));

    if (!verify_matrix(hC_ref, hC, M * N)) {
        printf("Vectorized kernel failed correctness check!\n");
        return EXIT_FAILURE;
    }
    printf("Vectorized kernel passed correctness verification!\n");

    const int warmup = 3;
    const int iters = 20;

    auto time_kernel = [&](auto launch_fn) -> double {
        for (int i = 0; i < warmup; ++i) launch_fn();
        CUDA_CHECK(cudaDeviceSynchronize());

        double t0 = now_seconds();
        for (int i = 0; i < iters; ++i) launch_fn();
        CUDA_CHECK(cudaDeviceSynchronize());
        double t1 = now_seconds();

        return (t1 - t0) / iters;
    };

    double t_naive = time_kernel([&] { run_sgemm_naive(M, N, K, alpha, dA, dB, beta, dC); });
    double t_1d = time_kernel([&] { run_sgemm_1d_blocktiling(M, N, K, alpha, dA, dB, beta, dC); });
    double t_2d = time_kernel([&] { run_sgemm_2d_blocktiling(M, N, K, alpha, dA, dB, beta, dC); });
    double t_vectorize = time_kernel([&] { run_sgemm_vectorize(M, N, K, alpha, dA, dB, beta, dC); });
    double t_cublas = time_kernel([&] { run_cublas(handle, M, N, K, alpha, dA, dB, beta, dC_ref); });

    double flops = 2.0 * (double)M * (double)N * (double)K;

    printf("\n%-25s %10s %12s\n", "kernel", "time(ms)", "TFLOPS");
    printf("%-25s %10.3f %12.2f\n", "naive", t_naive * 1e3, flops / t_naive / 1e12);
    printf("%-25s %10.3f %12.2f\n", "1d_blocktiling", t_1d * 1e3, flops / t_1d / 1e12);
    printf("%-25s %10.3f %12.2f\n", "2d_blocktiling", t_2d * 1e3, flops / t_2d / 1e12);
    printf("%-25s %10.3f %12.2f\n", "vectorized (BM=64,BN=128)", t_vectorize * 1e3, flops / t_vectorize / 1e12);
    printf("%-25s %10.3f %12.2f\n", "cublas", t_cublas * 1e3, flops / t_cublas / 1e12);

    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    CUDA_CHECK(cudaFree(dC_ref));

    free(hA);
    free(hB);
    free(hC);
    free(hC_ref);

    return EXIT_SUCCESS;
}