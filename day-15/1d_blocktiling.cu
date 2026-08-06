#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

// ---------------------------------------------------------------------------
// Error checking helper
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t err__ = (call);                                              \
    if (err__ != cudaSuccess) {                                              \
      fprintf(stderr, "[CUDA ERROR] %s:%d: %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(err__));                                    \
      exit(EXIT_FAILURE);                                                    \
    }                                                                        \
  } while (0)

#define CUBLAS_CHECK(call)                                                   \
  do {                                                                       \
    cublasStatus_t st__ = (call);                                            \
    if (st__ != CUBLAS_STATUS_SUCCESS) {                                     \
      fprintf(stderr, "[CUBLAS ERROR] %s:%d: status %d\n", __FILE__,         \
              __LINE__, (int)st__);                                          \
      exit(EXIT_FAILURE);                                                    \
    }                                                                        \
  } while (0)

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

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
  // Which BM x BN tile of C this block owns.
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  // Compute-time thread mapping: threadCol sweeps the BN dimension,
  // threadRow indexes which TM-strip within BM this thread owns.
  const int threadCol = threadIdx.x % BN;
  const int threadRow = threadIdx.x / BN;

  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  // Advance the base pointers to the start of this block's data.
  A += cRow * BM * K;
  B += cCol * BN;
  C += cRow * BM * N + cCol * BN;

  // Load-time thread mapping (different from the compute-time mapping
  // above): chosen purely so consecutive threadIdx.x values touch
  // consecutive addresses in global memory, for coalescing.
  const uint innerColA = threadIdx.x % BK;
  const uint innerRowA = threadIdx.x / BK;
  const uint innerColB = threadIdx.x % BN;
  const uint innerRowB = threadIdx.x / BN;

  float threadResults[TM] = {0.0f};

  for (uint bkIdx = 0; bkIdx < (uint)K; bkIdx += BK) {
    // Cooperatively load the current BM x BK slice of A and BK x BN slice
    // of B into shared memory.
    As[innerRowA * BK + innerColA] = A[innerRowA * K + innerColA];
    Bs[innerRowB * BN + innerColB] = B[innerRowB * N + innerColB];
    __syncthreads();

    A += BK;
    B += BK * N;

    // dotIdx is the OUTER loop so that tmpB is loaded from shared memory
    // once and reused across all TM accumulations below, instead of being
    // re-fetched from Bs inside the inner loop.
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

void run_sgemm_1d_blocktiling(int M, int N, int K, float alpha,
                               const float *A, const float *B, float beta,
                               float *C) {
  const int BM = 64;
  const int BN = 64;
  const int BK = 8;
  const int TM = 8;

  static_assert(BM == BN, "This load scheme requires BM == BN");
  static_assert(BM % TM == 0, "BM must be divisible by TM");
  static_assert((BM * BK) % 1 == 0, "sanity");

  dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
  dim3 blockDim((BM * BN) / TM); // 64*64/8 = 512 threads

  sgemm_1d_blocktiling<BM, BN, BK, TM>
      <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

void run_sgemm_naive(int M, int N, int K, float alpha, const float *A,
                      const float *B, float beta, float *C) {
  dim3 blockDim(32, 32);
  dim3 gridDim(CEIL_DIV(N, 32), CEIL_DIV(M, 32));
  sgemm_naive<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

void run_cublas(cublasHandle_t handle, int M, int N, int K, float alpha,
                 const float *A, const float *B, float beta, float *C) {
  CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                            B, N, A, K, &beta, C, N));
}

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

int main(void)
{
	int M=4086, N = 4096, K=4096; 

	printf("M=%d N=%d K=%d\n", M, N, K);

	const float alpha = 1.0f, beta = 0.0f;

	size_t sizeA = (size_t)M * K * sizeof(float);
	size_t sizeB = (size_t)K * N * sizeof(float); 
	size_t sizeC = (size_t)M * N * sizeof(float);

	float *hA = (float *)malloc(sizeA);
	float *hB = (float *)malloc(sizeB);
	float *hC = (float *)malloc(sizeC);
	float *hC_ref = (float *)malloc(sizeC);// result of cuBLAS
	
	srand(42);
	randomize_matrix(hA, M*K);
	randomize_matrix(hB, K*N);
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

	run_cublas(handle, M, N, K, alpha, dA, dB, beta, dC_ref);
        run_sgemm_1d_blocktiling(M, N, K, alpha, dA, dB, beta, dC);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaGetLastError());

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

	double t_naive = time_kernel([&] {
	run_sgemm_naive(M, N, K, alpha, dA, dB, beta, dC);
	});
	double t_1d = time_kernel([&] {
	run_sgemm_1d_blocktiling(M, N, K, alpha, dA, dB, beta, dC);
	});
	double t_cublas = time_kernel([&] {
	run_cublas(handle, M, N, K, alpha, dA, dB, beta, dC_ref);
	});

	double flops = 2.0 * (double)M * (double)N * (double)K;
	printf("\n%-20s %10s %12s\n", "kernel", "time(ms)", "GFLOP/s");
	printf("%-20s %10.3f %12.1f\n", "naive", t_naive * 1e3,
	 flops / t_naive / 1e9);
	printf("%-20s %10.3f %12.1f\n", "1d_blocktiling", t_1d * 1e3,
	 flops / t_1d / 1e9);
	printf("%-20s %10.3f %12.1f\n", "cublas", t_cublas * 1e3,
	 flops / t_cublas / 1e9);

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
