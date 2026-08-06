#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <cmath>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error %s at line %d\n", cudaGetErrorString(err), __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t st = call; \
        if (st != CUBLAS_STATUS_SUCCESS) { \
            fprintf(stderr, "cuBLAS Error %d at line %d\n", (int)st, __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void __launch_bounds__((BM * BN) / (TM * TN), 1)
sgemm_vectorized(int M, int N, int K, float alpha, const float * __restrict__ A,
                 const float * __restrict__ B, float beta, float * __restrict__ C) {
    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;

    const uint numThreads = (BM * BN) / (TM * TN);
    const int threadCol = threadIdx.x % (BN / TN);
    const int threadRow = threadIdx.x / (BN / TN);

    __shared__ float As[BK][BM]; // Transposed layout for A
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

struct ConfigResult {
    const char* name;
    double ms;
    double tflops;
};

int main() {
    int M = 4096, N = 4096, K = 4096;
    size_t sizeA = (size_t)M * K * sizeof(float);
    size_t sizeB = (size_t)K * N * sizeof(float);
    size_t sizeC = (size_t)M * N * sizeof(float);

    float *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, sizeA));
    CUDA_CHECK(cudaMalloc(&dB, sizeB));
    CUDA_CHECK(cudaMalloc(&dC, sizeC));
    CUDA_CHECK(cudaMemset(dA, 1, sizeA));
    CUDA_CHECK(cudaMemset(dB, 1, sizeB));

    float alpha = 1.0f, beta = 0.0f;
    double flops = 2.0 * (double)M * (double)N * (double)K;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    std::vector<ConfigResult> results;

    {
        dim3 gridDim(CEIL_DIV(N, 32), CEIL_DIV(M, 32));
        dim3 blockDim((32 * 32) / (2 * 8));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<32, 32, 8, 2, 8><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<32, 32, 8, 2, 8><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=32_BN=32_BK=8_TM=2_TN=8", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 32), CEIL_DIV(M, 32));
        dim3 blockDim((32 * 32) / (4 * 4));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<32, 32, 8, 4, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<32, 32, 8, 4, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=32_BN=32_BK=8_TM=4_TN=4", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 32), CEIL_DIV(M, 32));
        dim3 blockDim((32 * 32) / (2 * 4));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<32, 32, 16, 2, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<32, 32, 16, 2, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=32_BN=32_BK=16_TM=2_TN=4", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 32), CEIL_DIV(M, 32));
        dim3 blockDim((32 * 32) / (2 * 8));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<32, 32, 16, 2, 8><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<32, 32, 16, 2, 8><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=32_BN=32_BK=16_TM=2_TN=8", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 32), CEIL_DIV(M, 32));
        dim3 blockDim((32 * 32) / (4 * 4));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<32, 32, 16, 4, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<32, 32, 16, 4, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=32_BN=32_BK=16_TM=4_TN=4", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 32), CEIL_DIV(M, 32));
        dim3 blockDim((32 * 32) / (2 * 4));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<32, 32, 32, 2, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<32, 32, 32, 2, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=32_BN=32_BK=32_TM=2_TN=4", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 32), CEIL_DIV(M, 32));
        dim3 blockDim((32 * 32) / (2 * 8));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<32, 32, 32, 2, 8><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<32, 32, 32, 2, 8><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=32_BN=32_BK=32_TM=2_TN=8", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 32), CEIL_DIV(M, 32));
        dim3 blockDim((32 * 32) / (4 * 4));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<32, 32, 32, 4, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<32, 32, 32, 4, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=32_BN=32_BK=32_TM=4_TN=4", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 64), CEIL_DIV(M, 32));
        dim3 blockDim((32 * 64) / (2 * 16));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<32, 64, 8, 2, 16><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<32, 64, 8, 2, 16><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=32_BN=64_BK=8_TM=2_TN=16", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 64), CEIL_DIV(M, 32));
        dim3 blockDim((32 * 64) / (4 * 8));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<32, 64, 8, 4, 8><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<32, 64, 8, 4, 8><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=32_BN=64_BK=8_TM=4_TN=8", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 64), CEIL_DIV(M, 32));
        dim3 blockDim((32 * 64) / (8 * 4));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<32, 64, 8, 8, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<32, 64, 8, 8, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=32_BN=64_BK=8_TM=8_TN=4", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 64), CEIL_DIV(M, 32));
        dim3 blockDim((32 * 64) / (2 * 8));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<32, 64, 16, 2, 8><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<32, 64, 16, 2, 8><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=32_BN=64_BK=16_TM=2_TN=8", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 64), CEIL_DIV(M, 32));
        dim3 blockDim((32 * 64) / (2 * 16));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<32, 64, 16, 2, 16><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<32, 64, 16, 2, 16><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=32_BN=64_BK=16_TM=2_TN=16", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 64), CEIL_DIV(M, 32));
        dim3 blockDim((32 * 64) / (4 * 4));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<32, 64, 16, 4, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<32, 64, 16, 4, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=32_BN=64_BK=16_TM=4_TN=4", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 64), CEIL_DIV(M, 32));
        dim3 blockDim((32 * 64) / (4 * 8));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<32, 64, 16, 4, 8><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<32, 64, 16, 4, 8><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=32_BN=64_BK=16_TM=4_TN=8", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 64), CEIL_DIV(M, 32));
        dim3 blockDim((32 * 64) / (8 * 4));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<32, 64, 16, 8, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<32, 64, 16, 8, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=32_BN=64_BK=16_TM=8_TN=4", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 64), CEIL_DIV(M, 32));
        dim3 blockDim((32 * 64) / (2 * 4));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<32, 64, 32, 2, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<32, 64, 32, 2, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=32_BN=64_BK=32_TM=2_TN=4", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 64), CEIL_DIV(M, 32));
        dim3 blockDim((32 * 64) / (2 * 8));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<32, 64, 32, 2, 8><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<32, 64, 32, 2, 8><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=32_BN=64_BK=32_TM=2_TN=8", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 64), CEIL_DIV(M, 32));
        dim3 blockDim((32 * 64) / (2 * 16));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<32, 64, 32, 2, 16><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<32, 64, 32, 2, 16><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=32_BN=64_BK=32_TM=2_TN=16", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 64), CEIL_DIV(M, 32));
        dim3 blockDim((32 * 64) / (4 * 4));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<32, 64, 32, 4, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<32, 64, 32, 4, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=32_BN=64_BK=32_TM=4_TN=4", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 64), CEIL_DIV(M, 32));
        dim3 blockDim((32 * 64) / (4 * 8));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<32, 64, 32, 4, 8><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<32, 64, 32, 4, 8><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=32_BN=64_BK=32_TM=4_TN=8", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 64), CEIL_DIV(M, 32));
        dim3 blockDim((32 * 64) / (8 * 4));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<32, 64, 32, 8, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<32, 64, 32, 8, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=32_BN=64_BK=32_TM=8_TN=4", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 128), CEIL_DIV(M, 32));
        dim3 blockDim((32 * 128) / (4 * 16));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<32, 128, 8, 4, 16><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<32, 128, 8, 4, 16><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=32_BN=128_BK=8_TM=4_TN=16", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 128), CEIL_DIV(M, 32));
        dim3 blockDim((32 * 128) / (8 * 8));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<32, 128, 8, 8, 8><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<32, 128, 8, 8, 8><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=32_BN=128_BK=8_TM=8_TN=8", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 128), CEIL_DIV(M, 32));
        dim3 blockDim((32 * 128) / (16 * 4));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<32, 128, 8, 16, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<32, 128, 8, 16, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=32_BN=128_BK=8_TM=16_TN=4", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 128), CEIL_DIV(M, 32));
        dim3 blockDim((32 * 128) / (2 * 16));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<32, 128, 16, 2, 16><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<32, 128, 16, 2, 16><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=32_BN=128_BK=16_TM=2_TN=16", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 128), CEIL_DIV(M, 32));
        dim3 blockDim((32 * 128) / (4 * 8));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<32, 128, 16, 4, 8><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<32, 128, 16, 4, 8><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=32_BN=128_BK=16_TM=4_TN=8", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 128), CEIL_DIV(M, 32));
        dim3 blockDim((32 * 128) / (4 * 16));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<32, 128, 16, 4, 16><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<32, 128, 16, 4, 16><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=32_BN=128_BK=16_TM=4_TN=16", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 128), CEIL_DIV(M, 32));
        dim3 blockDim((32 * 128) / (8 * 4));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<32, 128, 16, 8, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<32, 128, 16, 8, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=32_BN=128_BK=16_TM=8_TN=4", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 128), CEIL_DIV(M, 32));
        dim3 blockDim((32 * 128) / (8 * 8));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<32, 128, 16, 8, 8><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<32, 128, 16, 8, 8><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=32_BN=128_BK=16_TM=8_TN=8", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 128), CEIL_DIV(M, 32));
        dim3 blockDim((32 * 128) / (16 * 4));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<32, 128, 16, 16, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<32, 128, 16, 16, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=32_BN=128_BK=16_TM=16_TN=4", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 128), CEIL_DIV(M, 32));
        dim3 blockDim((32 * 128) / (2 * 8));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<32, 128, 32, 2, 8><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<32, 128, 32, 2, 8><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=32_BN=128_BK=32_TM=2_TN=8", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 128), CEIL_DIV(M, 32));
        dim3 blockDim((32 * 128) / (2 * 16));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<32, 128, 32, 2, 16><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<32, 128, 32, 2, 16><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=32_BN=128_BK=32_TM=2_TN=16", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 128), CEIL_DIV(M, 32));
        dim3 blockDim((32 * 128) / (4 * 4));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<32, 128, 32, 4, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<32, 128, 32, 4, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=32_BN=128_BK=32_TM=4_TN=4", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 128), CEIL_DIV(M, 32));
        dim3 blockDim((32 * 128) / (4 * 8));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<32, 128, 32, 4, 8><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<32, 128, 32, 4, 8><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=32_BN=128_BK=32_TM=4_TN=8", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 128), CEIL_DIV(M, 32));
        dim3 blockDim((32 * 128) / (4 * 16));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<32, 128, 32, 4, 16><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<32, 128, 32, 4, 16><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=32_BN=128_BK=32_TM=4_TN=16", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 128), CEIL_DIV(M, 32));
        dim3 blockDim((32 * 128) / (8 * 4));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<32, 128, 32, 8, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<32, 128, 32, 8, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=32_BN=128_BK=32_TM=8_TN=4", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 128), CEIL_DIV(M, 32));
        dim3 blockDim((32 * 128) / (8 * 8));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<32, 128, 32, 8, 8><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<32, 128, 32, 8, 8><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=32_BN=128_BK=32_TM=8_TN=8", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 128), CEIL_DIV(M, 32));
        dim3 blockDim((32 * 128) / (16 * 4));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<32, 128, 32, 16, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<32, 128, 32, 16, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=32_BN=128_BK=32_TM=16_TN=4", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 256), CEIL_DIV(M, 32));
        dim3 blockDim((32 * 256) / (4 * 16));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<32, 256, 16, 4, 16><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<32, 256, 16, 4, 16><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=32_BN=256_BK=16_TM=4_TN=16", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 256), CEIL_DIV(M, 32));
        dim3 blockDim((32 * 256) / (8 * 8));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<32, 256, 16, 8, 8><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<32, 256, 16, 8, 8><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=32_BN=256_BK=16_TM=8_TN=8", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 256), CEIL_DIV(M, 32));
        dim3 blockDim((32 * 256) / (16 * 4));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<32, 256, 16, 16, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<32, 256, 16, 16, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=32_BN=256_BK=16_TM=16_TN=4", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 256), CEIL_DIV(M, 32));
        dim3 blockDim((32 * 256) / (2 * 16));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<32, 256, 32, 2, 16><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<32, 256, 32, 2, 16><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=32_BN=256_BK=32_TM=2_TN=16", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 256), CEIL_DIV(M, 32));
        dim3 blockDim((32 * 256) / (4 * 8));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<32, 256, 32, 4, 8><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<32, 256, 32, 4, 8><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=32_BN=256_BK=32_TM=4_TN=8", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 256), CEIL_DIV(M, 32));
        dim3 blockDim((32 * 256) / (4 * 16));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<32, 256, 32, 4, 16><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<32, 256, 32, 4, 16><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=32_BN=256_BK=32_TM=4_TN=16", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 256), CEIL_DIV(M, 32));
        dim3 blockDim((32 * 256) / (8 * 4));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<32, 256, 32, 8, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<32, 256, 32, 8, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=32_BN=256_BK=32_TM=8_TN=4", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 256), CEIL_DIV(M, 32));
        dim3 blockDim((32 * 256) / (8 * 8));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<32, 256, 32, 8, 8><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<32, 256, 32, 8, 8><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=32_BN=256_BK=32_TM=8_TN=8", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 256), CEIL_DIV(M, 32));
        dim3 blockDim((32 * 256) / (16 * 4));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<32, 256, 32, 16, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<32, 256, 32, 16, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=32_BN=256_BK=32_TM=16_TN=4", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 32), CEIL_DIV(M, 64));
        dim3 blockDim((64 * 32) / (2 * 16));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<64, 32, 8, 2, 16><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<64, 32, 8, 2, 16><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=64_BN=32_BK=8_TM=2_TN=16", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 32), CEIL_DIV(M, 64));
        dim3 blockDim((64 * 32) / (4 * 8));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<64, 32, 8, 4, 8><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<64, 32, 8, 4, 8><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=64_BN=32_BK=8_TM=4_TN=8", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 32), CEIL_DIV(M, 64));
        dim3 blockDim((64 * 32) / (8 * 4));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<64, 32, 8, 8, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<64, 32, 8, 8, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=64_BN=32_BK=8_TM=8_TN=4", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 32), CEIL_DIV(M, 64));
        dim3 blockDim((64 * 32) / (2 * 8));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<64, 32, 16, 2, 8><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<64, 32, 16, 2, 8><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=64_BN=32_BK=16_TM=2_TN=8", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 32), CEIL_DIV(M, 64));
        dim3 blockDim((64 * 32) / (2 * 16));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<64, 32, 16, 2, 16><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<64, 32, 16, 2, 16><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=64_BN=32_BK=16_TM=2_TN=16", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 32), CEIL_DIV(M, 64));
        dim3 blockDim((64 * 32) / (4 * 4));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<64, 32, 16, 4, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<64, 32, 16, 4, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=64_BN=32_BK=16_TM=4_TN=4", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 32), CEIL_DIV(M, 64));
        dim3 blockDim((64 * 32) / (4 * 8));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<64, 32, 16, 4, 8><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<64, 32, 16, 4, 8><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=64_BN=32_BK=16_TM=4_TN=8", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 32), CEIL_DIV(M, 64));
        dim3 blockDim((64 * 32) / (8 * 4));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<64, 32, 16, 8, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<64, 32, 16, 8, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=64_BN=32_BK=16_TM=8_TN=4", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 32), CEIL_DIV(M, 64));
        dim3 blockDim((64 * 32) / (2 * 4));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<64, 32, 32, 2, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<64, 32, 32, 2, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=64_BN=32_BK=32_TM=2_TN=4", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 32), CEIL_DIV(M, 64));
        dim3 blockDim((64 * 32) / (2 * 8));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<64, 32, 32, 2, 8><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<64, 32, 32, 2, 8><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=64_BN=32_BK=32_TM=2_TN=8", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 32), CEIL_DIV(M, 64));
        dim3 blockDim((64 * 32) / (2 * 16));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<64, 32, 32, 2, 16><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<64, 32, 32, 2, 16><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=64_BN=32_BK=32_TM=2_TN=16", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 32), CEIL_DIV(M, 64));
        dim3 blockDim((64 * 32) / (4 * 4));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<64, 32, 32, 4, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<64, 32, 32, 4, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=64_BN=32_BK=32_TM=4_TN=4", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 32), CEIL_DIV(M, 64));
        dim3 blockDim((64 * 32) / (4 * 8));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<64, 32, 32, 4, 8><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<64, 32, 32, 4, 8><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=64_BN=32_BK=32_TM=4_TN=8", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 32), CEIL_DIV(M, 64));
        dim3 blockDim((64 * 32) / (8 * 4));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<64, 32, 32, 8, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<64, 32, 32, 8, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=64_BN=32_BK=32_TM=8_TN=4", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 64), CEIL_DIV(M, 64));
        dim3 blockDim((64 * 64) / (2 * 16));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<64, 64, 8, 2, 16><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<64, 64, 8, 2, 16><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=64_BN=64_BK=8_TM=2_TN=16", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 64), CEIL_DIV(M, 64));
        dim3 blockDim((64 * 64) / (4 * 8));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<64, 64, 8, 4, 8><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<64, 64, 8, 4, 8><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=64_BN=64_BK=8_TM=4_TN=8", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 64), CEIL_DIV(M, 64));
        dim3 blockDim((64 * 64) / (4 * 16));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<64, 64, 8, 4, 16><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<64, 64, 8, 4, 16><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=64_BN=64_BK=8_TM=4_TN=16", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 64), CEIL_DIV(M, 64));
        dim3 blockDim((64 * 64) / (8 * 4));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<64, 64, 8, 8, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<64, 64, 8, 8, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=64_BN=64_BK=8_TM=8_TN=4", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 64), CEIL_DIV(M, 64));
        dim3 blockDim((64 * 64) / (8 * 8));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<64, 64, 8, 8, 8><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<64, 64, 8, 8, 8><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=64_BN=64_BK=8_TM=8_TN=8", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 64), CEIL_DIV(M, 64));
        dim3 blockDim((64 * 64) / (16 * 4));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<64, 64, 8, 16, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<64, 64, 8, 16, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=64_BN=64_BK=8_TM=16_TN=4", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 64), CEIL_DIV(M, 64));
        dim3 blockDim((64 * 64) / (2 * 8));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<64, 64, 16, 2, 8><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<64, 64, 16, 2, 8><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=64_BN=64_BK=16_TM=2_TN=8", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 64), CEIL_DIV(M, 64));
        dim3 blockDim((64 * 64) / (2 * 16));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<64, 64, 16, 2, 16><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<64, 64, 16, 2, 16><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=64_BN=64_BK=16_TM=2_TN=16", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 64), CEIL_DIV(M, 64));
        dim3 blockDim((64 * 64) / (4 * 4));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<64, 64, 16, 4, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<64, 64, 16, 4, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=64_BN=64_BK=16_TM=4_TN=4", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 64), CEIL_DIV(M, 64));
        dim3 blockDim((64 * 64) / (4 * 8));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<64, 64, 16, 4, 8><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<64, 64, 16, 4, 8><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=64_BN=64_BK=16_TM=4_TN=8", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 64), CEIL_DIV(M, 64));
        dim3 blockDim((64 * 64) / (4 * 16));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<64, 64, 16, 4, 16><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<64, 64, 16, 4, 16><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=64_BN=64_BK=16_TM=4_TN=16", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 64), CEIL_DIV(M, 64));
        dim3 blockDim((64 * 64) / (8 * 4));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<64, 64, 16, 8, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<64, 64, 16, 8, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=64_BN=64_BK=16_TM=8_TN=4", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 64), CEIL_DIV(M, 64));
        dim3 blockDim((64 * 64) / (8 * 8));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<64, 64, 16, 8, 8><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<64, 64, 16, 8, 8><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=64_BN=64_BK=16_TM=8_TN=8", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 64), CEIL_DIV(M, 64));
        dim3 blockDim((64 * 64) / (16 * 4));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<64, 64, 16, 16, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<64, 64, 16, 16, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=64_BN=64_BK=16_TM=16_TN=4", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 64), CEIL_DIV(M, 64));
        dim3 blockDim((64 * 64) / (2 * 4));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<64, 64, 32, 2, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<64, 64, 32, 2, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=64_BN=64_BK=32_TM=2_TN=4", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 64), CEIL_DIV(M, 64));
        dim3 blockDim((64 * 64) / (2 * 8));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<64, 64, 32, 2, 8><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<64, 64, 32, 2, 8><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=64_BN=64_BK=32_TM=2_TN=8", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 64), CEIL_DIV(M, 64));
        dim3 blockDim((64 * 64) / (2 * 16));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<64, 64, 32, 2, 16><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<64, 64, 32, 2, 16><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=64_BN=64_BK=32_TM=2_TN=16", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 64), CEIL_DIV(M, 64));
        dim3 blockDim((64 * 64) / (4 * 4));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<64, 64, 32, 4, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<64, 64, 32, 4, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=64_BN=64_BK=32_TM=4_TN=4", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 64), CEIL_DIV(M, 64));
        dim3 blockDim((64 * 64) / (4 * 8));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<64, 64, 32, 4, 8><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<64, 64, 32, 4, 8><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=64_BN=64_BK=32_TM=4_TN=8", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 64), CEIL_DIV(M, 64));
        dim3 blockDim((64 * 64) / (4 * 16));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<64, 64, 32, 4, 16><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<64, 64, 32, 4, 16><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=64_BN=64_BK=32_TM=4_TN=16", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 64), CEIL_DIV(M, 64));
        dim3 blockDim((64 * 64) / (8 * 4));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<64, 64, 32, 8, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<64, 64, 32, 8, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=64_BN=64_BK=32_TM=8_TN=4", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 64), CEIL_DIV(M, 64));
        dim3 blockDim((64 * 64) / (8 * 8));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<64, 64, 32, 8, 8><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<64, 64, 32, 8, 8><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=64_BN=64_BK=32_TM=8_TN=8", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 64), CEIL_DIV(M, 64));
        dim3 blockDim((64 * 64) / (16 * 4));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<64, 64, 32, 16, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<64, 64, 32, 16, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=64_BN=64_BK=32_TM=16_TN=4", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 128), CEIL_DIV(M, 64));
        dim3 blockDim((64 * 128) / (4 * 16));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<64, 128, 8, 4, 16><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<64, 128, 8, 4, 16><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=64_BN=128_BK=8_TM=4_TN=16", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 128), CEIL_DIV(M, 64));
        dim3 blockDim((64 * 128) / (8 * 8));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<64, 128, 8, 8, 8><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<64, 128, 8, 8, 8><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=64_BN=128_BK=8_TM=8_TN=8", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 128), CEIL_DIV(M, 64));
        dim3 blockDim((64 * 128) / (16 * 4));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<64, 128, 8, 16, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<64, 128, 8, 16, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=64_BN=128_BK=8_TM=16_TN=4", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 128), CEIL_DIV(M, 64));
        dim3 blockDim((64 * 128) / (2 * 16));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<64, 128, 16, 2, 16><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<64, 128, 16, 2, 16><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=64_BN=128_BK=16_TM=2_TN=16", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 128), CEIL_DIV(M, 64));
        dim3 blockDim((64 * 128) / (4 * 8));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<64, 128, 16, 4, 8><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<64, 128, 16, 4, 8><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=64_BN=128_BK=16_TM=4_TN=8", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 128), CEIL_DIV(M, 64));
        dim3 blockDim((64 * 128) / (4 * 16));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<64, 128, 16, 4, 16><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<64, 128, 16, 4, 16><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=64_BN=128_BK=16_TM=4_TN=16", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 128), CEIL_DIV(M, 64));
        dim3 blockDim((64 * 128) / (8 * 4));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<64, 128, 16, 8, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<64, 128, 16, 8, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=64_BN=128_BK=16_TM=8_TN=4", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 128), CEIL_DIV(M, 64));
        dim3 blockDim((64 * 128) / (8 * 8));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<64, 128, 16, 8, 8><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<64, 128, 16, 8, 8><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=64_BN=128_BK=16_TM=8_TN=8", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 128), CEIL_DIV(M, 64));
        dim3 blockDim((64 * 128) / (16 * 4));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<64, 128, 16, 16, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<64, 128, 16, 16, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=64_BN=128_BK=16_TM=16_TN=4", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 128), CEIL_DIV(M, 64));
        dim3 blockDim((64 * 128) / (2 * 8));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<64, 128, 32, 2, 8><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<64, 128, 32, 2, 8><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=64_BN=128_BK=32_TM=2_TN=8", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 128), CEIL_DIV(M, 64));
        dim3 blockDim((64 * 128) / (2 * 16));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<64, 128, 32, 2, 16><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<64, 128, 32, 2, 16><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=64_BN=128_BK=32_TM=2_TN=16", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 128), CEIL_DIV(M, 64));
        dim3 blockDim((64 * 128) / (4 * 4));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<64, 128, 32, 4, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<64, 128, 32, 4, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=64_BN=128_BK=32_TM=4_TN=4", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 128), CEIL_DIV(M, 64));
        dim3 blockDim((64 * 128) / (4 * 8));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<64, 128, 32, 4, 8><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<64, 128, 32, 4, 8><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=64_BN=128_BK=32_TM=4_TN=8", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 128), CEIL_DIV(M, 64));
        dim3 blockDim((64 * 128) / (4 * 16));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<64, 128, 32, 4, 16><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<64, 128, 32, 4, 16><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=64_BN=128_BK=32_TM=4_TN=16", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 128), CEIL_DIV(M, 64));
        dim3 blockDim((64 * 128) / (8 * 4));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<64, 128, 32, 8, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<64, 128, 32, 8, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=64_BN=128_BK=32_TM=8_TN=4", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 128), CEIL_DIV(M, 64));
        dim3 blockDim((64 * 128) / (8 * 8));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<64, 128, 32, 8, 8><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<64, 128, 32, 8, 8><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=64_BN=128_BK=32_TM=8_TN=8", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 128), CEIL_DIV(M, 64));
        dim3 blockDim((64 * 128) / (16 * 4));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<64, 128, 32, 16, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<64, 128, 32, 16, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=64_BN=128_BK=32_TM=16_TN=4", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 256), CEIL_DIV(M, 64));
        dim3 blockDim((64 * 256) / (4 * 16));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<64, 256, 16, 4, 16><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<64, 256, 16, 4, 16><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=64_BN=256_BK=16_TM=4_TN=16", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 256), CEIL_DIV(M, 64));
        dim3 blockDim((64 * 256) / (8 * 8));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<64, 256, 16, 8, 8><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<64, 256, 16, 8, 8><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=64_BN=256_BK=16_TM=8_TN=8", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 256), CEIL_DIV(M, 64));
        dim3 blockDim((64 * 256) / (16 * 4));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<64, 256, 16, 16, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<64, 256, 16, 16, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=64_BN=256_BK=16_TM=16_TN=4", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 256), CEIL_DIV(M, 64));
        dim3 blockDim((64 * 256) / (2 * 16));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<64, 256, 32, 2, 16><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<64, 256, 32, 2, 16><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=64_BN=256_BK=32_TM=2_TN=16", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 256), CEIL_DIV(M, 64));
        dim3 blockDim((64 * 256) / (4 * 8));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<64, 256, 32, 4, 8><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<64, 256, 32, 4, 8><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=64_BN=256_BK=32_TM=4_TN=8", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 256), CEIL_DIV(M, 64));
        dim3 blockDim((64 * 256) / (4 * 16));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<64, 256, 32, 4, 16><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<64, 256, 32, 4, 16><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=64_BN=256_BK=32_TM=4_TN=16", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 256), CEIL_DIV(M, 64));
        dim3 blockDim((64 * 256) / (8 * 4));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<64, 256, 32, 8, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<64, 256, 32, 8, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=64_BN=256_BK=32_TM=8_TN=4", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 256), CEIL_DIV(M, 64));
        dim3 blockDim((64 * 256) / (8 * 8));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<64, 256, 32, 8, 8><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<64, 256, 32, 8, 8><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=64_BN=256_BK=32_TM=8_TN=8", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 256), CEIL_DIV(M, 64));
        dim3 blockDim((64 * 256) / (16 * 4));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<64, 256, 32, 16, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<64, 256, 32, 16, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=64_BN=256_BK=32_TM=16_TN=4", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 32), CEIL_DIV(M, 128));
        dim3 blockDim((128 * 32) / (4 * 16));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<128, 32, 8, 4, 16><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<128, 32, 8, 4, 16><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=128_BN=32_BK=8_TM=4_TN=16", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 32), CEIL_DIV(M, 128));
        dim3 blockDim((128 * 32) / (8 * 8));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<128, 32, 8, 8, 8><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<128, 32, 8, 8, 8><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=128_BN=32_BK=8_TM=8_TN=8", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 32), CEIL_DIV(M, 128));
        dim3 blockDim((128 * 32) / (16 * 4));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<128, 32, 8, 16, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<128, 32, 8, 16, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=128_BN=32_BK=8_TM=16_TN=4", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 32), CEIL_DIV(M, 128));
        dim3 blockDim((128 * 32) / (2 * 16));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<128, 32, 16, 2, 16><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<128, 32, 16, 2, 16><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=128_BN=32_BK=16_TM=2_TN=16", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 32), CEIL_DIV(M, 128));
        dim3 blockDim((128 * 32) / (4 * 8));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<128, 32, 16, 4, 8><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<128, 32, 16, 4, 8><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=128_BN=32_BK=16_TM=4_TN=8", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 32), CEIL_DIV(M, 128));
        dim3 blockDim((128 * 32) / (4 * 16));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<128, 32, 16, 4, 16><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<128, 32, 16, 4, 16><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=128_BN=32_BK=16_TM=4_TN=16", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 32), CEIL_DIV(M, 128));
        dim3 blockDim((128 * 32) / (8 * 4));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<128, 32, 16, 8, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<128, 32, 16, 8, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=128_BN=32_BK=16_TM=8_TN=4", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 32), CEIL_DIV(M, 128));
        dim3 blockDim((128 * 32) / (8 * 8));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<128, 32, 16, 8, 8><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<128, 32, 16, 8, 8><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=128_BN=32_BK=16_TM=8_TN=8", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 32), CEIL_DIV(M, 128));
        dim3 blockDim((128 * 32) / (16 * 4));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<128, 32, 16, 16, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<128, 32, 16, 16, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=128_BN=32_BK=16_TM=16_TN=4", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 32), CEIL_DIV(M, 128));
        dim3 blockDim((128 * 32) / (2 * 8));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<128, 32, 32, 2, 8><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<128, 32, 32, 2, 8><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=128_BN=32_BK=32_TM=2_TN=8", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 32), CEIL_DIV(M, 128));
        dim3 blockDim((128 * 32) / (2 * 16));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<128, 32, 32, 2, 16><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<128, 32, 32, 2, 16><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=128_BN=32_BK=32_TM=2_TN=16", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 32), CEIL_DIV(M, 128));
        dim3 blockDim((128 * 32) / (4 * 4));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<128, 32, 32, 4, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<128, 32, 32, 4, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=128_BN=32_BK=32_TM=4_TN=4", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 32), CEIL_DIV(M, 128));
        dim3 blockDim((128 * 32) / (4 * 8));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<128, 32, 32, 4, 8><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<128, 32, 32, 4, 8><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=128_BN=32_BK=32_TM=4_TN=8", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 32), CEIL_DIV(M, 128));
        dim3 blockDim((128 * 32) / (4 * 16));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<128, 32, 32, 4, 16><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<128, 32, 32, 4, 16><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=128_BN=32_BK=32_TM=4_TN=16", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 32), CEIL_DIV(M, 128));
        dim3 blockDim((128 * 32) / (8 * 4));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<128, 32, 32, 8, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<128, 32, 32, 8, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=128_BN=32_BK=32_TM=8_TN=4", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 32), CEIL_DIV(M, 128));
        dim3 blockDim((128 * 32) / (8 * 8));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<128, 32, 32, 8, 8><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<128, 32, 32, 8, 8><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=128_BN=32_BK=32_TM=8_TN=8", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 32), CEIL_DIV(M, 128));
        dim3 blockDim((128 * 32) / (16 * 4));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<128, 32, 32, 16, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<128, 32, 32, 16, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=128_BN=32_BK=32_TM=16_TN=4", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 64), CEIL_DIV(M, 128));
        dim3 blockDim((128 * 64) / (4 * 16));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<128, 64, 8, 4, 16><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<128, 64, 8, 4, 16><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=128_BN=64_BK=8_TM=4_TN=16", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 64), CEIL_DIV(M, 128));
        dim3 blockDim((128 * 64) / (8 * 8));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<128, 64, 8, 8, 8><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<128, 64, 8, 8, 8><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=128_BN=64_BK=8_TM=8_TN=8", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 64), CEIL_DIV(M, 128));
        dim3 blockDim((128 * 64) / (16 * 4));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<128, 64, 8, 16, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<128, 64, 8, 16, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=128_BN=64_BK=8_TM=16_TN=4", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 64), CEIL_DIV(M, 128));
        dim3 blockDim((128 * 64) / (2 * 16));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<128, 64, 16, 2, 16><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<128, 64, 16, 2, 16><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=128_BN=64_BK=16_TM=2_TN=16", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 64), CEIL_DIV(M, 128));
        dim3 blockDim((128 * 64) / (4 * 8));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<128, 64, 16, 4, 8><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<128, 64, 16, 4, 8><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=128_BN=64_BK=16_TM=4_TN=8", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 64), CEIL_DIV(M, 128));
        dim3 blockDim((128 * 64) / (4 * 16));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<128, 64, 16, 4, 16><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<128, 64, 16, 4, 16><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=128_BN=64_BK=16_TM=4_TN=16", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 64), CEIL_DIV(M, 128));
        dim3 blockDim((128 * 64) / (8 * 4));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<128, 64, 16, 8, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<128, 64, 16, 8, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=128_BN=64_BK=16_TM=8_TN=4", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 64), CEIL_DIV(M, 128));
        dim3 blockDim((128 * 64) / (8 * 8));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<128, 64, 16, 8, 8><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<128, 64, 16, 8, 8><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=128_BN=64_BK=16_TM=8_TN=8", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 64), CEIL_DIV(M, 128));
        dim3 blockDim((128 * 64) / (16 * 4));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<128, 64, 16, 16, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<128, 64, 16, 16, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=128_BN=64_BK=16_TM=16_TN=4", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 64), CEIL_DIV(M, 128));
        dim3 blockDim((128 * 64) / (2 * 8));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<128, 64, 32, 2, 8><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<128, 64, 32, 2, 8><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=128_BN=64_BK=32_TM=2_TN=8", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 64), CEIL_DIV(M, 128));
        dim3 blockDim((128 * 64) / (2 * 16));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<128, 64, 32, 2, 16><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<128, 64, 32, 2, 16><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=128_BN=64_BK=32_TM=2_TN=16", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 64), CEIL_DIV(M, 128));
        dim3 blockDim((128 * 64) / (4 * 4));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<128, 64, 32, 4, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<128, 64, 32, 4, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=128_BN=64_BK=32_TM=4_TN=4", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 64), CEIL_DIV(M, 128));
        dim3 blockDim((128 * 64) / (4 * 8));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<128, 64, 32, 4, 8><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<128, 64, 32, 4, 8><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=128_BN=64_BK=32_TM=4_TN=8", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 64), CEIL_DIV(M, 128));
        dim3 blockDim((128 * 64) / (4 * 16));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<128, 64, 32, 4, 16><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<128, 64, 32, 4, 16><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=128_BN=64_BK=32_TM=4_TN=16", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 64), CEIL_DIV(M, 128));
        dim3 blockDim((128 * 64) / (8 * 4));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<128, 64, 32, 8, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<128, 64, 32, 8, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=128_BN=64_BK=32_TM=8_TN=4", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 64), CEIL_DIV(M, 128));
        dim3 blockDim((128 * 64) / (8 * 8));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<128, 64, 32, 8, 8><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<128, 64, 32, 8, 8><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=128_BN=64_BK=32_TM=8_TN=8", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 64), CEIL_DIV(M, 128));
        dim3 blockDim((128 * 64) / (16 * 4));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<128, 64, 32, 16, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<128, 64, 32, 16, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=128_BN=64_BK=32_TM=16_TN=4", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 128), CEIL_DIV(M, 128));
        dim3 blockDim((128 * 128) / (4 * 16));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<128, 128, 8, 4, 16><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<128, 128, 8, 4, 16><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=128_BN=128_BK=8_TM=4_TN=16", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 128), CEIL_DIV(M, 128));
        dim3 blockDim((128 * 128) / (8 * 8));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<128, 128, 8, 8, 8><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<128, 128, 8, 8, 8><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=128_BN=128_BK=8_TM=8_TN=8", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 128), CEIL_DIV(M, 128));
        dim3 blockDim((128 * 128) / (16 * 4));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<128, 128, 8, 16, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<128, 128, 8, 16, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=128_BN=128_BK=8_TM=16_TN=4", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 128), CEIL_DIV(M, 128));
        dim3 blockDim((128 * 128) / (2 * 16));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<128, 128, 16, 2, 16><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<128, 128, 16, 2, 16><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=128_BN=128_BK=16_TM=2_TN=16", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 128), CEIL_DIV(M, 128));
        dim3 blockDim((128 * 128) / (4 * 8));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<128, 128, 16, 4, 8><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<128, 128, 16, 4, 8><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=128_BN=128_BK=16_TM=4_TN=8", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 128), CEIL_DIV(M, 128));
        dim3 blockDim((128 * 128) / (4 * 16));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<128, 128, 16, 4, 16><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<128, 128, 16, 4, 16><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=128_BN=128_BK=16_TM=4_TN=16", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 128), CEIL_DIV(M, 128));
        dim3 blockDim((128 * 128) / (8 * 4));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<128, 128, 16, 8, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<128, 128, 16, 8, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=128_BN=128_BK=16_TM=8_TN=4", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 128), CEIL_DIV(M, 128));
        dim3 blockDim((128 * 128) / (8 * 8));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<128, 128, 16, 8, 8><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<128, 128, 16, 8, 8><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=128_BN=128_BK=16_TM=8_TN=8", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 128), CEIL_DIV(M, 128));
        dim3 blockDim((128 * 128) / (16 * 4));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<128, 128, 16, 16, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<128, 128, 16, 16, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=128_BN=128_BK=16_TM=16_TN=4", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 128), CEIL_DIV(M, 128));
        dim3 blockDim((128 * 128) / (2 * 8));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<128, 128, 32, 2, 8><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<128, 128, 32, 2, 8><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=128_BN=128_BK=32_TM=2_TN=8", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 128), CEIL_DIV(M, 128));
        dim3 blockDim((128 * 128) / (2 * 16));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<128, 128, 32, 2, 16><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<128, 128, 32, 2, 16><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=128_BN=128_BK=32_TM=2_TN=16", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 128), CEIL_DIV(M, 128));
        dim3 blockDim((128 * 128) / (4 * 4));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<128, 128, 32, 4, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<128, 128, 32, 4, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=128_BN=128_BK=32_TM=4_TN=4", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 128), CEIL_DIV(M, 128));
        dim3 blockDim((128 * 128) / (4 * 8));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<128, 128, 32, 4, 8><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<128, 128, 32, 4, 8><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=128_BN=128_BK=32_TM=4_TN=8", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 128), CEIL_DIV(M, 128));
        dim3 blockDim((128 * 128) / (4 * 16));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<128, 128, 32, 4, 16><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<128, 128, 32, 4, 16><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=128_BN=128_BK=32_TM=4_TN=16", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 128), CEIL_DIV(M, 128));
        dim3 blockDim((128 * 128) / (8 * 4));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<128, 128, 32, 8, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<128, 128, 32, 8, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=128_BN=128_BK=32_TM=8_TN=4", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 128), CEIL_DIV(M, 128));
        dim3 blockDim((128 * 128) / (8 * 8));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<128, 128, 32, 8, 8><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<128, 128, 32, 8, 8><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=128_BN=128_BK=32_TM=8_TN=8", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 128), CEIL_DIV(M, 128));
        dim3 blockDim((128 * 128) / (16 * 4));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<128, 128, 32, 16, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<128, 128, 32, 16, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=128_BN=128_BK=32_TM=16_TN=4", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 256), CEIL_DIV(M, 128));
        dim3 blockDim((128 * 256) / (4 * 16));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<128, 256, 16, 4, 16><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<128, 256, 16, 4, 16><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=128_BN=256_BK=16_TM=4_TN=16", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 256), CEIL_DIV(M, 128));
        dim3 blockDim((128 * 256) / (8 * 8));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<128, 256, 16, 8, 8><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<128, 256, 16, 8, 8><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=128_BN=256_BK=16_TM=8_TN=8", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 256), CEIL_DIV(M, 128));
        dim3 blockDim((128 * 256) / (16 * 4));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<128, 256, 16, 16, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<128, 256, 16, 16, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=128_BN=256_BK=16_TM=16_TN=4", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 256), CEIL_DIV(M, 128));
        dim3 blockDim((128 * 256) / (2 * 16));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<128, 256, 32, 2, 16><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<128, 256, 32, 2, 16><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=128_BN=256_BK=32_TM=2_TN=16", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 256), CEIL_DIV(M, 128));
        dim3 blockDim((128 * 256) / (4 * 8));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<128, 256, 32, 4, 8><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<128, 256, 32, 4, 8><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=128_BN=256_BK=32_TM=4_TN=8", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 256), CEIL_DIV(M, 128));
        dim3 blockDim((128 * 256) / (4 * 16));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<128, 256, 32, 4, 16><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<128, 256, 32, 4, 16><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=128_BN=256_BK=32_TM=4_TN=16", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 256), CEIL_DIV(M, 128));
        dim3 blockDim((128 * 256) / (8 * 4));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<128, 256, 32, 8, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<128, 256, 32, 8, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=128_BN=256_BK=32_TM=8_TN=4", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 256), CEIL_DIV(M, 128));
        dim3 blockDim((128 * 256) / (8 * 8));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<128, 256, 32, 8, 8><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<128, 256, 32, 8, 8><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=128_BN=256_BK=32_TM=8_TN=8", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 256), CEIL_DIV(M, 128));
        dim3 blockDim((128 * 256) / (16 * 4));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<128, 256, 32, 16, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<128, 256, 32, 16, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=128_BN=256_BK=32_TM=16_TN=4", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 32), CEIL_DIV(M, 256));
        dim3 blockDim((256 * 32) / (4 * 16));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<256, 32, 16, 4, 16><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<256, 32, 16, 4, 16><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=256_BN=32_BK=16_TM=4_TN=16", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 32), CEIL_DIV(M, 256));
        dim3 blockDim((256 * 32) / (8 * 8));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<256, 32, 16, 8, 8><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<256, 32, 16, 8, 8><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=256_BN=32_BK=16_TM=8_TN=8", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 32), CEIL_DIV(M, 256));
        dim3 blockDim((256 * 32) / (16 * 4));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<256, 32, 16, 16, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<256, 32, 16, 16, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=256_BN=32_BK=16_TM=16_TN=4", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 32), CEIL_DIV(M, 256));
        dim3 blockDim((256 * 32) / (2 * 16));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<256, 32, 32, 2, 16><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<256, 32, 32, 2, 16><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=256_BN=32_BK=32_TM=2_TN=16", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 32), CEIL_DIV(M, 256));
        dim3 blockDim((256 * 32) / (4 * 8));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<256, 32, 32, 4, 8><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<256, 32, 32, 4, 8><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=256_BN=32_BK=32_TM=4_TN=8", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 32), CEIL_DIV(M, 256));
        dim3 blockDim((256 * 32) / (4 * 16));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<256, 32, 32, 4, 16><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<256, 32, 32, 4, 16><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=256_BN=32_BK=32_TM=4_TN=16", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 32), CEIL_DIV(M, 256));
        dim3 blockDim((256 * 32) / (8 * 4));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<256, 32, 32, 8, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<256, 32, 32, 8, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=256_BN=32_BK=32_TM=8_TN=4", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 32), CEIL_DIV(M, 256));
        dim3 blockDim((256 * 32) / (8 * 8));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<256, 32, 32, 8, 8><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<256, 32, 32, 8, 8><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=256_BN=32_BK=32_TM=8_TN=8", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 32), CEIL_DIV(M, 256));
        dim3 blockDim((256 * 32) / (16 * 4));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<256, 32, 32, 16, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<256, 32, 32, 16, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=256_BN=32_BK=32_TM=16_TN=4", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 64), CEIL_DIV(M, 256));
        dim3 blockDim((256 * 64) / (4 * 16));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<256, 64, 16, 4, 16><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<256, 64, 16, 4, 16><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=256_BN=64_BK=16_TM=4_TN=16", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 64), CEIL_DIV(M, 256));
        dim3 blockDim((256 * 64) / (8 * 8));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<256, 64, 16, 8, 8><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<256, 64, 16, 8, 8><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=256_BN=64_BK=16_TM=8_TN=8", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 64), CEIL_DIV(M, 256));
        dim3 blockDim((256 * 64) / (16 * 4));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<256, 64, 16, 16, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<256, 64, 16, 16, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=256_BN=64_BK=16_TM=16_TN=4", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 64), CEIL_DIV(M, 256));
        dim3 blockDim((256 * 64) / (2 * 16));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<256, 64, 32, 2, 16><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<256, 64, 32, 2, 16><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=256_BN=64_BK=32_TM=2_TN=16", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 64), CEIL_DIV(M, 256));
        dim3 blockDim((256 * 64) / (4 * 8));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<256, 64, 32, 4, 8><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<256, 64, 32, 4, 8><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=256_BN=64_BK=32_TM=4_TN=8", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 64), CEIL_DIV(M, 256));
        dim3 blockDim((256 * 64) / (4 * 16));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<256, 64, 32, 4, 16><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<256, 64, 32, 4, 16><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=256_BN=64_BK=32_TM=4_TN=16", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 64), CEIL_DIV(M, 256));
        dim3 blockDim((256 * 64) / (8 * 4));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<256, 64, 32, 8, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<256, 64, 32, 8, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=256_BN=64_BK=32_TM=8_TN=4", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 64), CEIL_DIV(M, 256));
        dim3 blockDim((256 * 64) / (8 * 8));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<256, 64, 32, 8, 8><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<256, 64, 32, 8, 8><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=256_BN=64_BK=32_TM=8_TN=8", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 64), CEIL_DIV(M, 256));
        dim3 blockDim((256 * 64) / (16 * 4));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<256, 64, 32, 16, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<256, 64, 32, 16, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=256_BN=64_BK=32_TM=16_TN=4", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 128), CEIL_DIV(M, 256));
        dim3 blockDim((256 * 128) / (4 * 16));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<256, 128, 16, 4, 16><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<256, 128, 16, 4, 16><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=256_BN=128_BK=16_TM=4_TN=16", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 128), CEIL_DIV(M, 256));
        dim3 blockDim((256 * 128) / (8 * 8));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<256, 128, 16, 8, 8><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<256, 128, 16, 8, 8><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=256_BN=128_BK=16_TM=8_TN=8", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 128), CEIL_DIV(M, 256));
        dim3 blockDim((256 * 128) / (16 * 4));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<256, 128, 16, 16, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<256, 128, 16, 16, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=256_BN=128_BK=16_TM=16_TN=4", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 128), CEIL_DIV(M, 256));
        dim3 blockDim((256 * 128) / (2 * 16));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<256, 128, 32, 2, 16><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<256, 128, 32, 2, 16><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=256_BN=128_BK=32_TM=2_TN=16", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 128), CEIL_DIV(M, 256));
        dim3 blockDim((256 * 128) / (4 * 8));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<256, 128, 32, 4, 8><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<256, 128, 32, 4, 8><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=256_BN=128_BK=32_TM=4_TN=8", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 128), CEIL_DIV(M, 256));
        dim3 blockDim((256 * 128) / (4 * 16));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<256, 128, 32, 4, 16><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<256, 128, 32, 4, 16><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=256_BN=128_BK=32_TM=4_TN=16", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 128), CEIL_DIV(M, 256));
        dim3 blockDim((256 * 128) / (8 * 4));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<256, 128, 32, 8, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<256, 128, 32, 8, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=256_BN=128_BK=32_TM=8_TN=4", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 128), CEIL_DIV(M, 256));
        dim3 blockDim((256 * 128) / (8 * 8));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<256, 128, 32, 8, 8><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<256, 128, 32, 8, 8><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=256_BN=128_BK=32_TM=8_TN=8", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 128), CEIL_DIV(M, 256));
        dim3 blockDim((256 * 128) / (16 * 4));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<256, 128, 32, 16, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<256, 128, 32, 16, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=256_BN=128_BK=32_TM=16_TN=4", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 256), CEIL_DIV(M, 256));
        dim3 blockDim((256 * 256) / (4 * 16));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<256, 256, 16, 4, 16><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<256, 256, 16, 4, 16><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=256_BN=256_BK=16_TM=4_TN=16", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 256), CEIL_DIV(M, 256));
        dim3 blockDim((256 * 256) / (8 * 8));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<256, 256, 16, 8, 8><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<256, 256, 16, 8, 8><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=256_BN=256_BK=16_TM=8_TN=8", ms, tflops});
    }

    {
        dim3 gridDim(CEIL_DIV(N, 256), CEIL_DIV(M, 256));
        dim3 blockDim((256 * 256) / (16 * 4));

        // Warmup
        for (int i = 0; i < 3; ++i) {
            sgemm_vectorized<256, 256, 16, 16, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            sgemm_vectorized<256, 256, 16, 16, 4><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"BM=256_BN=256_BK=16_TM=16_TN=4", ms, tflops});
    }

    {
        cublasHandle_t handle;
        CUBLAS_CHECK(cublasCreate(&handle));
        for (int i = 0; i < 3; ++i) {
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, dB, N, dA, K, &beta, dC, N);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, dB, N, dA, K, &beta, dC, N);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({"cuBLAS Baseline", ms, tflops});
        cublasDestroy(handle);
    }

    std::sort(results.begin(), results.end(), [](const ConfigResult& a, const ConfigResult& b) {
        return a.tflops > b.tflops;
    });

    printf("\n================ AUTOTUNING LEADERBOARD ================\n");
    printf("%-35s | %-10s | %-10s\n", "Configuration", "Time (ms)", "TFLOPS");
    printf("--------------------------------------------------------\n");
    for (const auto& r : results) {
        printf("%-35s | %-10.3f | %-10.2f\n", r.name, r.ms, r.tflops);
    }
    printf("========================================================\n");

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    return 0;
}
