import os
import subprocess
import sys

# Search space dimensions
PARAM_GRID = {
    "BM": [32, 64, 128, 256],
    "BN": [32, 64, 128, 256],
    "BK": [8, 16, 32],
    "TM": [2, 4, 8, 16],
    "TN": [4, 8, 16],  # TN must be divisible by 4 for float4 C stores
}

MAX_SMEM_BYTES = 49152  # 48 KB standard SMEM ceiling per block


def is_valid_config(BM, BN, BK, TM, TN):
    # Rule 1: Tile dimensions must be divisible by thread per-register tiles
    if BM % TM != 0 or BN % TN != 0:
        return False

    num_threads = (BM * BN) // (TM * TN)

    # Rule 2: Thread limits
    if num_threads < 64 or num_threads > 1024 or num_threads % 32 != 0:
        return False

    # Rule 3: 128-bit (float4 = 4 floats) vector alignment requirements
    if BK % 4 != 0 or BN % 4 != 0 or TN % 4 != 0:
        return False

    # Rule 4: Vector GMEM -> SMEM tiling divisibility
    # Each thread loads float4 (4 floats) per load instruction
    floats_per_load_step = num_threads * 4

    tile_A_floats = BM * BK
    tile_B_floats = BK * BN

    # Tile size must be an exact multiple of thread loading capacity per step
    if tile_A_floats < floats_per_load_step or tile_A_floats % floats_per_load_step != 0:
        return False
    if tile_B_floats < floats_per_load_step or tile_B_floats % floats_per_load_step != 0:
        return False

    # Coalescing stride check along 4-wide float columns
    if num_threads % (BK // 4) != 0 or num_threads % (BN // 4) != 0:
        return False

    # Rule 5: SMEM Footprint Check (float = 4 bytes)
    smem_bytes = (tile_A_floats + tile_B_floats) * 4
    if smem_bytes > MAX_SMEM_BYTES:
        return False

    # Rule 6: Heuristic check for register pressure
    if TM * TN > 64:
        return False

    return True


def generate_cpp_source(valid_configs):
    cpp_code = """#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <cmath>

#define CUDA_CHECK(call) \\
    do { \\
        cudaError_t err = call; \\
        if (err != cudaSuccess) { \\
            fprintf(stderr, "CUDA Error %s at line %d\\n", cudaGetErrorString(err), __LINE__); \\
            exit(EXIT_FAILURE); \\
        } \\
    } while (0)

#define CUBLAS_CHECK(call) \\
    do { \\
        cublasStatus_t st = call; \\
        if (st != CUBLAS_STATUS_SUCCESS) { \\
            fprintf(stderr, "cuBLAS Error %d at line %d\\n", (int)st, __LINE__); \\
            exit(EXIT_FAILURE); \\
        } \\
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
"""

    for cfg in valid_configs:
        BM, BN, BK, TM, TN = cfg
        name_str = f"BM={BM}_BN={BN}_BK={BK}_TM={TM}_TN={TN}"
        cpp_code += f"""
    {{
        dim3 gridDim(CEIL_DIV(N, {BN}), CEIL_DIV(M, {BM}));
        dim3 blockDim(({BM} * {BN}) / ({TM} * {TN}));

        // Warmup
        for (int i = 0; i < 3; ++i) {{
            sgemm_vectorized<{BM}, {BN}, {BK}, {TM}, {TN}><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }}
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) {{
            sgemm_vectorized<{BM}, {BN}, {BK}, {TM}, {TN}><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        }}
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        results.push_back({{"{name_str}", ms, tflops}});
    }}
"""

    cpp_code += """
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

    printf("\\n================ AUTOTUNING LEADERBOARD ================\\n");
    printf("%-35s | %-10s | %-10s\\n", "Configuration", "Time (ms)", "TFLOPS");
    printf("--------------------------------------------------------\\n");
    for (const auto& r : results) {
        printf("%-35s | %-10.3f | %-10.2f\\n", r.name, r.ms, r.tflops);
    }
    printf("========================================================\\n");

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    return 0;
}
"""
    with open("tune_generated.cu", "w") as f:
        f.write(cpp_code)

def main():
    valid_configs = []
    total_searched = 0

    for BM in PARAM_GRID["BM"]:
        for BN in PARAM_GRID["BN"]:
            for BK in PARAM_GRID["BK"]:
                for TM in PARAM_GRID["TM"]:
                    for TN in PARAM_GRID["TN"]:
                        total_searched += 1
                        if is_valid_config(BM, BN, BK, TM, TN):
                            valid_configs.append((BM, BN, BK, TM, TN))

    print(
        f"Filtered {total_searched} parameter sets down to {len(valid_configs)} mathematically valid vectorized configurations."
    )

    if not valid_configs:
        print("No valid configurations found.")
        sys.exit(1)

    print("Generating C++ benchmark driver (tune_generated.cu)...")
    generate_cpp_source(valid_configs)

    print("Compiling with NVCC...")
    compile_cmd = "nvcc -O3 -lineinfo tune_generated.cu -o autotune_bin -lcublas"
    res = subprocess.run(compile_cmd, shell=True)

    if res.returncode != 0:
        print("NVCC Compilation Failed.")
        sys.exit(1)

    print("Running Autotuner Benchmarks...\n")
    subprocess.run("./autotune_bin", shell=True)


if __name__ == "__main__":
    main()