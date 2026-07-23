#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define FILTER_RADIUS 2                                   // 5x5 filter
#define IN_TILE_DIM   32                                   // block = 32x32 threads
#define OUT_TILE_DIM  (IN_TILE_DIM - 2*FILTER_RADIUS)       // = 28 output pixels per block

__constant__ float F_c[2*FILTER_RADIUS+1][2*FILTER_RADIUS+1];

__global__
void convolution_tiled_2D_const_mem_kernel(const float *N, float *P, int width, int height)
{
    int col = blockIdx.x*OUT_TILE_DIM + threadIdx.x - FILTER_RADIUS;
    int row = blockIdx.y*OUT_TILE_DIM + threadIdx.y - FILTER_RADIUS;

    __shared__ float N_s[IN_TILE_DIM][IN_TILE_DIM];

    // Load input tile (with halo) into shared memory, zero-padding out-of-bounds
    if (row >= 0 && row < height && col >= 0 && col < width)
        N_s[threadIdx.y][threadIdx.x] = N[row * width + col];
    else
        N_s[threadIdx.y][threadIdx.x] = 0.0f;

    __syncthreads();

    // Position of this thread within the *output* tile (excludes halo threads)
    int tileCol = threadIdx.x - FILTER_RADIUS;
    int tileRow = threadIdx.y - FILTER_RADIUS;

    if (col >= 0 && col < width && row >= 0 && row < height)
    {
        if (tileCol >= 0 && tileCol < OUT_TILE_DIM && tileRow >= 0 && tileRow < OUT_TILE_DIM)
        {
            float Pvalue = 0.0f;
            for (int fRow = 0; fRow < 2*FILTER_RADIUS+1; fRow++)
                for (int fCol = 0; fCol < 2*FILTER_RADIUS+1; fCol++)
                    Pvalue += F_c[fRow][fCol] * N_s[tileRow + fRow][tileCol + fCol];

            P[row * width + col] = Pvalue;
        }
    }
}

// ---------------------------------------------------------------------------
// Host wrapper: copies filter to constant memory, launches kernel, times it.
// ---------------------------------------------------------------------------
float convolution_gpu(const float *N_h, const float *filter_h, float *P_h, int width, int height)
{
    float *N_d, *P_d;
    size_t sizeN = width * height * sizeof(float);
    size_t sizeP = width * height * sizeof(float);

    cudaMalloc(&N_d, sizeN);
    cudaMalloc(&P_d, sizeP);
    cudaMemcpy(N_d, N_h, sizeN, cudaMemcpyHostToDevice);

    // Copy filter into __constant__ memory (note: symbol, not a device pointer)
    cudaMemcpyToSymbol(F_c, filter_h, (2*FILTER_RADIUS+1) * (2*FILTER_RADIUS+1) * sizeof(float));

    dim3 threads(IN_TILE_DIM, IN_TILE_DIM);
    dim3 blocks((width  + OUT_TILE_DIM - 1) / OUT_TILE_DIM,
                (height + OUT_TILE_DIM - 1) / OUT_TILE_DIM);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    convolution_tiled_2D_const_mem_kernel<<<blocks, threads>>>(N_d, P_d, width, height);
    cudaEventRecord(stop);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));

    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    cudaMemcpy(P_h, P_d, sizeP, cudaMemcpyDeviceToHost);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(N_d);
    cudaFree(P_d);

    return ms;
}

// ---------------------------------------------------------------------------
// CPU reference: same zero-padded convolution, done with plain nested loops.
// ---------------------------------------------------------------------------
void convolution_cpu(const float *N, const float *filter, float *P, int width, int height)
{
    for (int row = 0; row < height; row++)
    {
        for (int col = 0; col < width; col++)
        {
            float Pvalue = 0.0f;
            for (int fRow = 0; fRow < 2*FILTER_RADIUS+1; fRow++)
            {
                for (int fCol = 0; fCol < 2*FILTER_RADIUS+1; fCol++)
                {
                    int inRow = row + fRow - FILTER_RADIUS;
                    int inCol = col + fCol - FILTER_RADIUS;
                    if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width)
                        Pvalue += filter[fRow * (2*FILTER_RADIUS+1) + fCol] * N[inRow * width + inCol];
                }
            }
            P[row * width + col] = Pvalue;
        }
    }
}

int main()
{
    const int width  = 1024;
    const int height = 1024;
    const int fSize   = 2*FILTER_RADIUS+1;

    float *N       = (float*)malloc(width * height * sizeof(float));
    float *P_gpu   = (float*)malloc(width * height * sizeof(float));
    float *P_cpu   = (float*)malloc(width * height * sizeof(float));
    float *filter  = (float*)malloc(fSize * fSize * sizeof(float));

    // Random input image
    srand(42);
    for (int i = 0; i < width * height; i++)
        N[i] = (float)(rand() % 10);

    // Simple averaging (box blur) filter, normalized
    for (int i = 0; i < fSize * fSize; i++)
        filter[i] = 1.0f / (fSize * fSize);

    float gpu_ms = convolution_gpu(N, filter, P_gpu, width, height);
    convolution_cpu(N, filter, P_cpu, width, height);

    float max_err = 0.0f;
    for (int i = 0; i < width * height; i++)
    {
        float err = fabsf(P_gpu[i] - P_cpu[i]);
        if (err > max_err) max_err = err;
    }

    printf("Image size:      %dx%d\n", width, height);
    printf("Filter radius:   %d (%dx%d filter)\n", FILTER_RADIUS, fSize, fSize);
    printf("OUT_TILE_DIM:    %d, IN_TILE_DIM: %d\n", OUT_TILE_DIM, IN_TILE_DIM);
    printf("GPU time:        %8.3f ms\n", gpu_ms);
    printf("Max abs error:   %f -> %s\n", max_err, max_err < 1e-3f ? "PASS" : "FAIL");

    free(N); free(P_gpu); free(P_cpu); free(filter);
    return 0;
}
