#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Macro for catching CUDA errors 
#define cudaCheckError() { \
    cudaError_t e=cudaGetLastError(); \
    if(e!=cudaSuccess) { \
        printf("CUDA Error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); \
        exit(EXIT_FAILURE); \
    } \
}


// Constant memory requires a fixed size at compile time.
// A radius of 2 allows up to a 5x5x5 filter (125 elements).
#define MAX_FILTER_RADIUS 2 
#define MAX_FILTER_SIZE (5 * 5 * 5)
__constant__ float F[MAX_FILTER_SIZE];


// 2. The Kernel
// Notice that 'float *F' is no longer passed as a parameter.
__global__ void convolution_3D_const_mem_kernel(float *N, float *P, int r,
    int width, int height, int depth) {
    
    int outCol = blockIdx.x * blockDim.x + threadIdx.x;
    int outRow = blockIdx.y * blockDim.y + threadIdx.y;
    int outDepth = blockIdx.z * blockDim.z + threadIdx.z;
    
    // Safety check to prevent out-of-bounds memory writes
    if (outCol < width && outRow < height && outDepth < depth) {
        
        float Pvalue = 0.0f;
        
        for (int fDepth = 0; fDepth < 2*r+1; fDepth++) {
            for (int fRow = 0; fRow < 2*r+1; fRow++) {
                for (int fCol = 0; fCol < 2*r+1; fCol++) {
                    int inRow = outRow - r + fRow;
                    int inCol = outCol - r + fCol;
                    int inDepth = outDepth - r + fDepth;
                    
                    if (inRow >= 0 && inRow < height && 
                        inCol >= 0 && inCol < width && 
                        inDepth >= 0 && inDepth < depth) {
                        
                        int input_idx = inDepth * height * width + 
                                      inRow * width + 
                                      inCol;
                        int filter_idx = fDepth * (2*r+1) * (2*r+1) + 
                                       fRow * (2*r+1) + 
                                       fCol;
                        
                        // F is accessed directly from constant memory here
                        Pvalue += F[filter_idx] * N[input_idx];
                    }
                }
            }
        }
        
        P[outDepth * height * width + outRow * width + outCol] = Pvalue;
    }
}

int main()
{
    // Define volume dimensions and filter radius
    int width = 32;
    int height = 32;
    int depth = 32;
    int r = 1; // 3x3x3 filter
    int filter_dim = 2 * r + 1;
    int filter_elements = filter_dim * filter_dim * filter_dim;

    if (filter_elements > MAX_FILTER_SIZE) {
        printf("Error: Filter size exceeds allocated constant memory.\n");
        return -1;
    }

    size_t vol_bytes = width * height * depth * sizeof(float);
    size_t filter_bytes = filter_elements * sizeof(float);

    // Allocate memory on the Host (CPU)
    float *h_N = (float*)malloc(vol_bytes);
    float *h_F = (float*)malloc(filter_bytes);
    float *h_P = (float*)malloc(vol_bytes);

    // Initialize data with 1.0f for easy verification
    for(int i = 0; i < width * height * depth; i++) {
        h_N[i] = 1.0f;
    }
    for(int i = 0; i < filter_elements; i++) {
        h_F[i] = 1.0f; 
    }

    // Allocate memory on the Device (GPU) for N and P ONLY
    float *d_N, *d_P;
    cudaMalloc((void**)&d_N, vol_bytes);
    cudaMalloc((void**)&d_P, vol_bytes);
    cudaCheckError();

    // Copy Input Volume to Global Memory
    cudaMemcpy(d_N, h_N, vol_bytes, cudaMemcpyHostToDevice);
    
    // CRITICAL DIFFERENCE: Copy Filter to Constant Memory
    // We use cudaMemcpyToSymbol instead of cudaMemcpy
    cudaMemcpyToSymbol(F, h_F, filter_bytes);
    cudaCheckError();

    // Define Kernel Execution Configuration
    dim3 threadsPerBlock(8, 8, 8);
    dim3 blocksPerGrid(
        (width + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (height + threadsPerBlock.y - 1) / threadsPerBlock.y,
        (depth + threadsPerBlock.z - 1) / threadsPerBlock.z
    );

    printf("Launching 3D Convolution Kernel (Constant Memory)...\n");
    
    // Launch Kernel (Note: d_F is no longer a parameter)
    convolution_3D_const_mem_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_N, d_P, r, width, height, depth);
    cudaDeviceSynchronize(); 
    cudaCheckError();

    // Transfer results back
    cudaMemcpy(h_P, d_P, vol_bytes, cudaMemcpyDeviceToHost);

    // Verify results
    int center_idx = (depth/2) * width * height + (height/2) * width + (width/2);
    printf("Value at center (should be %d.0): %f\n", filter_elements, h_P[center_idx]);
    printf("Value at origin (should be 8.0): %f\n", h_P[0]);

    // Free Memory
    cudaFree(d_N);
    cudaFree(d_P);
    // Note: No need to free F since it was statically allocated in constant memory
    
    free(h_N);
    free(h_F);
    free(h_P);

    printf("Done.\n");
    return 0;
}