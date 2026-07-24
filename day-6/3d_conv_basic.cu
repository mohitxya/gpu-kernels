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

// Your verified kernel
__global__ void convolution_3D_basic_kernel(float *N, float *F, float *P, int r, int width, int height, int depth)
{
    int outCol = blockIdx.x*blockDim.x + threadIdx.x; 
    int outRow = blockIdx.y*blockDim.y + threadIdx.y; 
    int outDepth = blockIdx.z*blockDim.z + threadIdx.z; 
    
    if (outCol < width && outRow < height && outDepth < depth)
    {
        float Pvalue = 0.0f; 

        for(int fDepth = 0; fDepth < 2*r+1; fDepth++)
        {
            for(int fRow = 0; fRow < 2*r+1; fRow++)
            {
                for(int fCol = 0; fCol < 2*r+1; fCol++)
                {
                    int inDepth = outDepth -r +fDepth; 
                    int inRow = outRow -r + fRow; 
                    int inCol = outCol - r + fCol; 

                    if(inRow >= 0 && inRow < height &&
                    inCol >= 0 && inCol < width &&
                    inDepth >= 0 && inDepth < depth){
                        int inIndex = inDepth*width*height + inRow*width + inCol; 
                        int fIndex = fDepth*(2*r+1)*(2*r+1) + fRow*(2*r+1) + fCol; 

                        Pvalue += F[fIndex]*N[inIndex];
                    }
                }
            }
        }
        P[outDepth*width*height + outRow*width + outCol] = Pvalue;  
    }
}

int main()
{
    // 1. Define volume dimensions and filter radius
    int width = 32;
    int height = 32;
    int depth = 32;
    int r = 1; // 3x3x3 filter
    int filter_dim = 2 * r + 1;

    size_t vol_bytes = width * height * depth * sizeof(float);
    size_t filter_bytes = filter_dim * filter_dim * filter_dim * sizeof(float);

    // 2. Allocate memory on the Host (CPU)
    float *h_N = (float*)malloc(vol_bytes);
    float *h_F = (float*)malloc(filter_bytes);
    float *h_P = (float*)malloc(vol_bytes);

    // 3. Initialize data
    // Fill input volume and filter with 1.0f for easy verification
    for(int i = 0; i < width * height * depth; i++) {
        h_N[i] = 1.0f;
    }
    for(int i = 0; i < filter_dim * filter_dim * filter_dim; i++) {
        h_F[i] = 1.0f; 
    }

    // 4. Allocate memory on the Device (GPU)
    float *d_N, *d_F, *d_P;
    cudaMalloc((void**)&d_N, vol_bytes);
    cudaMalloc((void**)&d_F, filter_bytes);
    cudaMalloc((void**)&d_P, vol_bytes);
    cudaCheckError();

    // 5. Transfer data from Host to Device
    cudaMemcpy(d_N, h_N, vol_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_F, h_F, filter_bytes, cudaMemcpyHostToDevice);
    cudaCheckError();

    // 6. Define Kernel Execution Configuration
    // A standard 3D block size of 8x8x8 gives 512 threads per block
    dim3 threadsPerBlock(8, 8, 8);
    
    // Calculate grid size using ceiling division to cover the whole volume
    dim3 blocksPerGrid(
        (width + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (height + threadsPerBlock.y - 1) / threadsPerBlock.y,
        (depth + threadsPerBlock.z - 1) / threadsPerBlock.z
    );

    // 7. Launch Kernel
    printf("Launching 3D Convolution Kernel...\n");
    printf("Grid: (%d, %d, %d), Block: (%d, %d, %d)\n", 
           blocksPerGrid.x, blocksPerGrid.y, blocksPerGrid.z,
           threadsPerBlock.x, threadsPerBlock.y, threadsPerBlock.z);

    convolution_3D_basic_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_N, d_F, d_P, r, width, height, depth);
    cudaDeviceSynchronize(); // Wait for GPU to finish
    cudaCheckError();

    // 8. Transfer results back from Device to Host
    cudaMemcpy(h_P, d_P, vol_bytes, cudaMemcpyDeviceToHost);

    // 9. Verify results
    // Center of the volume: full filter overlap (3x3x3 = 27 elements)
    int center_idx = (depth/2) * width * height + (height/2) * width + (width/2);
    printf("Value at center (should be 27.0): %f\n", h_P[center_idx]);

    // Edge of the volume (0,0,0): only 2x2x2 = 8 filter elements overlap the volume
    printf("Value at origin (should be 8.0): %f\n", h_P[0]);

    // 10. Free Memory
    cudaFree(d_N);
    cudaFree(d_F);
    cudaFree(d_P);
    free(h_N);
    free(h_F);
    free(h_P);

    printf("Done.\n");
    return 0;
}