#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define FILTER_RADIUS 1 


// 8x8x8 = 512 threads per block (Max is 1024).
#define IN_TILE_DIM 8  
#define OUT_TILE_DIM ((IN_TILE_DIM) - 2*(FILTER_RADIUS))

// Constant memory for the filter, declared as a 3D array
__constant__ float F_c[2*FILTER_RADIUS+1][2*FILTER_RADIUS+1][2*FILTER_RADIUS+1];

// Macro for catching CUDA errors 
#define cudaCheckError() { \
    cudaError_t e=cudaGetLastError(); \
    if(e!=cudaSuccess) { \
        printf("CUDA Error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); \
        exit(EXIT_FAILURE); \
    } \
}

__global__ void convolution_tiled_3D_const_mem_kernel(float *N, float *P,
                                                    int width, int height, int depth) {
    // 1. Calculate global coordinates for loading the INPUT tile
    int col = blockIdx.x*OUT_TILE_DIM + threadIdx.x - FILTER_RADIUS;
    int row = blockIdx.y*OUT_TILE_DIM + threadIdx.y - FILTER_RADIUS;
    int dep = blockIdx.z*OUT_TILE_DIM + threadIdx.z - FILTER_RADIUS;

    // Allocate ultra-fast shared memory for the input tile
    __shared__ float N_s[IN_TILE_DIM][IN_TILE_DIM][IN_TILE_DIM];
    
    // 2. Collaborative Loading: Every thread loads exactly one voxel
    if(dep>=0 && dep<depth && row>=0 && row<height && col>=0 && col<width) {
        N_s[threadIdx.z][threadIdx.y][threadIdx.x] = N[dep*width*height + row*width + col];
    } else {
        // Zero-padding for ghost cells outside the actual image boundaries
        N_s[threadIdx.z][threadIdx.y][threadIdx.x] = 0.0f;
    }
    
    // 3. Barrier: Wait for all threads to finish loading
    __syncthreads();
    
    // 4. Calculate local coordinates within the OUTPUT tile
    int tileCol = threadIdx.x - FILTER_RADIUS;
    int tileRow = threadIdx.y - FILTER_RADIUS;
    int tileDep = threadIdx.z - FILTER_RADIUS;
    
    // 5. Compute Phase: Turn off halo threads, only inner threads calculate
    if (dep >= 0 && dep < depth && row >= 0 && row < height && col >= 0 && col < width) {
        if (tileCol>=0 && tileCol<OUT_TILE_DIM && 
            tileRow>=0 && tileRow<OUT_TILE_DIM &&
            tileDep>=0 && tileDep<OUT_TILE_DIM) {
            
            float Pvalue = 0.0f;
            
            for (int fDep = 0; fDep < 2*FILTER_RADIUS+1; fDep++) {
                for (int fRow = 0; fRow < 2*FILTER_RADIUS+1; fRow++) {
                    for (int fCol = 0; fCol < 2*FILTER_RADIUS+1; fCol++) {
                        // Accessing F_c from Constant Cache, N_s from Shared Memory
                        Pvalue += F_c[fDep][fRow][fCol] * 
                                 N_s[tileDep+fDep][tileRow+fRow][tileCol+fCol];
                    }
                }
            }
            
            // 6. Write back to global memory
            P[dep*width*height + row*width + col] = Pvalue;
        }
    }
}

int main() {
    // Volume dimensions
    int width = 32;
    int height = 32;
    int depth = 32;
    
    int filter_dim = 2 * FILTER_RADIUS + 1;
    int filter_elements = filter_dim * filter_dim * filter_dim;

    size_t vol_bytes = width * height * depth * sizeof(float);
    size_t filter_bytes = filter_elements * sizeof(float);

    // Allocate Host Memory
    float *h_N = (float*)malloc(vol_bytes);
    float *h_F = (float*)malloc(filter_bytes);
    float *h_P = (float*)malloc(vol_bytes);

    // Initialize with 1.0f for easy verification
    for(int i = 0; i < width * height * depth; i++) h_N[i] = 1.0f;
    for(int i = 0; i < filter_elements; i++) h_F[i] = 1.0f; 

    // Allocate Device Memory
    float *d_N, *d_P;
    cudaMalloc((void**)&d_N, vol_bytes);
    cudaMalloc((void**)&d_P, vol_bytes);
    cudaCheckError();

    // Copy Input Volume to Global Memory
    cudaMemcpy(d_N, h_N, vol_bytes, cudaMemcpyHostToDevice);
    
    // Copy Filter to Constant Memory Symbol
    cudaMemcpyToSymbol(F_c, h_F, filter_bytes);
    cudaCheckError();

    // Configure Execution
    // Block size uses the larger IN_TILE_DIM
    dim3 threadsPerBlock(IN_TILE_DIM, IN_TILE_DIM, IN_TILE_DIM);
    
    // Grid size is calculated using the smaller OUT_TILE_DIM
    // because each block only computes a tile of size OUT_TILE_DIM
    dim3 blocksPerGrid(
        (width + OUT_TILE_DIM - 1) / OUT_TILE_DIM,
        (height + OUT_TILE_DIM - 1) / OUT_TILE_DIM,
        (depth + OUT_TILE_DIM - 1) / OUT_TILE_DIM
    );

    printf("Launching 3D Tiled Convolution Kernel...\n");
    printf("Block size: %dx%dx%d (%d threads)\n", IN_TILE_DIM, IN_TILE_DIM, IN_TILE_DIM, IN_TILE_DIM*IN_TILE_DIM*IN_TILE_DIM);
    printf("Output tile size: %dx%dx%d\n", OUT_TILE_DIM, OUT_TILE_DIM, OUT_TILE_DIM);
    
    convolution_tiled_3D_const_mem_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_N, d_P, width, height, depth);
    cudaDeviceSynchronize(); 
    cudaCheckError();

    cudaMemcpy(h_P, d_P, vol_bytes, cudaMemcpyDeviceToHost);

    // Verify center (should equal number of elements in the filter)
    int center_idx = (depth/2) * width * height + (height/2) * width + (width/2);
    printf("Value at center (should be %d.0): %f\n", filter_elements, h_P[center_idx]);

    cudaFree(d_N);
    cudaFree(d_P);
    free(h_N);
    free(h_F);
    free(h_P);

    printf("Done.\n");
    return 0;
}