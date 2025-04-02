/**
 * Matrix Transpose Optimization with CUDA
 * 
 * This file implements matrix transpose:
 * 1. Naive implementation
 * 2. Shared memory optimization
 * 3. Shared Memory with Swizzling implementation (from Colfax post)
 * 4. Vectorized Memory Access with Transposition (from Lei Mao's blog)
 *
 * Includes timing code to measure performance.
 * 
 * References:
 * - Colfax Research: https://research.colfax-intl.com/tutorial-matrix-transpose-in-cutlass/
 * - Lei Mao's Blog: https://leimao.github.io/article/CUDA-Matrix-Multiplication-Optimization/
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <chrono>

// Matrix dimensions
#define M 1024  // Number of rows in input matrix
#define N 2048  // Number of columns in input matrix

__global__ void matrixTransposeNaive(float *input, float *output, int m, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < m && col < n) {
        // Input: row-major layout of m×n matrix
        // Output: row-major layout of n×m matrix (the transpose)
        output[col * m + row] = input[row * n + col];
    }
}

__global__ void matrixTransposeShared(float *input, float *output, int m, int n) {
    // Tile size
    const int TILE_SIZE = 32;
    
    // Shared memory with padding to avoid bank conflicts
    __shared__ float tile[TILE_SIZE][TILE_SIZE+1];  // +1 padding prevents bank conflicts
    
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    
    // Calculate global indices
    int row_in = by * TILE_SIZE + ty;
    int col_in = bx * TILE_SIZE + tx;
    
    // Read from input matrix (coalesced) and write to shared memory
    if (row_in < m && col_in < n) {
        tile[ty][tx] = input[row_in * n + col_in];
    }
    
    __syncthreads();
    
    // Calculate transposed indices
    int row_out = bx * TILE_SIZE + ty;  // Note: bx is used for row_out
    int col_out = by * TILE_SIZE + tx;  // Note: by is used for col_out
    
    // Write from shared memory to output (coalesced)
    if (row_out < n && col_out < m) {
        output[row_out * m + col_out] = tile[tx][ty];  // Note: tx and ty are swapped
    }
}

__global__ void matrixTransposeSwizzled(float *input, float *output, int m, int n) {
    // Tile size
    const int TILE_SIZE = 32;
    
    // Shared memory with swizzling pattern to avoid bank conflicts
    __shared__ float tile[TILE_SIZE][TILE_SIZE];
    
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    
    // Calculate global indices
    int row_in = by * TILE_SIZE + ty;
    int col_in = bx * TILE_SIZE + tx;

    // Compute linear index within the tile
    int linear_idx = ty * TILE_SIZE + tx;
    
    // Compute swizzled index for shared memory
    // This is equivalent to the Swizzle<5,0,5> mentioned in the article
    int swizzled_idx = linear_idx ^ ((linear_idx >> 5) & 0x1F);
    int swizzled_row = swizzled_idx / TILE_SIZE;
    int swizzled_col = swizzled_idx % TILE_SIZE;
    
    // Read from input matrix (coalesced) and write to shared memory (swizzled)
    if (row_in < m && col_in < n) {
        tile[swizzled_row][swizzled_col] = input[row_in * n + col_in];
    }
    
    __syncthreads();
    
    // Calculate transposed indices
    int row_out = bx * TILE_SIZE + tx;  // Swap tx and ty for transpose
    int col_out = by * TILE_SIZE + ty;
    
    // Re-compute linear index for reading transposed data
    linear_idx = ty * TILE_SIZE + tx;  // Note: tx and ty are swapped

    // Apply the same swizzling pattern for reading
    swizzled_idx = linear_idx ^ ((linear_idx >> 5) & 0x1F);
    swizzled_row = swizzled_idx / TILE_SIZE;
    swizzled_col = swizzled_idx % TILE_SIZE;
    
    // Write from shared memory to output (coalesced)
    if (row_out < n && col_out < m) {
        // Use swizzled index when reading from shared memory
        output[row_out * m + col_out] = tile[swizzled_row][swizzled_col];
    }
}

__global__ void matrixTransposeVectorized4(float *input, float *output, int m, int n) {
    const int TILE_SIZE = 32;
    __shared__ float tile[TILE_SIZE][TILE_SIZE + 1];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    // Compute input positions
    int row_in = by * TILE_SIZE + ty;           // row index of input matrix
    int col_in = bx * TILE_SIZE * 4 + tx * 4;   // each thread handles 4 elements (vectorized)

    float4 data;

    // load input elements into float4 with bounds checking: 
    // 4 contiguous elements from a row in global memory (vectorized access)
    data.x = (row_in < m && col_in + 0 < n) ? input[row_in * n + col_in + 0] : 0.0f;
    data.y = (row_in < m && col_in + 1 < n) ? input[row_in * n + col_in + 1] : 0.0f;
    data.z = (row_in < m && col_in + 2 < n) ? input[row_in * n + col_in + 2] : 0.0f;
    data.w = (row_in < m && col_in + 3 < n) ? input[row_in * n + col_in + 3] : 0.0f;

    // Transpose-on-load: Store in shared memory, effectively transposing during the load
    // The key optimization: swap row and column indices when storing
    // This transforms the data layout from row-major to column-major in shared memory
    tile[ty][tx * 4 + 0] = data.x;
    tile[ty][tx * 4 + 1] = data.y;
    tile[ty][tx * 4 + 2] = data.z;
    tile[ty][tx * 4 + 3] = data.w;

    __syncthreads();

    // Compute output positions (now transposed)
    // Swap block indices to achieve matrix transposition
    int row_out = bx * TILE_SIZE * 4 + tx * 4;
    int col_out = by * TILE_SIZE + ty;

    // If within bounds, write transposed elements back to global memory
    // Each thread writes 4 values down a column — resulting in coalesced writes after the transpose
    if (col_out < m) {
        if (row_out + 0 < n) output[(row_out + 0) * m + col_out] = tile[ty][tx * 4 + 0];
        if (row_out + 1 < n) output[(row_out + 1) * m + col_out] = tile[ty][tx * 4 + 1];
        if (row_out + 2 < n) output[(row_out + 2) * m + col_out] = tile[ty][tx * 4 + 2];
        if (row_out + 3 < n) output[(row_out + 3) * m + col_out] = tile[ty][tx * 4 + 3];
    }
}

void checkCudaError(cudaError_t error, const char *message) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s: %s\n", message, cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}

void initializeMatrix(float *matrix, int m, int n) {
    for (int i = 0; i < m * n; i++) {
        matrix[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

bool verifyTranspose(float *input, float *output, int m, int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if (input[i * n + j] != output[j * m + i]) {
                printf("Mismatch at [%d, %d]: Expected %f, got %f\n",
                       i, j, input[i * n + j], output[j * m + i]);
                return false;
            }
        }
    }
    return true;
}

int main() {
    float *h_input, *h_output;  // Host matrices
    float *d_input, *d_output;  // Device matrices
    
    size_t bytes_input = M * N * sizeof(float);
    size_t bytes_output = N * M * sizeof(float);
    
    // Allocate host memory
    h_input = (float*)malloc(bytes_input);
    h_output = (float*)malloc(bytes_output);
    
    // Initialize input matrix with random values
    srand(42); 
    initializeMatrix(h_input, M, N);
    
    // Allocate device memory
    checkCudaError(cudaMalloc(&d_input, bytes_input), "Allocating d_input");
    checkCudaError(cudaMalloc(&d_output, bytes_output), "Allocating d_output");
    
    // Copy input matrix from host to device
    checkCudaError(cudaMemcpy(d_input, h_input, bytes_input, cudaMemcpyHostToDevice), "Copying h_input to d_input");
    
    // Set up execution configuration
    dim3 blockDim(32, 32);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y);
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float elapsed_time;
    
    // ====================== NAIVE IMPLEMENTATION ======================
    printf("\n=== Naive Matrix Transpose Implementation ===\n");
    printf("Transposing a %d x %d matrix\n", M, N);
    
    cudaEventRecord(start);
    
    matrixTransposeNaive<<<gridDim, blockDim>>>(d_input, d_output, M, N);
    checkCudaError(cudaGetLastError(), "Launching naive transpose kernel");
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    
    printf("Naive transpose kernel execution time: %.2f ms\n", elapsed_time);
    
    // Copy result back to host
    checkCudaError(cudaMemcpy(h_output, d_output, bytes_output, cudaMemcpyDeviceToHost), "Copying d_output to h_output");
    
    // Verify the result
    bool verification_result = verifyTranspose(h_input, h_output, M, N);
    printf("Verification %s\n", verification_result ? "PASSED" : "FAILED");
    
    // ====================== SHARED MEMORY IMPLEMENTATION ======================
    printf("\n=== Shared Memory Matrix Transpose Implementation ===\n");
    
    cudaEventRecord(start);
    
    matrixTransposeShared<<<gridDim, blockDim>>>(d_input, d_output, M, N);
    checkCudaError(cudaGetLastError(), "Launching shared memory transpose kernel");
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    
    printf("Shared memory transpose kernel execution time: %.2f ms\n", elapsed_time);
    
    // Copy result back to host
    checkCudaError(cudaMemcpy(h_output, d_output, bytes_output, cudaMemcpyDeviceToHost), "Copying d_output to h_output");
    
    // Verify the result
    verification_result = verifyTranspose(h_input, h_output, M, N);
    printf("Verification %s\n", verification_result ? "PASSED" : "FAILED");

    // ====================== SHARED MEMORY WITH SWIZZLING IMPLEMENTATION ======================
    printf("\n=== Shared Memory with Swizzling Matrix Transpose Implementation ===\n");

    cudaEventRecord(start);

    matrixTransposeSwizzled<<<gridDim, blockDim>>>(d_input, d_output, M, N);
    checkCudaError(cudaGetLastError(), "Launching swizzled transpose kernel");

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);

    printf("Shared memory with swizzling transpose kernel execution time: %.2f ms\n", elapsed_time);

    // Copy result back to host
    checkCudaError(cudaMemcpy(h_output, d_output, bytes_output, cudaMemcpyDeviceToHost), "Copying d_output to h_output");

    // Verify the result
    verification_result = verifyTranspose(h_input, h_output, M, N);
    printf("Verification %s\n", verification_result ? "PASSED" : "FAILED");

    // ====================== VECTORIZED MEMORY ACCESS IMPLEMENTATION ======================
    printf("\n=== Vectorized Memory Access Matrix Transpose Implementation ===\n");

    // Vectorized version: 4 floats per thread in x-dimension
    dim3 blockDimVec(8, 32);  // 8*4 = 32 elements per row per tile
    dim3 gridDimVec((N + blockDimVec.x * 4 - 1) / (blockDimVec.x * 4),
                    (M + blockDimVec.y - 1) / blockDimVec.y);

    cudaEventRecord(start);

    matrixTransposeVectorized4<<<gridDimVec, blockDimVec>>>(d_input, d_output, M, N);
    checkCudaError(cudaGetLastError(), "Launching vectorized transpose kernel");

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);

    printf("Vectorized transpose kernel execution time: %.2f ms\n", elapsed_time);

    // Copy result back to host
    checkCudaError(cudaMemcpy(h_output, d_output, bytes_output, cudaMemcpyDeviceToHost), "Copying d_output to h_output");

    // Verify the result
    verification_result = verifyTranspose(h_input, h_output, M, N);
    printf("Verification %s\n", verification_result ? "PASSED" : "FAILED");

    // Clean up
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);
    
    return 0;
}

