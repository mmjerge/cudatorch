/**
 * Matrix Transpose Optimization with CUDA
 * 
 * This file implements matrix transpose:
 * 1. Naive implementation
 * 2. Shared memory optimization
 * 3.  Shared Memory with Swizzling implementation (from Colfax post)
 *
 * Includes timing code to measure performance.
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
    
    // Compute swizzled index for shared memory
    // This is equivalent to the Swizzle<5,0,5> mentioned in the article
    int linear_idx = ty * TILE_SIZE + tx;
    int swizzled_idx = linear_idx ^ ((linear_idx >> 5) << 5);
    int swizzled_row = swizzled_idx / TILE_SIZE;
    int swizzled_col = swizzled_idx % TILE_SIZE;
    
    // Read from input matrix (coalesced) and write to shared memory (swizzled)
    if (row_in < m && col_in < n) {
        tile[swizzled_row][swizzled_col] = input[row_in * n + col_in];
    }
    
    __syncthreads();
    
    // Calculate transposed indices
    int row_out = bx * TILE_SIZE + ty;  // Note: bx is used for row_out
    int col_out = by * TILE_SIZE + tx;  // Note: by is used for col_out
    
    // Compute swizzled index for reading from shared memory
    // reading from shared memory in transposed order
    linear_idx = tx * TILE_SIZE + ty;  // Note: tx and ty are swapped
    swizzled_idx = linear_idx ^ ((linear_idx >> 5) << 5);
    swizzled_row = swizzled_idx / TILE_SIZE;
    swizzled_col = swizzled_idx % TILE_SIZE;
    
    // Write from shared memory to output (coalesced)
    if (row_out < n && col_out < m) {
        // Use swizzled index when reading from shared memory
        output[row_out * m + col_out] = tile[swizzled_row][swizzled_col];
    }
}

__global__ void matrixTransposeSwizzledDebug(float *input, float *output, int m, int n) {
    // Tile size
    const int TILE_SIZE = 32;
    
    // Shared memory with swizzling pattern to avoid bank conflicts
    __shared__ float tile[TILE_SIZE][TILE_SIZE];
    
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    
    // Calculate global indices
    int row_in = by * TILE_SIZE + ty;
    int col_in = bx * TILE_SIZE + tx;
    
    // Compute swizzled index for shared memory
    int linear_idx = ty * TILE_SIZE + tx;
    int swizzled_idx = linear_idx ^ ((linear_idx >> 5) << 5);
    int swizzled_row = swizzled_idx / TILE_SIZE;
    int swizzled_col = swizzled_idx % TILE_SIZE;
    
    // Only have thread (0,0) in block (0,0) print debug info
    if (bx == 0 && by == 0 && tx == 0 && ty == 0) {
        printf("DEBUG - Block (0,0), Thread (0,0):\n");
        printf("Input value at [0,0]: %f\n", input[0]);
        printf("Write swizzle: linear_idx=%d, swizzled_idx=%d, row=%d, col=%d\n", 
               linear_idx, swizzled_idx, swizzled_row, swizzled_col);
    }
    
    // Read from input matrix (coalesced) and write to shared memory (swizzled)
    if (row_in < m && col_in < n) {
        // Use swizzled index to avoid bank conflicts
        tile[swizzled_row][swizzled_col] = input[row_in * n + col_in];
    }
    
    __syncthreads();  // Ensure all threads have loaded their data
    
    // Calculate transposed indices
    int row_out = bx * TILE_SIZE + ty;  // Note: bx is used for row_out
    int col_out = by * TILE_SIZE + tx;  // Note: by is used for col_out
    
    // Compute swizzled index for reading from shared memory
    linear_idx = tx * TILE_SIZE + ty;  // Note: tx and ty are swapped for transpose
    swizzled_idx = linear_idx ^ ((linear_idx >> 5) << 5);
    swizzled_row = swizzled_idx / TILE_SIZE;
    swizzled_col = swizzled_idx % TILE_SIZE;
    
    // More debug prints
    if (bx == 0 && by == 0 && tx == 0 && ty == 0) {
        printf("Read swizzle: linear_idx=%d, swizzled_idx=%d, row=%d, col=%d\n", 
               linear_idx, swizzled_idx, swizzled_row, swizzled_col);
        printf("Value in tile[%d][%d]: %f\n", swizzled_row, swizzled_col, tile[swizzled_row][swizzled_col]);
    }
    
    // Write from shared memory to output (coalesced)
    if (row_out < n && col_out < m) {
        // Use swizzled index when reading from shared memory
        output[row_out * m + col_out] = tile[swizzled_row][swizzled_col];
    }
    
    // Final debug print
    if (bx == 0 && by == 0 && tx == 0 && ty == 0) {
        printf("Output value at [0,0]: %f\n", output[0]);
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

void printMatrixDebug(float *h_input, float *h_output, int m, int n) {
    printf("\n=== DEBUG MATRIX VALUES ===\n");
    
    // Print the first few elements of input
    printf("Input Matrix (first 3x3 corner):\n");
    for (int i = 0; i < min(3, m); i++) {
        for (int j = 0; j < min(3, n); j++) {
            printf("%8.6f ", h_input[i * n + j]);
        }
        printf("\n");
    }
    
    // Print expected transpose results
    printf("\nExpected Transposed Values (first 3x3 corner):\n");
    for (int i = 0; i < min(3, n); i++) {
        for (int j = 0; j < min(3, m); j++) {
            printf("%8.6f ", h_input[j * n + i]);  // This is what we expect in the output
        }
        printf("\n");
    }
    
    // Print actual transpose results
    printf("\nActual Transposed Matrix (first 3x3 corner):\n");
    for (int i = 0; i < min(3, n); i++) {
        for (int j = 0; j < min(3, m); j++) {
            printf("%8.6f ", h_output[i * m + j]);
        }
        printf("\n");
    }
    
    // Check for mismatches in the first few elements
    printf("\nChecking for Mismatches in First Few Elements:\n");
    int mismatch_count = 0;
    for (int i = 0; i < min(5, n); i++) {
        for (int j = 0; j < min(5, m); j++) {
            float expected = h_input[j * n + i];
            float actual = h_output[i * m + j];
            if (fabs(expected - actual) > 1e-5) {
                printf("Mismatch at [%d,%d]: Expected %8.6f, Got %8.6f\n", 
                       i, j, expected, actual);
                mismatch_count++;
                if (mismatch_count >= 5) {
                    printf("Too many mismatches, stopping...\n");
                    return;
                }
            }
        }
    }
    
    if (mismatch_count == 0) {
        printf("No mismatches found in the first few elements!\n");
    }
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

    matrixTransposeSwizzledDebug<<<gridDim, blockDim>>>(d_input, d_output, M, N);
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
    // In your main function, add this after verifying the swizzled transpose:
    if (!verification_result) {
        printf("\n=== Debugging Swizzled Transpose Failure ===\n");
        
        // Run the debug kernel
        matrixTransposeSwizzledDebug<<<gridDim, blockDim>>>(d_input, d_output, M, N);
        checkCudaError(cudaGetLastError(), "Launching debug kernel");
        cudaDeviceSynchronize();  // Make sure to sync to see printf output
        
        // Copy result back for debugging
        checkCudaError(cudaMemcpy(h_output, d_output, bytes_output, cudaMemcpyDeviceToHost), 
                    "Copying debug output to host");
        
        // Print matrix values for visual inspection
        printMatrixDebug(h_input, h_output, M, N);
    }

    // Clean up
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);
    
    return 0;
}

