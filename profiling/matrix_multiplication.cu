/**
 * Matrix Multiplication Optimization with CUDA
 * 
 * This file contains three implementations of matrix multiplication:
 * 1. Naive implementation
 * 2. Shared memory optimization
 * 3. Tensor cores (WMMA) implementation
 *
 * Each implementation includes timing code to measure performance.
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <chrono>

#define N 1024  // Change as needed

__global__ void matrixMulNaive(float *a, float *b, float *c, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < n; i++) {
            sum += a[row * n + i] * b[i * n + col];
        }
        c[row * n + col] = sum;
    }
}

__global__ void matrixMulShared(float *a, float *b, float *c, int n) {
    // Tile size
    const int TILE_SIZE = 32;
    
    __shared__ float s_a[TILE_SIZE][TILE_SIZE];
    __shared__ float s_b[TILE_SIZE][TILE_SIZE];
    
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    for (int t = 0; t < (n + TILE_SIZE - 1) / TILE_SIZE; t++) {
        if (row < n && t * TILE_SIZE + tx < n) {
            s_a[ty][tx] = a[row * n + t * TILE_SIZE + tx];
        } else {
            s_a[ty][tx] = 0.0f;
        }
        
        if (t * TILE_SIZE + ty < n && col < n) {
            s_b[ty][tx] = b[(t * TILE_SIZE + ty) * n + col];
        } else {
            s_b[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        for (int i = 0; i < TILE_SIZE; i++) {
            sum += s_a[ty][i] * s_b[i][tx];
        }
        
        __syncthreads();
    }
    
    if (row < n && col < n) {
        c[row * n + col] = sum;
    }
}

__global__ void matrixMulTensorCores(half *a, half *b, float *c, int n) {
    // WMMA dimensions
    const int WMMA_M = 16;
    const int WMMA_N = 16;
    const int WMMA_K = 16;
    
    using namespace nvcuda::wmma;
    
    fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major> a_frag;
    fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, row_major> b_frag;
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    
    int warpM = (blockIdx.y * blockDim.y + threadIdx.y) / 32;
    int warpN = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    
    if (warpM * WMMA_M < n && warpN * WMMA_N < n) {
        fill_fragment(c_frag, 0.0f);
        
        for (int i = 0; i < n; i += WMMA_K) {
            if (i < n) {
                load_matrix_sync(a_frag, a + warpM * WMMA_M * n + i, n);
                load_matrix_sync(b_frag, b + i * n + warpN * WMMA_N, n);
                mma_sync(c_frag, a_frag, b_frag, c_frag);
            }
        }
        
        store_matrix_sync(c + warpM * WMMA_M * n + warpN * WMMA_N, c_frag, n, row_major);
    }
}

void checkCudaError(cudaError_t error, const char *message) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s: %s\n", message, cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}

void initializeMatrix(float *matrix, int n) {
    for (int i = 0; i < n * n; i++) {
        matrix[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

void convertToHalf(float *src, half *dst, int n) {
    for (int i = 0; i < n * n; i++) {
        dst[i] = __float2half(src[i]);
    }
}

bool verifyResults(float *a, float *b, float *c, int n) {
    float *verification = (float*)malloc(n * n * sizeof(float));
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            verification[i * n + j] = 0.0f;
            for (int k = 0; k < n; k++) {
                verification[i * n + j] += a[i * n + k] * b[k * n + j];
            }
        }
    }
    
    const float epsilon = 1e-2;  
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (fabs(verification[i * n + j] - c[i * n + j]) > epsilon) {
                printf("Verification failed at [%d, %d]: Expected %f, got %f\n", 
                       i, j, verification[i * n + j], c[i * n + j]);
                free(verification);
                return false;
            }
        }
    }
    
    free(verification);
    return true;
}

int main() {
    float *h_a, *h_b, *h_c;
    float *d_a, *d_b, *d_c;
    half *h_a_half, *h_b_half;
    half *d_a_half, *d_b_half;
    
    size_t bytes = N * N * sizeof(float);
    size_t bytes_half = N * N * sizeof(half);
    
    h_a = (float*)malloc(bytes);
    h_b = (float*)malloc(bytes);
    h_c = (float*)malloc(bytes);
    h_a_half = (half*)malloc(bytes_half);
    h_b_half = (half*)malloc(bytes_half);
    
    srand(42);
    initializeMatrix(h_a, N);
    initializeMatrix(h_b, N);
    
    convertToHalf(h_a, h_a_half, N);
    convertToHalf(h_b, h_b_half, N);
    
    checkCudaError(cudaMalloc(&d_a, bytes), "Allocating d_a");
    checkCudaError(cudaMalloc(&d_b, bytes), "Allocating d_b");
    checkCudaError(cudaMalloc(&d_c, bytes), "Allocating d_c");
    checkCudaError(cudaMalloc(&d_a_half, bytes_half), "Allocating d_a_half");
    checkCudaError(cudaMalloc(&d_b_half, bytes_half), "Allocating d_b_half");
    
    checkCudaError(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice), "Copying h_a to d_a");
    checkCudaError(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice), "Copying h_b to d_b");
    checkCudaError(cudaMemcpy(d_a_half, h_a_half, bytes_half, cudaMemcpyHostToDevice), "Copying h_a_half to d_a_half");
    checkCudaError(cudaMemcpy(d_b_half, h_b_half, bytes_half, cudaMemcpyHostToDevice), "Copying h_b_half to d_b_half");
    
    dim3 blockDim(32, 32);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float elapsed_time;
    
    // ====================== NAIVE IMPLEMENTATION ======================
    printf("\n=== Naive Implementation ===\n");
    
    cudaEventRecord(start);
    
    matrixMulNaive<<<gridDim, blockDim>>>(d_a, d_b, d_c, N);
    checkCudaError(cudaGetLastError(), "Launching naive kernel");
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    
    printf("Naive kernel execution time: %.2f ms\n", elapsed_time);
    
    checkCudaError(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost), "Copying d_c to h_c (naive)");
    
    printf("Verification %s\n", verifyResults(h_a, h_b, h_c, N) ? "PASSED" : "FAILED");
    
    // ====================== SHARED MEMORY IMPLEMENTATION ======================
    printf("\n=== Shared Memory Implementation ===\n");
    
    cudaEventRecord(start);
    
    matrixMulShared<<<gridDim, blockDim>>>(d_a, d_b, d_c, N);
    checkCudaError(cudaGetLastError(), "Launching shared memory kernel");
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    
    printf("Shared memory kernel execution time: %.2f ms\n", elapsed_time);
    
    checkCudaError(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost), "Copying d_c to h_c (shared)");
    
    printf("Verification %s\n", verifyResults(h_a, h_b, h_c, N) ? "PASSED" : "FAILED");
    
    // ====================== TENSOR CORES IMPLEMENTATION ======================
    int deviceId;
    cudaGetDevice(&deviceId);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceId);
    
    if (deviceProp.major >= 7) { 
        printf("\n=== Tensor Cores Implementation ===\n");
        
        dim3 wmmaBlockDim(128, 4);
        dim3 wmmaGridDim((N + (WMMA_M * 8 - 1)) / (WMMA_M * 8), (N + WMMA_N - 1) / WMMA_N);
        
        cudaEventRecord(start);
        
        matrixMulTensorCores<<<wmmaGridDim, wmmaBlockDim>>>(d_a_half, d_b_half, d_c, N);
        checkCudaError(cudaGetLastError(), "Launching tensor cores kernel");
        
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed_time, start, stop);
        
        printf("Tensor cores kernel execution time: %.2f ms\n", elapsed_time);
        
        checkCudaError(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost), "Copying d_c to h_c (tensor)");
        
        printf("Verification %s\n", verifyResults(h_a, h_b, h_c, N) ? "PASSED" : "FAILED");
    } else {
        printf("\nThis GPU does not support tensor cores. Skipping tensor cores implementation.\n");
    }
    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFree(d_a_half);
    cudaFree(d_b_half);
    free(h_a);
    free(h_b);
    free(h_c);
    free(h_a_half);
    free(h_b_half);
    
    return 0;
}
