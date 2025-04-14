/**
 * Matrix Transpose Optimization with CUDA
 * 
 * This file implements matrix transpose:
 * 1. Naive implementation
 * 2. Shared memory optimization
 * 3. Shared Memory with Swizzling implementation (from Colfax post)
 * 4. Vectorized Memory Access with Transposition (from Lei Mao's blog)
 * 5. Warp Shuffle implementation (using direct register-to-register transfers)
 *
 * Includes timing code to measure performance.
 * 
 * References:
 * - Colfax Research: https://research.colfax-intl.com/tutorial-matrix-transpose-in-cutlass/
 * - Lei Mao's Blog: https://leimao.github.io/article/CUDA-Matrix-Multiplication-Optimization/
 * - NVIDIA Developer Blog: https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <chrono>
#include <cublas_v2.h>
#include <fstream>

// Add CuTe includes
#include <cute/tensor.hpp>
#include <cute/algorithm/copy.hpp>

// Matrix dimensions
#define SMALL_SIZE 32
#define MEDIUM_SIZE 1024
#define LARGE_SIZE 8192
#define NON_SQUARE_M 1024
#define NON_SQUARE_N 2048

struct PerfResult {
    std::string gpuName;
    std::string implementation;
    int m;
    int n;
    float executionTimeMs;
    float throughputGBps;
    bool verificationPassed;
};

std::vector<PerfResult> allResults;

struct GPUInfo {
    char name[256];
    int major;
    int minor;
};

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

__global__ void matrixTransposeWarpShuffle(float *input, float *output, int m, int n) {
    // each warp will handle a 32x32 tile
    // the 32x32 tile will be processed in 32 segments, each of size 32x1
    // each thread in the warp is responsible for one element in each segment

    // get starting position of the warp's tile
    // int blockId = blockIdx.y * gridDim.x + blockIdx.x;
    int threadId = threadIdx.y * blockDim.x + threadIdx.x;
    // int warpId = threadId / 32;
    int laneId = threadId % 32; // which lane this thread occupies with its warp (thread's position within the warp (0-31)

    int baseRow = blockIdx.y * 32;
    int baseCol = blockIdx.x * 32;

    for (int i = 0; i < 32; i++) {
        int row = baseRow + i;
        int col =  baseCol + laneId;

        float value = 0.0f;
        if(row < m && col < n) {
            value = input[row * n + col];
        }

        // Transpose via warp shuffle:
        // In a traditional transpose, element at position (i,j) moves to position (j,i)
        // In this implementation, we process the matrix in 32×32 tiles, where:
        // - Each iteration 'i' processes one row of the tile
        // - Each thread at lane 'laneId' handles one column
        // For the shuffle operation:
        // - We need the value from position (laneId,i) to go to position (i,laneId)
        // - So each thread gets the value from the thread whose lane ID matches
        //   this thread's current row index 'i'

        // The __shfl_sync(mask, value, sourceLane, width) function allows each thread to receive
        // a value from another thread in the same warp specified by 'sourceLane'
        // Here, we use laneId as the sourceLane to get values from threads corresponding to
        // the properly transposed position
        float transposed = __shfl_sync(0xFFFFFFFF, value, laneId, 32);

        // For a transpose, if we read from (row,col), we should write to (col,row)
        // Since row = baseRow + i and col = baseCol + laneId, we should write to:
        int outRow = baseCol + laneId;  // Column becomes row
        int outCol = baseRow + i;       // Row becomes column
        
        // write transposed value to output
        if(outRow < n && outCol < m) {
            output[outRow * m + outCol] = transposed;
        }
    }
}

void matrixTransposeCublas(cublasHandle_t handle, float *d_input, float *d_output, int m, int n) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    // For cublasSgeam:
    // - First matrix is input (m×n)
    // - Output will be (n×m)
    // - We're setting the operation to transpose the first matrix
    
    // Important: Leading dimensions need to match the actual matrix storage:
    // - For input matrix (m×n), leading dimension is n
    // - For output matrix (n×m), leading dimension is m
    
    cublasSgeam(handle, 
                CUBLAS_OP_T,  // Transpose the input matrix
                CUBLAS_OP_N,  // No operation on the second matrix
                n, m,         // Dimensions of output matrix (n×m)
                &alpha,
                d_input, n,   // Input matrix with leading dimension n
                &beta,
                d_output, m,  // Use d_output as B matrix with leading dimension m
                d_output, m); // Output matrix with leading dimension m
}

template <typename T>
__global__ void matrixTransposeCuTeNaive(T* input, T* output, int m, int n) {
    // Create layouts for row-major matrices
    auto input_layout = cute::make_layout(cute::make_shape(m, n), 
                                         cute::make_stride(n, 1));
    auto output_layout = cute::make_layout(cute::make_shape(n, m), 
                                          cute::make_stride(m, 1));
    
    // Create tensors
    auto input_tensor = cute::make_tensor(input, input_layout);
    auto output_tensor = cute::make_tensor(output, output_layout);
    
    // Get thread coordinates
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < n && y < m) {
        // Transpose: output(x,y) = input(y,x)
        output_tensor(x, y) = input_tensor(y, x);
    }
}

template <typename T>
__global__ void matrixTransposeCuTeShared(T* input, T* output, int m, int n) {
    constexpr int TILE_DIM = 32;
    
    // Declare shared memory tile
    __shared__ T tile[TILE_DIM][TILE_DIM+1]; // +1 padding to avoid bank conflicts
    
    // Create layouts for row-major matrices
    auto input_layout = cute::make_layout(cute::make_shape(m, n), 
                                         cute::make_stride(n, 1));
    auto output_layout = cute::make_layout(cute::make_shape(n, m), 
                                          cute::make_stride(m, 1));
    
    // Create tensors
    auto input_tensor = cute::make_tensor(input, input_layout);
    auto output_tensor = cute::make_tensor(output, output_layout);
    
    // Calculate tile indices
    int block_row = blockIdx.y * TILE_DIM;
    int block_col = blockIdx.x * TILE_DIM;
    
    // Load data from input to shared memory
    if (block_row + threadIdx.y < m && block_col + threadIdx.x < n) {
        tile[threadIdx.y][threadIdx.x] = input_tensor(block_row + threadIdx.y, 
                                                     block_col + threadIdx.x);
    }
    
    __syncthreads();
    
    // Transpose coordinates for output
    int output_row = block_col + threadIdx.y;
    int output_col = block_row + threadIdx.x;
    
    // Write to output with transposed coordinates
    if (output_row < n && output_col < m) {
        output_tensor(output_row, output_col) = tile[threadIdx.x][threadIdx.y];
    }
}

template <typename T>
__global__ void matrixTransposeCuTeSwizzled(T* input, T* output, int m, int n) {
    constexpr int TILE_DIM = 32;
    
    // Declare shared memory with swizzling pattern
    __shared__ T tile[TILE_DIM][TILE_DIM];
    
    // Create layouts for row-major matrices
    auto input_layout = cute::make_layout(cute::make_shape(m, n), 
                                         cute::make_stride(n, 1));
    auto output_layout = cute::make_layout(cute::make_shape(n, m), 
                                          cute::make_stride(m, 1));
    
    // Create tensors
    auto input_tensor = cute::make_tensor(input, input_layout);
    auto output_tensor = cute::make_tensor(output, output_layout);
    
    // Calculate tile indices
    int block_row = blockIdx.y * TILE_DIM;
    int block_col = blockIdx.x * TILE_DIM;
    
    // Calculate linear index within tile
    int linear_idx = threadIdx.y * TILE_DIM + threadIdx.x;
    
    // Apply swizzle for shared memory
    int swizzled_idx = linear_idx ^ ((linear_idx >> 5) & 0x1F);
    int swizzled_row = swizzled_idx / TILE_DIM;
    int swizzled_col = swizzled_idx % TILE_DIM;
    
    // Load data using swizzled pattern
    if (block_row + threadIdx.y < m && block_col + threadIdx.x < n) {
        tile[swizzled_row][swizzled_col] = input_tensor(block_row + threadIdx.y, 
                                                       block_col + threadIdx.x);
    }
    
    __syncthreads();
    
    // Transpose coordinates for output
    int output_row = block_col + threadIdx.y;
    int output_col = block_row + threadIdx.x;
    
    // Recalculate linear index and swizzle for reading
    linear_idx = threadIdx.x * TILE_DIM + threadIdx.y;  // Note: x and y swapped
    swizzled_idx = linear_idx ^ ((linear_idx >> 5) & 0x1F);
    swizzled_row = swizzled_idx / TILE_DIM;
    swizzled_col = swizzled_idx % TILE_DIM;
    
    // Write to output with transposed and swizzled coordinates
    if (output_row < n && output_col < m) {
        output_tensor(output_row, output_col) = tile[swizzled_row][swizzled_col];
    }
}

// =============== HELPER FUNCTIONS ===============

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

struct MatrixSize {
    int m, n;
    const char* name;
};

// Create an array of sizes to test
MatrixSize matrix_sizes[] = {
    {SMALL_SIZE, SMALL_SIZE, "Small (32x32)"},
    {MEDIUM_SIZE, MEDIUM_SIZE, "Medium (1024x1024)"},
    // Uncomment for testing larger sizes later
    {LARGE_SIZE, LARGE_SIZE, "Large (8192x8192)"},
    {NON_SQUARE_M, NON_SQUARE_N, "Non-square (1024x2048)"}
};

float calculateTransposeThroughput(int m, int n, float timeMs) {
    // For transpose, we read m*n elements and write m*n elements (each 4 bytes)
    float total_bytes = 2.0f * m * n * sizeof(float);
    float timeS = timeMs / 1000.0f;  // Convert ms to seconds
    return (total_bytes / (1024.0f * 1024.0f * 1024.0f)) / timeS;  // Return in GB/s
}

std::vector<GPUInfo> detectGPUs() {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    
    std::vector<GPUInfo> gpus;
    
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, i);
        
        GPUInfo gpu;
        strcpy(gpu.name, deviceProp.name);
        gpu.major = deviceProp.major;
        gpu.minor = deviceProp.minor;
        
        gpus.push_back(gpu);
    }
    
    return gpus;
}

bool isTargetGPU(const char* name) {
    return (strstr(name, "RTX 2080 Ti") != NULL ||
            strstr(name, "A100") != NULL ||
            strstr(name, "H100") != NULL);
}

void saveResultsToCSV(const std::vector<PerfResult>& results, const char* filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        fprintf(stderr, "Error: Could not open file %s for writing\n", filename);
        return;
    }
    
    // Write CSV header
    file << "GPU,Implementation,M,N,Time_ms,Throughput_GBps,Verification\n";
    
    // Write results
    for (const auto& result : results) {
        file << result.gpuName << ","
             << result.implementation << ","
             << result.m << ","
             << result.n << ","
             << result.executionTimeMs << ","
             << result.throughputGBps << ","
             << (result.verificationPassed ? "PASSED" : "FAILED") << "\n";
    }
    
    file.close();
    printf("\nResults saved to %s\n", filename);
}

int main(int argc, char** argv) {
    // Parse command line args (if you want to add this capability)
    std::string targetGPU = "";  // Empty string means all GPUs
    
    // Detect available GPUs
    std::vector<GPUInfo> gpus = detectGPUs();
    std::vector<PerfResult> allResults;
    
    // Print detected GPUs
    printf("Detected GPUs:\n");
    for (const auto& gpu : gpus) {
        printf("  %s (SM %d.%d)\n", gpu.name, gpu.major, gpu.minor);
    }
    
    // Loop through each GPU
    for (int deviceId = 0; deviceId < gpus.size(); deviceId++) {
        const auto& gpu = gpus[deviceId];
        
        // Skip if not a target GPU or if filtering by name
        if (!isTargetGPU(gpu.name) || 
            (!targetGPU.empty() && strstr(gpu.name, targetGPU.c_str()) == NULL)) {
            printf("Skipping GPU: %s (not selected for testing)\n", gpu.name);
            continue;
        }
        
        printf("\n=== Testing on GPU: %s ===\n", gpu.name);
        
        // Set current device
        cudaSetDevice(deviceId);
        
        // Create CUDA events for timing
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        float elapsed_time;

        // cuBLAS handle
        cublasHandle_t cublasHandle;
        cublasCreate(&cublasHandle);
        
        // Loop through each matrix size
        for (int size_idx = 0; size_idx < sizeof(matrix_sizes)/sizeof(matrix_sizes[0]); size_idx++) {
            int M = matrix_sizes[size_idx].m;
            int N = matrix_sizes[size_idx].n;
            const char* size_name = matrix_sizes[size_idx].name;
            
            printf("\n\n========== Testing %s matrices on %s ==========\n", size_name, gpu.name);
            
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
            
            // ====================== NAIVE IMPLEMENTATION ======================
            printf("\n=== Naive Matrix Transpose Implementation ===\n");
            printf("Transposing a %d x %d matrix\n", M, N);
            
            cudaEventRecord(start);
            
            matrixTransposeNaive<<<gridDim, blockDim>>>(d_input, d_output, M, N);
            checkCudaError(cudaGetLastError(), "Launching naive transpose kernel");
            
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&elapsed_time, start, stop);
            
            float throughput_GBps = calculateTransposeThroughput(M, N, elapsed_time);
            printf("Naive transpose kernel execution time: %.2f ms\n", elapsed_time);
            printf("Naive transpose kernel throughput: %.2f GB/s\n", throughput_GBps);
            
            // Copy result back to host
            checkCudaError(cudaMemcpy(h_output, d_output, bytes_output, cudaMemcpyDeviceToHost), "Copying d_output to h_output");
            
            // Verify the result
            bool verification_result = verifyTranspose(h_input, h_output, M, N);
            printf("Verification %s\n", verification_result ? "PASSED" : "FAILED");
            
            // Store results
            PerfResult result;
            result.gpuName = gpu.name;
            result.implementation = "Naive";
            result.m = M;
            result.n = N;
            result.executionTimeMs = elapsed_time;
            result.throughputGBps = throughput_GBps;
            result.verificationPassed = verification_result;
            allResults.push_back(result);
            
            // ====================== SHARED MEMORY IMPLEMENTATION ======================
            printf("\n=== Shared Memory Matrix Transpose Implementation ===\n");
            
            cudaEventRecord(start);
            
            matrixTransposeShared<<<gridDim, blockDim>>>(d_input, d_output, M, N);
            checkCudaError(cudaGetLastError(), "Launching shared memory transpose kernel");
            
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&elapsed_time, start, stop);
            
            float throughput_GBps_shared = calculateTransposeThroughput(M, N, elapsed_time);
            printf("Shared memory transpose kernel execution time: %.2f ms\n", elapsed_time);
            printf("Shared memory transpose kernel throughput: %.2f GB/s\n", throughput_GBps_shared);
            
            // Copy result back to host
            checkCudaError(cudaMemcpy(h_output, d_output, bytes_output, cudaMemcpyDeviceToHost), "Copying d_output to h_output");
            
            // Verify the result
            verification_result = verifyTranspose(h_input, h_output, M, N);
            printf("Verification %s\n", verification_result ? "PASSED" : "FAILED");

            // Store results
            result.implementation = "Shared";
            result.executionTimeMs = elapsed_time;
            result.throughputGBps = throughput_GBps_shared;
            result.verificationPassed = verification_result;
            allResults.push_back(result);

            // ====================== SHARED MEMORY WITH SWIZZLING IMPLEMENTATION ======================
            printf("\n=== Shared Memory with Swizzling Matrix Transpose Implementation ===\n");

            cudaEventRecord(start);

            matrixTransposeSwizzled<<<gridDim, blockDim>>>(d_input, d_output, M, N);
            checkCudaError(cudaGetLastError(), "Launching swizzled transpose kernel");

            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&elapsed_time, start, stop);

            float throughput_GBps_swizzled = calculateTransposeThroughput(M, N, elapsed_time);
            printf("Shared memory with swizzling transpose kernel execution time: %.2f ms\n", elapsed_time);
            printf("Shared memory with swizzling transpose kernel throughput: %.2f GB/s\n", throughput_GBps_swizzled);

            // Copy result back to host
            checkCudaError(cudaMemcpy(h_output, d_output, bytes_output, cudaMemcpyDeviceToHost), "Copying d_output to h_output");

            // Verify the result
            verification_result = verifyTranspose(h_input, h_output, M, N);
            printf("Verification %s\n", verification_result ? "PASSED" : "FAILED");

            // Store results
            result.implementation = "Swizzled";
            result.executionTimeMs = elapsed_time;
            result.throughputGBps = throughput_GBps_swizzled;
            result.verificationPassed = verification_result;
            allResults.push_back(result);

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

            float throughput_GBps_vec = calculateTransposeThroughput(M, N, elapsed_time);
            printf("Vectorized transpose kernel execution time: %.2f ms\n", elapsed_time);
            printf("Vectorized transpose kernel throughput: %.2f GB/s\n", throughput_GBps_vec);

            // Copy result back to host
            checkCudaError(cudaMemcpy(h_output, d_output, bytes_output, cudaMemcpyDeviceToHost), "Copying d_output to h_output");

            // Verify the result
            verification_result = verifyTranspose(h_input, h_output, M, N);
            printf("Verification %s\n", verification_result ? "PASSED" : "FAILED");

            // Store results
            result.implementation = "Vectorized";
            result.executionTimeMs = elapsed_time;
            result.throughputGBps = throughput_GBps_vec;
            result.verificationPassed = verification_result;
            allResults.push_back(result);

            // ====================== WARP SHUFFLE IMPLEMENTATION ======================
            printf("\n=== Warp Shuffle Matrix Transpose Implementation ===\n");

            // configuring for warp-level operations
            dim3 blockDimWarp(32, 8);  // 256 threads per block (8 warps of 32 threads each)
            // grid dimensions to cover the entire matrix with 32x32 tiles:
            // using ceiling division (n+32-1)/32 to ensure we have enough blocks to cover any partial tiles at the edges
            // this ensures every matrix element is processed even when matrix dimensions aren't multiples of 32
            dim3 gridDimWarp((N + 32 - 1) / 32, (M + 32 - 1) / 32);

            cudaEventRecord(start);

            matrixTransposeWarpShuffle<<<gridDimWarp, blockDimWarp>>>(d_input, d_output, M, N);
            checkCudaError(cudaGetLastError(), "Launching warp shuffle transpose kernel");

            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&elapsed_time, start, stop);

            float throughput_GBps_warp = calculateTransposeThroughput(M, N, elapsed_time);
            printf("Warp shuffle transpose kernel execution time: %.2f ms\n", elapsed_time);
            printf("Warp shuffle transpose kernel throughput: %.2f GB/s\n", throughput_GBps_warp);

            // Copy result back to host
            checkCudaError(cudaMemcpy(h_output, d_output, bytes_output, cudaMemcpyDeviceToHost), "Copying d_output to h_output");

            // Verify the result
            verification_result = verifyTranspose(h_input, h_output, M, N);
            printf("Verification %s\n", verification_result ? "PASSED" : "FAILED");

            // Store results
            result.implementation = "WarpShuffle";
            result.executionTimeMs = elapsed_time;
            result.throughputGBps = throughput_GBps_warp;
            result.verificationPassed = verification_result;
            allResults.push_back(result);

            // ====================== CUBLAS IMPLEMENTATION ======================
            printf("\n=== cuBLAS Matrix Transpose Implementation ===\n");

            cudaEventRecord(start);

            matrixTransposeCublas(cublasHandle, d_input, d_output, M, N);
            checkCudaError(cudaGetLastError(), "Launching cuBLAS transpose");

            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&elapsed_time, start, stop);

            float throughput_GBps_cublas = calculateTransposeThroughput(M, N, elapsed_time);
            printf("cuBLAS transpose execution time: %.2f ms\n", elapsed_time);
            printf("cuBLAS transpose kernel throughput: %.2f GB/s\n", throughput_GBps_cublas);

            // Copy result back to host
            checkCudaError(cudaMemcpy(h_output, d_output, bytes_output, cudaMemcpyDeviceToHost), "Copying d_output to h_output");

            // Verify the result
            verification_result = verifyTranspose(h_input, h_output, M, N);
            printf("Verification %s\n", verification_result ? "PASSED" : "FAILED");

            // Store results
            result.implementation = "cuBLAS";
            result.executionTimeMs = elapsed_time;
            result.throughputGBps = throughput_GBps_cublas;
            result.verificationPassed = verification_result;
            allResults.push_back(result);

            // ====================== CUTE NAIVE IMPLEMENTATION ======================
            printf("\n=== CuTe Naive Matrix Transpose Implementation ===\n");

            cudaEventRecord(start);

            matrixTransposeCuTeNaive<<<gridDim, blockDim>>>(d_input, d_output, M, N);
            checkCudaError(cudaGetLastError(), "Launching CuTe naive transpose kernel");

            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&elapsed_time, start, stop);

            float throughput_GBps_cute_naive = calculateTransposeThroughput(M, N, elapsed_time);
            printf("CuTe naive transpose kernel execution time: %.2f ms\n", elapsed_time);
            printf("CuTe naive transpose kernel throughput: %.2f GB/s\n", throughput_GBps_cute_naive);

            // Copy result back to host
            checkCudaError(cudaMemcpy(h_output, d_output, bytes_output, cudaMemcpyDeviceToHost), "Copying d_output to h_output");

            // Verify the result
            verification_result = verifyTranspose(h_input, h_output, M, N);
            printf("Verification %s\n", verification_result ? "PASSED" : "FAILED");

            // Store results
            result.implementation = "CuTe_Naive";
            result.executionTimeMs = elapsed_time;
            result.throughputGBps = throughput_GBps_cute_naive;
            result.verificationPassed = verification_result;
            allResults.push_back(result);

            // ====================== CUTE SHARED MEMORY IMPLEMENTATION ======================
            printf("\n=== CuTe Shared Memory Matrix Transpose Implementation ===\n");

            cudaEventRecord(start);

            matrixTransposeCuTeShared<<<gridDim, blockDim>>>(d_input, d_output, M, N);
            checkCudaError(cudaGetLastError(), "Launching CuTe shared memory transpose kernel");

            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&elapsed_time, start, stop);

            float throughput_GBps_cute_shared = calculateTransposeThroughput(M, N, elapsed_time);
            printf("CuTe shared memory transpose kernel execution time: %.2f ms\n", elapsed_time);
            printf("CuTe shared memory transpose kernel throughput: %.2f GB/s\n", throughput_GBps_cute_shared);

            // Copy result back to host
            checkCudaError(cudaMemcpy(h_output, d_output, bytes_output, cudaMemcpyDeviceToHost), "Copying d_output to h_output");

            // Verify the result
            verification_result = verifyTranspose(h_input, h_output, M, N);
            printf("Verification %s\n", verification_result ? "PASSED" : "FAILED");

            // Store results
            result.implementation = "CuTe_Shared";
            result.executionTimeMs = elapsed_time;
            result.throughputGBps = throughput_GBps_cute_shared;
            result.verificationPassed = verification_result;
            allResults.push_back(result);

            // ====================== CUTE SWIZZLED IMPLEMENTATION ======================
            printf("\n=== CuTe Swizzled Matrix Transpose Implementation ===\n");

            cudaEventRecord(start);

            matrixTransposeCuTeSwizzled<<<gridDim, blockDim>>>(d_input, d_output, M, N);
            checkCudaError(cudaGetLastError(), "Launching CuTe swizzled transpose kernel");

            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&elapsed_time, start, stop);

            float throughput_GBps_cute_swizzled = calculateTransposeThroughput(M, N, elapsed_time);
            printf("CuTe swizzled transpose kernel execution time: %.2f ms\n", elapsed_time);
            printf("CuTe swizzled transpose kernel throughput: %.2f GB/s\n", throughput_GBps_cute_swizzled);

            // Copy result back to host
            checkCudaError(cudaMemcpy(h_output, d_output, bytes_output, cudaMemcpyDeviceToHost), "Copying d_output to h_output");

            // Verify the result
            verification_result = verifyTranspose(h_input, h_output, M, N);
            printf("Verification %s\n", verification_result ? "PASSED" : "FAILED");

            // Store results
            result.implementation = "CuTe_Swizzled";
            result.executionTimeMs = elapsed_time;
            result.throughputGBps = throughput_GBps_cute_swizzled;
            result.verificationPassed = verification_result;
            allResults.push_back(result);

            // Clean up for this matrix size
            cudaFree(d_input);
            cudaFree(d_output);
            free(h_input);
            free(h_output);
        }
        
        // Clean up CUDA events
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        
        // Destroy cuBLAS handle
        cublasDestroy(cublasHandle);
    }
    
    // Save results to CSV
    saveResultsToCSV(allResults, "matrix_transpose_performance.csv");
    
    return 0;
}
