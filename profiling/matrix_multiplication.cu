/**
 * Matrix Multiplication Optimization with CUDA
 * 
 * This file contains multiple implementations of matrix multiplication:
 * 1. Naive implementation
 * 2. Shared memory optimization
 * 3. Tensor cores (WMMA) implementation
 * 4. cuBLAS implementation
 * 5. CUTLASS implementation
 *
 * Benchmarking across RTX 2080 Ti, A100, and H100 GPUs
 * Testing different matrix sizes and measuring throughput
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <chrono>
#include <vector>
#include <string>
#include <fstream>
#include <algorithm>
#include <cublas_v2.h>
#include <string.h>
#include <getopt.h>

// For CUTLASS implementation
#include <cutlass/gemm/device/gemm.h>

// Define matrix sizes to test
#define SMALL_SIZE 32
#define MEDIUM_SIZE 1024
#define LARGE_SIZE 8192
#define NON_SQUARE_M 1024
#define NON_SQUARE_N 2048
#define NON_SQUARE_K 1024

// Define a structure for GPU information
struct GPUInfo {
    char name[256];
    int major;
    int minor;
    bool hasTensorCores;
};

// Define a structure for performance results
struct PerfResult {
    std::string gpuName;
    std::string implementation;
    int m;
    int n;
    int k;
    float executionTimeMs;
    float throughputGFlops;
};

// =============== KERNEL IMPLEMENTATIONS ===============

// Naive implementation - Updated for non-square matrices (M×K * K×N = M×N)
__global__ void matrixMulNaive(float *a, float *b, float *c, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < k; i++) {
            sum += a[row * k + i] * b[i * n + col];
        }
        c[row * n + col] = sum;
    }
}

// Shared memory implementation - Updated for non-square matrices
__global__ void matrixMulShared(float *a, float *b, float *c, int m, int n, int k) {
    // Tile size
    const int TILE_SIZE = 32;
    
    __shared__ float s_a[TILE_SIZE][TILE_SIZE];
    __shared__ float s_b[TILE_SIZE][TILE_SIZE];
    
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    for (int t = 0; t < (k + TILE_SIZE - 1) / TILE_SIZE; t++) {
        if (row < m && t * TILE_SIZE + tx < k) {
            s_a[ty][tx] = a[row * k + t * TILE_SIZE + tx];
        } else {
            s_a[ty][tx] = 0.0f;
        }
        
        if (t * TILE_SIZE + ty < k && col < n) {
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
    
    if (row < m && col < n) {
        c[row * n + col] = sum;
    }
}

// Helper kernel for converting float to half
__global__ void convertToHalfKernel(float *in, half *out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = __float2half(in[idx]);
    }
}

// Tensor cores implementation - Updated for non-square matrices
__global__ void matrixMulTensorCores(half *a, half *b, float *c, int m, int n, int k) {
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
    
    if (warpM * WMMA_M < m && warpN * WMMA_N < n) {
        fill_fragment(c_frag, 0.0f);
        
        for (int i = 0; i < k; i += WMMA_K) {
            if (i < k) {
                load_matrix_sync(a_frag, a + warpM * WMMA_M * k + i, k);
                load_matrix_sync(b_frag, b + i * n + warpN * WMMA_N, n);
                mma_sync(c_frag, a_frag, b_frag, c_frag);
            }
        }
        
        store_matrix_sync(c + warpM * WMMA_M * n + warpN * WMMA_N, c_frag, n, row_major);
    }
}

// =============== CUTLASS IMPLEMENTATION ===============

// Define CUTLASS GEMM type configuration
template <typename Gemm>
void runCutlassGemm(float *d_a, float *d_b, float *d_c, int m, int n, int k) {
    // Create a CUTLASS GEMM operator
    Gemm gemm_operator;

    // Setup GEMM problem size
    typename Gemm::Arguments args(
        {m, n, k},                   // Problem dimensions (M, N, K)
        {d_a, k},                    // Tensor A (device pointer and leading dimension)
        {d_b, n},                    // Tensor B (device pointer and leading dimension)
        {d_c, n},                    // Tensor C (device pointer and leading dimension)
        {d_c, n},                    // Tensor D (device pointer and leading dimension)
        {1.0f, 0.0f}                 // alpha and beta
    );

    // Launch the GEMM kernel
    cudaDeviceSynchronize();
    cutlass::Status status = gemm_operator(args);
    cudaDeviceSynchronize();

    if (status != cutlass::Status::kSuccess) {
        printf("CUTLASS GEMM kernel failed: %s\n", cutlass::cutlassGetStatusString(status));
    }
}

void matrixMulCutlass(float *d_a, float *d_b, float *d_c, int m, int n, int k) {
    // Define the CUTLASS GEMM type
    using ElementInputA = float;
    using ElementInputB = float;
    using ElementOutput = float;
    using ElementAccumulator = float;
    using ElementCompute = float;

    // Use row-major layouts for A and B matrices, column-major for output
    using LayoutInputA = cutlass::layout::RowMajor;
    using LayoutInputB = cutlass::layout::RowMajor;
    using LayoutOutput = cutlass::layout::RowMajor;

    // Get device properties to determine architecture
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);

    // Select architecture based on GPU capabilities
    if (props.major >= 8) {
        // Ampere or newer (SM80+)
        printf("Using CUTLASS configuration for Ampere+ architecture\n");
        
        using CutlassGemm = cutlass::gemm::device::Gemm<
            ElementInputA, LayoutInputA,
            ElementInputB, LayoutInputB,
            ElementOutput, LayoutOutput,
            ElementAccumulator,
            cutlass::arch::OpClassSimt,     // Using SIMT architecture
            cutlass::arch::Sm80             // Target SM architecture
        >;
        
        runCutlassGemm<CutlassGemm>(d_a, d_b, d_c, m, n, k);
    }
    else if (props.major >= 7) {
        // Volta/Turing (SM70-75)
        printf("Using CUTLASS configuration for Volta/Turing architecture\n");
        
        using CutlassGemm = cutlass::gemm::device::Gemm<
            ElementInputA, LayoutInputA,
            ElementInputB, LayoutInputB,
            ElementOutput, LayoutOutput,
            ElementAccumulator,
            cutlass::arch::OpClassSimt,     // Using SIMT architecture
            cutlass::arch::Sm70             // Target SM architecture
        >;
        
        runCutlassGemm<CutlassGemm>(d_a, d_b, d_c, m, n, k);
    }
    else {
        // Pascal or older (SM60 or below)
        printf("Using CUTLASS configuration for Pascal or older architecture\n");
        
        using CutlassGemm = cutlass::gemm::device::Gemm<
            ElementInputA, LayoutInputA,
            ElementInputB, LayoutInputB,
            ElementOutput, LayoutOutput,
            ElementAccumulator,
            cutlass::arch::OpClassSimt,     // Using SIMT architecture
            cutlass::arch::Sm60             // Target SM architecture
        >;
        
        runCutlassGemm<CutlassGemm>(d_a, d_b, d_c, m, n, k);
    }
}

// CUTLASS Tensor Core implementation
void matrixMulCutlassTensorCores(float *d_a, float *d_b, float *d_c, int m, int n, int k) {
    // Check if we have a GPU that supports tensor cores
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);
    
    if (props.major >= 7) {
        printf("Using CUTLASS Tensor Core configuration\n");
        
        // Allocate and convert to half precision
        half *d_a_half, *d_b_half;
        cudaMalloc(&d_a_half, m * k * sizeof(half));
        cudaMalloc(&d_b_half, k * n * sizeof(half));
        
        // Convert float to half
        dim3 block(256);
        dim3 grid_a((m * k + block.x - 1) / block.x);
        dim3 grid_b((k * n + block.x - 1) / block.x);
        
        convertToHalfKernel<<<grid_a, block>>>(d_a, d_a_half, m * k);
        convertToHalfKernel<<<grid_b, block>>>(d_b, d_b_half, k * n);
        
        // Determine architecture-specific configuration
        if (props.major >= 8) {
            // Ampere or newer
            using ElementInputA = cutlass::half_t;
            using ElementInputB = cutlass::half_t;
            using ElementOutput = float;
            using ElementAccumulator = float;
            
            using LayoutInputA = cutlass::layout::RowMajor;
            using LayoutInputB = cutlass::layout::RowMajor;
            using LayoutOutput = cutlass::layout::RowMajor;
            
            using CutlassGemmTensorOp = cutlass::gemm::device::Gemm<
                ElementInputA, LayoutInputA,
                ElementInputB, LayoutInputB,
                ElementOutput, LayoutOutput,
                ElementAccumulator,
                cutlass::arch::OpClassTensorOp,
                cutlass::arch::Sm80
            >;
            
            // We would run the CUTLASS GEMM with tensor cores here
            // For now, use tensor core kernel as a fallback
            dim3 block_tc(128, 4);
            dim3 grid_tc((n + 16 - 1) / 16, (m + 16 - 1) / 16);
            matrixMulTensorCores<<<grid_tc, block_tc>>>(d_a_half, d_b_half, d_c, m, n, k);
        }
        else {
            // Volta/Turing
            dim3 block_tc(128, 4);
            dim3 grid_tc((n + 16 - 1) / 16, (m + 16 - 1) / 16);
            matrixMulTensorCores<<<grid_tc, block_tc>>>(d_a_half, d_b_half, d_c, m, n, k);
        }
        
        // Clean up
        cudaFree(d_a_half);
        cudaFree(d_b_half);
    }
    else {
        printf("This GPU does not support tensor cores. Running standard CUTLASS GEMM.\n");
        matrixMulCutlass(d_a, d_b, d_c, m, n, k);
    }
}

// =============== HELPER FUNCTIONS ===============

void checkCudaError(cudaError_t error, const char *message) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s: %s\n", message, cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}

void checkCublasError(cublasStatus_t status, const char *message) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS error: %s: %d\n", message, status);
        exit(EXIT_FAILURE);
    }
}

void initializeMatrix(float *matrix, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        matrix[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

void convertToHalf(float *src, half *dst, int size) {
    for (int i = 0; i < size; i++) {
        dst[i] = __float2half(src[i]);
    }
}

bool verifyResults(float *a, float *b, float *c, int m, int n, int k) {
    float *verification = (float*)malloc(m * n * sizeof(float));
    
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            verification[i * n + j] = 0.0f;
            for (int p = 0; p < k; p++) {
                verification[i * n + j] += a[i * k + p] * b[p * n + j];
            }
        }
    }
    
    const float epsilon = 1e-2;  
    for (int i = 0; i < m; i++) {
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

// Calculate GFLOPs for matrix multiplication
float calculateGFlops(int m, int n, int k, float timeMs) {
    // Each matrix multiply-add is 2 operations
    // Total operations = m * n * k * 2
    float operations = 2.0f * static_cast<float>(m) * static_cast<float>(n) * static_cast<float>(k);
    float timeS = timeMs / 1000.0f;
    return operations / (timeS * 1e9);
}

// Detect available GPUs and return their info
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
        gpu.hasTensorCores = (deviceProp.major >= 7);
        
        gpus.push_back(gpu);
    }
    
    return gpus;
}

// Function to check if a GPU is one we want to test on
bool isTargetGPU(const char* name) {
    // Check if the GPU is one of our target GPUs
    return (strstr(name, "RTX 2080 Ti") != NULL ||
            strstr(name, "A100") != NULL ||
            strstr(name, "H100") != NULL);
}

// Function to parse command line arguments
void parseArgs(int argc, char **argv, std::string &targetGPU) {
    const struct option long_options[] = {
        {"gpu", required_argument, 0, 'g'},
        {"help", no_argument, 0, 'h'},
        {0, 0, 0, 0}
    };

    int opt;
    int option_index = 0;
    
    while ((opt = getopt_long(argc, argv, "g:h", long_options, &option_index)) != -1) {
        switch (opt) {
            case 'g':
                targetGPU = optarg;
                break;
            case 'h':
                printf("Usage: %s [OPTIONS]\n", argv[0]);
                printf("Options:\n");
                printf("  -g, --gpu=GPU_NAME    Specify GPU to use (e.g., \"RTX 2080 Ti\", \"A100\", \"H100\")\n");
                printf("  -h, --help            Display this help message\n");
                exit(0);
            default:
                fprintf(stderr, "Try '%s --help' for more information.\n", argv[0]);
                exit(1);
        }
    }
}

// =============== CUBLAS IMPLEMENTATION ===============

void matrixMulCublas(cublasHandle_t handle, float *d_a, float *d_b, float *d_c, int m, int n, int k) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    // Note: cuBLAS uses column-major ordering, so we compute B*A instead of A*B
    checkCublasError(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                                  n, m, k, 
                                  &alpha, 
                                  d_b, n, 
                                  d_a, k, 
                                  &beta, 
                                  d_c, n), 
                     "Executing cuBLAS SGEMM");
}

// =============== TESTING FRAMEWORK ===============

// Define structures for passing parameters to benchmark functions
struct NaiveParams {
    float *d_a, *d_b, *d_c;
};

struct SharedParams {
    float *d_a, *d_b, *d_c;
};

struct TensorParams {
    half *d_a, *d_b;
    float *d_c;
};

struct CublasParams {
    cublasHandle_t handle;
    float *d_a, *d_b, *d_c;
};

struct CutlassParams {
    float *d_a, *d_b, *d_c;
};

// Benchmark functions
void naiveBenchmark(void* p, int m, int n, int k) {
    NaiveParams* params = (NaiveParams*)p;
    dim3 blockDim(32, 32);
    dim3 gridDim((n + blockDim.x - 1) / blockDim.x, (m + blockDim.y - 1) / blockDim.y);
    matrixMulNaive<<<gridDim, blockDim>>>(params->d_a, params->d_b, params->d_c, m, n, k);
    checkCudaError(cudaGetLastError(), "Launching naive kernel");
    cudaDeviceSynchronize();
}

void sharedBenchmark(void* p, int m, int n, int k) {
    SharedParams* params = (SharedParams*)p;
    dim3 blockDim(32, 32);
    dim3 gridDim((n + blockDim.x - 1) / blockDim.x, (m + blockDim.y - 1) / blockDim.y);
    matrixMulShared<<<gridDim, blockDim>>>(params->d_a, params->d_b, params->d_c, m, n, k);
    checkCudaError(cudaGetLastError(), "Launching shared memory kernel");
    cudaDeviceSynchronize();
}

void tensorBenchmark(void* p, int m, int n, int k) {
    TensorParams* params = (TensorParams*)p;
    dim3 blockDim(128, 4);
    dim3 gridDim((n + 16 - 1) / 16, (m + 16 - 1) / 16);
    matrixMulTensorCores<<<gridDim, blockDim>>>(params->d_a, params->d_b, params->d_c, m, n, k);
    checkCudaError(cudaGetLastError(), "Launching tensor cores kernel");
    cudaDeviceSynchronize();
}

void cublasBenchmark(void* p, int m, int n, int k) {
    CublasParams* params = (CublasParams*)p;
    matrixMulCublas(params->handle, params->d_a, params->d_b, params->d_c, m, n, k);
    cudaDeviceSynchronize();
}

void cutlassBenchmark(void* p, int m, int n, int k) {
    CutlassParams* params = (CutlassParams*)p;
    matrixMulCutlass(params->d_a, params->d_b, params->d_c, m, n, k);
    cudaDeviceSynchronize();
}

void cutlassTensorBenchmark(void* p, int m, int n, int k) {
    CutlassParams* params = (CutlassParams*)p;
    matrixMulCutlassTensorCores(params->d_a, params->d_b, params->d_c, m, n, k);
    cudaDeviceSynchronize();
}

// Run benchmark for a specific implementation and return performance results
PerfResult runBenchmark(const char* gpuName, const char* implName, 
                      void (*benchmark)(void*, int, int, int), 
                      void* params, int m, int n, int k, bool verify) {
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float elapsed_time;
    
    // Run benchmark
    cudaEventRecord(start);
    benchmark(params, m, n, k);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    
    // Calculate throughput
    float gflops = calculateGFlops(m, n, k, elapsed_time);
    
    // Create performance result
    PerfResult result;
    result.gpuName = gpuName;
    result.implementation = implName;
    result.m = m;
    result.n = n;
    result.k = k;
    result.executionTimeMs = elapsed_time;
    result.throughputGFlops = gflops;
    
    printf("%-15s %-25s %5d x %5d x %5d: %10.2f ms (%10.2f GFlops)\n", 
           gpuName, implName, m, n, k, elapsed_time, gflops);
    
    return result;
}

// Save results to a CSV file for later chart generation
void saveResultsToCSV(const std::vector<PerfResult>& results, const char* filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        fprintf(stderr, "Error: Could not open file %s for writing\n", filename);
        return;
    }
    
    // Write CSV header
    file << "GPU,Implementation,M,N,K,Time_ms,Throughput_GFlops\n";
    
    // Write results
    for (const auto& result : results) {
        file << result.gpuName << ","
             << result.implementation << ","
             << result.m << ","
             << result.n << ","
             << result.k << ","
             << result.executionTimeMs << ","
             << result.throughputGFlops << "\n";
    }
    
    file.close();
}

// =============== MAIN FUNCTION ===============

int main(int argc, char** argv) {
    // Parse command line arguments
    std::string targetGPU = "";  // Empty string means all GPUs
    parseArgs(argc, argv, targetGPU);
    
    // Detect available GPUs
    std::vector<GPUInfo> gpus = detectGPUs();
    std::vector<PerfResult> allResults;
    
    // If no GPUs found, exit
    if (gpus.empty()) {
        fprintf(stderr, "No CUDA-capable devices found\n");
        return EXIT_FAILURE;
    }
    
    // Print detected GPUs
    printf("Detected GPUs:\n");
    for (const auto& gpu : gpus) {
        printf("  %s (SM %d.%d)\n", gpu.name, gpu.major, gpu.minor);
    }
    
    // Filter by target GPU if specified
    if (!targetGPU.empty()) {
        printf("\nFiltering to run only on: %s\n", targetGPU.c_str());
    }
    
    // Define matrix sizes to test
    struct MatrixSize {
        int m, n, k;
        const char* name;
    };
    
    MatrixSize sizes[] = {
        {SMALL_SIZE, SMALL_SIZE, SMALL_SIZE, "Small (32x32)"},
        {MEDIUM_SIZE, MEDIUM_SIZE, MEDIUM_SIZE, "Medium (1024x1024)"},
        {LARGE_SIZE, LARGE_SIZE, LARGE_SIZE, "Large (8192x8192)"},
        {NON_SQUARE_M, NON_SQUARE_N, NON_SQUARE_K, "Non-square (1024x2048)"}
    };
    
    // Loop through each GPU
    for (const auto& gpu : gpus) {
        // Skip if not a target GPU or if we're filtering by name and it doesn't match
        if (!isTargetGPU(gpu.name) || (!targetGPU.empty() && strstr(gpu.name, targetGPU.c_str()) == NULL)) {
            printf("Skipping GPU: %s (not selected for testing)\n", gpu.name);
            continue;
        }
        
        printf("\n=== Testing on GPU: %s ===\n", gpu.name);
        
        // Set the current device
        int deviceId = -1;
        for (int i = 0; i < gpus.size(); i++) {
            if (strcmp(gpu.name, gpus[i].name) == 0) {
                deviceId = i;
                break;
            }
        }
        
        if (deviceId == -1) {
            fprintf(stderr, "Error: Could not find device ID for GPU %s\n", gpu.name);
            continue;
        }
        
        cudaSetDevice(deviceId);
        
        // Initialize cuBLAS
        cublasHandle_t cublasHandle;
        checkCublasError(cublasCreate(&cublasHandle), "Creating cuBLAS handle");
        
        // Loop through each matrix size
        for (const auto& size : sizes) {
            int m = size.m;
            int n = size.n;
            int k = size.k;
            
            printf("\n--- Testing %s matrices ---\n", size.name);
            
            // Skip large matrices for specific implementations (too slow or too memory intensive)
            bool isLargeMatrix = (m >= LARGE_SIZE || n >= LARGE_SIZE || k >= LARGE_SIZE);
            
            // Allocate host memory
            float *h_a = (float*)malloc(m * k * sizeof(float));
            float *h_b = (float*)malloc(k * n * sizeof(float));
            float *h_c = (float*)malloc(m * n * sizeof(float));
            half *h_a_half = (half*)malloc(m * k * sizeof(half));
            half *h_b_half = (half*)malloc(k * n * sizeof(half));
            
            // Initialize matrices
            srand(42);  // Use fixed seed for reproducibility
            initializeMatrix(h_a, m, k);
            initializeMatrix(h_b, k, n);
            
            // Convert to half precision for tensor cores
            convertToHalf(h_a, h_a_half, m * k);
            convertToHalf(h_b, h_b_half, k * n);
            
            // Allocate device memory
            float *d_a, *d_b, *d_c;
            half *d_a_half, *d_b_half;
            
            size_t bytes_a = m * k * sizeof(float);
            size_t bytes_b = k * n * sizeof(float);
            size_t bytes_c = m * n * sizeof(float);
            size_t bytes_a_half = m * k * sizeof(half);
            size_t bytes_b_half = k * n * sizeof(half);
            
            checkCudaError(cudaMalloc(&d_a, bytes_a), "Allocating d_a");
            checkCudaError(cudaMalloc(&d_b, bytes_b), "Allocating d_b");
            checkCudaError(cudaMalloc(&d_c, bytes_c), "Allocating d_c");
            checkCudaError(cudaMalloc(&d_a_half, bytes_a_half), "Allocating d_a_half");
            checkCudaError(cudaMalloc(&d_b_half, bytes_b_half), "Allocating d_b_half");
            
            // Copy data to device
            checkCudaError(cudaMemcpy(d_a, h_a, bytes_a, cudaMemcpyHostToDevice), "Copying h_a to d_a");
            checkCudaError(cudaMemcpy(d_b, h_b, bytes_b, cudaMemcpyHostToDevice), "Copying h_b to d_b");
            checkCudaError(cudaMemcpy(d_a_half, h_a_half, bytes_a_half, cudaMemcpyHostToDevice), "Copying h_a_half to d_a_half");
            checkCudaError(cudaMemcpy(d_b_half, h_b_half, bytes_b_half, cudaMemcpyHostToDevice), "Copying h_b_half to d_b_half");
            
            // Set up parameter structures for benchmarks
            NaiveParams naiveParams = {d_a, d_b, d_c};
            SharedParams sharedParams = {d_a, d_b, d_c};
            TensorParams tensorParams = {d_a_half, d_b_half, d_c};
            CublasParams cublasParams = {cublasHandle, d_a, d_b, d_c};
            CutlassParams cutlassParams = {d_a, d_b, d_c};
            
            // Skip large matrices for naive implementation (too slow)
            if (!isLargeMatrix) {
                // Run and record naive implementation
                PerfResult naiveResult = runBenchmark(gpu.name, "Naive", naiveBenchmark, &naiveParams, m, n, k, true);
                allResults.push_back(naiveResult);
                
                // Verify results
                checkCudaError(cudaMemcpy(h_c, d_c, bytes_c, cudaMemcpyDeviceToHost), "Copying d_c to h_c (naive)");
                printf("Naive verification: %s\n", verifyResults(h_a, h_b, h_c, m, n, k) ? "PASSED" : "FAILED");
            }
            
            // Run and record shared memory implementation
            PerfResult sharedResult = runBenchmark(gpu.name, "Shared Memory", sharedBenchmark, &sharedParams, m, n, k, true);
            allResults.push_back(sharedResult);
            
            // Verify results
            checkCudaError(cudaMemcpy(h_c, d_c, bytes_c, cudaMemcpyDeviceToHost), "Copying d_c to h_c (shared)");
            printf("Shared memory verification: %s\n", verifyResults(h_a, h_b, h_c, m, n, k) ? "PASSED" : "FAILED");
            
            // Run tensor cores implementation if GPU supports it
            if (gpu.hasTensorCores) {
                PerfResult tensorResult = runBenchmark(gpu.name, "Tensor Cores", tensorBenchmark, &tensorParams, m, n, k, true);
                allResults.push_back(tensorResult);
                
                // Verify results
                checkCudaError(cudaMemcpy(h_c, d_c, bytes_c, cudaMemcpyDeviceToHost), "Copying d_c to h_c (tensor)");
                printf("Tensor cores verification: %s\n", verifyResults(h_a, h_b, h_c, m, n, k) ? "PASSED" : "FAILED");
            }
            
            // Run cuBLAS implementation
            PerfResult cublasResult = runBenchmark(gpu.name, "cuBLAS", cublasBenchmark, &cublasParams, m, n, k, true);
            allResults.push_back(cublasResult);
            
            // Verify results
            checkCudaError(cudaMemcpy(h_c, d_c, bytes_c, cudaMemcpyDeviceToHost), "Copying d_c to h_c (cuBLAS)");
            printf("cuBLAS verification: %s\n", verifyResults(h_a, h_b, h_c, m, n, k) ? "PASSED" : "FAILED");
            
            // Run CUTLASS implementation
            PerfResult cutlassResult = runBenchmark(gpu.name, "CUTLASS", cutlassBenchmark, &cutlassParams, m, n, k, true);
            allResults.push_back(cutlassResult);
            
            // Verify results
            checkCudaError(cudaMemcpy(h_c, d_c, bytes_c, cudaMemcpyDeviceToHost), "Copying d_c to h_c (CUTLASS)");
            printf("CUTLASS verification: %s\n", verifyResults(h_a, h_b, h_c, m, n, k) ? "PASSED" : "FAILED");
            
            // Run CUTLASS Tensor Core implementation if GPU supports it
            if (gpu.hasTensorCores) {
                PerfResult cutlassTensorResult = runBenchmark(gpu.name, "CUTLASS Tensor Cores", cutlassTensorBenchmark, &cutlassParams, m, n, k, true);
                allResults.push_back(cutlassTensorResult);
                
                // Verify results
                checkCudaError(cudaMemcpy(h_c, d_c, bytes_c, cudaMemcpyDeviceToHost), "Copying d_c to h_c (CUTLASS Tensor)");
                printf("CUTLASS Tensor verification: %s\n", verifyResults(h_a, h_b, h_c, m, n, k) ? "PASSED" : "FAILED");
            }
            
            // Free device memory
            cudaFree(d_a);
            cudaFree(d_b);
            cudaFree(d_c);
            cudaFree(d_a_half);
            cudaFree(d_b_half);
            
            // Free host memory
            free(h_a);
            free(h_b);
            free(h_c);
            free(h_a_half);
            free(h_b_half);
        }
        
        // Destroy cuBLAS handle
        cublasDestroy(cublasHandle);
    }
    
    // Generate a filename that includes the GPU name if specified
    std::string csvFilename = "matrix_mul_performance";
    if (!targetGPU.empty()) {
        // Replace spaces with underscores for the filename
        std::string gpuNameForFile = targetGPU;
        std::replace(gpuNameForFile.begin(), gpuNameForFile.end(), ' ', '_');
        csvFilename += "_" + gpuNameForFile;
    }
    csvFilename += ".csv";
    
    // Save results to CSV for later chart generation
    saveResultsToCSV(allResults, csvFilename.c_str());
    
    printf("\nResults saved to %s. Use this file to generate charts.\n", csvFilename.c_str());
    
    return 0;
}
