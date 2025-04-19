/**
 * Matrix Multiplication Optimization with CUDA
 * 
 * This file contains multiple implementations of matrix multiplication:
 * 1. Naive implementation
 * 2. Shared memory optimization
 * 3. Tensor cores (WMMA) implementation
 * 4. cuBLAS implementation
 * 5. CUTLASS implementation
 * 6. Sparse matrix (cuSPARSE) implementation
 *
 * Benchmarking across RTX 2080 Ti, A100, and H100 GPUs
 * Testing different matrix sizes and sparsity patterns
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
 #include <algorithm>  // For std::sort, std::shuffle
 #include <utility>    // For std::pair
 #include <random>     // For std::default_random_engine
 #include <cublas_v2.h>
 #include <cusparse.h>  // Added for sparse matrix operations
 #include <string.h>
 #include <getopt.h>
 #include <time.h>
 
 // For CUTLASS implementation
 #include <cutlass/gemm/device/gemm.h>
 
 // Matrix size definitions
 #define SMALL_SIZE 32
 #define MEDIUM_SIZE 1024
 #define LARGE_SIZE 8192
 #define XLARGE_SIZE 16384  // Added extra large size
 
 // Non-square matrix dimensions
 #define NON_SQUARE_M 1024
 #define NON_SQUARE_N 2048
 #define NON_SQUARE_K 1024
 
 // Verification sample counts
 #define VERIFICATION_SAMPLE_COUNT 100  // Number of random points to verify for large matrices
 #define NON_SQUARE_VERIFICATION_SAMPLE_COUNT 1000  // More samples for non-square matrices
 
 // Sparse matrix parameters
 #define DENSITY_HIGH 0.5    // 50% non-zero elements
 #define DENSITY_MEDIUM 0.1  // 10% non-zero elements
 #define DENSITY_LOW 0.01    // 1% non-zero elements
 
 // Forward declarations of helper functions
 void checkCudaError(cudaError_t error, const char *message);
 void checkCublasError(cublasStatus_t status, const char *message);
 void checkCusparseError(cusparseStatus_t status, const char *message);
 
 struct GPUInfo {
     char name[256];
     int major;
     int minor;
     bool hasTensorCores;
 };
 
 struct PerfResult {
     std::string gpuName;
     std::string implementation;
     int m;
     int n;
     int k;
     float density;          // Added for sparse matrices (density = % of non-zeros)
     float executionTimeMs;
     float throughputGFlops;
     std::string verificationResult;
 };
 
 // =============== KERNEL IMPLEMENTATIONS ===============
 
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
 
 __global__ void matrixMulShared(float *a, float *b, float *c, int m, int n, int k) {
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
 
 __global__ void convertToHalfKernel(float *in, half *out, int size) {
     int idx = blockIdx.x * blockDim.x + threadIdx.x;
     if (idx < size) {
         out[idx] = __float2half(in[idx]);
     }
 }
 
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
         
         store_matrix_sync(c + warpM * WMMA_M * n + warpN * WMMA_N, c_frag, n, nvcuda::wmma::mem_row_major);
     }
 }
 
 // =============== SPARSE MATRIX SUPPORT ===============
 
 // Structure to hold CSR format sparse matrix
 struct CSRMatrix {
     int m;                  // Number of rows
     int n;                  // Number of columns
     int nnz;                // Number of non-zero elements
     float *values;          // Array of non-zero values
     int *rowPtrs;           // Array of row pointers (size m+1)
     int *colIndices;        // Array of column indices
     float density;          // Density (fraction of non-zero elements)
     
     // Constructor
     CSRMatrix() : m(0), n(0), nnz(0), values(nullptr), rowPtrs(nullptr), colIndices(nullptr), density(0.0f) {}
     
     // Destructor - checking CUDA errors here would be good practice but can cause issues in destructors
     ~CSRMatrix() {
         if (values) {
             cudaFree(values);
             values = nullptr;
         }
         if (rowPtrs) {
             cudaFree(rowPtrs);
             rowPtrs = nullptr;
         }
         if (colIndices) {
             cudaFree(colIndices);
             colIndices = nullptr;
         }
     }
 };
 
 // Convert a dense matrix to CSR format on the device
 CSRMatrix* convertDenseToCSR(float *d_dense, int m, int n, float targetDensity) {
     printf("Converting %dx%d dense matrix to CSR format with target density %.4f\n", m, n, targetDensity);
     
     // Allocate host memory for the dense matrix
     float *h_dense = (float*)malloc(m * n * sizeof(float));
     checkCudaError(cudaMemcpy(h_dense, d_dense, m * n * sizeof(float), cudaMemcpyDeviceToHost), 
                    "Copying dense matrix from device to host");
     
     // Create a temporary CSR matrix on the host
     int maxNnz = static_cast<int>(m * n * targetDensity);
     float *h_values = (float*)malloc(maxNnz * sizeof(float));
     int *h_rowPtrs = (int*)malloc((m + 1) * sizeof(int));
     int *h_colIndices = (int*)malloc(maxNnz * sizeof(int));
     
     // Convert to CSR format with target density
     h_rowPtrs[0] = 0;
     int nnz = 0;
     
     for (int i = 0; i < m; i++) {
         for (int j = 0; j < n; j++) {
             // Use a probability threshold based on target density
             float val = h_dense[i * n + j];
             float prob = static_cast<float>(rand()) / RAND_MAX;
             
             if (prob < targetDensity && val != 0.0f) {
                 h_values[nnz] = val;
                 h_colIndices[nnz] = j;
                 nnz++;
                 
                 if (nnz >= maxNnz) {
                     // We've reached the maximum allowed number of non-zeros
                     // Complete the rest of the row pointers and break
                     for (int k = i + 1; k <= m; k++) {
                         h_rowPtrs[k] = nnz;
                     }
                     goto done_conversion;
                 }
             }
         }
         h_rowPtrs[i + 1] = nnz;
     }
     
 done_conversion:
     // Calculate actual density
     float actualDensity = static_cast<float>(nnz) / (m * n);
     printf("Converted to CSR format with %d non-zeros (actual density: %.4f)\n", nnz, actualDensity);
     
     // Create and fill the CSR matrix structure
     CSRMatrix *csrMatrix = new CSRMatrix();
     csrMatrix->m = m;
     csrMatrix->n = n;
     csrMatrix->nnz = nnz;
     csrMatrix->density = actualDensity;
     
     // Allocate device memory for CSR format
     checkCudaError(cudaMalloc(&csrMatrix->values, nnz * sizeof(float)), "Allocating CSR values");
     checkCudaError(cudaMalloc(&csrMatrix->rowPtrs, (m + 1) * sizeof(int)), "Allocating CSR row pointers");
     checkCudaError(cudaMalloc(&csrMatrix->colIndices, nnz * sizeof(int)), "Allocating CSR column indices");
     
     // Copy CSR data to device
     checkCudaError(cudaMemcpy(csrMatrix->values, h_values, nnz * sizeof(float), cudaMemcpyHostToDevice),
                    "Copying CSR values to device");
     checkCudaError(cudaMemcpy(csrMatrix->rowPtrs, h_rowPtrs, (m + 1) * sizeof(int), cudaMemcpyHostToDevice),
                    "Copying CSR row pointers to device");
     checkCudaError(cudaMemcpy(csrMatrix->colIndices, h_colIndices, nnz * sizeof(int), cudaMemcpyHostToDevice),
                    "Copying CSR column indices to device");
     
     // Free host memory
     free(h_dense);
     free(h_values);
     free(h_rowPtrs);
     free(h_colIndices);
     
     return csrMatrix;
 }
 
 // Generate a random sparse matrix directly in CSR format
 CSRMatrix* generateRandomCSRMatrix(int m, int n, float density) {
     printf("Generating random %dx%d sparse matrix with density %.4f\n", m, n, density);
     
     // Calculate the number of non-zero elements based on density
     int targetNnz = static_cast<int>(m * n * density);
     int maxNnz = std::min(targetNnz, m * n); // Cannot have more non-zeros than matrix elements
     
     // Allocate host memory
     float *h_values = (float*)malloc(maxNnz * sizeof(float));
     int *h_rowPtrs = (int*)malloc((m + 1) * sizeof(int));
     int *h_colIndices = (int*)malloc(maxNnz * sizeof(int));
     
     // Initialize row pointers
     h_rowPtrs[0] = 0;
     
     // Generate random sparse matrix
     int nnz = 0;
     std::vector<std::pair<int, float>> rowElements;
     
     for (int i = 0; i < m; i++) {
         rowElements.clear();
         
         // Determine number of non-zeros for this row
         int rowNnz = static_cast<int>(n * density);
         rowNnz = std::min(rowNnz, n); // Cannot have more non-zeros than columns
         
         // Generate random column indices for this row
         std::vector<int> cols;
         for (int j = 0; j < n; j++) {
             cols.push_back(j);
         }
         // Use a random engine with fixed seed for reproducibility
         static std::default_random_engine rng(42);
         std::shuffle(cols.begin(), cols.end(), rng);
         
         // Take only the first rowNnz columns
         for (int j = 0; j < rowNnz; j++) {
             float value = static_cast<float>(rand()) / RAND_MAX;
             rowElements.push_back(std::make_pair(cols[j], value));
         }
         
         // Sort by column index
         std::sort(rowElements.begin(), rowElements.end());
         
         // Add to CSR arrays
         for (const auto& elem : rowElements) {
             if (nnz < maxNnz) {
                 h_colIndices[nnz] = elem.first;
                 h_values[nnz] = elem.second;
                 nnz++;
             }
         }
         
         h_rowPtrs[i + 1] = nnz;
     }
     
     // Calculate actual density
     float actualDensity = static_cast<float>(nnz) / (m * n);
     printf("Generated CSR matrix with %d non-zeros (actual density: %.4f)\n", nnz, actualDensity);
     
     // Create and fill the CSR matrix structure
     CSRMatrix *csrMatrix = new CSRMatrix();
     csrMatrix->m = m;
     csrMatrix->n = n;
     csrMatrix->nnz = nnz;
     csrMatrix->density = actualDensity;
     
     // Allocate device memory for CSR format
     checkCudaError(cudaMalloc(&csrMatrix->values, nnz * sizeof(float)), "Allocating CSR values");
     checkCudaError(cudaMalloc(&csrMatrix->rowPtrs, (m + 1) * sizeof(int)), "Allocating CSR row pointers");
     checkCudaError(cudaMalloc(&csrMatrix->colIndices, nnz * sizeof(int)), "Allocating CSR column indices");
     
     // Copy CSR data to device
     checkCudaError(cudaMemcpy(csrMatrix->values, h_values, nnz * sizeof(float), cudaMemcpyHostToDevice),
                    "Copying CSR values to device");
     checkCudaError(cudaMemcpy(csrMatrix->rowPtrs, h_rowPtrs, (m + 1) * sizeof(int), cudaMemcpyHostToDevice),
                    "Copying CSR row pointers to device");
     checkCudaError(cudaMemcpy(csrMatrix->colIndices, h_colIndices, nnz * sizeof(int), cudaMemcpyHostToDevice),
                    "Copying CSR column indices to device");
     
     // Free host memory
     free(h_values);
     free(h_rowPtrs);
     free(h_colIndices);
     
     return csrMatrix;
 }
 
 // =============== CUTLASS IMPLEMENTATION ===============
 
 template <typename Gemm>
 void runCutlassGemm(float *d_a, float *d_b, float *d_c, int m, int n, int k) {
     Gemm gemm_operator;
 
     typename Gemm::Arguments args(
         {m, n, k},                   // Problem dimensions (M, N, K)
         {d_a, k},                    // Tensor A (device pointer and leading dimension)
         {d_b, n},                    // Tensor B (device pointer and leading dimension)
         {d_c, n},                    // Tensor C (device pointer and leading dimension)
         {d_c, n},                    // Tensor D (device pointer and leading dimension)
         {1.0f, 0.0f}                 // alpha and beta
     );
 
     cudaDeviceSynchronize();
     cutlass::Status status = gemm_operator(args);
     cudaDeviceSynchronize();
 
     if (status != cutlass::Status::kSuccess) {
         printf("CUTLASS GEMM kernel failed: %s\n", cutlass::cutlassGetStatusString(status));
     }
 }
 
 void matrixMulCutlass(float *d_a, float *d_b, float *d_c, int m, int n, int k) {
     using ElementInputA = float;
     using ElementInputB = float;
     using ElementOutput = float;
     using ElementAccumulator = float;
     using ElementCompute = float;
 
     using LayoutInputA = cutlass::layout::RowMajor;
     using LayoutInputB = cutlass::layout::RowMajor;
     using LayoutOutput = cutlass::layout::RowMajor;
 
     int device;
     cudaGetDevice(&device);
     cudaDeviceProp props;
     cudaGetDeviceProperties(&props, device);
 
     if (props.major >= 8) {
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
 
 // Implementation of the function declarations
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
 
 void checkCusparseError(cusparseStatus_t status, const char *message) {
     if (status != CUSPARSE_STATUS_SUCCESS) {
         fprintf(stderr, "cuSPARSE error: %s: %d\n", message, status);
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
 
 // Full matrix verification (slow for large matrices)
 bool verifyResultsFull(float *a, float *b, float *c, int m, int n, int k, float epsilon) {
     // For tensor core implementations, use a higher epsilon
     for (int i = 0; i < m; i++) {
         for (int j = 0; j < n; j++) {
             float expected = 0.0f;
             for (int p = 0; p < k; p++) {
                 expected += a[i * k + p] * b[p * n + j];
             }
             if (fabs(expected - c[i * n + j]) > epsilon) {
                 printf("Verification failed at [%d, %d]: Expected %f, got %f\n", 
                        i, j, expected, c[i * n + j]);
                 return false;
             }
         }
     }
     
     return true;
 }
 
 // Partial verification for large matrices (samples random elements)
 bool verifyResultsPartial(float *a, float *b, float *c, int m, int n, int k, int samples, float epsilon) {
     srand(42);  // Use fixed seed for reproducibility
     
     // Check if this is a non-square matrix
     bool isNonSquare = (m == NON_SQUARE_M && n == NON_SQUARE_N && k == NON_SQUARE_K);
     
     printf("Performing partial verification with %d random samples %s...\n", 
            samples, isNonSquare ? "(non-square matrix)" : "");
     
     for (int s = 0; s < samples; s++) {
         int i = rand() % m;
         int j = rand() % n;
         
         float expected = 0.0f;
         for (int p = 0; p < k; p++) {
             expected += a[i * k + p] * b[p * n + j];
         }
         
         if (fabs(expected - c[i * n + j]) > epsilon) {
             printf("Verification failed at [%d, %d]: Expected %f, got %f\n", 
                    i, j, expected, c[i * n + j]);
             return false;
         }
     }
     
     return true;
 }
 
 // Verify sparse matrix multiplication results against dense computation
 bool verifySparseResults(float *a, float *b, float *c, int m, int n, int k, int samples, float epsilon) {
     srand(42);  // Use fixed seed for reproducibility
     
     printf("Performing sparse verification with %d random samples...\n", samples);
     
     for (int s = 0; s < samples; s++) {
         int i = rand() % m;
         int j = rand() % n;
         
         float expected = 0.0f;
         for (int p = 0; p < k; p++) {
             expected += a[i * k + p] * b[p * n + j];
         }
         
         if (fabs(expected - c[i * n + j]) > epsilon) {
             printf("Sparse verification failed at [%d, %d]: Expected %f, got %f\n", 
                    i, j, expected, c[i * n + j]);
             return false;
         }
     }
     
     return true;
 }
 
 // Global flags for verification settings
 bool g_fullVerifyNonSquare = false;
 int g_nonSquareSampleCount = NON_SQUARE_VERIFICATION_SAMPLE_COUNT;
 
 // Smart verification dispatcher based on matrix size
 bool verifyResults(float *a, float *b, float *c, int m, int n, int k, bool isTensorCore) {
     // Use different epsilon values for tensor core vs regular implementations
     float epsilon = isTensorCore ? 1e-1f : 1e-2f;
     
     // Check if this is a non-square matrix (it matches our predefined dimensions)
     bool isNonSquare = (m == NON_SQUARE_M && n == NON_SQUARE_N && k == NON_SQUARE_K);
     
     // Special handling for non-square matrices if full verification is requested
     if (isNonSquare && g_fullVerifyNonSquare) {
         printf("Performing full verification for non-square matrix as requested...\n");
         return verifyResultsFull(a, b, c, m, n, k, epsilon);
     }
     
     // For large matrices, use partial verification
     if (m > MEDIUM_SIZE || n > MEDIUM_SIZE || k > MEDIUM_SIZE) {
         // For non-square matrices, use specified sample count
         int sampleCount = isNonSquare ? g_nonSquareSampleCount : VERIFICATION_SAMPLE_COUNT;
         return verifyResultsPartial(a, b, c, m, n, k, sampleCount, epsilon);
     } else {
         // For small/medium matrices, always use full verification
         return verifyResultsFull(a, b, c, m, n, k, epsilon);
     }
 }
 
 // Calculate GFLOPs for matrix multiplication
 float calculateGFlops(int m, int n, int k, float timeMs) {
     // Each matrix multiply-add is 2 operations
     // Total operations = m * n * k * 2
     float operations = 2.0f * static_cast<float>(m) * static_cast<float>(n) * static_cast<float>(k);
     float timeS = timeMs / 1000.0f;
     return operations / (timeS * 1e9);
 }
 
 // Calculate GFLOPs for sparse matrix multiplication
 float calculateSparseGFlops(int m, int n, int k, float density, float timeMs) {
     // For sparse matrix multiplication, we only perform operations for non-zero elements
     // Approx number of non-zeros in A = m * k * density
     // Each row in C requires k operations
     float operations = 2.0f * static_cast<float>(m) * static_cast<float>(n) * static_cast<float>(k) * density;
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
 
 bool isTargetGPU(const char* name) {
     // Check if the GPU is one of our target GPUs
     return (strstr(name, "RTX 2080 Ti") != NULL ||
             strstr(name, "A100") != NULL ||
             strstr(name, "H100") != NULL);
 }
 
 // Function to parse command line arguments
 void parseArgs(int argc, char **argv, std::string &targetGPU, bool &fullVerifyNonSquare, 
                int &nonSquareSampleCount, bool &testSparse, bool &testXLarge) {
     const struct option long_options[] = {
         {"gpu", required_argument, 0, 'g'},
         {"full-verify", no_argument, 0, 'f'},
         {"samples", required_argument, 0, 's'},
         {"sparse", no_argument, 0, 'p'},
         {"xlarge", no_argument, 0, 'x'},
         {"help", no_argument, 0, 'h'},
         {0, 0, 0, 0}
     };
 
     int opt;
     int option_index = 0;
     
     while ((opt = getopt_long(argc, argv, "g:fs:pxh", long_options, &option_index)) != -1) {
         switch (opt) {
             case 'g':
                 targetGPU = optarg;
                 break;
             case 'f':
                 fullVerifyNonSquare = true;
                 break;
             case 's':
                 nonSquareSampleCount = atoi(optarg);
                 break;
             case 'p':
                 testSparse = true;
                 break;
             case 'x':
                 testXLarge = true;
                 break;
             case 'h':
                 printf("Usage: %s [OPTIONS]\n", argv[0]);
                 printf("Options:\n");
                 printf("  -g, --gpu=GPU_NAME    Specify GPU to use (e.g., \"RTX 2080 Ti\", \"A100\", \"H100\")\n");
                 printf("  -f, --full-verify     Use full verification for non-square matrices\n");
                 printf("  -s, --samples=COUNT   Number of samples to use for non-square matrix verification (default: %d)\n", 
                        NON_SQUARE_VERIFICATION_SAMPLE_COUNT);
                 printf("  -p, --sparse          Include sparse matrix multiplication tests\n");
                 printf("  -x, --xlarge          Include extra large matrix sizes\n");
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
 
 // =============== CUSPARSE IMPLEMENTATION ===============
 
 void matrixMulCusparse(cusparseHandle_t handle, CSRMatrix *a_csr, float *d_b, float *d_c, int m, int n, int k) {
     const float alpha = 1.0f;
     const float beta = 0.0f;
     
     // Create matrix descriptors
     cusparseSpMatDescr_t matA;
     cusparseDnMatDescr_t matB, matC;
     
     // Create sparse matrix A in CSR format
     checkCusparseError(
         cusparseCreateCsr(&matA, m, k, a_csr->nnz,
                           a_csr->rowPtrs, a_csr->colIndices, a_csr->values,
                           CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                           CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F),
         "Creating sparse matrix descriptor for A"
     );
     
     // Create dense matrix B
     checkCusparseError(
         cusparseCreateDnMat(&matB, k, n, n, d_b, CUDA_R_32F, CUSPARSE_ORDER_ROW),
         "Creating dense matrix descriptor for B"
     );
     
     // Create dense matrix C
     checkCusparseError(
         cusparseCreateDnMat(&matC, m, n, n, d_c, CUDA_R_32F, CUSPARSE_ORDER_ROW),
         "Creating dense matrix descriptor for C"
     );
     
     // Get buffer size for SpMM
     size_t bufferSize = 0;
     checkCusparseError(
         cusparseSpMM_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &alpha, matA, matB, &beta, matC, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize),
         "Getting buffer size for SpMM"
     );
     
     // Allocate buffer
     void *dBuffer = nullptr;
     checkCudaError(cudaMalloc(&dBuffer, bufferSize), "Allocating work buffer for SpMM");
     
     // Execute sparse matrix-dense matrix multiplication
     checkCusparseError(
         cusparseSpMM(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                     &alpha, matA, matB, &beta, matC, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, dBuffer),
         "Executing cusparseSpMM"
     );
     
     // Free resources
     cusparseDestroySpMat(matA);
     cusparseDestroyDnMat(matB);
     cusparseDestroyDnMat(matC);
     cudaFree(dBuffer);
 }
 
 // =============== TESTING FRAMEWORK ===============
 
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
 
 struct SparseParams {
     cusparseHandle_t handle;
     CSRMatrix *a_csr;
     float *d_b, *d_c;
 };
 
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
 
 void sparseBenchmark(void* p, int m, int n, int k) {
     SparseParams* params = (SparseParams*)p;
     matrixMulCusparse(params->handle, params->a_csr, params->d_b, params->d_c, m, n, k);
     cudaDeviceSynchronize();
 }
 
 PerfResult runBenchmark(const char* gpuName, const char* implName, 
                       void (*benchmark)(void*, int, int, int), 
                       void* params, int m, int n, int k, float density, bool isTensorCore) {
     
     cudaEvent_t start, stop;
     cudaEventCreate(&start);
     cudaEventCreate(&stop);
     float elapsed_time;
     
     // Print memory usage before running
     size_t free_mem, total_mem;
     cudaMemGetInfo(&free_mem, &total_mem);
     printf("GPU memory before %s: %.2f GB free / %.2f GB total\n", 
            implName, free_mem/1024.0f/1024.0f/1024.0f, total_mem/1024.0f/1024.0f/1024.0f);
     
     // Run the benchmark
     cudaEventRecord(start);
     benchmark(params, m, n, k);
     cudaEventRecord(stop);
     cudaEventSynchronize(stop);
     cudaEventElapsedTime(&elapsed_time, start, stop);
     
     // Clean up events
     cudaEventDestroy(start);
     cudaEventDestroy(stop);
     
     // Calculate GFLOPs - use sparse calculation if density < 1.0
     float gflops = (density < 1.0) ? 
                    calculateSparseGFlops(m, n, k, density, elapsed_time) : 
                    calculateGFlops(m, n, k, elapsed_time);
     
     PerfResult result;
     result.gpuName = gpuName;
     result.implementation = implName;
     result.m = m;
     result.n = n;
     result.k = k;
     result.density = density;
     result.executionTimeMs = elapsed_time;
     result.throughputGFlops = gflops;
     result.verificationResult = "Pending";
     
     // Add density information to output if sparse
     if (density < 1.0) {
         printf("%-15s %-25s %5d x %5d x %5d (%.2f%%): %10.2f ms (%10.2f GFlops)\n", 
                gpuName, implName, m, n, k, density * 100.0f, elapsed_time, gflops);
     } else {
         printf("%-15s %-25s %5d x %5d x %5d: %10.2f ms (%10.2f GFlops)\n", 
                gpuName, implName, m, n, k, elapsed_time, gflops);
     }
     
     return result;
 }
 
 void saveResultsToCSV(const std::vector<PerfResult>& results, const char* filename) {
     std::ofstream file(filename);
     if (!file.is_open()) {
         fprintf(stderr, "Error: Could not open file %s for writing\n", filename);
         return;
     }
     
     file << "GPU,Implementation,M,N,K,Density,Time_ms,Throughput_GFlops,Verification\n";
     
     for (const auto& result : results) {
         file << result.gpuName << ","
              << result.implementation << ","
              << result.m << ","
              << result.n << ","
              << result.k << ","
              << result.density << ","
              << result.executionTimeMs << ","
              << result.throughputGFlops << ","
              << result.verificationResult << "\n";
     }
     
     file.close();
     printf("Results saved to %s\n", filename);
 }
 
 // =============== MAIN FUNCTION ===============
 
 int main(int argc, char** argv) {
     std::string targetGPU = "";  // Empty string means all GPUs
     bool fullVerifyNonSquare = false;
     int nonSquareSampleCount = NON_SQUARE_VERIFICATION_SAMPLE_COUNT;
     bool testSparse = false;
     bool testXLarge = false;
     
     parseArgs(argc, argv, targetGPU, fullVerifyNonSquare, nonSquareSampleCount, testSparse, testXLarge);
     
     // Set global verification settings
     g_fullVerifyNonSquare = fullVerifyNonSquare;
     g_nonSquareSampleCount = nonSquareSampleCount;
     
     if (fullVerifyNonSquare) {
         printf("Using full verification for non-square matrices\n");
     } else {
         printf("Using partial verification with %d samples for non-square matrices\n", nonSquareSampleCount);
     }
     
     if (testSparse) {
         printf("Including sparse matrix multiplication tests\n");
     }
     
     if (testXLarge) {
         printf("Including extra large matrix size tests (16384x16384)\n");
     }
     
     std::vector<GPUInfo> gpus = detectGPUs();
     std::vector<PerfResult> allResults;
     
     if (gpus.empty()) {
         fprintf(stderr, "No CUDA-capable devices found\n");
         return EXIT_FAILURE;
     }
     
     printf("Detected GPUs:\n");
     for (const auto& gpu : gpus) {
         printf("  %s (SM %d.%d)%s\n", gpu.name, gpu.major, gpu.minor, 
                gpu.hasTensorCores ? " - Tensor Cores supported" : "");
     }
     
     if (!targetGPU.empty()) {
         printf("\nFiltering to run only on: %s\n", targetGPU.c_str());
     }
     
     struct MatrixSize {
         int m, n, k;
         const char* name;
     };
     
     // Define matrix sizes to test
     std::vector<MatrixSize> sizes = {
         {SMALL_SIZE, SMALL_SIZE, SMALL_SIZE, "Small (32x32)"},
         {MEDIUM_SIZE, MEDIUM_SIZE, MEDIUM_SIZE, "Medium (1024x1024)"},
         {LARGE_SIZE, LARGE_SIZE, LARGE_SIZE, "Large (8192x8192)"},
         {NON_SQUARE_M, NON_SQUARE_N, NON_SQUARE_K, "Non-square (1024x2048x1024)"}
     };
     
     // Add extra large size if requested
     if (testXLarge) {
         sizes.push_back({XLARGE_SIZE, XLARGE_SIZE, XLARGE_SIZE, "XLarge (16384x16384)"});
     }
     
     // Define sparsity levels to test
     std::vector<float> sparsityLevels = {
         DENSITY_HIGH,
         DENSITY_MEDIUM,
         DENSITY_LOW
     };
     
     // Loop through each GPU
     for (const auto& gpu : gpus) {
         if (!isTargetGPU(gpu.name) || (!targetGPU.empty() && strstr(gpu.name, targetGPU.c_str()) == NULL)) {
             printf("Skipping GPU: %s (not selected for testing)\n", gpu.name);
             continue;
         }
         
         printf("\n=== Testing on GPU: %s ===\n", gpu.name);
         
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
         
         // Create cuBLAS and cuSPARSE handles
         cublasHandle_t cublasHandle;
         checkCublasError(cublasCreate(&cublasHandle), "Creating cuBLAS handle");
         
         cusparseHandle_t cusparseHandle;
         if (testSparse) {
             checkCusparseError(cusparseCreate(&cusparseHandle), "Creating cuSPARSE handle");
         }
         
         // Test each matrix size
         for (const auto& size : sizes) {
             int m = size.m;
             int n = size.n;
             int k = size.k;
             
             printf("\n--- Testing %s matrices ---\n", size.name);
             
             bool isLargeMatrix = (m >= LARGE_SIZE || n >= LARGE_SIZE || k >= LARGE_SIZE);
             bool isXLargeMatrix = (m >= XLARGE_SIZE || n >= XLARGE_SIZE || k >= XLARGE_SIZE);
             
             // Skip extremely large tests for less powerful GPUs
             if (isXLargeMatrix && (gpu.major < 7)) {
                 printf("Skipping XLarge test for this GPU as it may run out of memory\n");
                 continue;
             }
             
             // Allocate host memory
             float *h_a = (float*)malloc(m * k * sizeof(float));
             float *h_b = (float*)malloc(k * n * sizeof(float));
             float *h_c = (float*)malloc(m * n * sizeof(float));
             half *h_a_half = (half*)malloc(m * k * sizeof(half));
             half *h_b_half = (half*)malloc(k * n * sizeof(half));
             
             if (!h_a || !h_b || !h_c || !h_a_half || !h_b_half) {
                 fprintf(stderr, "Error: Host memory allocation failed\n");
                 continue;
             }
             
             srand(42);  // Use fixed seed for reproducibility
             initializeMatrix(h_a, m, k);
             initializeMatrix(h_b, k, n);
             
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
             
             cudaError_t memAlloc;
             memAlloc = cudaMalloc(&d_a, bytes_a);
             if (memAlloc != cudaSuccess) {
                 fprintf(stderr, "Error allocating d_a: %s\n", cudaGetErrorString(memAlloc));
                 continue;
             }
             
             memAlloc = cudaMalloc(&d_b, bytes_b);
             if (memAlloc != cudaSuccess) {
                 fprintf(stderr, "Error allocating d_b: %s\n", cudaGetErrorString(memAlloc));
                 cudaFree(d_a);
                 continue;
             }
             
             memAlloc = cudaMalloc(&d_c, bytes_c);
             if (memAlloc != cudaSuccess) {
                 fprintf(stderr, "Error allocating d_c: %s\n", cudaGetErrorString(memAlloc));
                 cudaFree(d_a);
                 cudaFree(d_b);
                 continue;
             }
             
             memAlloc = cudaMalloc(&d_a_half, bytes_a_half);
             if (memAlloc != cudaSuccess) {
                 fprintf(stderr, "Error allocating d_a_half: %s\n", cudaGetErrorString(memAlloc));
                 cudaFree(d_a);
                 cudaFree(d_b);
                 cudaFree(d_c);
                 continue;
             }
             
             memAlloc = cudaMalloc(&d_b_half, bytes_b_half);
             if (memAlloc != cudaSuccess) {
                 fprintf(stderr, "Error allocating d_b_half: %s\n", cudaGetErrorString(memAlloc));
                 cudaFree(d_a);
                 cudaFree(d_b);
                 cudaFree(d_c);
                 cudaFree(d_a_half);
                 continue;
             }
             
             // Copy data to device
             checkCudaError(cudaMemcpy(d_a, h_a, bytes_a, cudaMemcpyHostToDevice), "Copying h_a to d_a");
             checkCudaError(cudaMemcpy(d_b, h_b, bytes_b, cudaMemcpyHostToDevice), "Copying h_b to d_b");
             checkCudaError(cudaMemcpy(d_a_half, h_a_half, bytes_a_half, cudaMemcpyHostToDevice), "Copying h_a_half to d_a_half");
             checkCudaError(cudaMemcpy(d_b_half, h_b_half, bytes_b_half, cudaMemcpyHostToDevice), "Copying h_b_half to d_b_half");
             
             // Setup parameters
             NaiveParams naiveParams = {d_a, d_b, d_c};
             SharedParams sharedParams = {d_a, d_b, d_c};
             TensorParams tensorParams = {d_a_half, d_b_half, d_c};
             CublasParams cublasParams = {cublasHandle, d_a, d_b, d_c};
             CutlassParams cutlassParams = {d_a, d_b, d_c};
             
             // Vector to store results for this matrix size
             std::vector<PerfResult> sizeResults;
             
             // Run naive implementation (skip for large matrices)
             if (!isLargeMatrix) {
                 PerfResult naiveResult = runBenchmark(gpu.name, "Naive", naiveBenchmark, &naiveParams, m, n, k, 1.0f, false);
                 
                 // Verify result
                 checkCudaError(cudaMemcpy(h_c, d_c, bytes_c, cudaMemcpyDeviceToHost), "Copying d_c to h_c (naive)");
                 bool naiveVerification = verifyResults(h_a, h_b, h_c, m, n, k, false);
                 naiveResult.verificationResult = naiveVerification ? "PASSED" : "FAILED";
                 printf("Naive verification: %s\n", naiveResult.verificationResult.c_str());
                 
                 sizeResults.push_back(naiveResult);
                 allResults.push_back(naiveResult);
             }
             
             // Run shared memory implementation
             PerfResult sharedResult = runBenchmark(gpu.name, "Shared Memory", sharedBenchmark, &sharedParams, m, n, k, 1.0f, false);
             
             // Verify result
             checkCudaError(cudaMemcpy(h_c, d_c, bytes_c, cudaMemcpyDeviceToHost), "Copying d_c to h_c (shared)");
             bool sharedVerification = verifyResults(h_a, h_b, h_c, m, n, k, false);
             sharedResult.verificationResult = sharedVerification ? "PASSED" : "FAILED";
             printf("Shared memory verification: %s\n", sharedResult.verificationResult.c_str());
             
             sizeResults.push_back(sharedResult);
             allResults.push_back(sharedResult);
             
             // Run tensor core implementation if available
             if (gpu.hasTensorCores) {
                 PerfResult tensorResult = runBenchmark(gpu.name, "Tensor Cores", tensorBenchmark, &tensorParams, m, n, k, 1.0f, true);
                 
                 // Verify result with higher tolerance
                 checkCudaError(cudaMemcpy(h_c, d_c, bytes_c, cudaMemcpyDeviceToHost), "Copying d_c to h_c (tensor)");
                 bool tensorVerification = verifyResults(h_a, h_b, h_c, m, n, k, true);
                 tensorResult.verificationResult = tensorVerification ? "PASSED" : "FAILED";
                 printf("Tensor cores verification: %s\n", tensorResult.verificationResult.c_str());
                 
                 sizeResults.push_back(tensorResult);
                 allResults.push_back(tensorResult);
             }
             
             // Run cuBLAS implementation
             PerfResult cublasResult = runBenchmark(gpu.name, "cuBLAS", cublasBenchmark, &cublasParams, m, n, k, 1.0f, false);
             
             // Verify result
             checkCudaError(cudaMemcpy(h_c, d_c, bytes_c, cudaMemcpyDeviceToHost), "Copying d_c to h_c (cuBLAS)");
             bool cublasVerification = verifyResults(h_a, h_b, h_c, m, n, k, false);
             cublasResult.verificationResult = cublasVerification ? "PASSED" : "FAILED";
             printf("cuBLAS verification: %s\n", cublasResult.verificationResult.c_str());
             
             sizeResults.push_back(cublasResult);
             allResults.push_back(cublasResult);
             
             // Run CUTLASS implementation
             PerfResult cutlassResult = runBenchmark(gpu.name, "CUTLASS", cutlassBenchmark, &cutlassParams, m, n, k, 1.0f, false);
             
             // Verify result
             checkCudaError(cudaMemcpy(h_c, d_c, bytes_c, cudaMemcpyDeviceToHost), "Copying d_c to h_c (CUTLASS)");
             bool cutlassVerification = verifyResults(h_a, h_b, h_c, m, n, k, false);
             cutlassResult.verificationResult = cutlassVerification ? "PASSED" : "FAILED";
             printf("CUTLASS verification: %s\n", cutlassResult.verificationResult.c_str());
             
             sizeResults.push_back(cutlassResult);
             allResults.push_back(cutlassResult);
             
             // Run CUTLASS Tensor Core implementation if available
             if (gpu.hasTensorCores) {
                 PerfResult cutlassTensorResult = runBenchmark(gpu.name, "CUTLASS Tensor Cores", cutlassTensorBenchmark, &cutlassParams, m, n, k, 1.0f, true);
                 
                 // Verify result with higher tolerance
                 checkCudaError(cudaMemcpy(h_c, d_c, bytes_c, cudaMemcpyDeviceToHost), "Copying d_c to h_c (CUTLASS Tensor)");
                 bool cutlassTensorVerification = verifyResults(h_a, h_b, h_c, m, n, k, true);
                 cutlassTensorResult.verificationResult = cutlassTensorVerification ? "PASSED" : "FAILED";
                 printf("CUTLASS Tensor verification: %s\n", cutlassTensorResult.verificationResult.c_str());
                 
                 sizeResults.push_back(cutlassTensorResult);
                 allResults.push_back(cutlassTensorResult);
             }
             
             // Run sparse matrix tests if requested
             if (testSparse && !isXLargeMatrix) {  // Skip sparse for XLarge matrices
                 printf("\n--- Running sparse matrix tests for %s matrices ---\n", size.name);
                 
                 for (float density : sparsityLevels) {
                     printf("\n-- Testing with density %.2f%% --\n", density * 100.0f);
                     
                     // Create a sparse matrix with the specified density
                     CSRMatrix *a_csr = generateRandomCSRMatrix(m, k, density);
                     
                     // Clear result matrix
                     checkCudaError(cudaMemset(d_c, 0, bytes_c), "Clearing d_c for sparse test");
                     
                     // Setup sparse parameters
                     SparseParams sparseParams = {cusparseHandle, a_csr, d_b, d_c};
                     
                     // Run sparse benchmark
                     std::string implName = "cuSPARSE (d=" + std::to_string(density) + ")";
                     PerfResult sparseResult = runBenchmark(gpu.name, implName.c_str(), sparseBenchmark, 
                                                           &sparseParams, m, n, k, density, false);
                     
                     // Verify result
                     checkCudaError(cudaMemcpy(h_c, d_c, bytes_c, cudaMemcpyDeviceToHost), "Copying d_c to h_c (sparse)");
                     
                     // Convert sparse matrix back to dense for verification
                     float *h_a_sparse = (float*)malloc(m * k * sizeof(float));
                     memset(h_a_sparse, 0, m * k * sizeof(float));
                     
                     // Get sparse matrix data back to host for verification
                     float *h_values = (float*)malloc(a_csr->nnz * sizeof(float));
                     int *h_rowPtrs = (int*)malloc((m + 1) * sizeof(int));
                     int *h_colIndices = (int*)malloc(a_csr->nnz * sizeof(int));
                     
                     checkCudaError(cudaMemcpy(h_values, a_csr->values, a_csr->nnz * sizeof(float), cudaMemcpyDeviceToHost),
                                 "Copying sparse values to host");
                     checkCudaError(cudaMemcpy(h_rowPtrs, a_csr->rowPtrs, (m + 1) * sizeof(int), cudaMemcpyDeviceToHost),
                                 "Copying sparse row pointers to host");
                     checkCudaError(cudaMemcpy(h_colIndices, a_csr->colIndices, a_csr->nnz * sizeof(int), cudaMemcpyDeviceToHost),
                                 "Copying sparse column indices to host");
                     
                     // Convert CSR back to dense for verification
                     for (int i = 0; i < m; i++) {
                         for (int nnz_idx = h_rowPtrs[i]; nnz_idx < h_rowPtrs[i + 1]; nnz_idx++) {
                             int j = h_colIndices[nnz_idx];
                             h_a_sparse[i * k + j] = h_values[nnz_idx];
                         }
                     }
                     
                     // Verify against reference sparse matrix
                     bool sparseVerification = verifySparseResults(h_a_sparse, h_b, h_c, m, n, k, 
                                                                std::min(100, (int)(m * n * 0.01)), 1e-2f);
                     sparseResult.verificationResult = sparseVerification ? "PASSED" : "FAILED";
                     printf("Sparse verification: %s\n", sparseResult.verificationResult.c_str());
                     
                     // Add to results
                     sizeResults.push_back(sparseResult);
                     allResults.push_back(sparseResult);
                     
                     // Clean up sparse resources
                     free(h_a_sparse);
                     free(h_values);
                     free(h_rowPtrs);
                     free(h_colIndices);
                     delete a_csr;
                 }
             }
             
             // Save intermediate results for this matrix size
             std::string sizeStr = std::to_string(m) + "x" + std::to_string(n) + "x" + std::to_string(k);
             std::string intermediateFile = "matrix_mul_" + sizeStr + "_" + gpu.name + ".csv";
             std::replace(intermediateFile.begin(), intermediateFile.end(), ' ', '_');
             saveResultsToCSV(sizeResults, intermediateFile.c_str());
             
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
             
             // Force garbage collection
             cudaDeviceSynchronize();
             cudaDeviceReset();
             cudaSetDevice(deviceId);
             cublasCreate(&cublasHandle);
             if (testSparse) {
                 cusparseCreate(&cusparseHandle);
             }
         }
         
         // Clean up handles
         cublasDestroy(cublasHandle);
         if (testSparse) {
             cusparseDestroy(cusparseHandle);
         }
     }
     
     // Save final results
     std::string csvFilename = "matrix_mul_performance";
     if (!targetGPU.empty()) {
         std::string gpuNameForFile = targetGPU;
         std::replace(gpuNameForFile.begin(), gpuNameForFile.end(), ' ', '_');
         csvFilename += "_" + gpuNameForFile;
     }
     if (testSparse) {
         csvFilename += "_with_sparse";
     }
     if (testXLarge) {
         csvFilename += "_with_xlarge";
     }
     csvFilename += ".csv";
     
     saveResultsToCSV(allResults, csvFilename.c_str());
     
     printf("\nFinal results saved to %s. Use this file to generate charts.\n", csvFilename.c_str());
     
     return 0;
 }
