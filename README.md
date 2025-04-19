# Matrix Performance Benchmark

A comprehensive benchmark for comparing matrix multiplication performance across different GPU architectures and implementation methods.

## Features

- Multiple matrix multiplication implementations:
  - Naive CUDA implementation
  - Shared memory optimization
  - Tensor cores (WMMA) implementation
  - cuBLAS implementation
  - CUTLASS implementation
- Cross-GPU architecture testing (RTX 2080 Ti, A100, H100)
- Various matrix size testing:
  - Small (32×32)
  - Medium (1024×1024)
  - Large (8192×8192)
  - Non-square (1024×2048×1024)
- Performance measurements in GFLOPs
- Automated chart generation

## Building the Benchmark

```bash
# Basic compilation
nvcc -o matrix_multiplication ./profiling/matrix_multiplication.cu -lcublas -lcusparse -O3 -arch=sm_70

# For CUTLASS support
nvcc -o matrix_multiplication ./profiling/matrix_multiplication.cu -lcublas -lcusparse -I/path/to/cutlass/include -O3 -arch=sm_70
```

## Usage Instructions

### Running Locally

```bash
# Run on all available GPUs
./matrix_multiplication

# Run on a specific GPU type
./matrix_multiplication --gpu "RTX 2080 Ti"
./matrix_multiplication --gpu "A100"
./matrix_multiplication --gpu "H100"
```

### Running on Specific GPU Types with Direct Access

#### Option 1: Use CUDA_VISIBLE_DEVICES environment variable
```bash
# Run only on the first GPU in the system
CUDA_VISIBLE_DEVICES=0 ./matrix_multiplication

# Run only on the second GPU
CUDA_VISIBLE_DEVICES=1 ./matrix_multiplication
```

#### Option 2: SSH into specific machines
```bash
# SSH to machine with RTX 2080 Ti
ssh rtx2080ti-node "cd /path/to/benchmark && ./matrix_multiplication"

# SSH to machine with A100
ssh a100-node "cd /path/to/benchmark && ./matrix_multiplication"

# SSH to machine with H100
ssh h100-node "cd /path/to/benchmark && ./matrix_multiplication"
```

### Running on UVA HPC with Slurm

```bash
# Submit the job to the Slurm queue
sbatch run_benchmark.sh

# Use Slurm job arrays to run on multiple GPU types
sbatch --array=0-2 run_benchmark.sh
```

### Generating Performance Charts

```bash
python3 generate_charts.py matrix_mul_performance.csv
```

Charts will be saved to the `charts/` directory, showing performance comparisons across GPU architectures, matrix sizes, and implementation methods.

## Adding CUTLASS Support

The repository includes a full CUTLASS implementation that demonstrates how to leverage NVIDIA's CUTLASS library for high-performance matrix multiplication. To use the CUTLASS implementation:

1. Make sure you have the CUTLASS library installed
2. Include the CUTLASS headers in your compilation
3. The implementation supports both standard GEMM operations and tensor core operations when available

## Results

The benchmark produces a CSV file with performance results that includes:
- GPU name
- Implementation type
- Matrix dimensions
- Execution time (ms)
- Throughput (GFLOPs)

The chart generation script visualizes these results to show:
- Relative performance across GPU architectures
- Performance scaling with matrix size
- Comparison between different implementation methods

Charts will be saved to the charts/ directory, showing performance comparisons across:

Different GPU architectures (RTX 2080 Ti, A100, H100)
Various matrix sizes (32×32, 1024×1024, 8192×8192, 1024×2048)
All implementation methods (Naive, Shared Memory, Tensor Cores, cuBLAS, CUTLASS)

