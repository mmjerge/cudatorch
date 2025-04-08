# cudatorch

## Usage Instructions

### Building and Running the Benchmark

1. Compile the benchmark code:
   ```bash
   nvcc -o matrix_mul_benchmark matrix_mul_benchmark.cu -lcublas -O3 -arch=sm_80
   ```

2. Run the benchmark locally
   ```bash
   ./matrix_mul_benchmark
   ```

This will test all implementations across the available GPUs and save results to matrix_mul_performance.csv.

### Generating Performance Charts

Generate visualization charts from the benchmark results:

  ```python
  python3 generate_charts.py
  ```

Charts will be saved to the charts/ directory, showing performance comparisons across:

Different GPU architectures (RTX 2080 Ti, A100, H100)
Various matrix sizes (32×32, 1024×1024, 8192×8192, 1024×2048)
All implementation methods (Naive, Shared Memory, Tensor Cores, cuBLAS, CUTLASS)

