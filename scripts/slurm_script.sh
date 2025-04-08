#!/bin/bash
#SBATCH --job-name=matrix_mul_bench
#SBATCH --output=matrix_mul_bench_%j.out
#SBATCH --error=matrix_mul_bench_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# This script will run the matrix multiplication benchmark on UVA's Slurm system
# It will test on RTX 2080 Ti, A100, and H100 GPUs as available

# Print node and GPU information
echo "Running on node: $(hostname)"
echo "Available GPUs:"
nvidia-smi -L

# Compile the CUDA program
echo "Compiling matrix_mul_benchmark.cu..."
nvcc -o matrix_mul_benchmark matrix_mul_benchmark.cu -lcublas -O3 -arch=sm_80

# Check if compilation was successful
if [ $? -ne 0 ]; then
    echo "Compilation failed. Exiting."
    exit 1
fi

# Run the benchmark on the allocated GPU
echo "Running benchmark..."
./matrix_mul_benchmark

# Check if benchmark was successful
if [ $? -ne 0 ]; then
    echo "Benchmark failed. Exiting."
    exit 1
fi

# Generate charts if the CSV file exists
if [ -f "matrix_mul_performance.csv" ]; then
    echo "Generating charts..."
    python3 generate_charts.py
else
    echo "No performance data found. Charts not generated."
fi

echo "Benchmarking complete. Check charts directory for performance visualizations."

# To run on multiple specific GPUs using Slurm's job array feature:
# Submit with: sbatch --array=0-2 run_benchmark.sh
# Where 0 could be RTX 2080 Ti, 1 could be A100, 2 could be H100

# Uncomment and modify this section to use job arrays:
#GPU_TYPES=("RTX_2080_Ti" "A100" "H100")
#CURRENT_GPU=${GPU_TYPES[$SLURM_ARRAY_TASK_ID]}
#echo "Testing on GPU type: $CURRENT_GPU"
#
# Add logic to select appropriate partition based on GPU type
# For example:
#case $CURRENT_GPU in
#    "RTX_2080_Ti")
#        PARTITION="gpu-2080ti"
#        ;;
#    "A100")
#        PARTITION="gpu-a100"
#        ;;
#    "H100")
#        PARTITION="gpu-h100"
#        ;;
#esac
