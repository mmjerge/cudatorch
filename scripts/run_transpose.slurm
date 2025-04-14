#!/bin/bash
#SBATCH --job-name=matrix_transpose_bench
#SBATCH --output=../results/transpose/matrix_transpose_bench_%A_%a.out
#SBATCH --error=../results/transpose/matrix_transpose_bench_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=gpu

# Define GPU types, binaries, and GRES strings
GPU_TYPES=("RTX2080" "A100" "H100")
BINARIES=("matrix_transpose_RTX2080" "matrix_transpose_A100" "matrix_transpose_H100")
GPU_GRES=("gpu:nvidia_geforce_rtx_2080_ti:1" "gpu:nvidia_a100-pcie-40gb:1" "gpu:nvidia_h100_nvl:1")

CURRENT_GPU=${GPU_TYPES[$SLURM_ARRAY_TASK_ID]}
BINARY=${BINARIES[$SLURM_ARRAY_TASK_ID]}
CURRENT_GRES=${GPU_GRES[$SLURM_ARRAY_TASK_ID]}

echo "Running on node: $(hostname)"
echo "Target GPU: $CURRENT_GPU"
echo "Requested GRES: $CURRENT_GRES"
echo "Running binary: $BINARY"
echo "Available GPUs:"
nvidia-smi -L

# Load necessary modules
module load cuda/12.4.0

# Manually export correct CUDA paths for runtime
export CUDA_ROOT=/sw/ubuntu2204/ebu082024/software/common/core/cuda/12.4.0
export LD_LIBRARY_PATH=$CUDA_ROOT/lib64:$LD_LIBRARY_PATH

echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
ldd ./$BINARY | grep cublas

# Run the binary
./$BINARY


# Save results
mkdir -p ../results/transpose
if [ -f "matrix_transpose_performance.csv" ]; then
    mv matrix_transpose_performance.csv "../results/transpose/matrix_transpose_${CURRENT_GPU}_performance.csv"
    echo "Results saved to ../results/transpose/matrix_transpose_${CURRENT_GPU}_performance.csv"
else
    echo "No performance data file found."
fi
