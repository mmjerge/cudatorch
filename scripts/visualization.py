#!/usr/bin/env python3
"""
Improved Visualization Script for Matrix Multiplication Benchmark Results

This script reads CSV files from a directory containing benchmark results
and creates consolidated plots showing performance across different
implementations, matrix sizes, and GPUs.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import argparse
import os
import glob
from matplotlib.lines import Line2D

def load_data_from_directory(directory):
    """Load and combine benchmark data from all CSV files in the given directory."""
    print(f"Loading data from directory: {directory}...")
    
    # Find all CSV files in the directory
    csv_files = glob.glob(os.path.join(directory, "*.csv"))
    
    if not csv_files:
        raise ValueError(f"No CSV files found in directory: {directory}")
    
    # Load and combine all CSV files
    dfs = []
    for file in csv_files:
        print(f"  Loading {os.path.basename(file)}...")
        df = pd.read_csv(file)
        dfs.append(df)
    
    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(combined_df)} data points from {len(csv_files)} files")
    
    return combined_df

def preprocess_data(df):
    """Clean and preprocess the benchmark data."""
    # Create matrix size label
    df['MatrixSize'] = df.apply(lambda row: f"{row['M']}x{row['N']}x{row['K']}", axis=1)
    
    # Identify sparse matrices
    df['IsSparse'] = df['Density'] < 1.0
    df['DensityLabel'] = df.apply(lambda row: f"Density {row['Density']:.2f}" if row['IsSparse'] else "Dense", axis=1)
    
    # Create full implementation label that includes density for sparse implementations
    df['ImplementationFull'] = df.apply(
        lambda row: f"{row['Implementation']} ({row['DensityLabel']})" if row['IsSparse'] else row['Implementation'], 
        axis=1
    )
    
    # Check if verification passed
    df['VerificationPassed'] = df['Verification'] == "PASSED"
    
    # Standardize GPU names
    gpu_name_mapping = {
        'RTX 2080 Ti': 'NVIDIA GeForce RTX 2080 Ti',
        'A100': 'NVIDIA A100-SXM4-40GB',
        # Add more mappings as needed
    }
    
    # Apply the mapping if the GPU name exists in the mapping
    df['GPU_Display'] = df['GPU'].apply(lambda x: gpu_name_mapping.get(x, x))
    
    return df

def plot_throughput_consolidated(df, output_dir):
    """Create a consolidated plot for throughput showing all matrix sizes."""
    print("Creating consolidated throughput plot...")
    
    plt.figure(figsize=(16, 10))
    
    # Define markers for different GPU types
    gpu_markers = {
        'RTX 2080 Ti': 'o',
        'A100': 's',
        'H100': '^',
        'NVIDIA GeForce RTX 2080 Ti': 'D',  # Diamond
        'NVIDIA A100-SXM4-40GB': 'P',       # Plus (pentagon)
    }
    
    # Add more GPU types if needed with distinct markers
    marker_options = ['o', 's', '^', 'D', 'P', '*', 'X', 'h', 'p', '8']
    marker_index = 0
    for gpu in df['GPU'].unique():
        if gpu not in gpu_markers:
            gpu_markers[gpu] = marker_options[marker_index % len(marker_options)]
            marker_index += 1
    
    # Create color map for matrix sizes
    matrix_sizes = sorted(df['MatrixSize'].unique())
    size_colors = plt.cm.viridis(np.linspace(0, 1, len(matrix_sizes)))
    size_color_map = dict(zip(matrix_sizes, size_colors))
    
    # Plot data points with different markers for each GPU type
    for gpu in df['GPU'].unique():
        gpu_df = df[df['GPU'] == gpu]
        display_gpu = df[df['GPU'] == gpu]['GPU_Display'].iloc[0]
        
        for size in matrix_sizes:
            size_df = gpu_df[gpu_df['MatrixSize'] == size]
            
            if size_df.empty:
                continue
                
            for impl in size_df['Implementation'].unique():
                impl_df = size_df[size_df['Implementation'] == impl]
                
                # Filter for passed and failed verification
                passed = impl_df[impl_df['VerificationPassed']]
                failed = impl_df[~impl_df['VerificationPassed']]
                
                # Plot passed verification points
                if not passed.empty:
                    plt.scatter(
                        passed['Implementation'],
                        passed['Throughput_GFlops'],
                        marker=gpu_markers[gpu],
                        s=150,
                        color=size_color_map[size],
                        edgecolors='black',
                        alpha=0.8,
                        label=f"{display_gpu} - {size}" if impl == impl_df['Implementation'].iloc[0] else ""
                    )
                
                # Plot failed verification points with 'x' marker
                if not failed.empty:
                    plt.scatter(
                        failed['Implementation'],
                        failed['Throughput_GFlops'],
                        marker='x',
                        s=150,
                        color=size_color_map[size],
                        alpha=0.5
                    )
    
    plt.title('Matrix Multiplication Throughput Across All Matrix Sizes', fontsize=16)
    plt.ylabel('Throughput (GFlops)', fontsize=14)
    plt.xlabel('Implementation', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.yscale('log')  # Log scale for better visualization
    
    # Create custom legend for GPUs and matrix sizes
    gpu_legend_elements = []
    for gpu in df['GPU'].unique():
        display_gpu = df[df['GPU'] == gpu]['GPU_Display'].iloc[0]
        gpu_legend_elements.append(
            Line2D([0], [0], marker=gpu_markers[gpu], color='w', markerfacecolor='gray', 
                  markeredgecolor='black', markersize=10, label=display_gpu)
        )
    
    size_legend_elements = [
        Line2D([0], [0], marker='o', color=size_color_map[size], 
               markeredgecolor='black', markersize=10, label=size)
        for size in matrix_sizes
    ]
    
    verification_legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
               markeredgecolor='black', markersize=10, label='Verification Passed'),
        Line2D([0], [0], marker='x', color='gray', markersize=10, label='Verification Failed')
    ]
    
    # Add legends in separate locations
    plt.gca().add_artist(plt.legend(handles=size_legend_elements, 
                        loc='upper left', fontsize=12, title='Matrix Size'))
    plt.gca().add_artist(plt.legend(handles=gpu_legend_elements, 
                        loc='upper right', fontsize=12, title='GPU Type'))
    plt.legend(handles=verification_legend_elements, 
              loc='lower right', fontsize=12)
    
    plt.tight_layout()
    
    filename = "consolidated_throughput.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved {filename}")

def plot_execution_time_consolidated(df, output_dir):
    """Create a consolidated plot for execution time showing all matrix sizes."""
    print("Creating consolidated execution time plot...")
    
    plt.figure(figsize=(16, 10))
    
    # Define markers for different GPU types
    gpu_markers = {
        'RTX 2080 Ti': 'o',
        'A100': 's',
        'H100': '^',
        'NVIDIA GeForce RTX 2080 Ti': 'D',  # Diamond
        'NVIDIA A100-SXM4-40GB': 'P',       # Plus (pentagon)
    }
    
    # Add more GPU types if needed with distinct markers
    marker_options = ['o', 's', '^', 'D', 'P', '*', 'X', 'h', 'p', '8']
    marker_index = 0
    for gpu in df['GPU'].unique():
        if gpu not in gpu_markers:
            gpu_markers[gpu] = marker_options[marker_index % len(marker_options)]
            marker_index += 1
    
    # Create color map for matrix sizes
    matrix_sizes = sorted(df['MatrixSize'].unique())
    size_colors = plt.cm.plasma(np.linspace(0, 1, len(matrix_sizes)))
    size_color_map = dict(zip(matrix_sizes, size_colors))
    
    # Plot data points with different markers for each GPU type
    for gpu in df['GPU'].unique():
        gpu_df = df[df['GPU'] == gpu]
        display_gpu = df[df['GPU'] == gpu]['GPU_Display'].iloc[0]
        
        for size in matrix_sizes:
            size_df = gpu_df[gpu_df['MatrixSize'] == size]
            
            if size_df.empty:
                continue
                
            for impl in size_df['Implementation'].unique():
                impl_df = size_df[size_df['Implementation'] == impl]
                
                # Filter for passed and failed verification
                passed = impl_df[impl_df['VerificationPassed']]
                failed = impl_df[~impl_df['VerificationPassed']]
                
                # Plot passed verification points
                if not passed.empty:
                    plt.scatter(
                        passed['Implementation'],
                        passed['Time_ms'],
                        marker=gpu_markers[gpu],
                        s=150,
                        color=size_color_map[size],
                        edgecolors='black',
                        alpha=0.8,
                        label=f"{display_gpu} - {size}" if impl == impl_df['Implementation'].iloc[0] else ""
                    )
                
                # Plot failed verification points with 'x' marker
                if not failed.empty:
                    plt.scatter(
                        failed['Implementation'],
                        failed['Time_ms'],
                        marker='x',
                        s=150,
                        color=size_color_map[size],
                        alpha=0.5
                    )
    
    plt.title('Matrix Multiplication Execution Time Across All Matrix Sizes', fontsize=16)
    plt.ylabel('Execution Time (ms)', fontsize=14)
    plt.xlabel('Implementation', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.yscale('log')  # Log scale for better visualization
    
    # Create custom legend for GPUs and matrix sizes
    gpu_legend_elements = []
    for gpu in df['GPU'].unique():
        display_gpu = df[df['GPU'] == gpu]['GPU_Display'].iloc[0]
        gpu_legend_elements.append(
            Line2D([0], [0], marker=gpu_markers[gpu], color='w', markerfacecolor='gray', 
                  markeredgecolor='black', markersize=10, label=display_gpu)
        )
    
    size_legend_elements = [
        Line2D([0], [0], marker='o', color=size_color_map[size], 
               markeredgecolor='black', markersize=10, label=size)
        for size in matrix_sizes
    ]
    
    verification_legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
               markeredgecolor='black', markersize=10, label='Verification Passed'),
        Line2D([0], [0], marker='x', color='gray', markersize=10, label='Verification Failed')
    ]
    
    # Add legends in separate locations
    plt.gca().add_artist(plt.legend(handles=size_legend_elements, 
                        loc='upper left', fontsize=12, title='Matrix Size'))
    plt.gca().add_artist(plt.legend(handles=gpu_legend_elements, 
                        loc='upper right', fontsize=12, title='GPU Type'))
    plt.legend(handles=verification_legend_elements, 
              loc='lower right', fontsize=12)
    
    plt.tight_layout()
    
    filename = "consolidated_execution_time.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved {filename}")

def plot_gpu_comparison_consolidated(df, output_dir):
    """Create a consolidated bar chart comparing GPU performance across all matrix sizes."""
    print("Creating consolidated GPU comparison plot...")
    
    # Group by matrix size, implementation and GPU, get average throughput
    grouped_df = df.groupby(['MatrixSize', 'Implementation', 'GPU_Display'])['Throughput_GFlops'].mean().reset_index()
    
    matrix_sizes = sorted(grouped_df['MatrixSize'].unique())
    implementations = sorted(grouped_df['Implementation'].unique())
    gpus = sorted(grouped_df['GPU_Display'].unique())
    
    # Set up figure with subplots
    fig, axes = plt.subplots(len(matrix_sizes), 1, figsize=(14, 6 * len(matrix_sizes)), sharex=True)
    if len(matrix_sizes) == 1:
        axes = [axes]  # Make axes iterable if only one subplot
    
    for i, size in enumerate(matrix_sizes):
        ax = axes[i]
        size_df = grouped_df[grouped_df['MatrixSize'] == size]
        
        width = 0.8 / len(gpus)
        positions = np.arange(len(implementations))
        
        # Plot bars for each GPU
        for j, gpu in enumerate(gpus):
            gpu_data = []
            for impl in implementations:
                impl_data = size_df[(size_df['Implementation'] == impl) & (size_df['GPU_Display'] == gpu)]
                if not impl_data.empty:
                    gpu_data.append(impl_data['Throughput_GFlops'].values[0])
                else:
                    gpu_data.append(0)
            
            offset = (j - len(gpus)/2 + 0.5) * width
            ax.bar(positions + offset, gpu_data, width, label=gpu if i == 0 else "", 
                  alpha=0.8, edgecolor='black')
        
        ax.set_title(f'GPU Comparison for {size} Matrices', fontsize=14)
        ax.set_ylabel('Throughput (GFlops)', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7, axis='y')
        
        # Add y-axis labels and grid
        ax.set_xticks(positions)
        ax.set_xticklabels(implementations, rotation=45, ha='right', fontsize=10)
    
    # Add common x-axis label
    fig.text(0.5, 0.04, 'Implementation', ha='center', fontsize=14)
    
    # Add legend at the top
    fig.legend(gpus, loc='upper center', ncol=len(gpus), fontsize=12, 
              bbox_to_anchor=(0.5, 0.98), title='GPU Type')
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    filename = "consolidated_gpu_comparison.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved {filename}")

def plot_speedup_vs_naive_consolidated(df, output_dir):
    """Create a consolidated plot showing speedup relative to naive implementation across all matrix sizes."""
    print("Creating consolidated speedup comparison plot...")
    
    # Check if naive implementation exists in the data
    if 'Naive' not in df['Implementation'].values:
        print("  No naive implementation found, skipping speedup plot")
        return
    
    # Group by matrix size, implementation and GPU, get mean execution time
    grouped_df = df.groupby(['MatrixSize', 'Implementation', 'GPU_Display'])['Time_ms'].mean().reset_index()
    
    matrix_sizes = sorted(grouped_df['MatrixSize'].unique())
    implementations = sorted([i for i in grouped_df['Implementation'].unique() if i != 'Naive'])
    gpus = sorted(grouped_df['GPU_Display'].unique())
    
    # Create a figure with a subplot for each GPU
    fig, axes = plt.subplots(len(gpus), 1, figsize=(14, 6 * len(gpus)), sharex=True)
    if len(gpus) == 1:
        axes = [axes]  # Make axes iterable if only one subplot
    
    for i, gpu in enumerate(gpus):
        ax = axes[i]
        gpu_df = grouped_df[grouped_df['GPU_Display'] == gpu]
        
        # Set up width for bars
        width = 0.8 / len(matrix_sizes)
        positions = np.arange(len(implementations))
        
        # Plot bars for each matrix size
        for j, size in enumerate(matrix_sizes):
            size_df = gpu_df[gpu_df['MatrixSize'] == size]
            
            # Get naive implementation time for this size and GPU
            naive_time = size_df[size_df['Implementation'] == 'Naive']['Time_ms'].values
            
            if len(naive_time) == 0:
                continue
                
            naive_time = naive_time[0]
            
            # Calculate speedups for each implementation
            speedups = []
            for impl in implementations:
                impl_data = size_df[size_df['Implementation'] == impl]
                if not impl_data.empty:
                    impl_time = impl_data['Time_ms'].values[0]
                    speedup = naive_time / impl_time
                    speedups.append(speedup)
                else:
                    speedups.append(0)
            
            offset = (j - len(matrix_sizes)/2 + 0.5) * width
            ax.bar(positions + offset, speedups, width, label=size if i == 0 else "", 
                  alpha=0.8, edgecolor='black')
        
        ax.set_title(f'Implementation Speedup vs Naive for {gpu}', fontsize=14)
        ax.set_ylabel('Speedup Factor (higher is better)', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7, axis='y')
        
        # Add reference line at speedup = 1
        ax.axhline(y=1, color='red', linestyle='--', alpha=0.7)
        
        # Add y-axis labels and grid
        ax.set_xticks(positions)
        ax.set_xticklabels(implementations, rotation=45, ha='right', fontsize=10)
    
    # Add common x-axis label
    fig.text(0.5, 0.04, 'Implementation', ha='center', fontsize=14)
    
    # Add legend at the top
    fig.legend(matrix_sizes, loc='upper center', ncol=len(matrix_sizes), fontsize=12, 
              bbox_to_anchor=(0.5, 0.98), title='Matrix Size')
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    filename = "consolidated_speedup_vs_naive.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved {filename}")

def plot_sparse_comparison_consolidated(df, output_dir):
    """Create a consolidated plot comparing sparse implementations across different density levels."""
    print("Creating consolidated sparse comparison plot...")
    
    # Filter sparse implementations
    sparse_df = df[df['IsSparse']]
    
    if sparse_df.empty:
        print("  No sparse implementations found in data, skipping sparse comparison plots")
        return
    
    # Group by matrix size, GPU, and density
    grouped_df = sparse_df.groupby(['MatrixSize', 'GPU_Display', 'Density'])['Throughput_GFlops'].mean().reset_index()
    
    matrix_sizes = sorted(grouped_df['MatrixSize'].unique())
    gpus = sorted(grouped_df['GPU_Display'].unique())
    densities = sorted(grouped_df['Density'].unique())
    
    # Create figure with subplots for each matrix size
    fig, axes = plt.subplots(len(matrix_sizes), 1, figsize=(14, 5 * len(matrix_sizes)), sharex=True)
    if len(matrix_sizes) == 1:
        axes = [axes]  # Make axes iterable if only one subplot
    
    for i, size in enumerate(matrix_sizes):
        ax = axes[i]
        size_df = grouped_df[grouped_df['MatrixSize'] == size]
        
        width = 0.8 / len(gpus)
        positions = np.arange(len(densities))
        
        # Plot bars for each GPU
        for j, gpu in enumerate(gpus):
            gpu_df = size_df[size_df['GPU_Display'] == gpu]
            
            throughputs = []
            for density in densities:
                density_df = gpu_df[gpu_df['Density'] == density]
                if not density_df.empty:
                    throughputs.append(density_df['Throughput_GFlops'].values[0])
                else:
                    throughputs.append(0)
            
            offset = (j - len(gpus)/2 + 0.5) * width
            ax.bar(positions + offset, throughputs, width, label=gpu if i == 0 else "", 
                  alpha=0.8, edgecolor='black')
        
        ax.set_title(f'Sparse Matrix Multiplication Performance for {size} Matrices', fontsize=14)
        ax.set_ylabel('Throughput (GFlops)', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7, axis='y')
        
        # Add x-axis labels
        ax.set_xticks(positions)
        ax.set_xticklabels([f"{d:.2f}" for d in densities], fontsize=10)
    
    # Add common x-axis label
    fig.text(0.5, 0.04, 'Density', ha='center', fontsize=14)
    
    # Add legend at the top
    fig.legend(gpus, loc='upper center', ncol=len(gpus), fontsize=12, 
              bbox_to_anchor=(0.5, 0.98), title='GPU Type')
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    filename = "consolidated_sparse_comparison.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved {filename}")

def main():
    parser = argparse.ArgumentParser(description='Visualize matrix multiplication benchmark results')
    parser.add_argument('--input-dir', required=True, help='Input directory containing CSV files with benchmark results')
    parser.add_argument('--output-dir', '-o', default='plots', help='Output directory for plots')
    args = parser.parse_args()
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Created output directory: {args.output_dir}")
    
    df = load_data_from_directory(args.input_dir)
    df = preprocess_data(df)
    
    plot_throughput_consolidated(df, args.output_dir)
    plot_execution_time_consolidated(df, args.output_dir)
    plot_gpu_comparison_consolidated(df, args.output_dir)
    plot_speedup_vs_naive_consolidated(df, args.output_dir)
    plot_sparse_comparison_consolidated(df, args.output_dir)
    
    print(f"All consolidated plots saved to {args.output_dir}")

if __name__ == "__main__":
    main()