#!/usr/bin/env python3
"""
Visualization Script for Matrix Multiplication Benchmark Results

This script reads the CSV output from the matrix multiplication benchmark
and creates various scatter plots to visualize performance across different
implementations and GPUs.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import argparse
import os
from matplotlib.lines import Line2D

def load_data(filename):
    """Load the benchmark data from CSV file."""
    print(f"Loading data from {filename}...")
    df = pd.read_csv(filename)
    print(f"Loaded {len(df)} data points")
    return df

def preprocess_data(df):
    """Clean and preprocess the benchmark data."""
    df['MatrixSize'] = df.apply(lambda row: f"{row['M']}x{row['N']}x{row['K']}", axis=1)
    
    df['IsSparse'] = df['Density'] < 1.0
    df['DensityLabel'] = df.apply(lambda row: f"Density {row['Density']:.2f}" if row['IsSparse'] else "Dense", axis=1)
    
    df['ImplementationFull'] = df.apply(
        lambda row: f"{row['Implementation']} ({row['DensityLabel']})" if row['IsSparse'] else row['Implementation'], 
        axis=1
    )
    
    df['VerificationPassed'] = df['Verification'] == "PASSED"
    
    return df

def plot_throughput_by_implementation(df, output_dir):
    """Create scatter plots for throughput by implementation for each matrix size."""
    print("Creating throughput by implementation plots...")
    
    matrix_sizes = df['MatrixSize'].unique()
    
    for size in matrix_sizes:
        size_df = df[df['MatrixSize'] == size]
        
        plt.figure(figsize=(14, 8))
        
        gpu_markers = {
            'RTX 2080 Ti': 'o',
            'A100': 's',
            'H100': '^'
        }
        
        implementations = size_df['ImplementationFull'].unique()
        implementation_colors = plt.cm.viridis(np.linspace(0, 1, len(implementations)))
        implementation_color_map = dict(zip(implementations, implementation_colors))
        
        for i, gpu in enumerate(size_df['GPU'].unique()):
            gpu_df = size_df[size_df['GPU'] == gpu]
            
            for impl in gpu_df['ImplementationFull'].unique():
                impl_df = gpu_df[gpu_df['ImplementationFull'] == impl]
                
                passed = impl_df[impl_df['VerificationPassed']]
                failed = impl_df[~impl_df['VerificationPassed']]
                
                plt.scatter(
                    passed['ImplementationFull'] if not passed.empty else [],
                    passed['Throughput_GFlops'] if not passed.empty else [],
                    marker=gpu_markers.get(gpu, 'o'),
                    s=100,
                    color=implementation_color_map[impl],
                    edgecolors='black',
                    alpha=0.8,
                    label=f"{gpu} - {impl}" if i == 0 else ""
                )
                
                if not failed.empty:
                    plt.scatter(
                        failed['ImplementationFull'],
                        failed['Throughput_GFlops'],
                        marker='x',
                        s=100,
                        color=implementation_color_map[impl],
                        alpha=0.5
                    )
        
        plt.title(f'Matrix Multiplication Throughput for {size} Matrices', fontsize=16)
        plt.ylabel('Throughput (GFlops)', fontsize=14)
        plt.xlabel('Implementation', fontsize=14)
        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        gpu_legend_elements = [
            Line2D([0], [0], marker=marker, color='w', markerfacecolor='gray', 
                   markeredgecolor='black', markersize=10, label=gpu)
            for gpu, marker in gpu_markers.items()
        ]
        
        verification_legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                   markeredgecolor='black', markersize=10, label='Verification Passed'),
            Line2D([0], [0], marker='x', color='gray', markersize=10, label='Verification Failed')
        ]
        
        plt.legend(handles=gpu_legend_elements + verification_legend_elements, 
                  loc='upper left', fontsize=12, bbox_to_anchor=(1, 1))
        
        plt.tight_layout()
        
        filename = f"throughput_{size.replace('x', '_')}.png"
        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved {filename}")

def plot_execution_time_by_implementation(df, output_dir):
    """Create scatter plots for execution time by implementation for each matrix size."""
    print("Creating execution time by implementation plots...")
    
    matrix_sizes = df['MatrixSize'].unique()
    
    for size in matrix_sizes:
        size_df = df[df['MatrixSize'] == size]
        
        plt.figure(figsize=(14, 8))
        
        gpu_markers = {
            'RTX 2080 Ti': 'o',
            'A100': 's',
            'H100': '^'
        }
        
        implementations = size_df['ImplementationFull'].unique()
        implementation_colors = plt.cm.plasma(np.linspace(0, 1, len(implementations)))
        implementation_color_map = dict(zip(implementations, implementation_colors))
        
        for i, gpu in enumerate(size_df['GPU'].unique()):
            gpu_df = size_df[size_df['GPU'] == gpu]
            
            for impl in gpu_df['ImplementationFull'].unique():
                impl_df = gpu_df[gpu_df['ImplementationFull'] == impl]
                
                passed = impl_df[impl_df['VerificationPassed']]
                failed = impl_df[~impl_df['VerificationPassed']]
                
                plt.scatter(
                    passed['ImplementationFull'] if not passed.empty else [],
                    passed['Time_ms'] if not passed.empty else [],
                    marker=gpu_markers.get(gpu, 'o'),
                    s=100,
                    color=implementation_color_map[impl],
                    edgecolors='black',
                    alpha=0.8,
                    label=f"{gpu} - {impl}" if i == 0 else ""
                )
                
                if not failed.empty:
                    plt.scatter(
                        failed['ImplementationFull'],
                        failed['Time_ms'],
                        marker='x',
                        s=100,
                        color=implementation_color_map[impl],
                        alpha=0.5
                    )
        
        plt.title(f'Matrix Multiplication Execution Time for {size} Matrices', fontsize=16)
        plt.ylabel('Execution Time (ms)', fontsize=14)
        plt.xlabel('Implementation', fontsize=14)
        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.yscale('log')
        
        gpu_legend_elements = [
            Line2D([0], [0], marker=marker, color='w', markerfacecolor='gray', 
                   markeredgecolor='black', markersize=10, label=gpu)
            for gpu, marker in gpu_markers.items()
        ]
        
        verification_legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                   markeredgecolor='black', markersize=10, label='Verification Passed'),
            Line2D([0], [0], marker='x', color='gray', markersize=10, label='Verification Failed')
        ]
        
        plt.legend(handles=gpu_legend_elements + verification_legend_elements, 
                  loc='upper left', fontsize=12, bbox_to_anchor=(1, 1))
        
        plt.tight_layout()
        
        filename = f"execution_time_{size.replace('x', '_')}.png"
        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved {filename}")

def plot_gpu_comparison(df, output_dir):
    """Create bar charts comparing GPUs for each implementation."""
    print("Creating GPU comparison plots...")
    
    matrix_sizes = df['MatrixSize'].unique()
    
    for size in matrix_sizes:
        size_df = df[df['MatrixSize'] == size]
        
        plt.figure(figsize=(16, 10))
        
        implementations = []
        gpus = sorted(size_df['GPU'].unique())
        throughputs = []
        
        for impl in size_df['ImplementationFull'].unique():
            implementations.append(impl)
            impl_df = size_df[size_df['ImplementationFull'] == impl]
            
            gpu_throughputs = []
            for gpu in gpus:
                gpu_df = impl_df[impl_df['GPU'] == gpu]
                if not gpu_df.empty:
                    gpu_throughputs.append(gpu_df['Throughput_GFlops'].values[0])
                else:
                    gpu_throughputs.append(0)
            
            throughputs.append(gpu_throughputs)
        
        throughputs = np.array(throughputs)
        
        width = 0.8 / len(gpus)
        
        positions = np.arange(len(implementations))
        
        for i, gpu in enumerate(gpus):
            offset = (i - len(gpus)/2 + 0.5) * width
            plt.bar(positions + offset, throughputs[:, i], width, 
                   label=gpu, alpha=0.8, edgecolor='black')
        
        plt.title(f'GPU Comparison for {size} Matrices', fontsize=16)
        plt.ylabel('Throughput (GFlops)', fontsize=14)
        plt.xlabel('Implementation', fontsize=14)
        plt.xticks(positions, implementations, rotation=45, ha='right', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7, axis='y')
        plt.legend(title='GPU', fontsize=12)
        
        plt.tight_layout()
        
        
        filename = f"gpu_comparison_{size.replace('x', '_')}.png"
        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved {filename}")

def plot_sparse_comparison(df, output_dir):
    """Create plots comparing sparse implementations across different density levels."""
    print("Creating sparse comparison plots...")
    
    sparse_df = df[df['IsSparse']]
    
    if sparse_df.empty:
        print("  No sparse implementations found in data, skipping sparse comparison plots")
        return
    
    matrix_sizes = sparse_df['MatrixSize'].unique()
    
    for size in matrix_sizes:
        size_df = sparse_df[sparse_df['MatrixSize'] == size]
        
        if size_df.empty:
            continue
        
        plt.figure(figsize=(14, 8))
        
        gpus = size_df['GPU'].unique()
        densities = sorted(size_df['Density'].unique())
        
        width = 0.8 / len(gpus)
        
        positions = np.arange(len(densities))
        
        for i, gpu in enumerate(gpus):
            gpu_df = size_df[size_df['GPU'] == gpu]
            
            throughputs = []
            for density in densities:
                density_df = gpu_df[gpu_df['Density'] == density]
                if not density_df.empty:
                    throughputs.append(density_df['Throughput_GFlops'].values[0])
                else:
                    throughputs.append(0)
            
            offset = (i - len(gpus)/2 + 0.5) * width
            plt.bar(positions + offset, throughputs, width, 
                   label=gpu, alpha=0.8, edgecolor='black')
        
        plt.title(f'Sparse Matrix Multiplication Performance for {size} Matrices', fontsize=16)
        plt.ylabel('Throughput (GFlops)', fontsize=14)
        plt.xlabel('Density', fontsize=14)
        plt.xticks(positions, [f"{d:.2f}" for d in densities], fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7, axis='y')
        plt.legend(title='GPU', fontsize=12)
        
        plt.tight_layout()
        
        filename = f"sparse_comparison_{size.replace('x', '_')}.png"
        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved {filename}")

def plot_speedup_vs_naive(df, output_dir):
    """Create plots showing speedup relative to naive implementation."""
    print("Creating speedup comparison plots...")
    
    matrix_sizes = df['MatrixSize'].unique()
    
    for size in matrix_sizes:
        size_df = df[df['MatrixSize'] == size]
        
        naive_df = size_df[size_df['Implementation'] == 'Naive']
        if naive_df.empty:
            print(f"  No naive implementation found for {size}, skipping speedup plot")
            continue
        
        plt.figure(figsize=(14, 8))
        
        for gpu in size_df['GPU'].unique():
            gpu_df = size_df[size_df['GPU'] == gpu]
            
            naive_time = gpu_df[gpu_df['Implementation'] == 'Naive']['Time_ms'].values
            
            if len(naive_time) == 0:
                continue
                
            naive_time = naive_time[0]
            
            impl_times = []
            impl_names = []
            impl_speedups = []
            
            for impl in gpu_df['ImplementationFull'].unique():
                if 'Naive' in impl:
                    continue
                    
                impl_df = gpu_df[gpu_df['ImplementationFull'] == impl]
                if not impl_df.empty:
                    impl_time = impl_df['Time_ms'].values[0]
                    speedup = naive_time / impl_time
                    
                    impl_names.append(impl)
                    impl_times.append(impl_time)
                    impl_speedups.append(speedup)
            
            sorted_indices = np.argsort(impl_speedups)
            impl_names = [impl_names[i] for i in sorted_indices]
            impl_speedups = [impl_speedups[i] for i in sorted_indices]
            
            plt.barh(impl_names, impl_speedups, alpha=0.7, edgecolor='black', label=gpu)
        
        plt.title(f'Implementation Speedup vs Naive for {size} Matrices', fontsize=16)
        plt.xlabel('Speedup Factor (higher is better)', fontsize=14)
        plt.ylabel('Implementation', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7, axis='x')
        plt.legend(title='GPU', fontsize=12)
        
        plt.axvline(x=1, color='red', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        filename = f"speedup_vs_naive_{size.replace('x', '_')}.png"
        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved {filename}")

def main():
    parser = argparse.ArgumentParser(description='Visualize matrix multiplication benchmark results')
    parser.add_argument('input_csv', help='Input CSV file with benchmark results')
    parser.add_argument('--output-dir', '-o', default='plots', help='Output directory for plots')
    args = parser.parse_args()
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Created output directory: {args.output_dir}")
    
    df = load_data(args.input_csv)
    df = preprocess_data(df)
    
    plot_throughput_by_implementation(df, args.output_dir)
    plot_execution_time_by_implementation(df, args.output_dir)
    plot_gpu_comparison(df, args.output_dir)
    plot_sparse_comparison(df, args.output_dir)
    plot_speedup_vs_naive(df, args.output_dir)
    
    print(f"All plots saved to {args.output_dir}")

if __name__ == "__main__":
    main()