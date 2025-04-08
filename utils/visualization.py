#!/usr/bin/env python3
# Script to generate performance charts from benchmark results

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

def generate_charts(csv_file):
    # Load the CSV data
    df = pd.read_csv(csv_file)
    
    # Create output directory if it doesn't exist
    output_dir = 'charts'
    os.makedirs(output_dir, exist_ok=True)
    
    # Group by matrix size
    matrix_sizes = df.groupby(['M', 'N', 'K']).groups.keys()
    
    # Set color palette
    colors = sns.color_palette("tab10", n_colors=len(df['Implementation'].unique()))
    
    # Generate charts for each matrix size
    for size in matrix_sizes:
        m, n, k = size
        size_name = f"{m}x{n}" if m == k else f"{m}x{n}x{k}"
        
        # Filter data for this matrix size
        size_df = df[(df['M'] == m) & (df['N'] == n) & (df['K'] == k)]
        
        # Create a bar chart grouped by GPU and implementation
        plt.figure(figsize=(14, 8))
        
        # Group by GPU
        gpus = size_df['GPU'].unique()
        width = 0.8 / len(size_df['Implementation'].unique())
        
        # For each implementation, add a group of bars
        for i, implementation in enumerate(size_df['Implementation'].unique()):
            impl_df = size_df[size_df['Implementation'] == implementation]
            positions = [j + i * width for j in range(len(gpus))]
            
            plt.bar(positions, impl_df['Throughput_GFlops'], 
                    width=width, label=implementation, color=colors[i])
        
        # Set chart properties
        plt.title(f'Matrix Multiplication Performance for {size_name} Matrices', fontsize=16)
        plt.xlabel('GPU Architecture', fontsize=14)
        plt.ylabel('Throughput (GFlops)', fontsize=14)
        plt.xticks([i + (len(size_df['Implementation'].unique()) - 1) * width / 2 for i in range(len(gpus))], gpus)
        plt.legend(title='Implementation', fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add values on top of bars
        for i, implementation in enumerate(size_df['Implementation'].unique()):
            impl_df = size_df[size_df['Implementation'] == implementation]
            positions = [j + i * width for j in range(len(gpus))]
            
            for pos, value in zip(positions, impl_df['Throughput_GFlops']):
                plt.text(pos, value + max(size_df['Throughput_GFlops']) * 0.02, 
                        f'{value:.0f}', ha='center', fontsize=9)
        
        # Save the chart
        plt.tight_layout()
        plt.savefig(f'{output_dir}/performance_{size_name}.png', dpi=300)
        plt.close()
    
    # Create a summary chart comparing implementations across GPU architectures
    # Create a grouped bar chart for each matrix size
    plt.figure(figsize=(16, 10))
    
    # Use seaborn's FacetGrid for a multi-chart display
    g = sns.FacetGrid(df, col="M", row="N", hue="Implementation", 
                     height=4, aspect=1.2, legend_out=True)
    g.map(sns.barplot, "GPU", "Throughput_GFlops")
    g.add_legend(title="Implementation")
    g.fig.suptitle('Matrix Multiplication Performance Comparison', fontsize=18, y=1.02)
    g.set_axis_labels("GPU Architecture", "Throughput (GFlops)")
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/performance_summary.png', dpi=300)
    plt.close()
    
    print(f"Charts generated in the '{output_dir}' directory.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    else:
        csv_file = "matrix_mul_performance.csv"
    
    generate_charts(csv_file)
