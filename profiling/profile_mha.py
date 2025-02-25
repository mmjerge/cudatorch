import torch
import torch.nn as nn
from torch.autograd.profiler import profile, record_function
import os
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import time
from datetime import datetime
import json

# Parse command line arguments
parser = argparse.ArgumentParser(description='Enhanced PyTorch MultiheadAttention Profiler')
parser.add_argument('--embed_dim', type=int, default=512, help='Embedding dimension')
parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
parser.add_argument('--seq_len', type=int, default=10, help='Sequence length')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--warmup', type=int, default=5, help='Number of warmup iterations')
parser.add_argument('--iterations', type=int, default=20, help='Number of iterations for testing')
parser.add_argument('--precision', choices=['fp32', 'fp16', 'bf16'], default='fp32', help='Precision for computation')
parser.add_argument('--compare_compile', action='store_true', help='Compare with torch.compile (PyTorch 2.0+)')
parser.add_argument('--compare_flash', action='store_true', help='Compare with FlashAttention if available')
parser.add_argument('--benchmark_params', action='store_true', help='Run benchmarks across various parameter sets')
parser.add_argument('--log_dir', type=str, default='./profiling_logs', help='Directory for profiling logs')
parser.add_argument('--causal', action='store_true', help='Use causal attention mask')
args = parser.parse_args()

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
run_id = f"mha_prof_{args.embed_dim}d_{args.num_heads}h_{args.seq_len}s_{args.batch_size}b_{args.precision}_{timestamp}"

log_dir = args.log_dir
os.makedirs(log_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")

print(f"PyTorch Version: {torch.__version__}")
print(f"Configuration: {args.embed_dim}d, {args.num_heads}h, {args.seq_len}s, {args.batch_size}b, {args.precision}")

def get_memory_stats():
    if device.type == 'cuda':
        return {
            'allocated': torch.cuda.memory_allocated() / (1024**2),  # MB
            'reserved': torch.cuda.memory_reserved() / (1024**2),    # MB
            'max_allocated': torch.cuda.max_memory_allocated() / (1024**2)  # MB
        }
    return {'allocated': 0, 'reserved': 0, 'max_allocated': 0}

def get_tensor_dtype(precision):
    if precision == 'fp16':
        return torch.float16
    elif precision == 'bf16' and hasattr(torch, 'bfloat16'):
        return torch.bfloat16
    return torch.float32

dtype = get_tensor_dtype(args.precision)
mha = nn.MultiheadAttention(embed_dim=args.embed_dim, num_heads=args.num_heads).to(device)

if dtype != torch.float32:
    mha = mha.to(dtype=dtype)

query = torch.randn(args.seq_len, args.batch_size, args.embed_dim, device=device, dtype=dtype)
key = torch.randn(args.seq_len, args.batch_size, args.embed_dim, device=device, dtype=dtype)
value = torch.randn(args.seq_len, args.batch_size, args.embed_dim, device=device, dtype=dtype)

attn_mask = None
if args.causal:
    attn_mask = torch.triu(
        torch.ones(args.seq_len, args.seq_len, device=device) * float('-inf'), 
        diagonal=1
    )
    print("Using causal attention mask")

has_flash_attn = False
if args.compare_flash:
    try:
        from flash_attn import flash_attn_func
        
        def flash_attention(q, k, v, attn_mask=None):
            q = q.transpose(0, 1)  # [batch_size, seq_len, embed_dim]
            k = k.transpose(0, 1)
            v = v.transpose(0, 1)
            
            head_dim = args.embed_dim // args.num_heads
            q = q.view(args.batch_size, args.seq_len, args.num_heads, head_dim)
            k = k.view(args.batch_size, args.seq_len, args.num_heads, head_dim)
            v = v.view(args.batch_size, args.seq_len, args.num_heads, head_dim)
            
            output = flash_attn_func(q, k, v, causal=args.causal)
            output = output.view(args.batch_size, args.seq_len, args.embed_dim)
            
            return output.transpose(0, 1), None
        
        has_flash_attn = True
        print("FlashAttention is available for comparison")
    except ImportError:
        print("FlashAttention not installed. Skipping comparison.")

has_compile = False
if args.compare_compile:
    try:
        if hasattr(torch, 'compile'):
            compiled_mha = torch.compile(mha)
            has_compile = True
            print("torch.compile is available for comparison")
        else:
            print("torch.compile not available in this PyTorch version")
    except Exception as e:
        print(f"Error setting up torch.compile: {e}")

def run_detailed_profiling(model_func, name, input_data):
    q, k, v, mask = input_data
    
    torch.cuda.reset_peak_memory_stats()
    mem_before = get_memory_stats()
    
    with profile(with_stack=True, use_cuda=True, with_modules=True) as prof:
        with record_function(f"{name}_detailed"):
            output, attention = model_func(q, k, v, attn_mask=mask)
            torch.cuda.synchronize()
    
    mem_after = get_memory_stats()
    peak_mem = torch.cuda.max_memory_allocated() / (1024**2)  # MB
    
    mem_usage = {
        'allocated_mb': mem_after['allocated'] - mem_before['allocated'],
        'peak_mb': peak_mem
    }
    
    profiling_output = prof.key_averages(group_by_stack_n=10).table(sort_by="cuda_time_total", row_limit=30)
    print(f"\n{name} Profiling Results:")
    print(profiling_output)
    print(f"Memory allocated: {mem_usage['allocated_mb']:.2f} MB")
    print(f"Peak memory: {mem_usage['peak_mb']:.2f} MB")
    
    text_log_path = os.path.join(log_dir, f"{run_id}_{name}_profiling_log.txt")
    with open(text_log_path, "w") as f:
        f.write(profiling_output)
        f.write(f"\nMemory allocated: {mem_usage['allocated_mb']:.2f} MB\n")
        f.write(f"Peak memory: {mem_usage['peak_mb']:.2f} MB\n")
    
    chrome_trace_path = os.path.join(log_dir, f"{run_id}_{name}_trace.json")
    prof.export_chrome_trace(chrome_trace_path)
    
    stack_trace_path = os.path.join(log_dir, f"{run_id}_{name}_stacks.txt")
    prof.export_stacks(stack_trace_path, "self_cuda_time_total")
    

    kernel_data = []
    for item in prof.key_averages():
        if item.key.find("cuda") != -1:  # Only CUDA kernels
            kernel_data.append({
                'name': item.key,
                'cuda_time_total': item.cuda_time_total,
                'cuda_time_avg': item.cuda_time,
                'calls': item.count
            })
    
    kernel_data = sorted(kernel_data, key=lambda x: x['cuda_time_total'], reverse=True)[:10]
    
    if kernel_data:
        plt.figure(figsize=(12, 8))
        plt.barh([k['name'][:50] for k in kernel_data], [k['cuda_time_total'] for k in kernel_data])
        plt.title(f"Top 10 CUDA Kernels for {name}")
        plt.xlabel('CUDA Time (μs)')
        plt.ylabel('Kernel Name')
        plt.tight_layout()
        plt.savefig(os.path.join(log_dir, f"{run_id}_{name}_top_kernels.png"))
        plt.close()
    
    return {
        'profiling_output': profiling_output,
        'memory_usage': mem_usage,
        'kernel_data': kernel_data
    }

def run_timing_benchmark(model_func, name, input_data, num_iterations=100):
    q, k, v, mask = input_data
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    torch.cuda.reset_peak_memory_stats()
    mem_before = get_memory_stats()
    
    start_event.record()
    
    for _ in range(num_iterations):
        output, _ = model_func(q, k, v, attn_mask=mask)
    
    end_event.record()
    torch.cuda.synchronize()
    
    elapsed_time = start_event.elapsed_time(end_event)
    avg_time_ms = elapsed_time / num_iterations
    
    mem_after = get_memory_stats()
    peak_mem = torch.cuda.max_memory_allocated() / (1024**2)  # MB
    
    result = {
        'name': name,
        'avg_time_ms': avg_time_ms,
        'total_time_ms': elapsed_time,
        'iterations': num_iterations,
        'memory_allocated_mb': mem_after['allocated'] - mem_before['allocated'],
        'peak_memory_mb': peak_mem
    }
    
    print(f"\n{name} Timing Benchmark:")
    print(f"Avg time per iteration: {avg_time_ms:.4f} ms")
    print(f"Memory allocated: {result['memory_allocated_mb']:.2f} MB")
    print(f"Peak memory: {result['peak_memory_mb']:.2f} MB")
    
    return result

def benchmark_across_params():
    results = []
    
    configs = [
        (256, 4, args.seq_len, args.batch_size),
        (512, 8, args.seq_len, args.batch_size),
        (768, 12, args.seq_len, args.batch_size),
        (1024, 16, args.seq_len, args.batch_size),
        
        (args.embed_dim, args.num_heads, 8, args.batch_size),
        (args.embed_dim, args.num_heads, 16, args.batch_size),
        (args.embed_dim, args.num_heads, 32, args.batch_size),
        (args.embed_dim, args.num_heads, 64, args.batch_size),
        
        (args.embed_dim, args.num_heads, args.seq_len, 16),
        (args.embed_dim, args.num_heads, args.seq_len, 32),
        (args.embed_dim, args.num_heads, args.seq_len, 64),
        (args.embed_dim, args.num_heads, args.seq_len, 128)
    ]
    
    print("\nRunning parameter benchmarks:")
    for embed_dim, num_heads, seq_len, batch_size in configs:
        config_name = f"d{embed_dim}_h{num_heads}_s{seq_len}_b{batch_size}"
        print(f"\nTesting configuration: {config_name}")
        
        test_mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads).to(device)
        test_mha = test_mha.to(dtype=dtype)
        
        test_q = torch.randn(seq_len, batch_size, embed_dim, device=device, dtype=dtype)
        test_k = torch.randn(seq_len, batch_size, embed_dim, device=device, dtype=dtype)
        test_v = torch.randn(seq_len, batch_size, embed_dim, device=device, dtype=dtype)
        
        test_mask = None
        if args.causal:
            test_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=device) * float('-inf'),
                diagonal=1
            )
        
        for _ in range(3):
            test_mha(test_q, test_k, test_v, attn_mask=test_mask)
            torch.cuda.synchronize()
        
        result = run_timing_benchmark(
            test_mha, 
            f"mha_{config_name}", 
            (test_q, test_k, test_v, test_mask), 
            num_iterations=10
        )
        
        result.update({
            'embed_dim': embed_dim,
            'num_heads': num_heads, 
            'seq_len': seq_len,
            'batch_size': batch_size,
            'precision': args.precision
        })
        
        results.append(result)
    
    df = pd.DataFrame(results)
    
    def create_comparison_plot(data, x_col, title, filename):
        plt.figure(figsize=(12, 8))
        plt.bar(data[x_col].astype(str), data['avg_time_ms'])
        plt.title(title)
        plt.xlabel(x_col)
        plt.ylabel('Average Time (ms)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(log_dir, filename))
        plt.close()
    
    if len(df[df['embed_dim'] != args.embed_dim]) > 0:
        embed_dim_data = df[df['seq_len'] == args.seq_len][df['batch_size'] == args.batch_size][df['num_heads'].isin([4, 8, 12, 16])]
        create_comparison_plot(
            embed_dim_data, 
            'embed_dim', 
            f'MHA Performance vs Embedding Dimension (seq_len={args.seq_len}, batch_size={args.batch_size})', 
            f'{run_id}_embed_dim_comparison.png'
        )
    
    if len(df[df['seq_len'] != args.seq_len]) > 0:
        seq_len_data = df[df['embed_dim'] == args.embed_dim][df['batch_size'] == args.batch_size][df['num_heads'] == args.num_heads]
        create_comparison_plot(
            seq_len_data, 
            'seq_len', 
            f'MHA Performance vs Sequence Length (embed_dim={args.embed_dim}, num_heads={args.num_heads})', 
            f'{run_id}_seq_len_comparison.png'
        )
    
    if len(df[df['batch_size'] != args.batch_size]) > 0:
        batch_size_data = df[df['embed_dim'] == args.embed_dim][df['seq_len'] == args.seq_len][df['num_heads'] == args.num_heads]
        create_comparison_plot(
            batch_size_data, 
            'batch_size', 
            f'MHA Performance vs Batch Size (embed_dim={args.embed_dim}, num_heads={args.num_heads})', 
            f'{run_id}_batch_size_comparison.png'
        )
    
    with open(os.path.join(log_dir, f"{run_id}_parameter_benchmark_results.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

print(f"\nStarting profiling run {run_id}")

print("Warming up...")
for _ in range(args.warmup):
    mha(query, key, value, attn_mask=attn_mask)
    torch.cuda.synchronize()
print("Warm-up complete")

input_data = (query, key, value, attn_mask)
standard_results = run_detailed_profiling(mha, "standard_mha", input_data)

benchmark_results = []

standard_timing = run_timing_benchmark(mha, "standard_mha", input_data, args.iterations)
benchmark_results.append(standard_timing)

if has_flash_attn:
    try:
        flash_results = run_detailed_profiling(flash_attention, "flash_attn", input_data)
        
        flash_timing = run_timing_benchmark(flash_attention, "flash_attn", input_data, args.iterations)
        benchmark_results.append(flash_timing)
    except Exception as e:
        print(f"Error during FlashAttention profiling: {e}")

if has_compile:
    try:
        print("Warming up compiled model...")
        for _ in range(args.warmup):
            compiled_mha(query, key, value, attn_mask=attn_mask)
            torch.cuda.synchronize()
        
        compiled_results = run_detailed_profiling(compiled_mha, "compiled_mha", input_data)
        
        compiled_timing = run_timing_benchmark(compiled_mha, "compiled_mha", input_data, args.iterations)
        benchmark_results.append(compiled_timing)
    except Exception as e:
        print(f"Error during torch.compile profiling: {e}")

if args.benchmark_params:
    param_results = benchmark_across_params()

if len(benchmark_results) > 1:
    implementations = [r['name'] for r in benchmark_results]
    times = [r['avg_time_ms'] for r in benchmark_results]
    
    plt.figure(figsize=(10, 6))
    plt.bar(implementations, times)
    plt.title('Performance Comparison of Different Implementations')
    plt.xlabel('Implementation')
    plt.ylabel('Average Time per Iteration (ms)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, f"{run_id}_implementation_comparison.png"))
    plt.close()
    
    with open(os.path.join(log_dir, f"{run_id}_benchmark_results.json"), 'w') as f:
        json.dump(benchmark_results, f, indent=2)

print(f"\nProfiling complete. Results saved to {log_dir}")
print(f"Run ID: {run_id}")