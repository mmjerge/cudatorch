import torch
import torch.nn as nn
from torch.autograd.profiler import profile, record_function
import os

# Ensure the directory for logs exists
log_dir = "./profiling_logs"
os.makedirs(log_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

mha = nn.MultiheadAttention(embed_dim=512, num_heads=8).to(device)
query = torch.randn(10, 32, 512, device=device)  # (seq_len, batch_size, embed_dim)
key = torch.randn(10, 32, 512, device=device)
value = torch.randn(10, 32, 512, device=device)

# Warm up
for _ in range(5):
    mha(query, key, value)
torch.cuda.synchronize()

# Profile with function tracing
with profile(with_stack=True, use_cuda=True, with_modules=True) as prof:
    with record_function("multihead_attention_detailed"):
        output, attention = mha(query, key, value)
        torch.cuda.synchronize()

# Print results with source information
profiling_output = prof.key_averages(group_by_stack_n=10).table(sort_by="cuda_time_total", row_limit=30)
print(profiling_output)

# Save text log
text_log_path = os.path.join(log_dir, "torch_profiling_log.txt")
with open(text_log_path, "w") as f:
    f.write(profiling_output)

# Save to Chrome trace file (for visualization)
chrome_trace_path = os.path.join(log_dir, "torch_profiling_trace.json")
prof.export_chrome_trace(chrome_trace_path)

# Save to TensorBoard format
stack_trace_path = os.path.join(log_dir, "torch_profiling_stacks.txt")
prof.export_stacks(stack_trace_path, "self_cuda_time_total")