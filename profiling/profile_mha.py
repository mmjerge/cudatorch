import torch
import torch.nn as nn
import torch.profiler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

mha = nn.MultiheadAttention(embed_dim=512, num_heads=8).to(device)
query = torch.randn(10, 32, 512, device=device)  # (seq_len, batch_size, embed_dim)
key = torch.randn(10, 32, 512, device=device)
value = torch.randn(10, 32, 512, device=device)

# Profiling PyTorch's MultiheadAttention
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA
    ],
    schedule=torch.profiler.schedule(
        wait=1,
        warmup=1,
        active=3,
        repeat=1
    ),
    on_trace_ready=torch.profiler.tensorboard_trace_handler("./profiling_logs"),
    record_shapes=True,
    with_stack=True
) as prof:
    for step in range(5):  # Run 5 iterations
        output, attn_weights = mha(query, key, value)
        prof.step()  # Need to call step() when using schedule

# Print profiling results
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))