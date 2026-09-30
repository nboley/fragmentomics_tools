#!/usr/bin/env python
"""GPU benchmark: eager forward+backward at multiple batch sizes."""
import os, sys, time
import torch
from torch.profiler import ProfilerActivity, profile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
torch.set_float32_matmul_precision("high")

from background_model.cut_site_model import CutSiteFragmentModel

P = 2048
model = CutSiteFragmentModel(
    L_min=25, L_max=179, tile_size=P,
    N1=128, K=128, N2=32, H=32, L_chunk=32,
).cuda()
l_in = model.calc_input_region_size(P)

for B in [16, 32, 64]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    x = torch.randn(B, 4, l_in, device="cuda")

    try:
        # Warmup
        model.train()
        for _ in range(3):
            logits, ep, disp = model(x)
            loss = logits.sum()
            loss.backward()
            model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()

        # Time forward
        t0 = time.time()
        for _ in range(5):
            logits, ep, disp = model(x)
        torch.cuda.synchronize()
        fwd = (time.time() - t0) / 5

        # Time backward
        logits, ep, disp = model(x)
        loss = logits.sum()
        torch.cuda.synchronize()
        t0 = time.time()
        loss.backward()
        torch.cuda.synchronize()
        bwd = time.time() - t0
        model.zero_grad(set_to_none=True)

        vram = torch.cuda.max_memory_allocated() / 1e9
        print(f"B={B:3d}  fwd={fwd*1000:7.0f}ms  bwd={bwd*1000:7.0f}ms  total={((fwd+bwd)*1000):7.0f}ms  per_sample={fwd*1000/B:5.1f}ms  VRAM={vram:.2f}GB")
    except torch.cuda.OutOfMemoryError:
        print(f"B={B:3d}  OOM (peak={torch.cuda.max_memory_allocated()/1e9:.2f}GB)")
        torch.cuda.empty_cache()
        model.zero_grad(set_to_none=True)

# Profile at B=16 for kernel breakdown
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
B = 16
x = torch.randn(B, 4, l_in, device="cuda")
model.train()
model.zero_grad(set_to_none=True)

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    for _ in range(3):
        logits, ep, disp = model(x)
        loss = logits.sum()
        loss.backward()
        model.zero_grad(set_to_none=True)
torch.cuda.synchronize()

print("\n=== Top CUDA ops (B=16) ===")
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
print(f"\nPeak VRAM (profile): {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
