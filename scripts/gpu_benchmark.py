#!/usr/bin/env python
"""GPU benchmark: eager vs compiled forward+backward, plus torch.profiler."""
import os
import sys
import time
import subprocess

import torch
from torch.profiler import ProfilerActivity, profile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
torch.set_float32_matmul_precision("high")

from background_model.cut_site_model import CutSiteFragmentModel

# Check if gcc is available for torch.compile
def _has_gcc():
    try:
        subprocess.run(["gcc", "--version"], capture_output=True, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        # Try conda env gcc
        gcc = os.environ.get("CC", "")
        if gcc and os.path.exists(gcc):
            return True
        return False

DO_COMPILE = _has_gcc()
if not DO_COMPILE:
    print("WARNING: gcc not found, skipping torch.compile benchmark")

B = 16
model = CutSiteFragmentModel(
    L_min=25, L_max=179, tile_size=2048,
    N1=128, K=128, N2=32, H=32, L_chunk=32,
).cuda()
l_in = model.calc_input_region_size(2048)
x = torch.randn(B, 4, l_in, device="cuda")

# -- Eager benchmark --------------------------------------------------------
for _ in range(5):
    model(x)
torch.cuda.synchronize()

t0 = time.time()
for _ in range(10):
    logits, ep, disp = model(x)
torch.cuda.synchronize()
eager_fwd = (time.time() - t0) / 10

logits, ep, disp = model(x)
loss = logits.sum()
torch.cuda.synchronize()
t0 = time.time()
loss.backward()
torch.cuda.synchronize()
eager_bwd = time.time() - t0

print(f"Eager  forward:    {eager_fwd*1000:.0f}ms")
print(f"Eager  backward:   {eager_bwd*1000:.0f}ms")
print(f"Eager  total:      {(eager_fwd+eager_bwd)*1000:.0f}ms")
print(f"Eager  per-sample: {eager_fwd*1000/B:.1f}ms")

# -- Compiled benchmark -----------------------------------------------------
if DO_COMPILE:
    compiled = torch.compile(model)
    for _ in range(5):
        compiled(x)
    torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(10):
        logits, ep, disp = compiled(x)
    torch.cuda.synchronize()
    comp_fwd = (time.time() - t0) / 10

    logits, ep, disp = compiled(x)
    loss = logits.sum()
    torch.cuda.synchronize()
    t0 = time.time()
    loss.backward()
    torch.cuda.synchronize()
    comp_bwd = time.time() - t0

    print()
    print(f"Compiled forward:    {comp_fwd*1000:.0f}ms")
    print(f"Compiled backward:   {comp_bwd*1000:.0f}ms")
    print(f"Compiled total:      {(comp_fwd+comp_bwd)*1000:.0f}ms")
    print(f"Compiled per-sample: {comp_fwd*1000/B:.1f}ms")
    print()
    print(f"Speedup forward: {eager_fwd/comp_fwd:.2f}x")
    print(f"Speedup total:   {(eager_fwd+eager_bwd)/(comp_fwd+comp_bwd):.2f}x")
else:
    print("\n(torch.compile skipped — no gcc)")

# -- Profiler ---------------------------------------------------------------
out_dir = "/efs/analytics/nathanboley/background_model/cut_site_runs/gpu_profile"
os.makedirs(out_dir, exist_ok=True)

# Reset for clean profile
model.zero_grad(set_to_none=True)
torch.cuda.reset_peak_memory_stats()

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
) as prof:
    for _ in range(3):
        logits, ep, disp = model(x)
        loss = logits.sum()
        loss.backward()
        model.zero_grad(set_to_none=True)
torch.cuda.synchronize()

print()
print("=== Top CUDA ops ===")
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
print()
print("=== Top CPU ops ===")
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=15))

prof.export_chrome_trace(os.path.join(out_dir, "trace.json"))
print(f"\nChrome trace: {out_dir}/trace.json")
print(f"Peak VRAM: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
