#!/usr/bin/env python
"""Profile the REAL cut-site training step on GPU, to locate optimisation targets.

Profiles what production actually runs: real batches from the store, the hex
baseline added to the logits, the fragment multinomial NLL, backward, and the
optimiser step.

This previously benchmarked `loss = logits.sum()` on synthetic randn at
P=2048 with no hex tables and no optimiser. That is a much simpler backward
graph than the real multinomial NLL and a different tile size from production,
so its kernel breakdown did not describe the step we care about. A profile of
the wrong configuration is worse than no profile -- it gets optimised against.

Example
-------
    python scripts/gpu_benchmark.py \
        --store /efs/.../quiet_tile1536_v2.zarr --tile-size 1536 \
        --hex-tables /efs/.../stage1_hex_tables_v3_offset3.npz
"""
import argparse, os, sys, time

import torch
import zarr
from torch.profiler import ProfilerActivity, profile
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
torch.set_float32_matmul_precision("high")

from background_model.constants import L_MAX, L_MIN
from background_model.cut_site_model.model import CutSiteFragmentModel
from background_model.cut_site_model.store import CutSiteStoreConfig
from background_model.cut_site_model.dataset import (
    CutSiteFragmentDataset, cut_site_collate_fn,
)

_DTYPE = {"32": torch.float32, "bf16": torch.bfloat16, "16": torch.float16}


def to_device(batch, dev):
    """Collate emits tensors AND lists of variable-length tensors (frag_*)."""
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(dev, non_blocking=True)
        elif isinstance(v, list):
            out[k] = [t.to(dev, non_blocking=True) for t in v]
        else:
            out[k] = v
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--store", required=True)
    p.add_argument("--hex-tables", default=None,
                   help="include the Stage-2 log_p0 path, as production does")
    p.add_argument("--batch-size", type=int, default=32)
    # No default (owner decision 176).  The dataset reads the tile size from
    # the store, so --tile-size is required AND must match the store.
    p.add_argument("--tile-size", type=int, required=True,
                   help="required; must equal the store's tile_size")
    p.add_argument("--L-min", type=int, default=L_MIN)
    p.add_argument("--L-max", type=int, default=L_MAX)
    p.add_argument("--min-N", type=int, default=10)
    p.add_argument("--precision", default="bf16", choices=sorted(_DTYPE))
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--timed", type=int, default=5)
    p.add_argument("--profile-steps", type=int, default=3)
    p.add_argument("--rows", type=int, default=25)
    p.add_argument("--trace-out", default=None, help="chrome trace .json path")
    args = p.parse_args()

    store_cfg = CutSiteStoreConfig.from_json(
        zarr.open_group(args.store, mode="r").attrs["config_json"])
    if args.tile_size != store_cfg.tile_size:
        p.error(f"--tile-size {args.tile_size} != store tile_size "
                f"{store_cfg.tile_size}")

    if not torch.cuda.is_available():
        sys.exit("needs a GPU")
    dev = torch.device("cuda")
    amp_dtype = _DTYPE[args.precision]
    use_amp = args.precision != "32"

    model = CutSiteFragmentModel(
        L_min=args.L_min, L_max=args.L_max, tile_size=args.tile_size,
        N1=128, K=128, N2=32, H=32, L_chunk=32,
        dispersion_window_size=256, mult_weight=1.0, nb_w0=0.0,
        hex_tables_path=args.hex_tables, freeze_hex=args.hex_tables is not None,
    ).to(dev)
    model.train()
    # _step calls self.log, which needs a Trainer. This is a profiling harness,
    # so silence it rather than stand up a Trainer we would then be measuring.
    model.log = lambda *a, **k: None
    model.log_dict = lambda *a, **k: None

    ds = CutSiteFragmentDataset(
        store_path=args.store, model_input_size=model.model_input_size,
        model_L_min=args.L_min, model_L_max=args.L_max,
        split="train", min_N=args.min_N, train_mode=True, preload=True,
    )
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=4,
                    collate_fn=cut_site_collate_fn, persistent_workers=True,
                    drop_last=True)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    print(f"device={torch.cuda.get_device_name(0)}  B={args.batch_size}  "
          f"P={args.tile_size}  precision={args.precision}  "
          f"hex={'yes' if args.hex_tables else 'no'}  regions={len(ds)}")

    it = iter(dl)

    def one_step(batch, do_opt=True):
        with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            loss = model._step(batch, "train")
        loss.backward()
        if do_opt:
            opt.step()
        opt.zero_grad(set_to_none=True)
        return loss

    for _ in range(args.warmup):
        one_step(to_device(next(it), dev))
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    # -- forward / backward split ------------------------------------------
    fwd_t = bwd_t = opt_t = 0.0
    for _ in range(args.timed):
        batch = to_device(next(it), dev)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            loss = model._step(batch, "train")
        torch.cuda.synchronize(); t1 = time.perf_counter()
        loss.backward()
        torch.cuda.synchronize(); t2 = time.perf_counter()
        opt.step(); opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize(); t3 = time.perf_counter()
        fwd_t += t1 - t0; bwd_t += t2 - t1; opt_t += t3 - t2

    n = args.timed
    fwd, bwd, optms = fwd_t / n * 1e3, bwd_t / n * 1e3, opt_t / n * 1e3
    tot = fwd + bwd + optms
    print(f"\n--- per step (mean of {n}) ---")
    print(f"  forward   {fwd:8.1f} ms  ({100*fwd/tot:4.1f}%)")
    print(f"  backward  {bwd:8.1f} ms  ({100*bwd/tot:4.1f}%)   {bwd/max(fwd,1e-9):.1f}x forward")
    print(f"  optimiser {optms:8.1f} ms  ({100*optms/tot:4.1f}%)")
    print(f"  TOTAL     {tot:8.1f} ms   -> {args.batch_size/(tot/1e3):.1f} examples/s")
    print(f"  peak VRAM {torch.cuda.max_memory_allocated()/2**30:.2f} GiB alloc, "
          f"{torch.cuda.max_memory_reserved()/2**30:.2f} GiB reserved")

    # -- kernel breakdown ---------------------------------------------------
    torch.cuda.reset_peak_memory_stats()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=True) as prof:
        for _ in range(args.profile_steps):
            one_step(to_device(next(it), dev))
    torch.cuda.synchronize()

    print(f"\n=== top ops by SELF CUDA time (B={args.batch_size}, "
          f"{args.profile_steps} steps) ===")
    # self time, not total: total double-counts parents and makes every
    # wrapper look like a hotspot.
    print(prof.key_averages().table(sort_by="self_cuda_time_total",
                                    row_limit=args.rows))

    if args.trace_out:
        os.makedirs(os.path.dirname(args.trace_out), exist_ok=True)
        prof.export_chrome_trace(args.trace_out)
        print(f"\ntrace -> {args.trace_out}")


if __name__ == "__main__":
    main()
