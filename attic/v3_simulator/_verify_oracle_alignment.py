"""Independent verification of scripts/sim_oracle.py coordinate alignment.

Checks, in order:
  A. ground truth regime assumptions (log_jitter == 0, NB vs multinomial sampler)
  B. dataset val crop == store y_full[:, crop_start:crop_stop]  (byte-exact)
  C. shift-correlation of oracle propensity vs observed counts over the FULL
     l_target extent -- the peak MUST be at shift 0
  D. shift-correlation after the center-crop, on the frame the loss actually sees
  E. uniform NLL through the frozen loss path vs the analytic value
"""
import json
import os
import sys

import numpy as np
import torch
import zarr

sys.path.insert(0, "/home/nathanboley/src/fragmentomics_tools")

from scripts.sim_oracle import (  # noqa: E402
    compute_oracle_propensity_for_tile, TRACK_INDEX, N_TRACKS, FASTA,
)
from scripts.sim_fragments import GCBias2D  # noqa: E402
from background_model.config import PlumbingConfig  # noqa: E402
from background_model.dataset import BackgroundTileDataset  # noqa: E402
from background_model_core import MaskedMultinomialNLLLoss  # noqa: E402

SIM = "/efs/analytics/nathanboley/background_model/simulation_v3/A"
STORE = "/efs/analytics/nathanboley/background_model/simulation_v3/stores/sim_store_v3_A.zarr"

gt = np.load(os.path.join(SIM, "ground_truth.npz"), allow_pickle=True)
gtj = json.load(open(os.path.join(SIM, "ground_truth.json")))

print("=== A. ground-truth regime assumptions ===")
print("gt keys:", list(gt.keys()))
print("has hexamer_r (NB dispersion):", "hexamer_r" in gt)
print("json nb_dispersion field:", gtj.get("nb_dispersion", "<ABSENT>"))
lj = gt["log_jitter"]
print(f"log_jitter: shape={lj.shape} max|.|={np.abs(lj).max():.3e}  all-zero={np.all(lj==0)}")

tc = gt["target_counts"]
d0 = np.load(os.path.join(SIM, "sample_000.npz"))
obs_per_region = np.bincount(d0["region_idx"], minlength=tc.shape[1])
exact = np.array_equal(obs_per_region, tc[0].astype(np.int64))
print(f"sample_000 fragments-per-region == target_counts exactly: {exact}")
if not exact:
    rel = (obs_per_region - tc[0]) / np.maximum(tc[0], 1)
    print(f"  rel dev: mean={rel.mean():+.4f} sd={rel.std():.4f}  "
          f"(multinomial => 0/0; NB => nonzero sd)")

print()
print("=== B. dataset val crop vs store reconstruction ===")
root = zarr.open_group(STORE, mode="r")
cfg = PlumbingConfig.from_json(root.attrs["config_json"])
tile_size, l_target, jitter = cfg.tile_size, cfg.l_target, cfg.jitter
crop_start = (l_target - tile_size) // 2
crop_stop = crop_start + tile_size
print(f"tile_size={tile_size} l_target={l_target} jitter={jitter} "
      f"crop=[{crop_start},{crop_stop})")

ds = BackgroundTileDataset(store_path=STORE, model_input_size=tile_size,
                           split="val", sample_role="train", min_N=0,
                           train_mode=False, seed=1337)
print("len(ds) =", len(ds))

indptr = np.asarray(root["counts/indptr"][:])
c_pos = np.asarray(root["counts/pos"][:])
c_trk = np.asarray(root["counts/track"][:])
c_dat = np.asarray(root["counts/data"][:])
n_tiles = root["tiles/split"].shape[0]


def y_full_for(s, t):
    u = s * n_tiles + t
    lo, hi = int(indptr[u]), int(indptr[u + 1])
    y = np.zeros((N_TRACKS, l_target), dtype=np.float64)
    if hi > lo:
        np.add.at(y, (c_trk[lo:hi], c_pos[lo:hi]), c_dat[lo:hi].astype(np.float64))
    return y


rng = np.random.default_rng(0)
probe = rng.choice(len(ds), size=8, replace=False)
ok = True
for di in probe:
    s, t = ds.index[di]
    _, y, m = ds[int(di)]
    yf = y_full_for(s, t)
    same = np.array_equal(np.asarray(y, dtype=np.float64), yf[:, crop_start:crop_stop])
    ok &= same and bool(np.asarray(m).all())
print("ds[i].y == y_full[:, crop] for all probes:", ok)

print()
print("=== C/D. shift-correlation: oracle propensity vs observed counts ===")
contigs = root["tiles/contig"][:]
starts = root["tiles/start"][:]
stops = root["tiles/stop"][:]
splits = root["tiles/split"][:]
roles = root["samples/role"][:]
train_s = np.nonzero(roles == 0)[0]
val_t = np.nonzero(splits == 1)[0]

import pysam  # noqa: E402
fa = pysam.FastaFile(FASTA)
gcbias = GCBias2D(gt["bias_lengths"], gt["bias_gc_percents"], gt["bias_grid"])
w6 = gt["w6"]
len_vals = gt["len_vals"]
lpps = gt["len_p_per_sample"]

SHIFTS = np.arange(-260, 261, 1)
probe_tiles = val_t[rng.choice(len(val_t), size=6, replace=False)]
track_probe = [0, 3, 5, 9]  # (+,short,first) (+,long,first) (+,long,mid) (-,short,mid)

full_curves = []
crop_curves = []
for t in probe_tiles:
    prop = compute_oracle_propensity_for_tile(
        str(contigs[t]), int(starts[t]), int(stops[t]), l_target, w6, gcbias,
        len_vals, lpps, train_s, fa,
    )  # (16, C, l_target)
    prop_mean = prop.mean(axis=0)
    obs = np.zeros((N_TRACKS, l_target))
    for s in train_s:
        obs += y_full_for(s, t)

    for trk in track_probe:
        p = prop_mean[trk]
        o = obs[trk]
        # full-extent shift scan (valid interior only, to avoid edge effects)
        pad = 300
        oc = o[pad:l_target - pad]
        cur = []
        for sh in SHIFTS:
            pc = p[pad + sh:l_target - pad + sh]
            cur.append(np.corrcoef(oc, pc)[0, 1])
        full_curves.append(np.array(cur))

        # crop-frame scan: what the loss actually compares
        pcrop = prop_mean[trk, crop_start:crop_stop]
        ocrop = o[crop_start:crop_stop]
        pad2 = 280
        oc2 = ocrop[pad2:tile_size - pad2]
        cur2 = []
        for sh in SHIFTS:
            lo = pad2 + sh
            hi = tile_size - pad2 + sh
            if lo < 0 or hi > tile_size:
                cur2.append(np.nan)
                continue
            cur2.append(np.corrcoef(oc2, pcrop[lo:hi])[0, 1])
        crop_curves.append(np.array(cur2))

fa.close()
F = np.nanmean(np.stack(full_curves), axis=0)
C = np.nanmean(np.stack(crop_curves), axis=0)
i0 = int(np.where(SHIFTS == 0)[0][0])
print(f"FULL extent : argmax shift = {SHIFTS[np.nanargmax(F)]:+d}  "
      f"r(0)={F[i0]:.4f}  r(max)={np.nanmax(F):.4f}")
print(f"CROP frame  : argmax shift = {SHIFTS[np.nanargmax(C)]:+d}  "
      f"r(0)={C[i0]:.4f}  r(max)={np.nanmax(C):.4f}")
print("  neighbourhood r (full):",
      {int(s): round(float(F[i0 + s]), 4) for s in (-128, -2, -1, 0, 1, 2, 128)})

print()
print("=== E. uniform NLL: frozen-loss path vs analytic ===")
loss_fn = MaskedMultinomialNLLLoss()
tot, n = 0.0, 0
analytic = []
for di in probe:
    _, y, m = ds[int(di)]
    lg = torch.zeros(1, N_TRACKS, tile_size)
    v = loss_fn(lg, y.unsqueeze(0), m.unsqueeze(0)).item()
    tot += v
    n += 1
    nz = (y.sum(dim=-1) > 0).float()
    analytic.append(float(nz.mean()) * np.log(int(m.sum())))
print(f"loss-path uniform mean over {n} probes: {tot/n:.6f}")
print(f"analytic  log(L_valid)*frac_nonzero_tracks: {np.mean(analytic):.6f}")
print(f"log(2048) = {np.log(2048):.6f}")
