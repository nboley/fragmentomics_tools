#!/usr/bin/env python
"""GPU cost benchmark for the cut-site fragment (p, L) logit sweep.

MEASUREMENT ONLY. This is a faithful *cost skeleton* of the proposed
"cut-site fragment model", not a trained or correct model. Synthetic random
data is deliberate: we are measuring wall-clock ms/step and peak VRAM of the
forward + backward, not learning anything.

The open question this answers: is the exact (position p, fragment length L)
sweep affordable as a TRAINING loss at batch 64 on an A10G, compared against
the 330-360 ms/batch cost of current production background models?

Architecture per region (see the design doc; reproduced here so the skeleton
is self-contained):
  L1  cut-site bank        : Conv1d(4, N1=128, k=6, padding='same') over one-hot seq (len P)
  L2a endpoint branch      : two SHARED taps -> read L1 at p and at p+L-1.
                             Implemented as per-position linear projections
                             (left_proj, right_proj) then, for each L, a shifted
                             slice/gather -> NO per-L convolution.
  L2b interior branch      : a SECOND bank Conv1d(4, N2=32, k=6, padding='same');
                             prefix-summed ONCE, so the span aggregate for every
                             (p, L) is S[p+L] - S[p]. Both the raw difference AND
                             the length-normalised mean (diff / L) are supplied.
  L3  head                 : concat(2a, 2b) -> optional hidden(H) -> single logit per (p, L)
  loss: multinomial NLL of ~54 observed fragments per region; log Z is a
        streaming/chunked logsumexp over blocks of L (running-max via
        torch.logsumexp per chunk + a final combine), with each chunk wrapped in
        gradient checkpointing so peak activation is P x L_chunk x C, not
        P x L x C. Then .backward().

Grid measured: H in {0, 32, 64} x L_chunk in {32, 64, all} x precision
{bf16-mixed, fp32} x mode {naive=1 sample/region, shared=16 samples/region}.
The shared mode is the point of the exercise: the (p, L) logit map is
sequence-only, so one region forward serves S samples' fragment sets; we report
cost per (sample, region) example both ways.

Run on GPU via /batch-run --gpu a10g. Sends all output to
/efs/analytics/nathanboley/background_model/bench/.
"""

import argparse
import contextlib
import json
import os
import statistics
import time
from datetime import datetime, timezone

import torch
import torch.nn as nn
import torch.utils.checkpoint

# ---- fixed geometry --------------------------------------------------------
P = 2048                 # region length (positions)
L_MIN, L_MAX = 25, 256   # fragment lengths, inclusive
N_LEN = L_MAX - L_MIN + 1  # 232
BATCH = 64               # regions per forward
N_FRAG = 54              # observed fragments per (sample, region)
SHARED_SAMPLES = 16      # samples served by one region forward in "shared" mode

# feature widths (documented so the measured VRAM can be attributed)
N1 = 128                 # L1 cut-site bank channels
N2 = 32                  # L2b interior bank channels
E = 16                   # endpoint projection width (per tap, summed)
C_INTERIOR = 2 * N2      # raw diff (N2) + length-normalised mean (N2) = 64
C_CONCAT = E + C_INTERIOR  # input width to the head

WARMUP = 5
TIMED = 20


class FragLogitModel(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.hidden = hidden
        self.l1 = nn.Conv1d(4, N1, kernel_size=6, padding="same")
        self.l2b = nn.Conv1d(4, N2, kernel_size=6, padding="same")
        self.left_proj = nn.Linear(N1, E)
        self.right_proj = nn.Linear(N1, E)
        if hidden == 0:
            self.head = nn.Linear(C_CONCAT, 1)
        else:
            self.head = nn.Sequential(
                nn.Linear(C_CONCAT, hidden),
                nn.ReLU(),
                nn.Linear(hidden, 1),
            )

    def features(self, seq):
        """seq: (B, 4, P) one-hot -> (left, right, prefixS)."""
        a = self.l1(seq).transpose(1, 2)          # (B, P, N1)
        left = self.left_proj(a)                  # (B, P, E)
        right = self.right_proj(a)                # (B, P, E)
        b = self.l2b(seq).transpose(1, 2)         # (B, P, N2)
        # prefix sum with a leading zero row so S[p+L] - S[p] is the span sum
        zeros = torch.zeros_like(b[:, :1, :])
        prefixS = torch.cat([zeros, torch.cumsum(b, dim=1)], dim=1)  # (B, P+1, N2)
        return left, right, prefixS

    def head_apply(self, concat):
        return self.head(concat).squeeze(-1)


def build_length_index(lengths, device):
    """For a block of lengths, precompute the gather indices and validity mask.

    Returns:
      right_idx : (P, nL) = p + L - 1, clamped to [0, P-1]
      pref_idx  : (P, nL) = p + L,     clamped to [0, P]
      valid     : (P, nL) bool, True where the fragment fits entirely in [0, P)
      lengths_f : (1, nL, 1) float, for length-normalisation
    """
    p = torch.arange(P, device=device).unsqueeze(1)      # (P, 1)
    L = lengths.to(device).unsqueeze(0)                  # (1, nL)
    right_idx = (p + L - 1)
    pref_idx = (p + L)
    valid = pref_idx <= P                                # p + L <= P
    right_idx = right_idx.clamp(max=P - 1)
    pref_idx = pref_idx.clamp(max=P)
    lengths_f = L.float().view(1, -1, 1)
    return right_idx, pref_idx, valid, lengths_f


def chunk_logsumexp(model, left, right, prefixS, right_idx, pref_idx, valid, lengths_f):
    """logsumexp over all (p, L) pairs in one length block. Returns (B,).

    This is the memory-heavy step: it materialises (B, P, nL, C_CONCAT). Wrapped
    in gradient checkpointing by the caller so the activation is freed after the
    forward and recomputed in backward.
    """
    B = left.shape[0]
    nL = right_idx.shape[1]
    # endpoint branch (2a): left at p, right at p+L-1 (shared taps -> gather)
    r_gather = right[:, right_idx.reshape(-1), :].view(B, P, nL, E)   # (B,P,nL,E)
    endpoint = left.unsqueeze(2) + r_gather                          # (B,P,nL,E)
    # interior branch (2b): span aggregate = S[p+L] - S[p]
    s_hi = prefixS[:, pref_idx.reshape(-1), :].view(B, P, nL, N2)     # (B,P,nL,N2)
    s_lo = prefixS[:, :P, :].unsqueeze(2)                            # (B,P,1,N2)
    diff = s_hi - s_lo                                              # (B,P,nL,N2)
    mean = diff / lengths_f.unsqueeze(0)                            # length-normalised
    interior = torch.cat([diff, mean], dim=-1)                     # (B,P,nL,C_INTERIOR)
    concat = torch.cat([endpoint, interior], dim=-1)               # (B,P,nL,C_CONCAT)
    logits = model.head_apply(concat)                              # (B,P,nL)
    logits = logits.masked_fill(~valid.unsqueeze(0), float("-inf"))
    return torch.logsumexp(logits.reshape(B, -1), dim=1)          # (B,)


def compute_logZ(model, left, right, prefixS, blocks, use_ckpt):
    """Streaming logsumexp over length blocks -> (B,). Combine is a final
    logsumexp over the per-block results (logsumexp is associative)."""
    per_block = []
    for right_idx, pref_idx, valid, lengths_f in blocks:
        if use_ckpt:
            lse = torch.utils.checkpoint.checkpoint(
                chunk_logsumexp, model, left, right, prefixS,
                right_idx, pref_idx, valid, lengths_f, use_reentrant=False,
            )
        else:
            lse = chunk_logsumexp(
                model, left, right, prefixS,
                right_idx, pref_idx, valid, lengths_f,
            )
        per_block.append(lse)
    return torch.logsumexp(torch.stack(per_block, dim=1), dim=1)   # (B,)


def compute_obs_logits(model, left, right, prefixS, p_idx, L_idx):
    """Logits at observed fragment pairs. p_idx, L_idx: (n,) -> (B, n).

    Cheap: only n pairs, no full grid. Used for the NLL numerator.
    """
    r_i = (p_idx + L_idx - 1).clamp(max=P - 1)
    endpoint = left[:, p_idx, :] + right[:, r_i, :]                # (B, n, E)
    s_hi = prefixS[:, (p_idx + L_idx).clamp(max=P), :]             # (B, n, N2)
    s_lo = prefixS[:, p_idx, :]                                    # (B, n, N2)
    diff = s_hi - s_lo
    mean = diff / L_idx.float().view(1, -1, 1)
    concat = torch.cat([endpoint, diff, mean], dim=-1)            # (B, n, C_CONCAT)
    return model.head_apply(concat)                               # (B, n)


def make_observed(device, n_sets):
    """Sample n_sets independent fragment sets, each N_FRAG valid (p, L) pairs.
    Returns p_idx, L_idx each (n_sets, N_FRAG)."""
    L = torch.randint(L_MIN, L_MAX + 1, (n_sets, N_FRAG), device=device)
    # valid start: 0 .. P - L  ->  p = floor(rand * (P - L + 1))
    span = (P - L + 1).float()
    p = (torch.rand(n_sets, N_FRAG, device=device) * span).long()
    return p, L


def run_cell(hidden, l_chunk, precision, mode, device):
    """Measure one grid cell. Returns a dict of results (or OOM marker)."""
    n_samples = SHARED_SAMPLES if mode == "shared" else 1

    # length blocks for the streaming logsumexp
    all_lengths = torch.arange(L_MIN, L_MAX + 1)
    if l_chunk >= N_LEN:
        chunk_edges = [all_lengths]
        use_ckpt = False  # single block: nothing to stream, checkpoint pointless
        l_chunk_report = N_LEN
    else:
        chunk_edges = [all_lengths[i:i + l_chunk] for i in range(0, N_LEN, l_chunk)]
        use_ckpt = True
        l_chunk_report = l_chunk
    blocks = [build_length_index(le, device) for le in chunk_edges]

    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if precision == "bf16"
        else contextlib.nullcontext()
    )

    try:
        model = FragLogitModel(hidden).to(device)
        opt_free = lambda: model.zero_grad(set_to_none=True)
        seq = torch.zeros(BATCH, 4, P, device=device)
        seq[torch.arange(BATCH)[:, None], torch.randint(0, 4, (BATCH, P), device=device),
            torch.arange(P)[None, :]] = 1.0

        # one fragment set per sample (shared across the region batch is fine for cost)
        obs = [make_observed(device, BATCH) for _ in range(n_samples)]

        def forward_loss():
            with autocast_ctx:
                left, right, prefixS = model.features(seq)
                logZ = compute_logZ(model, left, right, prefixS, blocks, use_ckpt)  # (B,)
                total = 0.0
                for p_idx0, L_idx0 in obs:
                    # same pair-set shape per sample; index by [:, k] not needed
                    ol = compute_obs_logits(model, left, right, prefixS,
                                            p_idx0[0], L_idx0[0])  # (B, N_FRAG)
                    total = total + (-ol.sum(dim=1) + N_FRAG * logZ).mean()
            return total / n_samples

        # warmup (full fwd+bwd)
        for _ in range(WARMUP):
            loss = forward_loss()
            loss.backward()
            opt_free()
        torch.cuda.synchronize()

        # forward-only timing (grad-enabled, matches training forward; no backward)
        fwd_times = []
        for _ in range(TIMED):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            loss = forward_loss()
            torch.cuda.synchronize()
            fwd_times.append((time.perf_counter() - t0) * 1e3)
            del loss

        # forward+backward timing + peak VRAM
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        fb_times = []
        for _ in range(TIMED):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            loss = forward_loss()
            loss.backward()
            torch.cuda.synchronize()
            fb_times.append((time.perf_counter() - t0) * 1e3)
            opt_free()
            del loss
        peak_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

        n_examples = BATCH * n_samples
        fwd_ms = statistics.median(fwd_times)
        fb_ms = statistics.median(fb_times)
        result = {
            "hidden": hidden,
            "l_chunk": l_chunk_report,
            "precision": precision,
            "mode": mode,
            "n_samples_per_region": n_samples,
            "n_examples": n_examples,
            "fwd_ms": round(fwd_ms, 2),
            "fwd_bwd_ms": round(fb_ms, 2),
            "fwd_bwd_ms_per_example": round(fb_ms / n_examples, 4),
            "peak_vram_mb": round(peak_mb, 1),
            "oom": False,
        }
    except torch.cuda.OutOfMemoryError as exc:
        result = {
            "hidden": hidden, "l_chunk": (N_LEN if l_chunk >= N_LEN else l_chunk),
            "precision": precision, "mode": mode,
            "n_samples_per_region": n_samples, "oom": True,
            "error": str(exc)[:200],
        }
    finally:
        try:
            del model, seq
        except NameError:
            pass
        torch.cuda.empty_cache()

    return result


def fmt_table(rows):
    hdr = ("H", "L_chunk", "prec", "mode", "examples",
           "fwd ms", "fwd+bwd ms", "ms/example", "peak VRAM MB")
    lines = ["| " + " | ".join(hdr) + " |",
             "|" + "|".join(["---"] * len(hdr)) + "|"]
    for r in rows:
        if r.get("oom"):
            cells = (str(r["hidden"]), str(r["l_chunk"]), r["precision"], r["mode"],
                     "-", "OOM", "OOM", "-", "OOM")
        else:
            cells = (str(r["hidden"]), str(r["l_chunk"]), r["precision"], r["mode"],
                     str(r["n_examples"]), f'{r["fwd_ms"]:.1f}', f'{r["fwd_bwd_ms"]:.1f}',
                     f'{r["fwd_bwd_ms_per_example"]:.3f}', f'{r["peak_vram_mb"]:.0f}')
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="/efs/analytics/nathanboley/background_model/bench")
    ap.add_argument("--commit", default=os.environ.get("BENCH_COMMIT", "unknown"))
    ap.add_argument("--dirty", default=os.environ.get("BENCH_DIRTY", "unknown"))
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA required; run via /batch-run --gpu a10g"
    device = torch.device("cuda")
    torch.manual_seed(0)
    os.makedirs(args.outdir, exist_ok=True)

    meta = {
        "commit": args.commit,
        "tree_dirty": args.dirty,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "torch_version": torch.__version__,
        "device_name": torch.cuda.get_device_name(0),
        "geometry": {"P": P, "L_MIN": L_MIN, "L_MAX": L_MAX, "N_LEN": N_LEN,
                     "BATCH": BATCH, "N_FRAG": N_FRAG, "SHARED_SAMPLES": SHARED_SAMPLES},
        "widths": {"N1": N1, "N2": N2, "E": E, "C_CONCAT": C_CONCAT},
        "n_pairs_per_region": int(sum(P - L + 1 for L in range(L_MIN, L_MAX + 1))),
        "warmup": WARMUP, "timed": TIMED,
    }
    print(json.dumps(meta, indent=2), flush=True)

    grid_H = [0, 32, 64]
    grid_chunk = [32, 64, N_LEN]
    grid_prec = ["bf16", "fp32"]
    grid_mode = ["naive", "shared"]

    rows = []
    for prec in grid_prec:
        for mode in grid_mode:
            for hidden in grid_H:
                for chunk in grid_chunk:
                    r = run_cell(hidden, chunk, prec, mode, device)
                    rows.append(r)
                    tag = "OOM" if r.get("oom") else (
                        f'fwd={r["fwd_ms"]:.1f} fb={r["fwd_bwd_ms"]:.1f} '
                        f'vram={r["peak_vram_mb"]:.0f}MB')
                    print(f'[cell] H={hidden} chunk={chunk} {prec} {mode}: {tag}',
                          flush=True)

    table = fmt_table(rows)
    print("\n" + table, flush=True)

    out = {"meta": meta, "rows": rows}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = os.path.join(args.outdir, f"bench_fragment_logit_sweep_{stamp}.json")
    md_path = os.path.join(args.outdir, f"bench_fragment_logit_sweep_{stamp}.md")
    with open(json_path, "w") as fh:
        json.dump(out, fh, indent=2)
    with open(md_path, "w") as fh:
        fh.write(f"# Fragment (p,L) logit sweep — GPU cost benchmark\n\n")
        fh.write(f"commit `{args.commit}` (tree dirty: {args.dirty}) — "
                 f"{meta['device_name']}, torch {meta['torch_version']}\n\n")
        fh.write(f"{meta['n_pairs_per_region']:,} (p,L) pairs per region; "
                 f"batch {BATCH}; {N_FRAG} observed frags/example.\n\n")
        fh.write(table + "\n")
    print(f"\nwrote {json_path}\nwrote {md_path}", flush=True)


if __name__ == "__main__":
    main()
