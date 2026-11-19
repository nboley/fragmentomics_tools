#!/usr/bin/env python
"""Write the two reference hex-table .npz files the scorer can load.

  uniform   all four hexamer tables AND p_L constant.  Every cell of the
            domain then gets ln|D| = ln(tile_size * n_L * 2).
  unif_pL   all four hexamer tables constant; p_L copied from --p-L-from.
            The domain probability collapses to p_L(L) / (tile_size * 2).

Same schema as count_tables.py's tables: start_fwd, stop_fwd, start_rev,
stop_rev (4096,) probabilities summing to 1, p_L (n_L,), k, L_min, L_max, and
the 3' registration entry.  The constant tables score identically under any 3'
convention, so the entry only satisfies the model's loader, which refuses an
.npz without it.

Usage:
    python scripts/cut_site_model/baselines/emit_reference_npz.py \\
        --p-L-from N0.npz --kind unif_pL --output unif_pL.npz
"""

import argparse
import os
import sys

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
sys.path.insert(0, _REPO)

from background_model.cut_site_model.model import hex_registration_record  # noqa: E402

TABLES = ("start_fwd", "stop_fwd", "start_rev", "stop_rev")


def reference_tables(source: dict, kind: str) -> dict:
    k, L_min, L_max = (int(source[f]) for f in ("k", "L_min", "L_max"))
    n_L = L_max - L_min + 1
    p_L = np.asarray(source["p_L"], dtype=np.float64)
    if p_L.shape != (n_L,):
        raise ValueError(f"p_L shape {p_L.shape} != ({n_L},)")
    if kind == "uniform":
        p_L = np.full(n_L, 1.0 / n_L)
    elif kind != "unif_pL":
        raise ValueError(f"kind must be uniform or unif_pL, not {kind!r}")
    out = {t: np.full(4 ** k, 1.0 / 4 ** k) for t in TABLES}
    out.update(p_L=p_L, k=np.int64(k), L_min=np.int64(L_min), L_max=np.int64(L_max),
               **hex_registration_record())
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--p-L-from", required=True,
                   help=".npz supplying p_L, k, L_min, L_max")
    p.add_argument("--kind", required=True, choices=["uniform", "unif_pL"])
    p.add_argument("--output", required=True)
    args = p.parse_args(argv)
    np.savez(args.output, **reference_tables(np.load(args.p_L_from), args.kind))


if __name__ == "__main__":
    main()
