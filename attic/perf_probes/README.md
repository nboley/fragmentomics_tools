# attic/perf_probes — REFERENCE ONLY, NOT LIVE CODE

One-off performance probes, kept for the evidence they produced rather than to be
re-run. **Do not import from here.** Their paths are not maintained.

| File | What it measured |
|---|---|
| `_bench_h5_handle_reuse.py` | Timing plus bit-identity evidence for the `load_fragment_arrays` h5 handle-reuse change. Run once against the pristine `dataframe.py` and once against the change, then `--diff a.json b.json`. The comparison is over per-region, per-field sha256 digests of the raw array bytes plus dtype, so a dtype change, a reordering or one flipped mantissa bit all show up — which a spot-check of a few numbers would not. |

Moved here from `scripts/` on 2026-10-10 (owner) while it was still untracked.
Untracked and stale is the worst combination: git holds no copy to recover, and
nothing updates the paths inside when the tree moves.
