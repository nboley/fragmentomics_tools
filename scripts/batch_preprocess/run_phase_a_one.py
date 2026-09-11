"""Phase A driver for ONE sample (one Batch array task).

Maps the array index -> library via draw50.tsv, points the worker at the
locally-downloaded h5, runs `_worker_process_sample` over all tiles, and writes
the shard (<library>.npz) plus a timing sidecar (<library>.timing.json).

Timing is measured around the worker call only (wall via perf_counter; CPU/RSS
via resource.getrusage), independent of the surrounding S3 I/O.
"""

import argparse
import json
import os
import resource
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bg_batch_common import build_config, load_drawn_sheet  # noqa: E402

from background_model.preprocess import _worker_process_sample, build_tiles  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs-dir", required=True)
    ap.add_argument("--index", type=int, required=True)
    ap.add_argument("--shard-dir", required=True)
    ap.add_argument("--ref", default="hg38")
    a = ap.parse_args()

    os.makedirs(a.shard_dir, exist_ok=True)
    cfg = build_config(a.inputs_dir)
    hash8 = cfg.config_hash8()
    sheet = load_drawn_sheet(a.inputs_dir)
    if a.index < 0 or a.index >= len(sheet):
        raise SystemExit(f"index {a.index} out of range 0..{len(sheet) - 1}")

    row = sheet.iloc[a.index].to_dict()
    library = str(row["library"])
    # Point the worker at the locally-downloaded h5 (overrides the manifest key).
    row["h5_path"] = os.path.join(a.inputs_dir, "h5", f"{library}.h5")

    tiles = build_tiles(cfg.region_beds, cfg.tile_size, cfg.jitter, cfg.rf_budget, a.ref)

    t0 = time.perf_counter()
    lib, ok = _worker_process_sample(row, tiles, cfg, a.shard_dir, a.ref)
    wall = time.perf_counter() - t0
    ru = resource.getrusage(resource.RUSAGE_SELF)

    shard_path = os.path.join(a.shard_dir, f"{library}.npz")
    timing = {
        "library": library,
        "index": a.index,
        "config_hash8": hash8,
        "ok": bool(ok),
        "n_tiles": len(tiles),
        "wall_sec": round(wall, 2),
        "user_cpu_sec": round(ru.ru_utime, 2),
        "sys_cpu_sec": round(ru.ru_stime, 2),
        "max_rss_mb": round(ru.ru_maxrss / 1024.0, 1),
        "shard_bytes": os.path.getsize(shard_path) if os.path.exists(shard_path) else 0,
    }
    with open(os.path.join(a.shard_dir, f"{library}.timing.json"), "w") as f:
        json.dump(timing, f, indent=2)
    print("TIMING " + json.dumps(timing))

    if not ok:
        raise SystemExit(f"worker failed for {library} (see {library}.error)")


if __name__ == "__main__":
    main()
