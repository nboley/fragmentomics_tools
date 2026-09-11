"""Phase B driver: assemble all 50 per-sample shards into the zarr store.

Assumes the shard dir has been populated (all <library>.npz for the draw) and
the FASTA + blacklist + region BED are present locally. Writes
bg_store_<hash8>.zarr + bg_store_<hash8>.config.json into the output dir.
Phase B is idempotent (rebuilds from shards under a .building/ dir, atomic
rename at the end).
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bg_batch_common import build_config, load_drawn_sheet  # noqa: E402

from background_model.preprocess import build_tiles, run_phase_b  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs-dir", required=True)
    ap.add_argument("--shard-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--ref", default="hg38")
    a = ap.parse_args()

    os.makedirs(a.output_dir, exist_ok=True)
    cfg = build_config(a.inputs_dir)
    sheet = load_drawn_sheet(a.inputs_dir)
    tiles = build_tiles(cfg.region_beds, cfg.tile_size, cfg.jitter, cfg.rf_budget, a.ref)

    # Fail fast with a clear message if any shard is missing.
    missing = [
        lib for lib in sheet["library"]
        if not os.path.exists(os.path.join(a.shard_dir, f"{lib}.npz"))
    ]
    if missing:
        raise SystemExit(f"Phase B missing {len(missing)} shard(s): {missing[:5]}...")

    store_path = run_phase_b(cfg, sheet, tiles, a.shard_dir, a.output_dir, a.ref)
    print(f"STORE_PATH {store_path}")
    print(f"STORE_NAME {cfg.store_name()}")
    print(f"CONFIG_HASH8 {cfg.config_hash8()}")


if __name__ == "__main__":
    main()
