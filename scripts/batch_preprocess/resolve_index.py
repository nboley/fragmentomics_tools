"""Print '<library> <config_hash8>' for an array index.

Used by task_entrypoint.sh to (a) resume-check the shard in S3 before
downloading the h5, and (b) name the shard object.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bg_batch_common import build_config, load_drawn_sheet  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs-dir", required=True)
    ap.add_argument("--index", type=int, required=True)
    a = ap.parse_args()

    cfg = build_config(a.inputs_dir)
    sheet = load_drawn_sheet(a.inputs_dir)
    if a.index < 0 or a.index >= len(sheet):
        raise SystemExit(f"index {a.index} out of range 0..{len(sheet) - 1}")
    library = str(sheet.iloc[a.index]["library"])
    print(f"{library} {cfg.config_hash8()}")


if __name__ == "__main__":
    main()
