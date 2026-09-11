"""Shared helpers for the AWS Batch background-model preprocess drivers.

Runs INSIDE the Batch container (biomarker image), which already has
fragmentomics_tools + fragments_h5 installed. Only the `background_model`
package is shipped via the repo clone (added to PYTHONPATH by the entrypoint),
so this module and the drivers next to it import `background_model.*` and
construct a PlumbingConfig from the locally-downloaded input files.

The store the config identifies is the USER-LOCKED default:
  - region set:   training_tiles.bed  (DEFAULT, not nogap; md5 8fe67ad4...)
  - blacklist:    ENCODE hg38-blacklist.v2  (confirmed store blacklist_bed)
  - sample pool:  ibd_quiescent.tsv (213-row Remission pool; md5 8cc70a7c...)
  - draw:         seed 1337, 40 train / 10 heldout  (materialized in draw50.tsv)
All other PlumbingConfig fields are library defaults. config_hash8 = b67d7c95.
"""

import os

import pandas as pd

from background_model.config import PlumbingConfig

# Canonical filenames within the staged inputs/ prefix.
INPUT_FILES = {
    "sample_sheet": "ibd_quiescent.tsv",
    "region_bed": "training_tiles.bed",
    "blacklist": "blacklist_encode_v2.bed",
    "fasta": "hg38.fa",
    "draw": "draw50.tsv",
}


def build_config(inputs_dir: str) -> PlumbingConfig:
    """Construct the store's PlumbingConfig from locally-staged input paths.

    Construction reads no file contents; config_hash8() lazily md5s the
    sample_sheet, blacklist, region BED and the FASTA .fai (NOT the 3 GB FASTA).
    Phase A therefore needs only the small files + the .fai present; the FASTA
    itself is required only by Phase B (sequence write).
    """
    return PlumbingConfig(
        sample_sheet=os.path.join(inputs_dir, INPUT_FILES["sample_sheet"]),
        region_beds={"train_pool": os.path.join(inputs_dir, INPUT_FILES["region_bed"])},
        blacklist_bed=os.path.join(inputs_dir, INPUT_FILES["blacklist"]),
        fasta=os.path.join(inputs_dir, INPUT_FILES["fasta"]),
    )


def load_drawn_sheet(inputs_dir: str) -> pd.DataFrame:
    """Load the frozen 50-sample drawn sheet (draw50.tsv).

    This IS the drawn sheet (roles pre-assigned: 0=train, 1=heldout) in the
    canonical sample-axis order for the store. The array task index maps
    directly to a row here; Phase B iterates it in order. h5_path is the
    manifest key (metadata only in Phase B); Phase A overrides it with the
    locally-downloaded h5 path.
    """
    return pd.read_csv(os.path.join(inputs_dir, INPUT_FILES["draw"]), sep="\t")
