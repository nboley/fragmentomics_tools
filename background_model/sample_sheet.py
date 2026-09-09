"""Sample-sheet builder for the background model preprocessing pipeline.

Reads a DataManifest TSV + clinical CSV, joins on library to pull the single
clinical column (ENDO_CATEGORY) that crosses the boundary, filters to the
quiescent pool, and emits a 4-column TSV suitable for PlumbingConfig.sample_sheet.

Usage:
    python -m background_model.sample_sheet \\
        --manifest manifests/ibd.data_manifest.tsv \\
        --clinical pooled_clinical.csv \\
        --output sample_sheet.tsv \\
        [--quiescent-categories Asymptomatic,Remission] \\
        [--sync]
"""

import argparse
import json
import sys

import pandas as pd


DEFAULT_QUIESCENT_CATEGORIES = {"Asymptomatic", "Remission"}


def library_from_key(key: str) -> str:
    """Derive the library name from a DataManifest key.

    Real IBD manifest keys look like ``NC-<seqrun>/<lib>.hg38.fragments.h5``;
    the library is the file basename with the fragments-h5 suffix stripped.
    """
    basename = str(key).rsplit("/", 1)[-1]
    suffix = ".hg38.fragments.h5"
    if basename.endswith(suffix):
        return basename[: -len(suffix)]
    return basename.split(".", 1)[0]


def parse_manifest(manifest_path: str) -> pd.DataFrame:
    """Parse a DataManifest TSV, extracting library/seqrun/h5_path from the notes JSON."""
    # DataManifest v3 files carry '#'-prefixed metadata header lines above the
    # real column header; comment='#' skips them (harmless for plain TSVs).
    df = pd.read_csv(manifest_path, sep="\t", comment="#")

    records = []
    for _, row in df.iterrows():
        notes = row.get("notes", "{}")
        if isinstance(notes, str):
            try:
                notes_dict = json.loads(notes)
            except json.JSONDecodeError:
                notes_dict = {}
        else:
            notes_dict = {}

        # Real manifest notes carry no 'library'; derive it from the key basename.
        library = notes_dict.get("library") or row.get("library") or library_from_key(
            row.get("key", "")
        )
        seqrun = notes_dict.get("seqrun", "")
        h5_path = row.get("path", row.get("key", ""))

        records.append({
            "library": str(library),
            "seqrun": str(seqrun),
            "h5_path": str(h5_path),
        })

    return pd.DataFrame(records)


def build_sample_sheet(
    manifest_path: str,
    clinical_path: str,
    quiescent_categories: set = None,
) -> pd.DataFrame:
    """Build the sample sheet by joining manifest with clinical data.

    Returns a DataFrame with columns: library, h5_path, seqrun, endo_category.
    """
    if quiescent_categories is None:
        quiescent_categories = DEFAULT_QUIESCENT_CATEGORIES

    manifest_df = parse_manifest(manifest_path)

    clinical_df = pd.read_csv(clinical_path)
    # Normalize column names: look for ENDO_CATEGORY or endo_category
    col_map = {c.lower(): c for c in clinical_df.columns}
    endo_col = col_map.get("endo_category", col_map.get("endocategory"))
    lib_col = col_map.get(
        "library",
        col_map.get("library_name", col_map.get("sample_id", col_map.get("sampleid"))),
    )

    if endo_col is None:
        raise ValueError(
            f"Clinical CSV must have an ENDO_CATEGORY column; "
            f"found: {list(clinical_df.columns)}"
        )
    if lib_col is None:
        raise ValueError(
            f"Clinical CSV must have a library/sample_id column; "
            f"found: {list(clinical_df.columns)}"
        )

    clinical_df = clinical_df[[lib_col, endo_col]].rename(
        columns={lib_col: "library", endo_col: "endo_category"}
    )
    clinical_df["library"] = clinical_df["library"].astype(str)

    merged = manifest_df.merge(clinical_df, on="library", how="inner")

    # Filter to quiescent pool
    sheet = merged[merged["endo_category"].isin(quiescent_categories)].copy()
    sheet = sheet[["library", "h5_path", "seqrun", "endo_category"]]
    sheet = sheet.drop_duplicates(subset=["library"]).reset_index(drop=True)

    return sheet


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Build a sample sheet for background model preprocessing."
    )
    parser.add_argument("--manifest", required=True, help="DataManifest TSV path")
    parser.add_argument("--clinical", required=True, help="Pooled clinical CSV path")
    parser.add_argument("--output", required=True, help="Output TSV path")
    parser.add_argument(
        "--quiescent-categories",
        default="Asymptomatic,Remission",
        help="Comma-separated ENDO_CATEGORY values for the quiescent pool",
    )
    parser.add_argument(
        "--sync",
        action="store_true",
        help="Sync h5 files via DataManifest before writing (serial)",
    )
    args = parser.parse_args(argv)

    quiescent = set(args.quiescent_categories.split(","))
    sheet = build_sample_sheet(args.manifest, args.clinical, quiescent)

    if args.sync:
        try:
            from datamanifest import DataManifest
        except ImportError:
            print("ERROR: --sync requires the datamanifest package", file=sys.stderr)
            sys.exit(1)
        dm = DataManifest(args.manifest)
        resolved_paths = []
        for _, row in sheet.iterrows():
            local = dm.sync_and_get(row["h5_path"])
            resolved_paths.append(str(local.path))
        sheet["h5_path"] = resolved_paths

    sheet.to_csv(args.output, sep="\t", index=False)
    print(f"Wrote {len(sheet)} samples to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
