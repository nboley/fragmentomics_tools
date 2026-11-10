#!/usr/bin/env python
"""Resolve IBD library names to on-disk fragment-h5 paths by globbing.

Why glob, not template
----------------------
Every real fragment-h5 in the data cache carries a per-file md5 prefix that is
NOT recorded in the project sheets, so the path CANNOT be reconstructed by
string templating from the library name.  It must be discovered by globbing the
cache for the file whose name ends in the library's stem.  (Confirmed empirically:
0 of the paths stored in ``draw50.tsv``/``ibd_quiescent.tsv`` resolve as-is.)

Two on-disk layouts coexist under the cache root
------------------------------------------------
1. ``ibd/frag_h5s/<seqrun>/<md5>-<library>.hg38.fragments.h5`` — keeps the full
   library name (``-Lib1``) and the ``.hg38`` infix.
2. flat at the cache root: ``<md5>-<n>-<stem>.fragments.h5`` — the ``-Lib<k>``
   suffix and the ``.hg38`` infix are dropped, and an extra ``-<n>-`` count
   field sits between the md5 and the stem.

When a library resolves under BOTH layouts we prefer (1), the ``ibd/frag_h5s``
match, and record which layout each row came from so the provenance of every
resolved path is inspectable.

Growth
------
Another process is actively staging more h5s into ``ibd/frag_h5s`` while this
runs, so the resolvable set GROWS over time.  This script bakes in no count and
is safe to re-run; re-run it to refresh the resolved sheet.

Ambiguity
---------
If a single library matches more than one file within a layout, that is a real
problem (duplicate stems, stale copies) and the script FAILS LOUDLY with
``AmbiguousMatch`` rather than silently taking the first match.

Output
------
A TSV with columns ``sample_name``, ``path`` (the two the counting script
requires) plus ``seqrun`` and ``layout`` for provenance, one row per resolvable
library.  Unresolved libraries are reported to the log and omitted.
"""
import argparse
import csv
import logging
import re
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_CACHE_ROOT = Path(
    "/efs/analytics/nathanboley/biomarker-projects/data_cache"
)
DEFAULT_INPUT_SHEET = "data/sample_sheets/ibd_quiescent.tsv"
DEFAULT_OUTPUT_SHEET = "data/sample_sheets/ibd_quiescent.resolved.tsv"


class AmbiguousMatch(Exception):
    """More than one file matched a single library within one layout."""


def flat_stem(library: str) -> str:
    """Return the library stem used by the flat cache-root layout.

    The flat layout drops a trailing ``-Lib<k>`` suffix (e.g. ``RD-50442-Lib1``
    -> ``RD-50442``).  Libraries without such a suffix pass through unchanged.
    """
    return re.sub(r"-Lib\d+$", "", library)


def _single_match(matches: list, layout: str, library: str) -> Path:
    """Return the sole match, or raise AmbiguousMatch when there is >1."""
    if len(matches) > 1:
        raise AmbiguousMatch(
            f"{layout}: library {library!r} matched {len(matches)} files: "
            + ", ".join(sorted(p.name for p in matches))
        )
    return matches[0] if matches else None


def find_ibd(cache_root: Path, seqrun: str, library: str):
    """Match ``ibd/frag_h5s/<seqrun>/<md5>-<library>.hg38.fragments.h5``.

    Returns the Path or None. Raises AmbiguousMatch on >1 match.
    """
    seqrun_dir = cache_root / "ibd" / "frag_h5s" / seqrun
    matches = list(seqrun_dir.glob(f"*-{library}.hg38.fragments.h5"))
    return _single_match(matches, "ibd/frag_h5s", library)


def find_flat(cache_root: Path, library: str):
    """Match the flat cache-root ``<md5>-<n>-<stem>.fragments.h5`` layout.

    Returns the Path or None. Raises AmbiguousMatch on >1 match.
    """
    stem = flat_stem(library)
    matches = list(cache_root.glob(f"*-{stem}.fragments.h5"))
    return _single_match(matches, "flat", library)


def resolve_library(cache_root: Path, seqrun: str, library: str):
    """Resolve one library to (path, layout).

    Prefers the ``ibd/frag_h5s`` layout over the flat cache root when a library
    resolves both ways.  Returns ``(None, None)`` when neither layout matches.
    Propagates AmbiguousMatch when a layout has >1 candidate.
    """
    ibd = find_ibd(cache_root, seqrun, library)
    if ibd is not None:
        return str(ibd), "ibd"
    flat = find_flat(cache_root, library)
    if flat is not None:
        return str(flat), "flat"
    return None, None


def resolve_sheet(input_sheet: str, cache_root: Path) -> tuple:
    """Resolve every row of *input_sheet*.

    Returns ``(resolved_rows, unresolved_libraries)`` where each resolved row is
    a dict with ``sample_name``, ``path``, ``seqrun`` and ``layout``.
    """
    with open(input_sheet, newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        rows = list(reader)
    if not rows:
        sys.exit(f"Input sheet {input_sheet} has no data rows")
    for col in ("library", "seqrun"):
        if col not in rows[0]:
            sys.exit(f"Input sheet must have a {col!r} column")

    resolved, unresolved = [], []
    for row in rows:
        library, seqrun = row["library"], row["seqrun"]
        path, layout = resolve_library(cache_root, seqrun, library)
        if path is None:
            unresolved.append(library)
            continue
        resolved.append({
            "sample_name": library,
            "path": path,
            "seqrun": seqrun,
            "layout": layout,
        })
    return resolved, unresolved


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input-sheet", default=DEFAULT_INPUT_SHEET)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_SHEET)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    resolved, unresolved = resolve_sheet(args.input_sheet, args.cache_root)

    n_ibd = sum(1 for r in resolved if r["layout"] == "ibd")
    n_flat = sum(1 for r in resolved if r["layout"] == "flat")
    logger.info(
        "Resolved %d/%d libraries (ibd=%d, flat=%d); %d unresolved",
        len(resolved), len(resolved) + len(unresolved),
        n_ibd, n_flat, len(unresolved),
    )
    if unresolved:
        logger.info("Unresolved libraries: %s", ", ".join(sorted(unresolved)))

    with open(args.output, "w", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["sample_name", "path", "seqrun", "layout"],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(resolved)
    logger.info("Wrote %d rows to %s", len(resolved), args.output)


if __name__ == "__main__":
    main()
