#!/usr/bin/env python
"""Generate the synthetic TSS fixture used by test_dataframe.py.

The fixture is a headered, gzipped TSV with columns matching BED6 positional
order (contig, start, stop, name, peak_tpm, strand) so that both
``pd.read_table`` and ``BedReader.load_dataframe`` can consume it.

Construction is fully deterministic — no random state.  The constants below
are duplicated in ``test_dataframe.py``; changing them here without updating
the test will cause assertion failures, which is the point.

Run from the repo root::

    python test/generate_tss_fixture.py
"""

import gzip
from pathlib import Path

# ── Construction parameters ──────────────────────────────────────────────
TOTAL_ROWS = 200

# Contiguous blocks so that per-chrom counts are trivially verifiable.
ROWS_PER_CHROM = {
    "chr1": 50,
    "chr2": 40,
    "chr3": 30,
    "chr4": 20,
    "chr5": 15,
    "chr6": 10,
    "chr7": 10,
    "chrX": 25,
}
assert sum(ROWS_PER_CHROM.values()) == TOTAL_ROWS

# peak_tpm for row *i* (0-indexed global row number) = (i + 1) * 0.25
# Range: [0.25, 50.0].  Rows with peak_tpm > 10: i >= 40 → 160 rows.

REGION_WIDTH = 100  # half-open [start, stop)
REGION_SPACING = 500  # between successive starts within a chromosome
STRANDS = ["+", "-", "."]

OUT_DIR = Path(__file__).parent / "data"
OUT_FILE = OUT_DIR / "tss.synthetic.hg19.bed.gz"

# ── Generate ─────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    global_i = 0
    for chrom, n in ROWS_PER_CHROM.items():
        for local_i in range(n):
            start = 10000 + local_i * REGION_SPACING
            stop = start + REGION_WIDTH
            name = f"TSS_{global_i:04d}"
            peak_tpm = (global_i + 1) * 0.25
            strand = STRANDS[global_i % len(STRANDS)]
            rows.append(f"{chrom}\t{start}\t{stop}\t{name}\t{peak_tpm}\t{strand}")
            global_i += 1

    assert global_i == TOTAL_ROWS

    header = "contig\tstart\tstop\tname\tpeak_tpm\tstrand"
    with gzip.open(OUT_FILE, "wt") as fh:
        fh.write(header + "\n")
        for row in rows:
            fh.write(row + "\n")

    print(f"Wrote {TOTAL_ROWS} rows to {OUT_FILE}")


if __name__ == "__main__":
    main()
