# Golden test fixtures — provenance

These files back `tests/test_bg_golden_counts.py` (the brief's one guard on
`fragment_array` counting). Do not edit by hand.

## Source

Copied 2026-08-27 (read-only) from the `fragments_h5` repo's own test corpus:

    /home/nathanboley/src/fragments_h5/tests/data/

(GitHub `KariusDx/fragments_h5`.) These are the same real BAM/FASTA fixtures the
`fragments_h5` suite uses.

| File | Origin | Notes |
|------|--------|-------|
| `small.chr6.bam` (+ `.bai`) | fragments_h5 tests/data | ~2600 PE reads on chr6 (genome-wide), 604 duplicate-flagged |
| `test_duplicates.bam` (+ `.bai`) | fragments_h5 tests/data | 2 PE pairs at chr6:99110000-99110116 (frag len 116, strand +); 1 pair is duplicate-flagged and coordinate-identical to the other |
| `GRCh38.p12.genome.chr6_99110000_99130000.fa.gz` (+ `.fai`, `.gzi`) | fragments_h5 tests/data | bgzipped chr6 (full contig length 170,805,979 in the `.fai`; only 99,110,000–99,130,000 is populated) |

## Built fixtures

`golden.small.chr6.frag.h5` and `golden.test_duplicates.frag.h5` are built ONCE
by `make_golden_fixture.sh` using the **production** fragments_h5 CLI (so the
golden count test is not circular):

    PYTHON=/home/nathanboley/miniconda3/envs/biomarker_env/bin/python \
        bash tests/data/make_golden_fixture.sh

Exact commands run (2026-08-27, biomarker_env, fragments_h5 editable install):

    python -m fragments_h5.main small.chr6.bam       golden.small.chr6.frag.h5       --fasta GRCh38.p12.genome.chr6_99110000_99130000.fa.gz
    python -m fragments_h5.main test_duplicates.bam  golden.test_duplicates.frag.h5  --fasta GRCh38.p12.genome.chr6_99110000_99130000.fa.gz --include-duplicates

Notes:
- The `build-fragments-h5` console script is not on PATH in `biomarker_env`;
  invoke the module (`python -m fragments_h5.main`) instead.
- `test_duplicates` is built with `--include-duplicates` on purpose, so the
  coordinate-identical pair enters the h5 and the golden test exercises
  read-time `RegionFragmentArray.drop_duplicate_fragments()`.
- Built h5 sizes: `golden.small.chr6.frag.h5` ~808 KB;
  `golden.test_duplicates.frag.h5` ~538 KB.
