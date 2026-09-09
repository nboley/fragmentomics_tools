# Sample-sheet QC — background-model-v2 quiescent pool (Phase 3.2)

Date: 2026-09-09. Branch: `background-model-v2`.

Builder `background_model/sample_sheet.py` run against the REAL IBD manifest +
pooled clinical CSV for the first time (Phase 1 built it against the manifest
FORMAT only). This note records the QC that gates the Phase 3.3 h5 download and
the 40/10 train/held-out split.

## Inputs

- Manifest: `.../tf_binding_site_classification/projects/ibd/manifests/ibd.data_manifest.tsv`
  (DataManifest v3, keys `NC-<seqrun>/<lib>.hg38.fragments.h5`).
- Clinical: `.../ibd_consolidated/ibd_analysis/data/metadata/Pooled_Meta_Data_IBD_Foundation_with_ResultIDs_09_16_2025.csv`
  (join key `library_name`; only `ENDO_CATEGORY` crosses the boundary).

## Output sheet (gitignored — not committed)

- `data/sample_sheets/ibd_quiescent.tsv`
- Columns: `library, h5_path, seqrun, endo_category`
- Rows: **213**
- md5: **8cc70a7c78bc11c9423f633c291f8362**

## Join QC

- Total manifest entries: **763** (763 unique libraries).
- Manifest libs present in clinical `library_name`: **751**.
- Join MISSES — no usable `ENDO_CATEGORY`: **167**
  - not in clinical at all: **12** — engineered/analytical controls, not
    patient libs (e.g. `AC-124104-Lib1_DC4-16709_S22`,
    `EC-124102-Lib1_DC4-16709_S24`).
  - in clinical but blank/NaN `ENDO_CATEGORY`: **155** — RD libs from a later
    round with no endoscopy category (e.g. `RD-124105-Lib1` … `RD-124112-Lib1`).
- Manifest libs joined WITH an `ENDO_CATEGORY`: **596**.

## ENDO_CATEGORY breakdown (all 763 manifest entries)

| ENDO_CATEGORY            | count |
|-------------------------|-------|
| Remission               | 213   |
| Mild                    | 170   |
| (no clinical / blank)   | 167   |
| Moderate                | 126   |
| Severe                  | 87    |

## Quiescent pool — the number awaited since Phase 1

- Contract quiescent set = `{Asymptomatic, Remission}`.
- **FINDING: this CSV has NO `Asymptomatic` value.** The only `ENDO_CATEGORY`
  levels present are `Mild / Moderate / Remission / Severe` (verified: no
  case/whitespace variant of "asympt" exists). The `{Asymptomatic, Remission}`
  filter therefore resolves to **Remission-only**.
- **Quiescent pool size: 213 (all Remission).**
- Decision: 213 ≫ the ~50-sample scale target, so the **40 train / 10 held-out
  split stands comfortably** (4.3× headroom). No relaxation of the quiescent
  definition is needed to reach 50. If a broader pool is ever wanted,
  `Mild` (170) is the natural next tier — but that is an algorithmic/definition
  change requiring owner approval, not made here.

## Quiescent pool seqrun spread

Pool spans **33 seqruns**, well distributed (max 15, most 6–7 per seqrun):

- NC-13909: 15; NC-13907: 8; NC-13925/NC-14150/NC-14192/NC-14193/NC-14198/
  NC-14199/DC4-17405/DC4-17453/DC4-17454: 7 each; the bulk of the remaining ~22
  seqruns: 6 each; NC-13910: 1.
- Ample flowcell diversity for a simple random 40/10 draw (flowcell
  stratification is explicitly rejected per the brief).

## Projected h5 sync size (gates Phase 3.3 download)

Sum of manifest `size` fields for the quiescent pool:

- Per-sample: mean **0.429 GB** (min 0.103, max 1.825 GB).
- Pool TOTAL (all 213): **91.3 GB**.
- **Projected 50-sample draw (mean × 50): ≈ 21.4 GB.**

50-sample sync is modest (~21 GB); syncing the full pool would be ~91 GB.

## Code adaptation (committed separately)

`background_model/sample_sheet.py` needed three minimal real-file adaptations,
all test-preserving (existing `TestSampleSheetBuilder` + full bg suite green,
191 passed):

1. `pd.read_csv(..., comment="#")` — skip the v3 `#`-prefixed metadata header.
2. `library_from_key()` — derive library from the key basename (real notes JSON
   carries no `library` field; the prior fallback used the full key path).
3. Recognize `library_name` as the clinical join column (was falling through to
   `SampleID`, which lacks the `-Lib1` suffix and would not have matched).
