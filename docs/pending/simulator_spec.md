# Simulator spec

## 1. Inputs

| # | Input | Source |
|---|-------|--------|
| 1 | Fragment h5 | manifest key `ibd/frag_h5s/<SEQRUN>/<SAMPLE>-Lib1.hg38.fragments.h5` |
| 2 | Regions BED | `quiet_v2_pad1200_repeats_removed_tile1536.bed` |
| 3 | Reference FASTA | `s3://karius-biomarker-data-assets/pipeline-assets/hg38/sequence.fa.bgz` |

Manifest: `~/src/biomarker-projects/tf_binding_site_classification/projects/ibd/manifests/ibd.data_manifest.tsv`
— 763 h5, mirror `s3://karius-biomarker-data-assets/projects`.

### Regions BED options

In the main checkout's gitignored `data/region_sets/`.

| BED | Regions | Tile | Repeats |
|---|---:|---|---|
| `quiet_v2_pad1200_repeats_removed_tile1536.bed` | **66,649** | 1536 | removed |
| `quiet_v2_pad1200_repeats_removed_tile2560.bed` | 11,505 | 2560 | removed |
| `quiet_v2_pad1200_repeats_kept_tile1536.bed` | 904,975 | 1536 | kept |
| `quiet_v2_pad1200_repeats_kept_tile2560.bed` | 523,604 | 2560 | kept |

## 2. Build

Four stages. Stages 1 and 2 share one pass over the h5. Stage 3 consumes
stage 2.

1. **Observed hexamer counts** `C(h)` — cut-site 6-mers of real fragments.
   Per sample.
2. **Fragment-length distribution** `f(L)` — `FragmentLengthDist`, built from
   the SRDF after §3 admission, so the same population `C(h)` counts.
   Per sample.
3. **Uniform hexamer counts** `N(h)` — the same tabulation as stage 1 with
   every start site equally likely. `N_start` is sequence-only, so per region
   set; `N_end` is `f(L)`-weighted.
4. **Region counts** — fragments per region.

## 3. Admission

A fragment is admitted to a region when:

- `min(mapq_read1, mapq_read2) >= 10`
- it survives dedup on `(start, stop)`
- its **start** lies in `[gstart, gstop)`
- its **length** lies in `[L_MIN, L_MAX]` = `[25, 180]`

In that order.

### Sequence frame

    left_pad  = HEX_HALF = 3
    right_pad = max_fl + HEX_HALF = 183

A hexamer index then equals its region-local coordinate. `max_fl` is `L_MAX`
for the fragment pass and the `FragmentLengthDist` bound for the uniform pass;
after §3 admission they are the same 180.

## 4. Sample

`n` comes from the region counts. Per fragment:

**0. Strand** — Bernoulli. It selects which tables the next two steps use,
because a minus-strand fragment's genomic start is its 3′ end:

| | genomic start `i` | genomic stop `i + L` |
|---|---|---|
| plus | `r_start_fwd(hex_fwd(i))` | `r_end_fwd(hex_fwd(i+L))` |
| minus | `r_end_rev(hex_rc(i))` | `r_start_rev(hex_rc(i+L))` |

**1. Start** — genomic start `i` over `[gstart, gstop)`, weighted by the left
column and normalised **within the region**:

    p[i] = r_start(hex(i))        for i in [gstart, gstop), 0 if hex(i) invalid
    P(i) = p[i] / p.sum()

**2. Stop** — given `i`, build the weight vector over lengths and normalise it
over the **valid** entries only:

    w[l] = f(l) · r_end(hex(i + l))     for every l in [L_MIN, L_MAX]
                                        yielding a VALID fragment, else 0
    P(l | i) = w[l] / w.sum()

A length is valid when both cut-site hexamers are ACGT-only. The denominator is
therefore **per start** — two starts in the same region normalise over different
length sets, so `w.sum()` is not a region constant.

## 5. Propensity

A hexamer's weight is observed over expected-under-uniform:

    r(h) = C(h) / N(h)

§4 normalises it per region for starts and per start for stops.

`C` and `N` are one tabulation counted two ways, so both cover the same
candidate positions and the same validity gate. `N_start` weights every
position 1; `N_end` weights by `f(L)` per route, which per position is

    w(i) = F(min(i, max_fl)) − F(max(min_fl − 1, i − region_len))

for the `FragmentLengthDist` CDF `F` over its own support. `f` reaches only the
edge ramp: interior positions carry weight 1 under any normalised `f`. The
`_rev` tables divide by the permuted expectation,
`N_rc == N_fwd[rc_permutation()]`.

## 6. Code

`background_model/simulator/count_hexamers_rdf.py`. Self-contained: stdlib,
numpy, pandas, `fragmentomics_tools.dataframe`.

| Function | Does |
|---|---|
| `filter_fragments` | dedup, length filter, admission. All filtering |
| `cut_site_hexamers` | per region → `start_hex`, `stop_hex`, `strand` |
| `counts_from_hexamers` | genomic start/stop + strand → the four tables |
| `count_sample` | the three passes end to end → `C(h)` |
| `FragmentLengthDist` | `counts`, `densities`, `min_fl`, `max_fl`, cached CDF |
| `uniform_hexamer_counts` | → `N(h)` |
| `fl_end_weight` | `w(i)` above |
| `propensities` | `C / N` |
| `sample_region` | draw `n` fragments → `starts_0`, `lengths`, `is_plus` |
| `hexamer_indices` | sliding 6-mer encode; `str`/`bytes`/`uint8`, case-folded |
| `rc_permutation` | RC as a 4096 permutation, derived from the encoder |

`C(h)`: `attach_fragment_arrays(callback=filter_fragments)` →
`attach_sequence` → one `parallel_apply` of `cut_site_hexamers` → four
`bincount`s.

`N(h)`: serial sequence walk.

`FragmentLengthDist` is built by `from_dataframe` (columns
`fragment_length`, `count`; absent lengths densify to zero) or `from_srdf`,
which delegates to it. Its three arrays are read-only, since an in-place edit
would leave the cached CDF stale.

`sample_region` is vectorised over fragments: all `n` in a region share its
arrays, so the length draw is one `(n, n_lengths)` block.

## Open

- **Per-region count hardcoded to `54`** (`sampler.py::_REGION_COUNTS`).
- **Hexamer tables in the simulator are synthetic log-normals**, not `r(h)`.
- **No tests.**
- **Second copy of the encoder** in `simulator/precompute.py`, until the old
  simulator is deleted.
- **Bulk GC is not modelled.** The length marginal matches by construction;
  the `(length, GC)` joint does not.
- **A start with no valid length is dropped**, so that region yields fewer than
  `n` fragments. `P(start)` does not condition on a valid fragment existing.
  Not observed on the 1536 tiles — 38,637 drawn for 38,637 requested — but it
  is a silent shortfall where it does occur.
