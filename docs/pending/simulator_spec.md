# The cut-site fragment simulator

This simulator produces cfDNA fragments whose true generating probability is
known. Real data never offers that. With it, a model can be asked two questions
it cannot otherwise be asked: does it recover the parameters that made the data,
and how close is it to the best score any model could reach?

## 1. What the model captures

A fragment is a cut at each end of a double-stranded molecule. The simulator
models where those cuts fall and how far apart they are.

- **Cut-site sequence preference.** Some 6-base sequences are cut far more often
  than others. That preference is measured on a real sample, so simulated
  fragments carry the same sequence signature at their ends.
- **Fragment length**, drawn from the distribution measured on the same sample.
- **Strand**, a fair coin. A fragment has no intrinsic orientation, and adapter
  ligation is symmetric, so `p_plus = 0.5` is a construction, not a fit. It still
  matters: a minus-strand fragment's genomic start is its 3′ end, so the two
  strands read different tables.
- **Depth.** Each region gets as many fragments as the real sample had there.

**Not captured:** bulk GC (the length profile matches, the length-GC joint does
not); anything in a fragment's interior; and lengths outside 25–180 bases, which
are filtered out of the real data before measurement.

## 2. How a fragment is drawn

Everything rests on the propensity of each of the 4096 6-mers:

    r(h) = C(h) / N(h)

`C(h)` counts how often `h` was a cut site in the real sample. `N(h)` counts how
often it would have been if cutting ignored sequence. Their ratio is how much
that sequence is preferred. There are four tables, since a cut can be at a
molecule's 5′ or 3′ end, on either strand.

```
         region  [gstart ................................ gstop)
                      |<----------- L ----------->|
    ..ACGTAC          |                           |          GGATCC..
      ^^^^^^          start                      stop          ^^^^^^
    6-mer at the start                              6-mer at the end
    picks the position                              picks the length

    pad 3 |<-------------- region_len -------------->| pad 183
```

To draw one fragment in a region:

1. **Strand**, by a fair coin. This selects which tables steps 2 and 3 read.
2. **Start**, with probability proportional to the propensity of the 6-mer there.
   Positions from which no fragment of any valid length could be made are
   excluded first, then the rest are renormalised.
3. **Length**, with probability proportional to the propensity of the 6-mer at
   the resulting far end, times the measured length distribution.
4. **Reject repeats.** A `(start, stop)` pair already drawn in this region sends
   both the start and the length back to be drawn again.

The probability of the resulting fragment is recorded.

A length can be impossible three ways: a non-ACGT base at the far end, a 6-mer
never observed as a cut site in this sample, or a length never observed. The draw
does not distinguish them; it only asks whether any weight remains.

## 3. What a run produces

| Output | Content |
|---|---|
| BED | one row per simulated fragment, 8 columns |
| `.p.tsv.gz` sidecar | `contig, start, stop, strand, p` — every fragment's generating probability |
| `run.json` | inputs, seed, counts, timings, `oracle_nll` |
| fragment h5 | optional, built from the BED |

The sidecar is what makes the simulation useful. Because repeats are redrawn,
`(contig, start, stop, strand)` is unique, so it joins one-to-one against the
fragments.

**The oracle**, `-mean(log p)`, is the score a model would get if it knew the
generating parameters exactly. No model can beat it. The figure in `run.json` is
pooled over every fragment and the whole length range; a model evaluated on a
subset of regions or a cropped window needs its oracle recomputed over that same
domain, which `scripts/simulator/cut_site_oracle_nll.py` does.

## 4. Running it

    python scripts/simulator/run_cut_site_simulator.py \
      --sample-id SAMPLE \
      --fragments-h5 /efs/analytics/nathanboley/ibd/frag_h5s/SAMPLE.fragments.h5 \
      --region-bed data/region_sets/quiet_v2_pad1200_repeats_removed_tile1536.bed \
      --fasta /efs/analytics/nathanboley/test_fragments_h5/GRCh38.p12.genome.fa.gz \
      --out-dir /efs/analytics/nathanboley/<run> \
      --seed N

A full run covers 66,649 regions in about 7 minutes.
`scripts/simulator/measure_cut_site_hexamers.py` runs the measurement alone.

Region sets live in the main checkout's gitignored `data/region_sets/`:
`..._removed_tile1536` (66,649 regions), `..._removed_tile2560` (11,505),
`..._kept_tile1536` (904,975), `..._kept_tile2560` (523,604).

Runs use GRCh38.p12 at the path above. A store's `config_hash` includes the
reference index, so artifacts built against a different reference will not
compare as equal even where the sequence is identical. Record the reference in
any output.

## 5. Reproducibility

**The same seed gives byte-identical output at any worker count** — BED, sidecar
and statistics alike.

Each region draws from its own stream, `default_rng([seed, region_index])`, where
`region_index` is the region's position in the input region set, not its row in
whatever frame reaches the draw. The pair keeps replicates under neighbouring
seeds independent; both values must fit in 32 bits. The uniform counts `N(h)` are
computed in fixed blocks of 256 regions, never per worker, because the block size
fixes the summation order and so the last bits of the result.

Regression references, both RD-56804 at seed 20261008:
`/efs/analytics/nathanboley/sim_d166_lowfix_w1/` (200 regions) and
`sim_d166_w1_2k/` (2,000). Compare sidecars **after decompressing** — gzip
records a timestamp, so compressed bytes are never stable.

## 6. Code

| Module | Holds | Depends on |
|---|---|---|
| `background_model/constants.py` | `L_MIN`, `L_MAX`, `N_LENGTHS`, `KMER`, `HEX_HALF`, `NHEX` | `tracks` |
| `background_model/hexamers.py` | 6-mer encoding, RC permutation | numpy, `constants` |
| `background_model/fragment_lengths.py` | `FragmentLengthDist`, `fl_end_weight` | numpy |
| `background_model/simulator/measure.py` | measurement: `C`, `N`, `f(L)`, `r` | pandas, `fragmentomics_tools` |
| `background_model/simulator/draw.py` | the draw, BED and sidecar writer | the above |

`L_MIN = 25` and `L_MAX = 180` derive from the locked `tracks.FL_BANDS`; `L_MAX`
is inclusive. `fragment_lengths.py` sits outside `simulator/` on purpose:
model-side code needs it and must not import the simulator.

Entry points: `measure_sample` (the whole measurement for one sample),
`simulate_fragments_to_bed` (draws every region in parallel, writes the output),
`sample_region`, `start_distribution` (the per-region start and length
distributions the draw uses), `oracle_nll`.

## 7. Rules that must not change without measurement

Each was wrong once, or is load-bearing in a way the code does not show.

- **A fragment belongs to the region containing its START**, which is what makes
  it belong to exactly one tile. The sequence frame is padded asymmetrically for
  the same reason — 3 bases left, 183 right — so a fragment whose end runs past
  the region edge still has a readable 6-mer, without widening the set of
  positions a fragment may start from.

- **Start tables divide by the start expectation, end tables by the end
  expectation — by POSITION, not by table name.** The `_rev` tables look crossed
  relative to their names and are correct that way, because a minus-strand
  fragment's 5′ cut site sits at its genomic stop. Reversing this pairing cost a
  median 1.2% and maximum 16% error per propensity cell, silently, with every
  total intact.

- **The two expectations differ.** `N_start` weights every position equally.
  `N_end` is length-weighted and ramps near region edges, because an end near an
  edge is reachable by fewer lengths.

  ![End weight across one tile and across three adjacent tiles](simulator_spec_figs/end_weight.png)

  The ramp is correct rather than a loss: the excluded routes are counted in the
  neighbouring tile, and `C` is tabulated under the same partition, so adjacent
  tiles sum to exactly 1 in the interior (above, measured). An isolated tile with
  no neighbour is genuinely depleted near its edges, which matters to anyone
  choosing a crop width or computing an oracle.

- **The length bound is applied once**, when the fragment arrays are built. The
  sampler draws over the measured distribution's own support and does not know
  `L_MIN` or `L_MAX`. Do not add a second check downstream.

- **Mapping quality is filtered before deduplication**, because dedup keeps the
  first occurrence of a `(start, stop)` pair, not the best-mapped one. The
  remaining filters commute.

- **Asking for more fragments than a region has positions raises**, as does
  exceeding twice the requested count in draw attempts. Both raise rather than
  silently emitting fewer fragments than asked for.

- **`r` is a definition, not an estimate.** The simulator draws from `r` and the
  oracle is computed from the same `r`, so they agree by construction. `C` and
  `N` apply their validity gates differently; that changes what the model is, not
  whether it is right. It would matter only if `r` were ever read as a
  measurement of biology rather than as a generating parameter.

## 8. Tests

`tests/test_hexamers.py`, `test_simulator_measure.py`, `test_simulator_draw.py`,
`test_simulator_propensity_denominators.py` and `test_cut_site_hygiene.py`, with
fixtures in `tests/conftest.py` and an independent encoder oracle in
`tests/cut_site_oracle.py` that imports nothing from the repo.
`scripts/_mutation_test.py` carries mutations that must each turn a test red; run
only the entries a change touches.

Run the suite with `make test`, never bare pytest.
