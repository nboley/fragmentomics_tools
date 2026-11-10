# The cut-site fragment simulator

This simulator produces cfDNA fragments whose true generating probability is
known. Real data never offers that. With it, a model can be asked two questions
it cannot otherwise be asked: does it recover the parameters that made the data,
and how close is it to the best score any model could reach?

## 1. What the model captures

A fragment is a cut at each end of a double-stranded molecule. The simulator
models where those cuts fall and how far apart they are.

**Cut-site sequence preference.** Some 6-base sequences are cut far more often
than others. The simulator measures that preference on a real sample and draws
from it, so simulated fragments carry the same sequence signature at their ends.

**Fragment length.** Lengths are drawn from the distribution measured on the
same sample, so the simulated length profile matches the real one.

**Strand.** A fragment has no intrinsic orientation. The strand label only
records which end became read 1, and adapter ligation is symmetric, so strand is
a fair coin, `p_plus = 0.5`. It is not fitted. It still matters to the draw,
because a minus-strand fragment's genomic start is its 3′ end, so the two strands
read different tables.

**Depth.** Each region receives as many fragments as were observed there in the
real sample.

### What it does not capture

- **Bulk GC content.** The length profile matches by construction; the joint
  distribution of length and GC does not.
- **Anything outside the ends.** The interior sequence of a fragment plays no
  part in whether it is drawn.
- **Fragments shorter than 25 or longer than 180 bases.** They are filtered out
  of the real data before anything is measured, so the simulator never sees them
  and never produces them.

## 2. How a fragment is drawn

Everything rests on one quantity, the propensity

    r(h) = C(h) / N(h)

for each of the 4096 possible 6-mers `h`. `C(h)` counts how often `h` was
actually a cut site in the real sample. `N(h)` counts how often it would have
been, had cutting been indifferent to sequence. Their ratio is how much that
sequence is preferred.

There are four tables, because a cut can be at a molecule's 5′ or 3′ end, on
either strand.

To draw one fragment in a region:

1. **Pick a strand** by a fair coin. This selects which tables the next two steps
   read.
2. **Pick a start position**, with probability proportional to the propensity of
   the 6-mer there. Positions from which no fragment of any valid length could
   be made are excluded first, and the remaining probabilities are renormalised.
3. **Pick a length**, with probability proportional to the propensity of the
   6-mer at the resulting far end, multiplied by the measured length
   distribution.
4. **Reject repeats.** If this `(start, stop)` pair was already drawn in this
   region, both the start and the length are drawn again.

The probability of the fragment that results is recorded.

A length can be impossible for three reasons: the sequence at the far end
contains a non-ACGT base, that 6-mer was never observed as a cut site in this
sample, or the length itself was never observed. The draw does not distinguish
them; it only asks whether any weight remains.

## 3. What a run produces

A run takes a fragment h5, a region set and a reference FASTA, and writes:

| Output | Content |
|---|---|
| BED | one row per simulated fragment, 8 columns |
| `.p.tsv.gz` sidecar | `contig, start, stop, strand, p` — the generating probability of every fragment |
| `run.json` | inputs, seed, counts, timings, and `oracle_nll` |
| fragment h5 | optional, built from the BED |

The sidecar is what makes the simulation useful. Because repeats are redrawn,
`(contig, start, stop, strand)` is unique, so the sidecar joins one-to-one
against the fragments.

**The oracle.** `oracle_nll = -mean(log p)` is the score a model would get if it
knew the generating parameters exactly. No model can beat it. Note that this
figure is pooled over every fragment and the whole length range; a model
evaluated on a subset of regions, or on a cropped window, needs its oracle
recomputed over that same domain, which
`scripts/simulator/cut_site_oracle_nll.py` does.

## 4. Running it

    python scripts/simulator/run_cut_site_simulator.py \
      --sample-id SAMPLE \
      --fragments-h5 /efs/analytics/nathanboley/ibd/frag_h5s/SAMPLE.fragments.h5 \
      --region-bed data/region_sets/quiet_v2_pad1200_repeats_removed_tile1536.bed \
      --fasta /efs/analytics/nathanboley/test_fragments_h5/GRCh38.p12.genome.fa.gz \
      --out-dir /efs/analytics/nathanboley/<run> \
      --seed N

`scripts/simulator/measure_cut_site_hexamers.py` runs the measurement alone and
writes `C`, `N`, `r`, `f(L)` and the per-region counts to an `.npz`.

A full run over the default region set covers 66,649 regions in about 7 minutes.

**Region sets** live in the main checkout's gitignored `data/region_sets/`:

| BED | Regions | Tile |
|---|---:|---|
| `quiet_v2_pad1200_repeats_removed_tile1536.bed` | 66,649 | 1536 |
| `quiet_v2_pad1200_repeats_removed_tile2560.bed` | 11,505 | 2560 |
| `quiet_v2_pad1200_repeats_kept_tile1536.bed` | 904,975 | 1536 |
| `quiet_v2_pad1200_repeats_kept_tile2560.bed` | 523,604 | 2560 |

**Reference.** Runs use GRCh38.p12 at the path above. Patch releases do not
change primary assembly sequence, but a store's `config_hash` includes the
reference index, so artifacts built against a different reference will not
compare as equal even where the sequence is identical. Record the reference in
any output.

## 5. Reproducibility

**The same seed gives byte-identical output at any worker count** — BED, sidecar
and statistics alike.

Each region draws from its own stream, `default_rng([seed, region_index])`, where
`region_index` is the region's position in the input region set, not its row in
whatever frame reaches the draw. The pair is what keeps replicates under
neighbouring seeds independent. Both values must fit in 32 bits.

The uniform counts `N(h)` are computed in fixed blocks of 256 regions, not per
worker, because the block size determines the floating-point summation order and
therefore the last bits of the result.

Two reference runs exist for regression checks, both RD-56804 at seed 20261008:
`/efs/analytics/nathanboley/sim_d166_lowfix_w1/` (200 regions) and
`sim_d166_w1_2k/` (2,000 regions). Compare the sidecar after decompressing it —
gzip records a timestamp, so the compressed bytes are never stable.

## 6. Code

| Module | Holds | Depends on |
|---|---|---|
| `background_model/constants.py` | `L_MIN`, `L_MAX`, `N_LENGTHS`, `KMER`, `HEX_HALF`, `NHEX` | `tracks` |
| `background_model/hexamers.py` | 6-mer encoding, reverse-complement permutation | numpy, `constants` |
| `background_model/fragment_lengths.py` | `FragmentLengthDist`, `fl_end_weight` | numpy |
| `background_model/simulator/measure.py` | measurement on real data: `C`, `N`, `f(L)`, `r` | pandas, `fragmentomics_tools` |
| `background_model/simulator/draw.py` | the draw, and the BED and sidecar writer | the above |

`L_MIN = 25` and `L_MAX = 180` derive from `tracks.FL_BANDS`, which is locked.
`L_MAX` is inclusive.

The entry points most callers need:

| Function | Does |
|---|---|
| `measure_sample` | the whole measurement for one sample, returning `C`, `N`, `f(L)`, `r` and the per-region counts |
| `simulate_fragments_to_bed` | draws every region in parallel, writes the BED and sidecar |
| `sample_region` | draws one region's fragments |
| `start_distribution` | the per-region start and length distributions the draw uses |
| `oracle_nll` | `-mean(log p)` |

`fragment_lengths.py` is deliberately outside `simulator/`: model-side code needs
`FragmentLengthDist` and `fl_end_weight`, and must not import the simulator.

## 7. Rules that must not change without measurement

Each of these was wrong once, or is load-bearing in a way that is not obvious
from the code.

- **A fragment belongs to the region containing its START.** This is what makes
  each fragment belong to exactly one tile. The sequence frame is padded
  asymmetrically for the same reason: 3 bases on the left, 183 on the right, so
  a fragment whose end runs past the region edge still has a readable 6-mer,
  without extending the set of positions a fragment may start from.

- **Start tables divide by the start expectation, end tables by the end
  expectation — by POSITION, not by table name.** The `_rev` tables look crossed
  relative to their names and are correct that way: a minus-strand fragment's 5′
  cut site sits at its genomic stop. Reversing this pairing produced a median
  1.2% and maximum 16% error per propensity cell, silently, with every total
  intact.

- **The two expectations differ.** `N_start` weights every position equally.
  `N_end` is weighted by the length distribution and ramps near region edges,
  because an end near an edge is reachable by fewer lengths. The ramp is correct,
  not a loss: the routes it excludes are counted in the neighbouring tile, and
  `C` is tabulated under the same partition. An isolated tile with no neighbour
  is genuinely depleted near its edges, which matters to anyone choosing a crop
  width or computing an oracle.

- **The length bound is applied once**, when the fragment arrays are built. The
  sampler draws over the measured distribution's own support and does not know
  `L_MIN` or `L_MAX`. Do not add a second check downstream.

- **Mapping quality is filtered before deduplication.** Deduplication keeps the
  first occurrence of a `(start, stop)` pair, not the best-mapped one, so the
  order matters. The remaining filters commute.

- **Requesting more fragments than a region has positions raises**, as does
  exceeding twice the requested count in draw attempts. Both raise rather than
  silently emitting fewer fragments than asked for.

- **`r` is a definition, not an estimate.** The simulator draws from `r` and the
  oracle is computed from the same `r`, so they are consistent by construction.
  `C` and `N` apply their validity gates differently; this changes what the model
  is, not whether it is right, and is settled. It would matter only if `r` were
  ever read as a measurement of biology rather than as a generating parameter.

## 8. Tests

`tests/test_hexamers.py`, `tests/test_simulator_measure.py`,
`tests/test_simulator_draw.py`, `tests/test_simulator_propensity_denominators.py`
and `tests/test_cut_site_hygiene.py`, with shared fixtures in
`tests/conftest.py` and an independent encoder oracle in
`tests/cut_site_oracle.py` that imports nothing from the repo.

`scripts/_mutation_test.py` carries mutations that must each turn at least one
test red. Run only the entries a change touches; the full sweep is slow.

Run the suite with `make test`, never bare pytest.
