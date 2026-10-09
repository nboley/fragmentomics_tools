"""The cut-site simulator's draw: fragments from ``r(h)``, ``f(L)`` and per-region counts.

Inputs come from ``background_model.cut_site_stats`` (``count_sample``,
``uniform_hexamer_counts``, ``propensities``, ``FragmentLengthDist``); encoding
is ``background_model.hexamers``.  Authority: ``docs/pending/simulator_spec.md``.
Driver: ``scripts/run_cut_site_simulator.py``.

``simulate_fragments_to_bed`` is parallel over regions, and **byte-identical
across worker counts** (owner decision 166): each region draws from its own
stream ``default_rng([seed, region_index])``, and every reduction is grouped
independently of ``n_workers``.  See ``region_rng``, and ``UNIFORM_BLOCK_SIZE``
in ``cut_site_stats`` for the same guarantee on ``N(h)``.

The sampler takes the per-site RATE ``r(h)`` from
``cut_site_stats.propensities``, not raw counts: it re-enumerates candidate
positions itself, so counts would apply hexamer abundance twice.
"""

from __future__ import annotations

from typing import Dict

import gzip

import numpy as np
import pandas as pd

from fragmentomics_tools.dataframe import DataFrameBase

from background_model.constants import HEX_HALF, L_MAX
from background_model.cut_site_stats import FragmentLengthDist
from background_model.hexamers import hexamer_indices

# Seeds and region indices are each ONE 32-bit word; see region_rng.
_SEED_WORD_MAX: int = 2 ** 32


def sample_region(
    sequence,
    region_len: int,
    n: int,
    *,
    r: Dict[str, np.ndarray],
    fl: "FragmentLengthDist",
    p_plus: float,
    rng,
    _dup_counter=None,
):
    """Draw ``n`` fragments in one region.

    Returns ``(starts_0, lengths, is_plus, probs)``.

    Per fragment: strand, then start, then length.

    - **Strand** ~ Bernoulli(``p_plus``).  The strand split is exactly
      ``Binomial(n, p_plus)`` and is never revisited.
    - **Start** ``i`` over ``[0, region_len)``, proportional to the start-side
      propensity AND gated on the start having at least one valid length
      (``live``).  Normalised within the strand.
    - **Length** over ``[min_fl, max_fl]``, proportional to
      ``end_p x f(L)``, normalised over ``L``.
    - **Duplicates** on ``(start, start + length)`` are redrawn.  The key is
      strand-blind, matching the read-time dedup convention.
    - **Feasibility**: ``n > region_len`` raises ``ValueError``.

    ``probs`` is the first-draw marginal probability of each fragment::

        p = P(strand) * P(start | strand) * P(length | start, strand)

    The full weight block ``W_s`` is computed once for all positions per
    strand, then ``t_s = W_s.sum(axis=1)`` gives the per-start total.
    ``live = t_s > 0`` restricts starts to those with at least one valid
    length.  The initial draw is vectorised; only duplicate collisions
    enter a scalar redraw loop.
    """
    if n > region_len:
        raise ValueError(
            f"requested n={n} fragments but region has only "
            f"{region_len} positions"
        )
    seq = np.frombuffer(bytes(sequence).upper(), dtype=np.uint8)
    fwd, rc, valid = hexamer_indices(seq)
    Ls = np.arange(fl.min_fl, fl.max_fl + 1)
    pos = np.arange(region_len)
    ends_all = pos[:, None] + Ls[None, :]

    n_plus = int(rng.binomial(n, p_plus))
    used = set()
    out_s, out_L, out_p, out_prob = [], [], [], []
    n_dup_redraws = 0

    for is_plus, k in ((True, n_plus), (False, n - n_plus)):
        if k == 0:
            continue
        track = fwd if is_plus else rc
        s_tab = r["start_fwd"] if is_plus else r["end_rev"]
        e_tab = r["end_fwd"] if is_plus else r["start_rev"]
        p_strand = np.float64(p_plus if is_plus else (1.0 - p_plus))

        W_s = e_tab[track[ends_all]] * valid[ends_all] * fl.densities[None, :]
        t_s = W_s.sum(axis=1, dtype=np.float64)
        live = t_s > 0

        a_s = s_tab[track[pos]] * valid[pos]
        a_s_live = a_s * live
        tot = a_s_live.sum(dtype=np.float64)
        if tot <= 0:
            continue

        start_probs = a_s_live / tot

        n_distinct = int(((a_s_live > 0)[:, None] & (W_s > 0)).sum())

        starts = rng.choice(pos, size=k, replace=True, p=start_probs)
        w = W_s[starts]
        row_sums = t_s[starts]
        cdf = np.cumsum(w / row_sums[:, None], axis=1)
        pick = (cdf < rng.random((k, 1))).sum(axis=1)
        lengths = Ls[pick]

        result_starts = np.empty(k, dtype=np.int64)
        result_lengths = np.empty(k, dtype=np.int64)
        for j in range(k):
            s, L = int(starts[j]), int(lengths[j])
            while (s, s + L) in used:
                n_dup_redraws += 1
                # Total draw attempts for the region are n + n_dup_redraws.
                # Cap them at 2n, so the raise fires once redraws exceed n.
                # The counter is region-scoped (initialised before the strand
                # loop), so this is a budget for the whole region rather than
                # per fragment or per strand.
                #
                # Why 2n does not false-positive: n <= region_len is already
                # enforced above, and each start admits ~len(Ls) lengths, so
                # the live (start, L) space is ~len(Ls) times larger than n
                # and expected redraws are well under 1% of n.  Exceeding n
                # redraws means the live space is pathologically small -- not
                # that collisions were unlucky -- so failing loudly beats
                # spinning.  Do not loosen this without redoing that
                # arithmetic.
                if n_dup_redraws > n:
                    raise RuntimeError(
                        f"duplicate redraws ({n_dup_redraws}) exceeded n={n}, "
                        f"i.e. more than 2n={2 * n} total draw attempts for "
                        f"this region; {len(used)} of {n_distinct} distinct "
                        f"(start, stop) pairs used in this strand block"
                    )
                s = int(rng.choice(pos, p=start_probs))
                w_row = W_s[s]
                cdf_row = np.cumsum(w_row / t_s[s])
                l_idx = int((cdf_row < rng.random()).sum())
                L = int(Ls[l_idx])
            used.add((s, s + L))
            result_starts[j] = s
            result_lengths[j] = L

        l_indices = result_lengths - fl.min_fl
        p_start_vals = start_probs[result_starts]
        p_length_vals = (
            W_s[result_starts, l_indices] / t_s[result_starts]
        )
        probs = p_strand * p_start_vals * p_length_vals

        out_s.append(result_starts)
        out_L.append(result_lengths)
        out_p.append(np.full(k, is_plus))
        out_prob.append(probs)

    if _dup_counter is not None:
        _dup_counter[0] += n_dup_redraws

    if not out_s:
        z = np.zeros(0, dtype=np.int64)
        return z, z, np.zeros(0, dtype=bool), np.zeros(0, dtype=np.float64)
    return (np.concatenate(out_s).astype(np.int64),
            np.concatenate(out_L).astype(np.int64),
            np.concatenate(out_p),
            np.concatenate(out_prob))


def oracle_nll(probs: np.ndarray) -> float:
    """``-mean(log(p))`` in float64."""
    probs = np.asarray(probs, dtype=np.float64)
    if probs.size == 0:
        return float("nan")
    return float(-np.log(probs).mean(dtype=np.float64))


def _as_seed_word(value, name: str) -> int:
    """``value`` as a Python int in ``[0, 2**32)``, or raise."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError(f"{name} must be an int, got {value!r}")
    value = int(value)
    if not 0 <= value < _SEED_WORD_MAX:
        raise ValueError(
            f"{name}={value} is outside [0, 2**32); see region_rng for why "
            f"a wider value would let two different (seed, region) pairs "
            f"share a stream"
        )
    return value


def region_rng(seed: int, region_index: int) -> np.random.Generator:
    """The draw stream for one region: ``default_rng([seed, region_index])``.

    Owner decision 166.  Each region's draw depends on ``(seed,
    region_index)`` and nothing else -- not on the worker count, not on which
    other regions are drawn, not on the order they run in.

    **The PAIR, not ``seed + region_index``.**  With a sum, region 1 of seed
    ``s`` and region 0 of seed ``s + 1`` share a stream, so replicates under
    neighbouring seeds would be correlated.

    **Both values must be ONE 32-bit word, so both are bounded to
    ``[0, 2**32)``.**  ``SeedSequence`` flattens the list into 32-bit words
    before hashing, so a wider seed spills into the region slot: measured,
    ``[7 + 3 * 2**32, 0]`` and ``[7, 3]`` give the SAME stream.  With both
    values one word wide the mapping from pair to stream is one-to-one.  (A
    related quirk is harmless: ``[s, 0]`` equals ``default_rng(s)``, because
    the entropy pool zero-pads.)
    """
    return np.random.default_rng([_as_seed_word(seed, "seed"),
                                  _as_seed_word(region_index, "region_index")])


def _draw_one_region(sequence, region_len: int, n: int, region_index: int,
                     *, seed: int, r, fl, p_plus: float) -> dict:
    """One region's draw on its own stream.  Runs in a worker.

    The duplicate-redraw count travels back in the RETURN value: a counter
    mutated inside a forked worker is the worker's copy and never reaches the
    parent.
    """
    if n == 0:
        z = np.zeros(0, dtype=np.int64)
        return dict(starts_0=z, lengths=z, is_plus=np.zeros(0, dtype=bool),
                    probs=np.zeros(0, dtype=np.float64), n_dup_redraws=0)
    dup_counter = [0]
    starts_0, lengths, is_plus, probs = sample_region(
        sequence, region_len, n, r=r, fl=fl, p_plus=p_plus,
        rng=region_rng(seed, region_index), _dup_counter=dup_counter,
    )
    return dict(starts_0=starts_0, lengths=lengths, is_plus=is_plus,
                probs=probs, n_dup_redraws=dup_counter[0])


def simulate_fragments_to_bed(
    srdf,
    out_path: str,
    *,
    r: Dict[str, np.ndarray],
    fl: "FragmentLengthDist",
    region_counts,
    seed: int,
    region_index=None,
    p_plus: float = 0.5,
    l_max: int = L_MAX,
    mapq: int = 60,
    sample_id: str | None = None,
    p_sidecar_path: str | None = None,
    n_workers: int | None = None,
    verbose: bool = False,
) -> dict:
    """Draw fragments for every region and write an 8-column BED + a p sidecar.

    The BED is the input to ``fragments_h5.build_fragments_h5``, which needs it
    bgzipped and tabix-indexed -- that is the caller's next step, not this
    function's.

    ``region_counts`` is the per-row ``n``, as returned by
    ``cut_site_stats.count_srdf``.
    ``seed`` is required rather than defaulted: an unseeded run cannot be
    reproduced, and the seed is this artifact's only provenance.

    **Seeding (owner decision 166).**  Row ``k`` draws from
    ``region_rng(seed, region_index[k])`` -- its own stream, shared with no
    other region.  ``region_index`` is the region's position in the INPUT
    region set, a property of the region rather than of this frame, so a
    filtered or reordered ``srdf`` draws exactly the same fragments for the
    regions it keeps.  It is taken from the argument if given, else from the
    frame's ``region_index`` column, which ``cut_site_stats.count_sample``
    attaches.  There is no positional fallback: ``arange(len(srdf))`` would
    silently re-key every region of a filtered frame.  Values must be unique.

    **Parallel, and byte-identical for every ``n_workers``.**  Regions are
    drawn through ``parallel_apply`` (``None`` = every CPU, ``1`` =
    in-process); the results come back in row order and are assembled exactly
    as a serial loop would, and ``n_dup_redraws`` is summed from per-region
    return values.  The BED, the sidecar and the stats are therefore
    identical whatever the worker count.

    Column layout, verified against ``fragments_h5.fragment.tsv_to_fragments``:

        contig  start  stop  <empty>  0  strand  mapq  mapq

    - **Exactly 8 columns.** 7 is rejected outright by the reader, and a row
      whose column count differs from the first row's is skipped.
    - **Column 4 is EMPTY, not** ``"."``. The reader takes
      ``parts[3] if parts[3] else None``, and ``"."`` is truthy -- it would
      write a literal ``"."`` cell barcode into the h5. Empty writes none.
    - **MAPQ is written explicitly** in columns 7 and 8, and must be 0-255.
      Omitting it stores a 255 sentinel that reads back as ``-1``, and the
      model's ``min_mapq=10`` then drops **every** fragment. The default 60
      clears that with room to spare.
    - 0-based half-open, matching ``starts_0``/``stops_0``.

    Output is sorted by ``(contig, start, stop)`` because tabix requires each
    contig's records contiguous and position-ordered. Sorting here is why
    ``sample_region``'s draw order is not part of its contract.

    **Sidecar:** a gzipped TSV next to the BED (``<prefix>.p.tsv.gz``) carrying
    ``(contig, start, stop, strand, p)`` for every drawn fragment.  With
    duplicates redrawn the ``(contig, start, stop, strand)`` key is unique, so
    a join against the h5 is exact and one-to-one.  ``p`` is written as
    ``%.17g`` to round-trip float64 exactly.  Read-time dedup removes nothing.

    Next step, verified end to end against the real reader::

        gz = pysam.tabix_index(bed_path, preset="bed", force=True)
        build_fragments_h5(gz, out_h5, fasta_filename=...)   # FASTA required

    **``pysam.tabix_index`` CONSUMES the plain BED** -- measured: after the call
    only ``sim.bed.gz`` and ``sim.bed.gz.tbi`` remain. Do not plan to re-read or
    hash the plain file afterwards. Convenient for the two-output contract,
    since the intermediate deletes itself, but surprising if unexpected.

    Returns a stats dict including ``oracle_nll`` and ``n_dup_redraws``.
    ``n_drawn == n_requested`` in all practical cases; ``n_short_regions`` is
    retained for interface continuity but is permanently zero.
    """
    if out_path.endswith(".gz"):
        raise ValueError(
            f"write a PLAIN bed, got {out_path!r}. pandas would gzip it, and "
            f"tabix needs BGZIP, which gzip is not -- the index step would fail "
            f"on a file that looks correct. bgzip it as a separate step."
        )
    region_counts = np.asarray(region_counts)
    if region_counts.shape != (len(srdf),):
        raise ValueError(
            f"region_counts has shape {region_counts.shape}, expected "
            f"({len(srdf)},) -- one entry per row of srdf, in row order"
        )
    for col in ("contig", "start", "stop", "fragment_array", "sequence"):
        if col not in srdf.columns:
            raise ValueError(f"srdf has no {col!r} column")
    seed = _as_seed_word(seed, "seed")

    expected_seq_len_extra = 2 * HEX_HALF + l_max

    # Validate every row in the parent, before anything forks, so a bad frame
    # fails with its row number rather than from inside a worker.
    region_lens = np.empty(len(srdf), dtype=np.int64)
    for i, (contig, gstart, gstop, fa, seq) in enumerate(zip(
        srdf["contig"], srdf["start"], srdf["stop"],
        srdf["fragment_array"], srdf["sequence"],
    )):
        region_len = int(gstop) - int(gstart)
        region_lens[i] = region_len
        if fa.length != region_len:
            raise AssertionError(
                f"row {i}: fragment_array.length {fa.length} != stop-start "
                f"{region_len}. Admission used the former and the sequence "
                f"frame the latter, so they must agree."
            )
        if len(seq) != region_len + expected_seq_len_extra:
            raise AssertionError(
                f"row {i} ({contig}:{gstart}-{gstop}): sequence is {len(seq)} b, "
                f"expected {region_len + expected_seq_len_extra} "
                f"(region {region_len} + left_pad {HEX_HALF} + right_pad "
                f"{l_max + HEX_HALF}). A short sequence means the flank was "
                f"truncated -- a region within {l_max + HEX_HALF} b of a contig "
                f"end cannot supply it -- and every hexamer index would shift."
            )

    if region_index is None:
        if "region_index" not in srdf.columns:
            raise ValueError(
                "srdf has no 'region_index' column and none was passed. Each "
                "region's stream is keyed on its position in the INPUT region "
                "set (count_sample attaches it); a positional default would "
                "silently re-key every region of a filtered frame."
            )
        region_index = srdf["region_index"]
    region_index = np.asarray(region_index)
    if region_index.shape != (len(srdf),):
        raise ValueError(
            f"region_index has shape {region_index.shape}, expected "
            f"({len(srdf)},) -- one entry per row of srdf, in row order"
        )
    region_index = np.array([_as_seed_word(v, "region_index")
                             for v in region_index.tolist()], dtype=np.int64)
    if np.unique(region_index).size != region_index.size:
        raise ValueError(
            "region_index has duplicate values, so two rows would draw from "
            "the same stream. A frame with several samples per region has "
            "one row per (sample, region) pair and is not supported here."
        )

    chunks = []
    n_requested = n_drawn = n_short = n_dup_redraws = 0
    if len(srdf):
        work = DataFrameBase(pd.DataFrame({
            "sequence": srdf["sequence"].to_numpy(),
            "region_len": region_lens,
            "n": region_counts.astype(np.int64),
            "region_index": region_index,
        }))
        draws = work.parallel_apply(
            lambda row: _draw_one_region(
                row["sequence"], int(row["region_len"]), int(row["n"]),
                int(row["region_index"]),
                seed=seed, r=r, fl=fl, p_plus=p_plus,
            ),
            n_workers=n_workers,
            verbose=verbose,
        )
        # Records come back in ROW order whatever process drew them, so
        # everything below is the serial assembly.
        draws = draws.itertuples(index=False)
    else:
        draws = iter(())

    for i, (contig, gstart, d) in enumerate(zip(
        srdf["contig"], srdf["start"], draws,
    )):
        n = int(region_counts[i])
        n_requested += n
        n_dup_redraws += int(d.n_dup_redraws)
        starts_0, lengths, is_plus, probs = (
            d.starts_0, d.lengths, d.is_plus, d.probs
        )
        if n == 0:
            continue
        n_drawn += len(starts_0)
        if len(starts_0) < n:
            n_short += 1
        if not len(starts_0):
            continue

        starts = int(gstart) + starts_0
        stops = starts + lengths
        strands = np.where(is_plus, "+", "-")

        chunks.append(pd.DataFrame({
            "contig": contig,
            "start": starts,
            "stop": stops,
            "name": "",
            "score": 0,
            "strand": strands,
            "mapq1": mapq,
            "mapq2": mapq,
            "p": probs,
        }))

    if chunks:
        bed = pd.concat(chunks, ignore_index=True)
        bed.sort_values(["contig", "start", "stop"], kind="stable", inplace=True)
    else:
        bed = pd.DataFrame(columns=["contig", "start", "stop", "name", "score",
                                    "strand", "mapq1", "mapq2", "p"])

    bed_out = bed[["contig", "start", "stop", "name", "score",
                   "strand", "mapq1", "mapq2"]]
    bed_out.to_csv(out_path, sep="\t", header=False, index=False)

    # Sidecar: (contig, start, stop, strand, p) with provenance header.
    if out_path.endswith(".bed"):
        sidecar_default = out_path[:-4] + ".p.tsv.gz"
    else:
        sidecar_default = out_path + ".p.tsv.gz"
    sidecar_path = p_sidecar_path or sidecar_default
    with gzip.open(sidecar_path, "wt") as f:
        meta_parts = [f"seed={seed}"]
        if sample_id is not None:
            meta_parts.append(f"sample={sample_id}")
        f.write(f"# {' '.join(meta_parts)}\n")
        f.write("contig\tstart\tstop\tstrand\tp\n")
        for row in bed.itertuples(index=False):
            f.write(
                f"{row.contig}\t{row.start}\t{row.stop}\t"
                f"{row.strand}\t{row.p:.17g}\n"
            )

    all_p = bed["p"].to_numpy(dtype=np.float64) if len(bed) else np.array(
        [], dtype=np.float64
    )

    return dict(
        n_regions=len(srdf),
        n_requested=n_requested,
        n_drawn=n_drawn,
        n_short_regions=n_short,
        n_rows_written=len(bed),
        oracle_nll=oracle_nll(all_p),
        n_dup_redraws=n_dup_redraws,
        p_sidecar=sidecar_path,
    )
