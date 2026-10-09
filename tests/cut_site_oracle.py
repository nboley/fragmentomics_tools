"""Independent oracle for cut-site hexamer tests.

Imports NOTHING from ``background_model`` or ``fragmentomics_tools``.
Every function is plain Python (or numpy for array results).  The AST
check ``test_oracle_is_independent`` (``tests/test_cut_site_hygiene.py``)
enforces this.

The oracle provides a second derivation of every quantity the module
computes. A test that compares the module's output to the oracle's is
comparing two independent implementations, so a bug must exist in at
least two codebases to pass.
"""

from itertools import product

import numpy as np

BASES = "ACGT"
KMER = 6
NHEX = 4 ** KMER  # 4096
_COMP = str.maketrans("ACGTacgt", "TGCAtgca")


def IDX(h: str) -> int:
    """Base-4 big-endian index. A=0, C=1, G=2, T=3, case-folded."""
    h = h.upper()
    val = 0
    for ch in h:
        val = val * 4 + BASES.index(ch)
    return val


def RC(h: str) -> str:
    """Reverse complement of a hexamer string."""
    return h.translate(_COMP)[::-1]


def DECODE(i: int) -> str:
    """Index to hexamer string."""
    chars = []
    for _ in range(KMER):
        chars.append(BASES[i % 4])
        i //= 4
    return "".join(reversed(chars))


def hex_at(genome: str, c: int, offset: int = 0) -> str:
    """The hexamer centred on cut site ``c``: ``genome[c-offset-3:c-offset+3]``.

    ``offset`` is the genomic coordinate of ``genome[0]``.
    """
    pos = c - offset
    if pos < 3:
        return ""
    return genome[pos - 3: pos + 3]


def valid(h: str) -> bool:
    """True when all 6 characters are in ACGTacgt."""
    return len(h) == KMER and all(ch in "ACGTacgt" for ch in h)


def decode_strand(x) -> str:
    """Bytes or str to '+'/'-'. Anything else raises."""
    if isinstance(x, bytes):
        x = x.decode("ascii")
    if x not in ("+", "-"):
        raise ValueError(f"unrecognised strand label: {x!r}")
    return x


def de_bruijn(k: int, n: int) -> str:
    """Linear de Bruijn sequence B(k, n) via Lyndon words.

    Every k-mer over an alphabet of size n occurs exactly once as a
    substring.  Length = k^n + (k-1).
    """
    alphabet = list(range(n))
    seq = []
    a = [0] * (k * n)

    def _db(t, p):
        if t > n:
            if n % p == 0:
                seq.extend(a[1:p + 1])
        else:
            a[t] = a[t - p]
            _db(t + 1, p)
            for j in range(a[t - p] + 1, k):
                a[t] = j
                _db(t + 1, t)

    _db(1, 1)
    digits = seq + seq[:n - 1]
    return "".join(BASES[d] for d in digits)


def all_palindromes() -> set:
    """The 64 self-complementary hexamers."""
    return {h for h in ("".join(p) for p in product(BASES, repeat=KMER))
            if RC(h) == h}


def bruteforce_count(
    fragments,
    regions,
    genome,
    *,
    min_mapq=10,
    l_min=25,
    l_max=180,
    genome_offset=0,
):
    """Brute-force hexamer counting.

    ``fragments``: list of (start, stop, strand_str, mapq1, mapq2).
    ``regions``: list of (gstart, gstop).
    ``genome``: reference string; ``genome[0]`` is at genomic coordinate
    ``genome_offset``.

    Returns ``(region_counts, tables)``.
    ``region_counts[i]`` = admitted fragments in region i.
    ``tables`` = dict of four int64 (NHEX,) arrays.
    """
    mapq_pass = [(s, e, st, m1, m2)
                 for s, e, st, m1, m2 in fragments
                 if min(m1, m2) >= min_mapq]
    seen = set()
    deduped = []
    for s, e, st, m1, m2 in mapq_pass:
        if (s, e) not in seen:
            seen.add((s, e))
            deduped.append((s, e, st, m1, m2))
    length_pass = [(s, e, st, m1, m2) for s, e, st, m1, m2 in deduped
                   if l_min <= (e - s) <= l_max]

    region_counts = np.zeros(len(regions), dtype=np.int64)
    tables = {k: np.zeros(NHEX, dtype=np.int64)
              for k in ("start_fwd", "end_fwd", "start_rev", "end_rev")}

    for s, e, st, m1, m2 in length_pass:
        for ri, (g0, g1) in enumerate(regions):
            if g0 <= s < g1:
                region_counts[ri] += 1
                s_hex = hex_at(genome, s, genome_offset)
                e_hex = hex_at(genome, e, genome_offset)
                if not valid(s_hex) or not valid(e_hex):
                    continue
                strand = decode_strand(st)
                if strand == "+":
                    tables["start_fwd"][IDX(s_hex)] += 1
                    tables["end_fwd"][IDX(e_hex)] += 1
                else:
                    tables["start_rev"][IDX(RC(e_hex))] += 1
                    tables["end_rev"][IDX(RC(s_hex))] += 1
                break
    return region_counts, tables


def enumerate_expectation(
    genome, regions, fl_densities, min_fl, max_fl, genome_offset=0,
):
    """N_start and N_end by literal enumeration.

    ``fl_densities``: array of length max_fl - min_fl + 1, fl_densities[l - min_fl] = f(l).
    ``genome_offset``: genomic coordinate of ``genome[0]``.
    Returns ``(N_start, N_end)`` as float64 arrays of shape (NHEX,).
    """
    N_start = np.zeros(NHEX, dtype=np.float64)
    N_end = np.zeros(NHEX, dtype=np.float64)

    for g0, g1 in regions:
        R = g1 - g0
        for s in range(R):
            c = g0 + s
            h = hex_at(genome, c, genome_offset)
            if valid(h):
                N_start[IDX(h)] += 1

        for i in range(R + max_fl):
            c = g0 + i
            h = hex_at(genome, c, genome_offset)
            if not valid(h):
                continue
            w = 0.0
            for l_idx, L in enumerate(range(min_fl, max_fl + 1)):
                s_of_this_end = i - L
                if 0 <= s_of_this_end < R:
                    w += fl_densities[l_idx]
            if w > 0:
                N_end[IDX(h)] += w

    return N_start, N_end


def enumerate_null_counts(
    genome, regions, fl_densities, min_fl, max_fl, p_plus=0.5,
    genome_offset=0,
):
    """Expected counts under the uniform-start null, as float tables.

    Every ``(start, length)`` pair contributes its weight to both the
    start and end tables, split by ``p_plus``.
    """
    tables = {k: np.zeros(NHEX, dtype=np.float64)
              for k in ("start_fwd", "end_fwd", "start_rev", "end_rev")}

    perm = np.empty(NHEX, dtype=np.int64)
    for i in range(NHEX):
        perm[i] = IDX(RC(DECODE(i)))

    for g0, g1 in regions:
        R = g1 - g0
        for s in range(R):
            for l_idx, L in enumerate(range(min_fl, max_fl + 1)):
                f_l = fl_densities[l_idx]
                if f_l == 0:
                    continue
                e = s + L
                c_s = g0 + s
                c_e = g0 + e
                h_s = hex_at(genome, c_s, genome_offset)
                h_e = hex_at(genome, c_e, genome_offset)
                if not valid(h_s) or not valid(h_e):
                    continue
                idx_s = IDX(h_s)
                idx_e = IDX(h_e)
                tables["start_fwd"][idx_s] += p_plus * f_l
                tables["end_fwd"][idx_e] += p_plus * f_l
                tables["start_rev"][perm[idx_e]] += (1 - p_plus) * f_l
                tables["end_rev"][perm[idx_s]] += (1 - p_plus) * f_l

    return tables


def sampler_start_weights(
    genome_seq_upper, region_len, r, track_name="fwd",
):
    """Raw start weights as the module computes them.

    ``genome_seq_upper``: the padded sequence (left_pad=3), uppercased.
    Returns the unnormalised weight array over [0, region_len).
    """
    weights = np.zeros(region_len, dtype=np.float64)
    s_tab = r["start_fwd"] if track_name == "fwd" else r["end_rev"]
    for i in range(region_len):
        h = genome_seq_upper[i: i + KMER]
        if not valid(h):
            continue
        if track_name == "fwd":
            idx = IDX(h)
        else:
            idx = IDX(RC(h))
        weights[i] = s_tab[idx]
    return weights
