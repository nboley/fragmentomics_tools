#!/usr/bin/env python3
"""Programmatic mutation testing for the cut-site simulator.

Targets ``background_model/hexamers.py``, ``background_model/cut_site_stats.py``
and ``background_model/simulator/draw.py`` (one file, ``count_hexamers_rdf.py``,
until decision 169 split it), plus a few library and oracle files.

Applies each mutation as a text substitution, runs the test suite,
records which tests fail, reverts, and prints a matrix.

**Exit status is the verdict.** 0 only if every non-VOID mutation's search
text was found, the replacement changed the file, and at least one test went
red.  A search text that matches nothing used to be recorded as "skip" and
the run still looked green -- which is how M6 and M40b went silently void
when unrelated edits moved their anchor text.  Now a skip, a mutation no test
catches, a timeout or an error all exit 1.  A mutation whose test run fails to
COLLECT (a ``SyntaxError`` or ``ImportError`` in the mutated module) is also an
error, not a catch: pytest reports every baseline test as absent, which used
to read as "every test went red" and exit 0.  Collection failure is detected
from pytest's own return code and summary text, never inferred from which
tests are missing.

**Every non-VOID anchor is checked before anything runs** (``check_anchor``):
its search text must occur exactly once in the target file, and that
occurrence must lie in code, not wholly inside a string, docstring or
comment.  A failing anchor exits 1 before the baseline.  Without it, an
anchor duplicated by a later edit mutates whichever copy comes first, and one
that only survives in a docstring mutates text that never executes.

VOID entries are listed in ``VOID`` with the reason.  A VOID mutation is
expected NOT to match; if its text reappears the run fails too, since the
entry then needs re-examining rather than ignoring.

NEVER commits. Verifies the tree is clean before and after.  Run it against
a copy (``MUTATION_WORKTREE``), never a tree other agents are using.
"""

import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import tokenize

# MUTATION_WORKTREE points the harness at a copy, so a mutation never touches a
# tree that other agents are running tests or code from.  No default: see the
# guard at the top of main(), which refuses to run at all unless the env var
# is set AND differs from FORBIDDEN_WORKTREE below.
WORKTREE = os.environ.get("MUTATION_WORKTREE")

# The shared worktree this harness must NEVER run mutations against. Kept only
# as the forbidden path to check against, not as a default.
FORBIDDEN_WORKTREE = (
    "/home/nathanboley/src/fragmentomics_tools/.claude/worktrees/background-model-work")
TEST_FILE = "tests/test_cut_site_simulator.py"
PROP_TEST_FILE = "tests/test_simulator_propensity_denominators.py"

PYTHON = "/home/nathanboley/miniconda3/envs/biomarker_env/bin/python"
PYTHON_BIN = os.path.dirname(PYTHON)

# Each mutation: (id, description, file_to_mutate_relative, old_text, new_text)
# file_to_mutate_relative is relative to WORKTREE
MUTATIONS = [
    # M1: swap G↔T in _BASE_LUT
    ("M1", "swap G↔T in _BASE_LUT",
     "background_model/hexamers.py",
     'for _code, _base in enumerate("ACGT"):',
     'for _code, _base in enumerate("ACTG"):'),

    # M2: reverse _POW (little-endian)
    ("M2", "reverse _POW (little-endian)",
     "background_model/hexamers.py",
     "_POW = (4 ** np.arange(KMER - 1, -1, -1)).astype(np.int64)",
     "_POW = (4 ** np.arange(KMER)).astype(np.int64)"),

    # M3: drop lowercase from _BASE_LUT
    ("M3", "drop lowercase from _BASE_LUT",
     "background_model/hexamers.py",
     "for _code, _base in enumerate(\"ACGT\"):\n    _BASE_LUT[ord(_base)] = _code\n    _BASE_LUT[ord(_base.lower())] = _code",
     "for _code, _base in enumerate(\"ACGT\"):\n    _BASE_LUT[ord(_base)] = _code"),

    # M4: complement without reverse in rc computation
    ("M4", "complement without reverse (rc line)",
     "background_model/hexamers.py",
     "rc = (3 - safe)[:, ::-1] @ _POW",
     "rc = (3 - safe) @ _POW"),

    # M7: drop valid mask in cut_site_hexamers
    ("M7", "drop valid mask in cut_site_hexamers",
     "background_model/cut_site_stats.py",
     "    ok = s_ok & e_ok\n    return pd.DataFrame({\n        \"start_hex\": s_hex[ok].astype(np.int32),\n        \"stop_hex\": e_hex[ok].astype(np.int32),\n        \"strand\": np.asarray(rfa.fragment_strands)[ok],\n    })",
     "    ok = np.ones(len(s_ok), dtype=bool)\n    return pd.DataFrame({\n        \"start_hex\": s_hex[ok].astype(np.int32),\n        \"stop_hex\": e_hex[ok].astype(np.int32),\n        \"strand\": np.asarray(rfa.fragment_strands)[ok],\n    })"),

    # M8: swap start_rev/end_rev sources
    ("M8", "swap start_rev/end_rev sources",
     "background_model/cut_site_stats.py",
     '    counts["start_rev"] = np.bincount(perm[e[~plus]], minlength=NHEX)\n    counts["end_rev"] = np.bincount(perm[s[~plus]], minlength=NHEX)',
     '    counts["start_rev"] = np.bincount(perm[s[~plus]], minlength=NHEX)\n    counts["end_rev"] = np.bincount(perm[e[~plus]], minlength=NHEX)'),

    # M9: omit perm on minus strand
    ("M9", "omit perm on minus strand",
     "background_model/cut_site_stats.py",
     '    counts["start_rev"] = np.bincount(perm[e[~plus]], minlength=NHEX)\n    counts["end_rev"] = np.bincount(perm[s[~plus]], minlength=NHEX)',
     '    counts["start_rev"] = np.bincount(e[~plus], minlength=NHEX)\n    counts["end_rev"] = np.bincount(s[~plus], minlength=NHEX)'),

    # M10: whole swap (plus = strand != "+")
    ("M10", "whole plus/minus swap",
     "background_model/cut_site_stats.py",
     '    plus = df["strand"].to_numpy() == "+"',
     '    plus = df["strand"].to_numpy() != "+"'),

    # M19: return C (no division) -- make propensities return counts not ratios
    ("M19", "propensities returns C (no division)",
     "background_model/cut_site_stats.py",
     "        r = np.zeros(NHEX, dtype=np.float64)\n        r[ok] = counts[name][ok] / d[ok]\n        out[name] = r",
     "        r = np.zeros(NHEX, dtype=np.float64)\n        r[ok] = counts[name][ok].astype(np.float64)\n        out[name] = r"),

    # M20: >= instead of > for min_expected
    ("M20", ">= instead of > for min_expected",
     "background_model/cut_site_stats.py",
     "        ok = d > min_expected",
     "        ok = d >= min_expected"),

    # M26: ignore p_plus (always 0.5)
    ("M26", "ignore p_plus (always 0.5)",
     "background_model/simulator/draw.py",
     "    n_plus = int(rng.binomial(n, p_plus))",
     "    n_plus = int(rng.binomial(n, 0.5))"),

    # M31: bytes comparison for strand
    ("M31", "bytes comparison for strand",
     "background_model/cut_site_stats.py",
     '    plus = df["strand"].to_numpy() == "+"',
     '    plus = df["strand"].to_numpy() == b"+"'),

    # M33: valid always True
    ("M33", "valid always True",
     "background_model/hexamers.py",
     "    valid = ~(win == 255).any(axis=1)",
     "    valid = np.ones(win.shape[0], dtype=bool)"),

    # M37: emit rows unsorted
    ("M37", "emit BED rows unsorted",
     "background_model/simulator/draw.py",
     '        bed.sort_values(["contig", "start", "stop"], kind="stable", inplace=True)',
     '        pass  # sorting removed'),

    # M38: densify without normalisation
    ("M38", "densify without normalisation",
     "background_model/cut_site_stats.py",
     "        self.densities = counts / np.float64(total)",
     "        self.densities = counts.astype(np.float64)"),

    # M41: restore pre-fix pairing (swap start_rev/end_rev denominators)
    ("M41", "restore pre-fix denominator pairing",
     "background_model/cut_site_stats.py",
     '        "start_rev": n_end.astype(np.float64)[perm],\n        "end_rev": n_start.astype(np.float64)[perm],',
     '        "start_rev": n_start.astype(np.float64)[perm],\n        "end_rev": n_end.astype(np.float64)[perm],'),

    # M42: drop valid in uniform_hexamer_counts (start)
    ("M42", "drop valid in uniform_hexamer_counts starts",
     "background_model/cut_site_stats.py",
     "            s_valid = valid[s_idx]\n            N_start += np.bincount(fwd[s_idx][s_valid], minlength=NHEX)",
     "            N_start += np.bincount(fwd[s_idx], minlength=NHEX)"),

    # ── Round 2: the 20 previously-unverified rows ─────────────────────────

    # M5: window [c-2,c+4) instead of [c-3,c+3)
    ("M5", "shift the hexamer window by 1 (frame)",
     "background_model/hexamers.py",
     "    fwd, _rc, valid = hexamer_indices(\n        seq[pos[:, None].astype(np.int64) + np.arange(KMER)[None, :]].reshape(-1)\n    )",
     "    fwd, _rc, valid = hexamer_indices(\n        seq[pos[:, None].astype(np.int64) + np.arange(1, KMER + 1)[None, :]].reshape(-1)\n    )"),

    # M6: left_pad=0 in count_sample
    ("M6", "left_pad=0 in count_sample",
     "background_model/cut_site_stats.py",
     # Anchored on the one line: 3895d08 put a comment block between it and
     # `srdf = (`, which silently turned the two-line pattern into a SKIP.
     "    left_flank, right_flank = HEX_HALF, l_max + HEX_HALF\n",
     "    left_flank, right_flank = 0, l_max + HEX_HALF\n"),

    # M11: MAPQ > instead of >=
    ("M11", "MAPQ > instead of >= (library)",
     "fragmentomics_tools/fragment_array/fragment_array.py",
     "            mask = mapq_vals >= min_mapq",
     "            mask = mapq_vals > min_mapq"),

    # M13: subset_fragment_lengths(l_min, l_max) -- half-open drops L_MAX
    ("M13", "half-open drops L_MAX=180",
     "background_model/cut_site_stats.py",
     "    fa = fa.drop_duplicate_fragments().subset_fragment_lengths(l_min, l_max + 1)",
     "    fa = fa.drop_duplicate_fragments().subset_fragment_lengths(l_min, l_max)"),

    # M16: right_pad=l_max (drops HEX_HALF, truncates overhang coverage)
    ("M16", "right_pad=l_max instead of l_max+HEX_HALF",
     "background_model/cut_site_stats.py",
     # Same drift as M6: anchored on the one line.
     "    left_flank, right_flank = HEX_HALF, l_max + HEX_HALF\n",
     "    left_flank, right_flank = HEX_HALF, l_max\n"),

    # M17: fl_end_weight off-by-one (drop the -1 on min_fl)
    ("M17", "fl_end_weight: max(min_fl, i-R) instead of max(min_fl-1, i-R)",
     "background_model/cut_site_stats.py",
     "        - fl.cdf_at(np.maximum(fl.min_fl - 1, i - region_len))",
     "        - fl.cdf_at(np.maximum(fl.min_fl, i - region_len))"),

    # M18: N_end over the region only (no flank)
    ("M18", "N_end computed over region_len only, no flank",
     "background_model/cut_site_stats.py",
     "            w = fl_end_weight(n_hex, region_len, fl)\n            m = valid & (w > 0)",
     "            w = fl_end_weight(region_len, region_len, fl)\n            m = valid & (w > 0)"),

    # M21: minus s_tab = start_rev instead of end_rev
    ("M21", "minus s_tab=start_rev instead of end_rev",
     "background_model/simulator/draw.py",
     '        s_tab = r["start_fwd"] if is_plus else r["end_rev"]',
     '        s_tab = r["start_fwd"] if is_plus else r["start_rev"]'),

    # M22: minus reads the fwd track instead of rc
    ("M22", "minus on fwd track instead of rc",
     "background_model/simulator/draw.py",
     "        track = fwd if is_plus else rc",
     "        track = fwd"),

    # M23: drop the valid mask on starts (retargeted: w_s renamed to a_s)
    ("M23", "drop valid mask on starts",
     "background_model/simulator/draw.py",
     "        a_s = s_tab[track[pos]] * valid[pos]",
     "        a_s = s_tab[track[pos]]"),

    # M27: plus uses minus tables end to end
    ("M27", "plus uses minus (end_rev/start_rev) tables",
     "background_model/simulator/draw.py",
     '        s_tab = r["start_fwd"] if is_plus else r["end_rev"]\n        e_tab = r["end_fwd"] if is_plus else r["start_rev"]',
     '        s_tab = r["end_rev"] if is_plus else r["end_rev"]\n        e_tab = r["start_rev"] if is_plus else r["start_rev"]'),

    # M28: strand column flipped (retargeted: np.where moved to strands var)
    ("M28", "writer strand column flipped",
     "background_model/simulator/draw.py",
     '        strands = np.where(is_plus, "+", "-")',
     '        strands = np.where(is_plus, "-", "+")'),

    # M29: 1-based start
    ("M29", "writer start is 1-based",
     "background_model/simulator/draw.py",
     "        starts = int(gstart) + starts_0",
     "        starts = int(gstart) + starts_0 + 1"),

    # M30: remove the U1 coercion on fragment_strands (library)
    ("M30", "remove U1 coercion on fragment_strands (library)",
     "fragmentomics_tools/fragment_array/fragment_array.py",
     '        self.fragment_strands = fragment_strands\n        if self.fragment_strands is not None:\n            self.fragment_strands = numpy.asarray(self.fragment_strands, dtype="U1")',
     '        self.fragment_strands = fragment_strands'),

    # M32: clip stops_0 to the region before the hexamer lookup
    ("M32", "clip stops_0 to region before hexamer lookup",
     "background_model/cut_site_stats.py",
     "    e_hex, e_ok = hexamers_at(sequence, rfa.stops_0)",
     "    e_hex, e_ok = hexamers_at(sequence, np.minimum(rfa.stops_0, rfa.length - 1))"),

    # M34: pad a truncated contig-end fetch with N instead of raising (library)
    ("M34", "pad truncated contig-end fetch with N instead of raising (library)",
     "fragmentomics_tools/region.py",
     '        seq = reference_fasta.fetch(self.chrom, start, stop).encode()\n        if len(seq) != stop - start:\n            raise ValueError(\n                f"{self}: fetched {len(seq)} bases, expected {stop - start} "\n                f"-- the fetch was truncated, which happens when the padded "\n                f"span runs off the end of {self.chrom}"\n            )\n        return seq',
     '        seq = reference_fasta.fetch(self.chrom, start, stop).encode()\n        if len(seq) != stop - start:\n            seq = seq + b"N" * (stop - start - len(seq))\n        return seq'),

    # M35: dedup key includes strand (library)
    ("M35", "dedup key includes strand (library)",
     "fragmentomics_tools/fragment_array/fragment_array.py",
     "    def drop_duplicate_fragments(self):\n        _, indices = np.unique(\n            np.array([self.starts_0, self.stops_0]), axis=1, return_index=True\n        )\n        return self.subset(indices)",
     '    def drop_duplicate_fragments(self):\n        strand_code = np.asarray([1 if s == "+" else 0 for s in self.fragment_strands])\n        _, indices = np.unique(\n            np.array([self.starts_0, self.stops_0, strand_code]), axis=1, return_index=True\n        )\n        return self.subset(indices)'),

    # M36: skip the length filter entirely
    ("M36", "skip the length filter entirely",
     "background_model/cut_site_stats.py",
     "    fa = fa.drop_duplicate_fragments().subset_fragment_lengths(l_min, l_max + 1)",
     "    fa = fa.drop_duplicate_fragments()"),

    # M39: VOID (see VOID below) -- the one-empty-strand raise was deleted by
    # owner decision 2026-10-08, so the pattern no longer exists in the source.
    ("M39", "delete the one-empty-strand raise (VOID: guard removed)",
     "background_model/cut_site_stats.py",
     "        if not n_plus or not n_minus:\n            raise AssertionError(",
     "        if not n_plus and not n_minus:\n            raise AssertionError("),

    # M24: drop valid on ends in W_s (the three-factor weight line)
    ("M24", "drop valid on ends in W_s",
     "background_model/simulator/draw.py",
     "        W_s = e_tab[track[ends_all]] * valid[ends_all] * fl.densities[None, :]",
     "        W_s = e_tab[track[ends_all]] * fl.densities[None, :]"),

    # ── Owner decision 166: seeding and parallel determinism ──────────────

    ("D1", "reseed with seed + i instead of the pair",
     "background_model/simulator/draw.py",
     '    return np.random.default_rng([_as_seed_word(seed, "seed"),\n                                  _as_seed_word(region_index, "region_index")])',
     '    return np.random.default_rng(_as_seed_word(seed, "seed")\n                                 + _as_seed_word(region_index, "region_index"))'),

    ("D2", "one rng shared across regions (per process)",
     "background_model/simulator/draw.py",
     "        rng=region_rng(seed, region_index), _dup_counter=dup_counter,",
     '        rng=globals().setdefault("_MUT_RNG", region_rng(seed, 0)), _dup_counter=dup_counter,'),

    ("D3", "N(h) reduction grouping follows n_workers",
     "background_model/cut_site_stats.py",
     "    los = np.arange(0, n, block_size, dtype=np.int64)",
     "    block_size = max(1, -(-n // (n_workers or __import__('os').cpu_count())))\n    los = np.arange(0, n, block_size, dtype=np.int64)"),

    ("D4", "dup counts lost when drawn in a forked worker",
     "background_model/simulator/draw.py",
     "                probs=probs, n_dup_redraws=dup_counter[0])",
     "                probs=probs, n_dup_redraws=(dup_counter[0] if __import__('multiprocessing').parent_process() is None else 0))"),

    ("D4b", "dup counts dropped on every path",
     "background_model/simulator/draw.py",
     "                probs=probs, n_dup_redraws=dup_counter[0])",
     "                probs=probs, n_dup_redraws=0)"),

    ("D5", "region_index from row position, not the index label",
     "background_model/cut_site_stats.py",
     "        rdf = rdf.assign(region_index=np.asarray(rdf.index, dtype=np.int64))",
     "        rdf = rdf.assign(region_index=np.arange(len(rdf), dtype=np.int64))"),

    # M40a: oracle imports background_model
    ("M40a", "oracle imports background_model",
     "tests/cut_site_oracle.py",
     "from itertools import product\n\nimport numpy as np",
     "from itertools import product\n\nimport numpy as np\nimport background_model"),

    # M40b: module imports simulator.precompute (unreachable, AST-visible only)
    ("M40b", "module imports simulator.precompute",
     "background_model/cut_site_stats.py",
     # 3895d08 added DataFrameBase to this import, which had made it a SKIP.
     "from fragmentomics_tools.dataframe import (\n    DataFrameBase,\n    SampleAndRegionDataFrame,\n    SampleDataFrame,\n)",
     "from fragmentomics_tools.dataframe import (\n    DataFrameBase,\n    SampleAndRegionDataFrame,\n    SampleDataFrame,\n)\nif False:\n    from background_model.simulator.precompute import hexamer_indices as _unused"),

    # ── Owner decision 169: layer split (hexamers <- cut_site_stats <- draw) ──

    ("L1", "hexamers imports pandas",
     "background_model/hexamers.py",
     "import numpy as np\n",
     "import numpy as np\nimport pandas as _pd  # noqa: F401\n"),

    ("L2", "cut_site_stats imports upward from simulator",
     "background_model/cut_site_stats.py",
     "from background_model.hexamers import (\n    HEX_HALF,\n    NHEX,\n    hexamer_indices,\n    hexamers_at,\n    rc_permutation,\n)",
     "from background_model.hexamers import (\n    HEX_HALF,\n    NHEX,\n    hexamer_indices,\n    hexamers_at,\n    rc_permutation,\n)\nif False:\n    from background_model.simulator.draw import sample_region as _unused"),
]

# Mutations whose target code was REMOVED on purpose, id -> reason.  Their
# search text is expected to match nothing.  Data, not a comment, so the exit
# status can tell an intended void from a pattern that silently drifted.
VOID = {
    "M39": "the one-empty-strand raise in count_srdf was deleted by owner "
           "decision 2026-10-08; there is no guard left to mutate",
}
_ids = [m[0] for m in MUTATIONS]
assert len(_ids) == len(set(_ids)), "duplicate mutation id"
assert set(VOID) <= set(_ids), f"VOID names unknown ids: {set(VOID) - set(_ids)}"


def read_file(path):
    with open(path) as f:
        return f.read()


def write_file(path, content):
    with open(path, "w") as f:
        f.write(content)


# Tokens that are not code for the anchor check: text inside them can be
# matched by a search string without the mutation touching behaviour.
_NON_CODE_TOKENS = {tokenize.STRING, tokenize.COMMENT, tokenize.NL,
                    tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT,
                    tokenize.ENDMARKER}


def check_anchor(source, old_text):
    """``None`` if ``old_text`` is a sound anchor in ``source``, else why not.

    Sound means it occurs EXACTLY ONCE, and that occurrence overlaps at least
    one code token -- a name, operator or number -- rather than lying wholly
    inside a string, docstring or comment.  ``str.replace(..., 1)`` mutates
    the FIRST occurrence, so a second copy makes the target ambiguous; and a
    copy inside a docstring or comment mutates text that never runs, which
    reads as an uncaught mutation, or worse as a catch by a doctest.
    """
    n = source.count(old_text)
    if n != 1:
        return f"search text occurs {n} times, expected exactly 1"
    start = source.index(old_text)
    end = start + len(old_text)
    # tokenize reports (row, col) with col in characters, so map rows to
    # absolute offsets.
    line_starts = [0]
    for line in source.splitlines(keepends=True):
        line_starts.append(line_starts[-1] + len(line))
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except (tokenize.TokenError, SyntaxError) as e:
        return f"target does not tokenize: {e}"
    for tok in tokens:
        if tok.type in _NON_CODE_TOKENS:
            continue
        t0 = line_starts[tok.start[0] - 1] + tok.start[1]
        t1 = line_starts[tok.end[0] - 1] + tok.end[1]
        if t0 < end and start < t1:
            return None
    return "search text lies only inside strings, docstrings or comments"


def clean_pycache():
    """Remove all __pycache__ directories and .pyc files."""
    for root, dirs, files in os.walk(WORKTREE):
        for d in dirs:
            if d == "__pycache__":
                shutil.rmtree(os.path.join(root, d), ignore_errors=True)
        for f in files:
            if f.endswith(".pyc"):
                os.remove(os.path.join(root, f))


def check_clean():
    r = subprocess.run(
        ["git", "-C", WORKTREE, "diff", "--stat", "--",
         "background_model/", "tests/", "fragmentomics_tools/"],
        capture_output=True, text=True,
    )
    return r.stdout.strip() == ""


def run_tests():
    """Run the test files.

    Returns ``(results, returncode, collection_failed)`` where ``results`` is
    a dict of pytest node id (``tests/file.py::Class::test_name[param]``) ->
    ``'passed'/'failed'/'error'``, parsed from ``-v`` output.  Keyed by the
    full node id, not the bare test name: two tests sharing a name in
    different classes or files would otherwise overwrite each other, and one
    going red could be masked by the other passing.
    ``collection_failed`` is True when pytest aborted before running
    anything -- a ``SyntaxError`` or ``ImportError`` in the mutated module --
    in which case every baseline test is simply ABSENT from ``results``,
    which looks identical to "every test failed" unless checked separately.
    """
    r = subprocess.run(
        [PYTHON, "-m", "pytest",
         os.path.join(WORKTREE, TEST_FILE),
         os.path.join(WORKTREE, PROP_TEST_FILE),
         "-v", "--tb=no",
         ],
        capture_output=True, text=True,
        cwd=WORKTREE,
        timeout=300,
        env={
            **os.environ,
            "PYTHONDONTWRITEBYTECODE": "1",
            "PATH": PYTHON_BIN + ":" + os.environ.get("PATH", ""),
        },
    )
    results = {}
    for line in r.stdout.splitlines():
        line = line.strip()
        for status_word in ("PASSED", "FAILED", "ERROR"):
            if f" {status_word}" in line and "::" in line:
                # Node id: everything before the status word.
                # Format: tests/file.py::Class::test_name PASSED [ xx%]
                test_id = line.split(f" {status_word}")[0].strip()
                # A teardown error prints a second line for the same id
                # (PASSED, then ERROR); never let a later PASSED hide it.
                if results.get(test_id, "passed") == "passed":
                    results[test_id] = status_word.lower()
                break

    lowered = r.stdout.lower()
    collection_failed = (
        r.returncode in (2, 3, 4)
        or "during collection" in lowered
        or "error collecting" in lowered
    )
    return results, r.returncode, collection_failed


def main():
    # Refuse to run at all against the shared worktree -- touches no file.
    if not WORKTREE:
        print("ERROR: MUTATION_WORKTREE is not set. There is no default; this "
              "harness mutates files in place and must never run against the "
              "shared worktree. Point it at a disposable copy, e.g. "
              "MUTATION_WORKTREE=/tmp/revreorg/copy2.", file=sys.stderr)
        return 2
    if os.path.realpath(WORKTREE) == os.path.realpath(FORBIDDEN_WORKTREE):
        print(f"ERROR: MUTATION_WORKTREE resolves to the shared worktree "
              f"({FORBIDDEN_WORKTREE}). Refusing to run mutations against it "
              f"-- point it at a disposable copy instead.", file=sys.stderr)
        return 2

    selected = set(sys.argv[1:]) if len(sys.argv) > 1 else None
    if selected and selected - set(_ids):
        # Otherwise a typo selects nothing, runs nothing and exits 0.
        print(f"ERROR: unknown mutation id(s): "
              f"{', '.join(sorted(selected - set(_ids)))}", file=sys.stderr)
        return 2
    mutations = [m for m in MUTATIONS if m[0] in selected] if selected else MUTATIONS

    # Anchor check, before the baseline: cheap, and a bad anchor makes every
    # later verdict about that mutation meaningless.
    bad_anchors = []
    for mid, _desc, relpath, old_text, _new in mutations:
        if mid in VOID:
            continue
        why = check_anchor(read_file(os.path.join(WORKTREE, relpath)), old_text)
        if why:
            bad_anchors.append(mid)
            print(f"ERROR: {mid} anchor in {relpath}: {why}")
    if bad_anchors:
        print(f"ANCHOR CHECK FAILED for {len(bad_anchors)} mutation(s): "
              f"{', '.join(bad_anchors)}. Nothing was run.")
        return 1
    n_checked = sum(1 for m in mutations if m[0] not in VOID)
    print(f"Anchor check: {n_checked} anchor(s), each exactly once and in code.")

    # Clean pycache and verify clean tree
    clean_pycache()
    if not selected and not check_clean():
        print("ERROR: tree is not clean in background_model/ or tests/. Aborting.")
        sys.exit(1)

    # Run baseline
    print("Running baseline tests...")
    baseline, baseline_rc, baseline_collection_failed = run_tests()
    if baseline_collection_failed:
        print(f"ERROR: baseline failed to collect (pytest rc={baseline_rc}). "
              f"Cannot compute a red/green matrix against a broken baseline.")
        return 1
    n_baseline_pass = sum(1 for v in baseline.values() if v == "passed")
    print(f"Baseline: {n_baseline_pass} passed, {len(baseline) - n_baseline_pass} other")

    results = {}

    for mid, desc, relpath, old_text, new_text in mutations:
        filepath = os.path.join(WORKTREE, relpath)
        original = read_file(filepath)

        if mid in VOID:
            if old_text in original:
                print(f"{mid}: VOID entry MATCHED in {relpath} -- re-examine it")
                results[mid] = {"desc": desc, "status": "void_matched", "red_tests": [],
                                "note": "VOID entry's text reappeared"}
            else:
                print(f"{mid}: VOID - {VOID[mid]}")
                results[mid] = {"desc": desc, "status": "void", "red_tests": [],
                                "note": VOID[mid]}
            continue

        if old_text not in original:
            print(f"{mid}: SKIP - old_text not found in {relpath}")
            results[mid] = {"desc": desc, "status": "skip", "red_tests": [], "note": "pattern not found"}
            continue

        # Apply mutation
        mutated = original.replace(old_text, new_text, 1)
        if mutated == original:
            print(f"{mid}: SKIP - replacement had no effect")
            results[mid] = {"desc": desc, "status": "skip", "red_tests": [], "note": "no effect"}
            continue

        write_file(filepath, mutated)
        clean_pycache()

        try:
            print(f"  {mid} ({desc}): running tests...", end="", flush=True)
            mut_results, mut_rc, mut_collection_failed = run_tests()

            missing_baseline_tests = [
                tname for tname, bstatus in baseline.items()
                if bstatus == "passed" and tname not in mut_results
            ]

            if mut_collection_failed or missing_baseline_tests:
                # Collection failure means pytest ran nothing, so every
                # baseline-passing test is ABSENT from mut_results -- that
                # used to be parsed as "status != passed" and counted as RED,
                # which is how a SyntaxError or ImportError in the mutated
                # module exited 0. Neither case is a catch; both are a
                # broken run.
                note_parts = []
                if mut_collection_failed:
                    note_parts.append(f"collection failed (pytest rc={mut_rc})")
                if missing_baseline_tests:
                    note_parts.append(
                        f"{len(missing_baseline_tests)} baseline test(s) "
                        f"absent from mutated results")
                note = "; ".join(note_parts)
                results[mid] = {"desc": desc, "status": "error", "red_tests": [],
                                "note": note}
                print(f" ERROR: {note}")
            else:
                # Find tests that went red (were passing in baseline, now
                # failing or erroring -- legitimately, since collection
                # succeeded and every baseline test is present).
                red_tests = []
                for tname, bstatus in baseline.items():
                    if bstatus == "passed":
                        mstatus = mut_results.get(tname, "missing")
                        if mstatus != "passed":
                            red_tests.append(tname)

                # Also find tests that went from failing to passing (unexpected)
                green_tests = []
                for tname, bstatus in baseline.items():
                    if bstatus != "passed":
                        mstatus = mut_results.get(tname, "missing")
                        if mstatus == "passed":
                            green_tests.append(tname)

                results[mid] = {
                    "desc": desc,
                    "status": "done",
                    "red_tests": red_tests,
                    "green_tests": green_tests,
                    "n_red": len(red_tests),
                }
                print(f" {len(red_tests)} red")
        except subprocess.TimeoutExpired:
            print(f" TIMEOUT")
            results[mid] = {"desc": desc, "status": "timeout", "red_tests": []}
        except Exception as e:
            print(f" ERROR: {e}")
            results[mid] = {"desc": desc, "status": "error", "red_tests": [], "note": str(e)}
        finally:
            # ALWAYS revert
            write_file(filepath, original)
            clean_pycache()

    # Verify tree clean after
    clean_after = check_clean()
    if not clean_after:
        print("\nWARNING: tree is NOT clean after mutation testing!")
    else:
        print("\nTree is clean after mutation testing.")

    # Print matrix
    print("\n" + "=" * 100)
    print("MUTATION TESTING RESULTS")
    print("=" * 100)
    print(f"{'ID':<6} {'Description':<45} {'#Red':>5}  {'Tests that went red'}")
    print("-" * 100)

    uncaught, skipped, broken, voided = [], [], [], []
    for mid, desc, _, _, _ in mutations:
        r = results.get(mid, {"status": "missing", "red_tests": [], "desc": desc})
        if r["status"] == "void":
            print(f"{mid:<6} {desc:<45} {'VOID':>5}  {r.get('note', '')}")
            voided.append(mid)
        elif r["status"] in ("skip", "void_matched"):
            print(f"{mid:<6} {desc:<45} {'SKIP':>5}  {r.get('note', '')}")
            skipped.append(mid)
        elif r["status"] == "timeout":
            print(f"{mid:<6} {desc:<45} {'T/O':>5}")
            broken.append(mid)
        elif r["status"] == "error":
            print(f"{mid:<6} {desc:<45} {'ERR':>5}  {r.get('note', '')}")
            broken.append(mid)
        else:
            n = r.get("n_red", 0)
            tests = ", ".join(r["red_tests"][:5])
            if len(r["red_tests"]) > 5:
                tests += f" (+{len(r['red_tests']) - 5} more)"
            print(f"{mid:<6} {desc:<45} {n:>5}  {tests}")
            if n == 0:
                uncaught.append(mid)

    n_red = sum(1 for m in mutations
                if results.get(m[0], {}).get("status") == "done"
                and results[m[0]].get("n_red", 0) > 0)
    print("\n" + "-" * 100)
    print(f"{len(mutations)} run: {n_red} RED, {len(voided)} VOID "
          f"({', '.join(voided) or '-'}), {len(skipped)} SKIPPED, "
          f"{len(uncaught)} UNCAUGHT, {len(broken)} timeout/error")
    if skipped:
        print(f"SKIPPED (search text matched nothing, or replacement had no "
              f"effect): {', '.join(skipped)}")
    if uncaught:
        print(f"UNCAUGHT MUTATIONS: {', '.join(uncaught)}")
    if broken:
        print(f"TIMEOUT/ERROR: {', '.join(broken)}")
    if not (skipped or uncaught or broken):
        print("ALL MUTATIONS MATCHED AND CAUGHT")

    # Dump full results to JSON
    out_path = os.path.join(WORKTREE, "scripts/_mutation_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results written to {out_path}")

    return 1 if (skipped or uncaught or broken or not clean_after) else 0


if __name__ == "__main__":
    sys.exit(main())
