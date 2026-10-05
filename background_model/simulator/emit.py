"""BED emission and manifest — Step 6 of the simulator.

Writes one sorted, bgzipped, tabix-indexed 8-column BED per sample, then
(optionally) calls ``build-fragments-h5`` to produce the fragment h5.

The manifest carries the three factor arrays (hexamer tables as DataFrames
keyed by the 6-mer string, the ``(L, gc_bin)`` predict LUT, and
``marginal_fl``) plus provenance fields.  Together these are sufficient to
reconstruct ``w`` via ``build_region_weights`` — the load-bearing property
of the output.

Hexamer table format (owner decision 81): DataFrames keyed by the hexamer
STRING, not bare 4096-element arrays ordered by an implicit integer code.
This makes the k-mer ordering a private detail on each side; a mismatched
table fails to join rather than silently misaligning.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from background_model.simulator.precompute import NHEX, hexamer_vocabulary
from background_model.simulator.weights import (
    L_MAX,
    L_MIN,
    HexamerTables,
)

# Module-level cache (computed once).
_VOCAB: Optional[np.ndarray] = None


def _get_vocab() -> np.ndarray:
    global _VOCAB
    if _VOCAB is None:
        _VOCAB = hexamer_vocabulary()
    return _VOCAB


# ── hexamer tables <-> DataFrames ────────────────────────────────────────

_VALID_TABLE_NAMES = frozenset(HexamerTables._fields)


def hex_table_to_dataframe(table: np.ndarray, table_name: str) -> pd.DataFrame:
    """Convert a 4096-element weight array to a DataFrame keyed by hexamer string.

    The ``table_name`` column travels with the data so that a DataFrame loaded
    into the wrong slot (e.g. ``start_fwd`` loaded as ``end_fwd``) is detected
    on reconstruction rather than silently misrouting weights.

    Parameters
    ----------
    table : ndarray, shape (4096,)
        Weight values indexed by the integer hexamer code.
    table_name : str
        One of ``"start_fwd"``, ``"end_fwd"``, ``"start_rev"``, ``"end_rev"``.

    Returns
    -------
    DataFrame with columns ``["hexamer", "weight", "table_name"]``, 4096 rows.
    """
    if table_name not in _VALID_TABLE_NAMES:
        raise ValueError(
            f"table_name {table_name!r} not in {sorted(_VALID_TABLE_NAMES)}"
        )
    vocab = _get_vocab()
    return pd.DataFrame({
        "hexamer": [v.decode() for v in vocab],
        "weight": table,
        "table_name": table_name,
    })


def dataframe_to_hex_table(
    df: pd.DataFrame, expected_name: Optional[str] = None,
) -> np.ndarray:
    """Reconstruct a 4096-element weight array from a hexamer-keyed DataFrame.

    The join is on the hexamer STRING — if the DataFrame was produced under a
    different k-mer convention, the join will misalign and weights will be wrong,
    which is exactly what the string key exists to make detectable (a bare array
    would silently misalign).

    Parameters
    ----------
    df : DataFrame
        Must have columns ``"hexamer"`` and ``"weight"``, and optionally
        ``"table_name"`` (added by ``hex_table_to_dataframe``).
    expected_name : str, optional
        The slot this table is being loaded into (e.g. ``"start_fwd"``).
        If the DataFrame carries a ``table_name`` column, every row's value
        must equal *expected_name*; a mismatch means the table was built for
        a different slot and is being loaded into the wrong one.

    Raises ``ValueError`` if the DataFrame does not contain exactly the 4096
    expected hexamers, or if the embedded table name disagrees with the slot.
    """
    if len(df) != NHEX:
        raise ValueError(
            f"DataFrame has {len(df)} rows, expected exactly {NHEX}. "
            f"A truncated table yields zero-weight hexamers that are "
            f"undetectable downstream (normalisation preserves Σ w = 1)."
        )

    hex_col = df["hexamer"]
    n_unique = hex_col.nunique()
    if n_unique != NHEX:
        raise ValueError(
            f"DataFrame has {n_unique} unique hexamer strings but expected "
            f"{NHEX}. Duplicates or missing entries make the table silently wrong."
        )

    # Slot-swap guard: if the DataFrame carries an embedded table_name,
    # it must match the slot we are loading it into.
    if expected_name is not None and "table_name" in df.columns:
        embedded_names = df["table_name"].unique()
        if len(embedded_names) != 1 or embedded_names[0] != expected_name:
            raise ValueError(
                f"Hex table slot mismatch: DataFrame carries "
                f"table_name={embedded_names.tolist()!r} but is being loaded "
                f"into slot {expected_name!r}. This means the tables were "
                f"swapped — the weights would be silently misrouted."
            )

    # Reindex onto the vocabulary ORDER, rather than sorting the frame.
    # Sorting by the hexamer string happens to give the same result today
    # because `hexamer_indices` is a base-4 ACGT encoder, so its index order
    # coincides with lexicographic order -- but nothing enforces that, and
    # `hexamer_vocabulary` exists specifically so no caller has to depend on
    # it.  Binding to the vocabulary keeps the encoding owned in one place.
    vocab_strs = [v.decode() for v in _get_vocab()]
    weights = df.set_index("hexamer")["weight"].reindex(vocab_strs)

    missing = weights.isna()
    if missing.any():
        names = missing.index[missing].tolist()
        raise ValueError(
            f"{int(missing.sum())} vocabulary hexamer(s) absent from the "
            f"DataFrame, e.g. {names[:5]} — not in the vocabulary derived "
            f"from hexamer_indices. NaN here is deliberate: a zero-filled "
            f"array would leave these at 0.0, which normalisation hides."
        )
    return weights.to_numpy(dtype=np.float64)


def hex_tables_to_dict(
    hex_tables: HexamerTables,
) -> Dict[str, pd.DataFrame]:
    """Convert all four hexamer tables to a dict of DataFrames."""
    return {
        name: hex_table_to_dataframe(getattr(hex_tables, name), name)
        for name in HexamerTables._fields
    }


def dict_to_hex_tables(d: Dict[str, pd.DataFrame]) -> HexamerTables:
    """Reconstruct ``HexamerTables`` from a dict of DataFrames.

    Each DataFrame is loaded into the slot whose key matches its dict key,
    and ``dataframe_to_hex_table`` validates that the embedded ``table_name``
    (if present) agrees with the slot.  A swapped pair (e.g. ``start_fwd``
    data loaded into the ``end_fwd`` slot) raises ``ValueError``.
    """
    return HexamerTables(**{
        name: dataframe_to_hex_table(d[name], expected_name=name)
        for name in HexamerTables._fields
    })


# ── BED emission ─────────────────────────────────────────────────────────

_MAPQ = 60  # All simulated reads get MAPQ 60.


def write_bed(
    bed_path: str,
    contig: str,
    gstart: int,
    starts: np.ndarray,
    stops: np.ndarray,
    strands: np.ndarray,
) -> None:
    """Append fragment records to a BED file (unsorted, uncompressed).

    8-column BED format:
    ``contig  start  stop  name  score  strand  mapq1  mapq2``

    The ``start``/``stop`` are region-local; ``gstart`` is added to convert
    to genomic coordinates.
    """
    with open(bed_path, "a") as f:
        for i in range(len(starts)):
            gs = gstart + int(starts[i])
            ge = gstart + int(stops[i])
            strand = str(strands[i])
            f.write(
                f"{contig}\t{gs}\t{ge}\t.\t0\t{strand}\t{_MAPQ}\t{_MAPQ}\n"
            )


def sort_bgzip_tabix(
    bed_path: str,
    bgzip_path: Optional[str] = None,
) -> str:
    """Sort a BED file, bgzip it, and create a tabix index.

    Preflight: verifies that ``sort``, ``bgzip``, and ``tabix`` are on
    ``PATH`` before doing any work.  CLAUDE.md warns these binaries are
    frequently absent in sandboxes and Batch containers; a missing binary
    raises ``FileNotFoundError`` AFTER the entire sampling run has been
    paid for.  Checking here fails fast with a clear message.

    The sort uses ``LC_ALL=C`` so that contig collation is byte-order
    regardless of the ambient locale — ``tabix`` requires a specific
    lexicographic order, and locale-dependent sort (e.g. ``en_US.UTF-8``)
    can reorder contigs.

    Parameters
    ----------
    bed_path : str
        Path to the unsorted BED file.
    bgzip_path : str, optional
        Output path for the bgzipped file.  Defaults to ``bed_path + ".gz"``.

    Returns
    -------
    str
        Path to the bgzipped file.

    Raises
    ------
    FileNotFoundError
        If any of ``sort``, ``bgzip``, ``tabix`` is not found on ``PATH``.
    """
    missing = [cmd for cmd in ("sort", "bgzip", "tabix")
               if shutil.which(cmd) is None]
    if missing:
        raise FileNotFoundError(
            f"Required binaries not found on PATH: {missing}. "
            f"In conda envs they ship in bin/; in Batch containers "
            f"they may need to be installed or added to PATH."
        )

    if bgzip_path is None:
        bgzip_path = bed_path + ".gz"

    sorted_path = bed_path + ".sorted"
    # sort by contig (lexicographic) then by start (numeric).
    # LC_ALL=C forces byte-order collation — required by tabix.
    sort_env = {**os.environ, "LC_ALL": "C"}
    subprocess.run(
        ["sort", "-k1,1", "-k2,2n", bed_path, "-o", sorted_path],
        check=True,
        env=sort_env,
    )
    # bgzip
    with open(bgzip_path, "wb") as out_f:
        subprocess.run(
            ["bgzip", "-c", sorted_path],
            stdout=out_f,
            check=True,
        )
    # tabix
    subprocess.run(
        ["tabix", "-p", "bed", bgzip_path],
        check=True,
    )
    # clean up intermediates
    os.remove(bed_path)
    os.remove(sorted_path)
    return bgzip_path


# ── manifest ─────────────────────────────────────────────────────────────

def _hash_file(path: str) -> str:
    """SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _hash_array(arr: np.ndarray) -> str:
    """SHA-256 hex digest of an array's raw bytes."""
    return hashlib.sha256(arr.tobytes()).hexdigest()


class ManifestMismatch(RuntimeError):
    """A manifest is being loaded into an environment it was not built against.

    An error rather than a warning because ``w`` is RECONSTRUCTED from the
    manifest rather than stored (owner decision 79e).  The hashes are the only
    thing standing between "the same weights the sampler drew from" and
    "different weights that still satisfy every invariant": a mismatched
    reference or region set changes ``hex_fwd``/``hex_rc``/``cum_gc`` and
    therefore changes ``w``, while ``Sum_Omega w == 1`` and the exact-half
    strand marginal still hold afterwards.  Nothing downstream would notice.
    """


def _repo_root() -> str:
    """Return the repo root inferred from this file's location.

    ``emit.py`` is at ``<repo>/background_model/simulator/emit.py``, so the
    repo root is three directories up.
    """
    return os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)
    )))


def _to_repo_relative(path: str, repo_root: Optional[str] = None) -> str:
    """Convert *path* to a repo-relative form.

    Resolution rule (Finding 1 fix):

    - **Absolute** paths are used as-is.
    - **Relative** paths are resolved against *repo_root*, NOT the current
      working directory.  This is the key invariant: a manifest stores
      repo-relative paths (e.g. ``scripts/run_simulator.py``), and at load
      time ``git_blob_sha`` passes that path back here.  If resolution used
      ``os.path.abspath`` (which prepends CWD), the path would only resolve
      when CWD happens to be the repo root — but production Batch jobs run
      from ``cd /tmp`` with ``PYTHONPATH`` set, so CWD is almost never the
      repo root.

    Raises ``ValueError`` if the result escapes the repo root (starts with
    ``..``), which means the file is outside the repository and cannot be
    addressed by ``git rev-parse HEAD:<relpath>``.
    """
    if repo_root is None:
        repo_root = _repo_root()
    if os.path.isabs(path):
        abs_path = path
    else:
        abs_path = os.path.join(repo_root, path)
    rel = os.path.relpath(abs_path, repo_root)
    if rel.startswith(".."):
        raise ValueError(
            f"Path {path!r} resolves outside the repo root {repo_root!r} "
            f"(relative: {rel!r}). The manifest records repo-relative paths "
            f"so that provenance checks work from any checkout."
        )
    return rel


def git_blob_sha(path: str, repo_root: Optional[str] = None) -> Optional[str]:
    """Git blob SHA of ``path`` at HEAD, or ``None`` if untracked/uncommitted.

    *path* may be absolute or repo-relative.  It is resolved to a repo-relative
    form internally.  If the path is outside the repo, ``ValueError`` is raised
    rather than silently returning ``None`` — a path that cannot be resolved
    must not degrade to "unverified and silent".

    The BLOB sha, not the repo commit sha, deliberately: a commit sha changes
    on any commit anywhere in the repo, so verifying against it would reject
    manifests over edits that cannot possibly affect this file.  The blob sha
    changes if and only if this file's content changes, which is the question
    actually being asked.

    Returns ``None`` when the file is untracked or uncommitted.  That is a
    normal state for a driver under development, so it is recorded as ``None``
    rather than raising -- but it is recorded, so the gap is visible in the
    manifest instead of being indistinguishable from a file that was never
    considered.
    """
    if repo_root is None:
        repo_root = _repo_root()
    rel = _to_repo_relative(path, repo_root)
    try:
        out = subprocess.run(
            ["git", "-C", repo_root, "rev-parse", f"HEAD:{rel}"],
            capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


#: Manifest format version.  BUMP THIS when a field that ``w`` reconstruction
#: depends on is added or changes meaning, and teach ``load_manifest`` to reject
#: the versions it can no longer interpret.  v1 -> v2 added ``pad``: a v1
#: manifest does not record the geometry, so reconstructing from it would
#: silently fall back to whatever ``MAX_FL_HALF`` happens to be in the code
#: doing the loading.
MANIFEST_VERSION = 2


def write_manifest(
    manifest_path: str,
    *,
    hex_tables: HexamerTables,
    predict_lut: np.ndarray,
    marginal_fl: np.ndarray,
    region_set_name: str,
    region_set_hash: str,
    reference_name: str,
    reference_hash: str,
    region_len: int,
    pad: int,
    l_min: int = L_MIN,
    l_max: int = L_MAX,
    fl_bands: Sequence[Tuple[int, int]] = ((25, 110), (110, 180)),
    per_region_counts: Dict[str, int] | None = None,
    rng_seed: int | None = None,
    commit_sha: str | None = None,
    simulator_script: str | None = None,
    simulator_script_sha: str | None = None,
) -> None:
    """Write the simulation manifest to a JSON file.

    The manifest carries the realised factor arrays (not the recipe) plus
    provenance fields, so that ``w`` can be reconstructed from the manifest
    alone via ``build_region_weights``.

    Hexamer tables are stored as DataFrames keyed by the hexamer string
    (owner decision 81).
    """
    # Convert hex tables to JSON-serialisable dicts
    hex_dfs = hex_tables_to_dict(hex_tables)
    hex_json = {
        name: df.to_dict(orient="list")
        for name, df in hex_dfs.items()
    }

    # Store the simulator script as a repo-relative path so provenance
    # checks work from any checkout or worktree, not just the one that
    # wrote the manifest.
    script_rel: Optional[str] = None
    if simulator_script is not None:
        script_rel = _to_repo_relative(simulator_script)

    manifest = {
        "version": MANIFEST_VERSION,
        "hex_tables": hex_json,
        "predict_lut": predict_lut.tolist(),
        "marginal_fl": marginal_fl.tolist(),
        "region_set_name": region_set_name,
        "region_set_hash": region_set_hash,
        "reference_name": reference_name,
        "reference_hash": reference_hash,
        "region_len": region_len,
        # Recorded EXPLICITLY rather than derived as l_max // 2.  The identity
        # happens to hold for the default pad, but `midpoint_index_arrays`
        # deliberately refuses a pad default precisely because re-deriving the
        # geometry is the failure mode; a manifest that stores the value
        # actually used does not care whether the identity still holds.
        "pad": pad,
        "l_min": l_min,
        "l_max": l_max,
        "fl_bands": [list(b) for b in fl_bands],
        "per_region_counts": per_region_counts,
        "rng_seed": rng_seed,
        "commit_sha": commit_sha,
        # Which script produced this, and the git blob sha of that exact file.
        # `commit_sha` is repo-wide and moves on unrelated commits; this pair
        # names one file and changes only when that file changes.
        # Stored as a REPO-RELATIVE path so the check works from any checkout.
        "simulator_script": script_rel,
        "simulator_script_sha": (
            simulator_script_sha
            if simulator_script_sha is not None
            else (git_blob_sha(simulator_script) if simulator_script else None)
        ),
    }

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)


class ManifestVerificationIncomplete(ValueError):
    """A manifest records a provenance check but the caller did not supply
    the input needed to run it.

    This is distinct from ``ManifestMismatch`` (a check ran and disagreed).
    Here the check COULD NOT run because the caller omitted the input.

    This exception enforces the **mandatory** tier of the provenance contract:
    ``reference`` and ``region_set``.  If the manifest records a hash for
    either of these and the caller does not supply the corresponding path,
    ``load_manifest`` raises this exception — a skipped check is genuinely
    unrepresentable for these inputs.

    The ``simulator_script`` check is **best-effort**: if ``git`` cannot
    resolve the recorded path, the check does not run and
    ``"simulator_script:unresolvable"`` is recorded in ``verified`` instead.
    See ``load_manifest`` for the full contract.
    """


def load_manifest(
    manifest_path: str,
    *,
    fasta_path: Optional[str] = None,
    region_set_path: Optional[str] = None,
    verify: bool = True,
) -> Dict[str, Any]:
    """Load a simulation manifest, reconstruct the factors, and VERIFY provenance.

    Verifying, not merely recording.  ``w`` is reconstructed from this manifest
    rather than stored (owner decision 79e), so reconstruction fidelity is the
    deliverable.  A manifest built against one hg38 patch and reloaded against
    another yields different ``hex``/``cum_gc`` and therefore different ``w``,
    and every invariant -- ``Sum_Omega w == 1``, the exact-half strand marginal,
    the domain size -- still passes.  The recorded hashes are the only detector.

    When ``verify=True`` (the default), the provenance contract has two tiers:

    **Mandatory** (``reference``, ``region_set``):
        If the manifest records a hash, the caller MUST supply the
        corresponding path.  Omitting it raises
        ``ManifestVerificationIncomplete`` — a skipped check is genuinely
        unrepresentable for these inputs.

    **Best-effort** (``simulator_script``):
        If the manifest records a ``simulator_script`` with a
        ``simulator_script_sha``, the loader calls ``git_blob_sha`` to
        resolve it.  A resolved sha that disagrees is a hard error
        (``ManifestMismatch``).  But if ``git_blob_sha`` returns ``None``
        — because ``git`` is not installed (AWS Batch containers) or the
        script is not tracked — the check does not run and
        ``"simulator_script:unresolvable"`` is appended to ``verified``.
        That entry means *the check did not run*, not that it passed.

    :param verify: set False to load without checking. Explicit, so that
        bypassing provenance is a visible decision at the call site.
    :raises ManifestMismatch: if any check that ran disagreed.
    :raises ManifestVerificationIncomplete: if a mandatory check could not
        run because the caller did not supply the input.
    """
    with open(manifest_path) as f:
        raw = json.load(f)

    # ── format version: checked ALWAYS, independent of `verify` ───────────
    # `verify` governs provenance HASHES — whether the recorded inputs are the
    # inputs on disk.  The version governs whether this file can be interpreted
    # at all, which is a prior question.  A v1 manifest records no `pad`, so
    # reconstructing `w` from it would silently adopt whatever MAX_FL_HALF the
    # loading code happens to hold; that is the gap the field was added to
    # close, so `verify=False` must not reopen it.
    version = raw.get("version")
    if version != MANIFEST_VERSION:
        raise ManifestMismatch(
            f"Unsupported manifest version in {manifest_path}.\n"
            f"  recorded : {version!r}\n"
            f"  supported: {MANIFEST_VERSION}\n"
            f"v1 manifests predate the `pad` field and therefore do not record "
            f"the midpoint geometry. Reconstructing w from one would use the "
            f"loader's own MAX_FL_HALF rather than the value the sampler drew "
            f"with, and every normalisation invariant would still hold — so the "
            f"error would be silent. Re-run the simulation to obtain a v"
            f"{MANIFEST_VERSION} manifest; there is deliberately no bypass."
        )

    verified: List[str] = []
    if verify:
        # ── file-hash checks: reference and region_set ────────────────
        checks = (
            ("reference", fasta_path, raw.get("reference_hash"),
             raw.get("reference_name")),
            ("region_set", region_set_path, raw.get("region_set_hash"),
             raw.get("region_set_name")),
        )
        for label, path, recorded, name in checks:
            if recorded is None:
                # Manifest does not claim this check — nothing to do.
                continue
            if path is None:
                raise ManifestVerificationIncomplete(
                    f"Manifest records a {label} hash ({name}: {recorded[:16]}…) "
                    f"but no {label} path was supplied to load_manifest. "
                    f"Pass the path to verify, or pass verify=False to skip "
                    f"all checks."
                )
            actual = _hash_file(path)
            if actual != recorded:
                raise ManifestMismatch(
                    f"{label} hash mismatch for {manifest_path}.\n"
                    f"  manifest recorded : {recorded}  ({name})\n"
                    f"  file on disk      : {actual}  ({path})\n"
                    f"Reconstructing w against this file would give DIFFERENT "
                    f"weights from the ones the sampler drew from, and every "
                    f"normalisation invariant would still hold. Pass "
                    f"verify=False only if you intend that."
                )
            verified.append(label)

        # ── simulator script check ────────────────────────────────────
        script = raw.get("simulator_script")
        recorded_sha = raw.get("simulator_script_sha")
        if script and recorded_sha:
            actual_sha = git_blob_sha(script)
            if actual_sha is not None and actual_sha != recorded_sha:
                raise ManifestMismatch(
                    f"simulator script changed since this manifest was written.\n"
                    f"  script            : {script}\n"
                    f"  manifest recorded : {recorded_sha}\n"
                    f"  current blob sha  : {actual_sha}\n"
                    f"The code that produced these fragments is not the code "
                    f"present now. Pass verify=False to load anyway."
                )
            if actual_sha is not None:
                verified.append("simulator_script")
            else:
                # Script path is recorded but unresolvable (e.g. not in a
                # git repo, or on a Batch container with no git).  Record
                # the gap explicitly so it is visible, not silent.
                verified.append("simulator_script:unresolvable")

    # Reconstruct hex tables from string-keyed DataFrames
    hex_dfs = {
        name: pd.DataFrame(raw["hex_tables"][name])
        for name in HexamerTables._fields
    }
    hex_tables = dict_to_hex_tables(hex_dfs)

    predict_lut = np.array(raw["predict_lut"], dtype=np.float64)
    marginal_fl = np.array(raw["marginal_fl"], dtype=np.float64)

    return {
        "hex_tables": hex_tables,
        "predict_lut": predict_lut,
        "marginal_fl": marginal_fl,
        "region_set_name": raw["region_set_name"],
        "region_set_hash": raw["region_set_hash"],
        "reference_name": raw["reference_name"],
        "reference_hash": raw["reference_hash"],
        "region_len": raw["region_len"],
        # Required, not .get() -- the version guard above has already rejected
        # any manifest that predates this field, so a KeyError here would mean
        # a v2 manifest was written without it.
        "pad": raw["pad"],
        "l_min": raw["l_min"],
        "l_max": raw["l_max"],
        "fl_bands": [tuple(b) for b in raw["fl_bands"]],
        "per_region_counts": raw.get("per_region_counts"),
        "rng_seed": raw.get("rng_seed"),
        "commit_sha": raw.get("commit_sha"),
        "simulator_script": raw.get("simulator_script"),
        "simulator_script_sha": raw.get("simulator_script_sha"),
        # Which provenance checks ACTUALLY ran. Empty means nothing was
        # compared -- either verify=False, or no paths were supplied.
        "verified": verified,
    }


def build_fragments_h5(
    bed_gz_path: str,
    h5_path: str,
    fasta_path: str,
) -> None:
    """Call ``build-fragments-h5`` to produce a fragment h5 from a bgzipped BED.

    This delegates to the production tool, which computes GC from the FASTA
    (not from the BED).  GC in the h5 therefore comes from the real reference
    through production code — the simulator does not emit GC.
    """
    subprocess.run(
        [
            "build-fragments-h5",
            bed_gz_path,
            h5_path,
            "--fasta", fasta_path,
        ],
        check=True,
    )
