# attic/v3_simulator — REFERENCE ONLY, NOT LIVE CODE

The v3-era simulator and its oracle/scoring cluster. Kept for reference during the
simulator rewrite. **Do not import from here, and do not treat anything here as
describing current behaviour.**

## Why this was moved

A model agent read `sim_fragments.py` as the live simulator and concluded the
simulator admits fragments by *containment* (`p ∈ [0, region_len - L]`, a
length-dependent support). It does not. The live path is
`background_model/simulator/weights.py::midpoint_index_arrays`, which gives
`j ∈ [0, region_len)` — length-INDEPENDENT. The agent's snippet had a scalar `L`
loop, which cannot be live code: `541941e` vectorised that across all 156 lengths.

The agent reached a confident wrong conclusion purely because this code was still
sitting in `scripts/` reading like production. That is the failure this directory
exists to prevent.

## What is here

| File | What it was |
|---|---|
| `sim_fragments.py` | The v3 simulator. Containment edge rule, superseded by the midpoint rule in `a352c34` |
| `sim_oracle.py` | Oracle NLL for the **v3** store. Its containment geometry was CORRECT for v3 |
| `sim_build_store.py` | Built v3 zarr stores from `sim_fragments.py` `.npz` output; the old 12-band track class |
| `sim_evaluate.py`, `sim_fit_gc_bias.py`, `sim_train_all.sh` | v3 evaluation / fitting / driver |
| `nb_oracle_*.py`, `score_v3nb_multinomial.py`, `_oracle_*.py` | The separate v3nb (negative-binomial) store's oracle and scoring |
| `_verify_oracle_alignment.py` | Independent coordinate-alignment check for `sim_oracle.py` (crop frame, shift-correlation peak at 0, uniform NLL through the frozen loss) |
| `_inspect_store_v3a.py`, `_write_oracle_json.py` | One-off v3_A probes: dump store facts; assemble `simulation_v3/A/oracle.json` from two `sim_oracle.py` runs |
| `_chain_1536.sh` | One-off 2026-09-26 chain: wait for the 1536 sim, build its store, submit the KEN run, compute its oracle anchors |

The four `_`-prefixed files above arrived later (owner, 2026-10-10). They had been
sitting UNTRACKED in the main checkout's `scripts/` (and one in a worktree's),
invoking paths this reorganisation moved — `python -m background_model.train` and
`scripts/sim_build_store.py`. Untracked and stale is the worst combination: git
holds no copy, so they cannot be recovered once lost, and nothing updates their
paths when the tree moves. They are kept here as a record of how the v3 anchors
were produced. **Their paths are NOT updated and they are not expected to run.**

Note `sim_oracle.py` here is the LATER version, which derives all geometry from the
store config at run time. The main checkout also held an untracked earlier copy
with the geometry hardcoded (`l_target=2304`, `tile_size=2048`); it was not
imported here, because a second, worse copy of one rule is the hazard this
directory exists to prevent.

None of these import `background_model.simulator`, and nothing outside this
directory imported them at the time of the move — that is why the move was safe.

## What was deliberately LEFT in scripts/

- **`cut_site_oracle.py`** — the CURRENT oracle. It imports `build_region_weights`
  and `precompute_region` from the live simulator, so it inherits the live geometry
  rather than restating it. Not v3.
- **`analyze_hexamer_distributions.py`** — contains a copy of `build_w6` marked
  "copied from sim_fragments.py (DO NOT MODIFY)". It is self-contained (no import),
  so it still runs, but it carries a snapshot of retired code. If it outlives the
  rewrite, that copy is the next thing to mislead someone. *(Since moved to
  `attic/hexamer_prior_pipeline/scripts/` with the counter whose Parquet it
  reads, owner decision 189.)*

## Still-live files that narrate the retired rule

Moving these scripts does NOT fix every source of the same confusion. Two
surviving files still describe containment as though it were current:

- `background_model/simulator/weights.py` (module docstring, ~line 33) — mentions
  `region_len - L + 1` valid positions.
- `tests/test_simulator_weights.py` (~line 530) — "the old containment rule gave
  `region_len - L + 1` positions".

The test comment is accurate and load-bearing (it documents what changed). The
`weights.py` docstring is the one worth rewriting, since it sits in the live file.
