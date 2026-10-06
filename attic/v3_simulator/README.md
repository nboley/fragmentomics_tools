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

None of these import `background_model.simulator`, and nothing outside this
directory imported them at the time of the move — that is why the move was safe.

## What was deliberately LEFT in scripts/

- **`cut_site_oracle.py`** — the CURRENT oracle. It imports `build_region_weights`
  and `precompute_region` from the live simulator, so it inherits the live geometry
  rather than restating it. Not v3.
- **`analyze_hexamer_distributions.py`** — contains a copy of `build_w6` marked
  "copied from sim_fragments.py (DO NOT MODIFY)". It is self-contained (no import),
  so it still runs, but it carries a snapshot of retired code. If it outlives the
  rewrite, that copy is the next thing to mislead someone.

## Still-live files that narrate the retired rule

Moving these scripts does NOT fix every source of the same confusion. Two
surviving files still describe containment as though it were current:

- `background_model/simulator/weights.py` (module docstring, ~line 33) — mentions
  `region_len - L + 1` valid positions.
- `tests/test_simulator_weights.py` (~line 530) — "the old containment rule gave
  `region_len - L + 1` positions".

The test comment is accurate and load-bearing (it documents what changed). The
`weights.py` docstring is the one worth rewriting, since it sits in the live file.
