# attic/pre_rewrite_simulator_docs — SUPERSEDED, NOT A DESCRIPTION OF THE CODE

Quarantined 2026-10-07 by owner decision. **Nothing here describes current
behaviour.** `docs/pending/simulator_spec.md` is the authority on the simulator;
the implementation is `background_model/hexamers.py`,
`background_model/cut_site_stats.py` and `background_model/simulator/draw.py`
(one file, `count_hexamers_rdf.py`, until 2026-10-09). The code these documents
describe is in `attic/pre_rewrite_simulator/`.

These were moved rather than deleted because all four were **untracked**, so
there was no git history to recover them from, and `simulator_basic_inputs.md`
is the only surviving record of the design that decisions 126-134 were written
against. Deletion would have been irreversible; this is not. Same reasoning as
`attic/v3_simulator`.

## Why each is here

| File | Superseded by | How it misleads |
|---|---|---|
| `simulator_basic_inputs.md` (46 KB) | `simulator_spec.md` | Treats capture/GC modelling and a `predict(L, gc)` surface as live — both dropped by owner decision — and states admission by **midpoint**. Admission is start-in-region. A research agent read this and reported capture as live; the error was caught only because the coordinator re-checked. |
| `h5_derived_counts_and_fl.md` (25 KB) | `simulator_basic_inputs.md`, itself superseded | Same capture/midpoint/count assumptions, one generation further back. |
| `simulator_output_design.md` (43 KB) | nothing — the implementation did not follow it | Written 2026-10-06 for the emit path, **never reviewed**, and part of it was mid-edit when its agent was stopped. It proposes a two-file output contract with an HDF5 metadata sidecar. **None of that exists.** What was actually built is `simulate_fragments_to_bed` (`a13d25a`), a single 8-column BED. Its verified `file:line` research on the h5 writer is still useful; its design is not what shipped. |
| `count_hexamers.py` (311 lines) | `count_hexamers_rdf.py` | Hand-rolled predecessor of the counting path, using **midpoint** membership. The library path is also faster (4.08 vs 4.31 ms/region), so there is no performance argument for keeping a second implementation of one rule. |

## The pattern these four illustrate

Four stale simulator documents accumulated in `docs/pending/` while the code was
rewritten underneath them, and `background_model/simulator/__init__.py` still
points readers at a doc asserting midpoint admission. Two separate agents reached
confident wrong conclusions from stale files that were still sitting in live
directories. A superseded design belongs here, not next to the current one.
