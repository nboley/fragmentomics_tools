# Critical Review: `bias_correction` (Background Model)

**Date:** 2026-08-25
**Scope:** `fragmentomics_tools/bias_correction/` (model.py, data.py, layers.py, loss.py, train.py, predict.py) plus integration points in `dataframe.py` and `fragment_array.py`
**Grade: D** — the core modeling idea is sound and worth reviving, but the code is research-notebook-quality: two loss families crash in training, the QC/stats path is stale against the training path, the weight-application code has a silent selection-logic bug, and there are zero tests.

---

## 1. Crash Bugs (code paths that cannot run)

| # | Location | Bug |
|---|----------|-----|
| C1 | `model.py:228-243` (`training_step`/`validation_step`) | Passes `total_count=` to the loss, but `NegativeBinomialNLLLoss.forward` (loss.py:151) and `NegativeBinomialNLLLossOld.forward` (loss.py:205) and `NegativeBinomialFixedTotalCountNLLLoss.forward` (loss.py:241) accept only `(input, target)` → **TypeError**. Only `multinomial`/`binomial` losses can train at all. |
| C2 | `model.py:86,90` | `logits_to_probs` used in `_build_dist` for the `negative_binomial_old`/`_fixed_total_count` branches but never imported in model.py → **NameError**. |
| C3 | `predict.py:89` | `self.max()` inside free function `make_tracks` → **NameError** whenever `sharey=True`. |
| C4 | `loss.py:98` | `sqrt` never imported → **NameError** if `scale_loss_to_d=True`. |
| C5 | `layers.py:70` | `coo_matrix` never imported → **NameError** on the sparse path of `jitter_matrix`. |
| C6 | `data.py:53-55` (`build_gene_coverage_counts`) | `build_counts(record)["a", "b"]` indexes a Series with a tuple → **KeyError** (needs a list: `[["a", "b"]]`). |
| C7 | `dataframe.py:1875` (`_set_fragment_array_weights_from_pred_record`) | Starts with `assert False` — intentionally disabled dead code left in place. |
| C8 | `train.py:26-28` | `sys.path.insert` of `/home/nboley/src/biomarker-projects/projects/` + imports from `fragmentomics.data`/`fragmentomics.lib` — a hard dependency on an uninstalled external repo under another user's home dir. Training is unrunnable as checked in. |

The pattern in C1-C6: every branch except the one exercised in the last experiment (`multinomial`) has rotted. The NB code paths are load-bearing in `train.py:180` (fine-tuning from an NB checkpoint) yet cannot execute.

## 2. Silent Wrong-Result Bugs

| # | Location | Bug |
|---|----------|-----|
| S1 | `dataframe.py:1928-1940` (`_set_fragment_array_weights_from_weights_record`) | Fragment selection mask is `((fl >= lb) & (fl <= ub)) \| (strand == s)` — strand is **OR'd** where it must be **AND'd**. A fragment on the matching strand but outside the length band still gets a weight; a fragment in the band gets weights from *both* strands' tracks (later loop iterations silently overwrite earlier ones since masks overlap across strand/band/cov-type combinations). The final weight per fragment depends on dict/loop iteration order. This is the function that applies the background model to data — the whole point of the package — and it assigns wrong weights silently. |
| S2 | `model.py:294` (`predict_from_fasta`) | Column list is `["pred_dist." for c in self.output_columns]` — every column gets the identical name `"pred_dist."` (missing `+ c`). Downstream `df[col]` returns a DataFrame instead of a Series, or the wrong track. |
| S3 | `model.py:93-95` vs `predict.py:55-71` | Multinomial dist is built with `n=1`, so `dist.mean()` is a probability vector summing to 1, while `calc_stats` compares `dist.mean().sum()` (≡1) against raw observed counts and `make_qc_plots` plots them as "Counts". Ratio-based panels survive; absolute-count panels are meaningless. No rescaling by observed total anywhere in the stats path. |
| S4 | `data.py:73` | `self.total_count = 1000` hardcoded; the real per-sample computation is commented out on the same line. Affects binomial loss semantics and any total-count-dependent likelihood. |
| S5 | `train.py:62-78, 100-108` | `sample(...)` calls without `random_state` in `build_ctcf_rdfs`/`build_promoter_rdfs` → non-reproducible train/val splits (only `build_gene_rdfs` seeds). |
| S6 | `model.py:326` | `predict_weights_from_rdf` clips weights to hardcoded `[0.5, max_scaling_factor]`, while `fragment_array.from_fname` exposes `min_background_scaling_factor`/`max_background_scaling_factor` — two disconnected clamp mechanisms for the same concept. |

## 3. API Drift / Two Incompatible Generations Coexisting

This is the strongest evidence a rewrite is warranted — the package contains **two generations of the same system** interleaved:

1. **Track naming schemes.** The old scheme (`fwd_start_counts`, `bkwd_stop_counts`, ...) is the default `output_columns` in `BackgroundModelModule.__init__` (model.py:177) and the only scheme `calc_stats`/`make_qc_plots` (predict.py) understand. The new scheme (`strand_+__fl_40_65__coverage_first`, ...) is what `data.py` produces and `train.py` uses. **A model trained with the current pipeline cannot be QC'd with the current QC code.**
2. **Ghost class.** `fragment_array.py:1858` type-hints `background_model: Optional["SeqToEndpointsMultiResModel"]` — a class that exists nowhere in the repo. The real class is `BackgroundModelModule`. The integration point in the core data structure references a deleted prior iteration.
3. **Dead vs. live weight appliers.** `_set_fragment_array_weights_from_pred_record` (assert-False'd) vs `_set_fragment_array_weights_from_weights_record` (live, but with bug S1) — the pred-record path also hardcodes fl bands while the weights path reads them from column names.
4. **Five loss variants** (`multinomial`, `binomial`, `negative_binomial`, `negative_binomial_old`, `negative_binomial_fixed_total_count`), of which three crash (C1) and one is explicitly "old". This is an experiment log, not an API.

## 4. Design Issues

- **`FragmentEndpointsDataset.__init__` does everything eagerly** (data.py:70-120): cross-joins samples × regions, loads and merges all fragment arrays, one-hot-encodes all sequence, attaches blacklists — all in RAM, recomputed on every run, no on-disk cache. For 10k regions × N samples this is the dominant cost and it's unresumable.
- **`.cuda()` calls inside `Dataset.__getitem__`** (data.py:152-154) — breaks `num_workers>0` (CUDA-in-fork), pins the dataset to GPU, and defeats Lightning's device management. Also allocates the constant `total_count` tensor per item.
- **Lightning misuse:** no `save_hyperparameters()`; custom `save()`/`load()` smuggle `model_params` inside the weights `state_dict` (model.py:158-171). This breaks under `torch>=2.6` where `torch.load` defaults to `weights_only=True` (non-tensor dict entry → load failure), and forfeits Lightning checkpointing/resume entirely.
- **Fake schema plumbing:** data.py:102-104 sets `sample_id="merged"`, `frag_h5="none"` just to satisfy `SampleAndRegionDataFrame`'s required columns — the abstraction doesn't fit the merged-sample use case.
- **Magic constants in the NB parameterization:** `sigmoid(x - 2)`, `exp(-(x - 6))` (loss.py:147-148, model.py:152-156) — mean is structurally bounded to (0,1) with no comment on why; parameterization is duplicated between loss.py and model.py and can drift.
- **Mutable default argument** for `output_columns` (model.py:177).
- **Import-time side effects:** `warnings.filterwarnings` at model.py module level; `os.environ["NCCL_P2P_DISABLE"]`, `set_start_method`, `set_float32_matmul_precision` at train.py import.
- **Dead code / debris:** `train.py:173` `return` mid-`main()` with unreachable fine-tuning loop after it; commented-out MultiIndex columns (model.py:313-316); duplicated "Building observed arrays (4/5)" print (model.py:347,354); copy-pasted "Multinomial" docstrings on all three NB losses; `self.num_workers = 64` with a `# TODO -- fix these` (model.py:222-224).

## 5. Portability / Reproducibility

- **Hardcoded absolute paths, duplicated 3×:** `FASTA_PATH = /home/nboley/src/Ravel/...` appears independently in model.py:52, data.py:16, predict.py:22 (and points at another user's home directory). Blacklist bed (`/scratch/karius/annotation/mappability...`) hardcoded in data.py:118 and predict.py:159; gencode, repeats, marker-gene, DHS paths hardcoded in predict.py/train.py; model checkpoints under `/scratch/karius/bias_correction_model/`.
- **No configuration mechanism** — no CLI, no config file, no env vars. Everything requires editing source.
- **Zero tests.** No unit tests for the losses (where 3 of 5 variants crash), the receptive-field arithmetic (`calc_input_region_size`), track-name round-tripping, or the weight application (bug S1 would have been caught by a 5-line test).
- Regex escape `"\d+"` in a plain string (data.py:26) — `SyntaxWarning` on modern Python.

## 6. What's Worth Keeping (the good parts)

- **The core idea and architecture are sound:** sequence → per-position fragment-endpoint distributions via a dilated ResNet (BPNet-style), trained on regions where fragmentation should be sequence-driven (off genes, non-CTCF DHS), evaluated where biology should cause deviations. This is a well-established, defensible design.
- `ResNetDilatedBlock` / `SpatialDropout` (layers.py) are reasonable and nearly self-contained.
- `MultinomialNLLLoss` — the one loss that works — is the right default for profile prediction.
- The structured track naming (`strand_x__fl_a_b__coverage_t`) with round-trip functions is a good convention; keep the new scheme, delete the old.
- `calc_input_region_size` receptive-field bookkeeping for unpadded convolutions is correct in spirit and just needs a test.
- The tiling/windowed whole-contig prediction approach in `predict_from_fasta` is the right shape for genome-scale inference.

## 7. Recommendation: Rewrite the scaffolding, keep the model core

A "big refactor" and a rewrite converge here, because the salvageable core is small (~300 lines: layers, multinomial loss, the conv stack, track naming). Suggested shape:

1. **Pick one generation.** New track names only; delete old default `output_columns`, fix or delete `calc_stats`/`make_qc_plots` to use them; delete `SeqToEndpointsMultiResModel` ghost hint; delete the assert-False weight applier and `NegativeBinomialNLLLossOld`.
2. **One loss to start** (multinomial). Re-add NB later *with tests* if overdispersion is actually needed — decide then, not by keeping five broken variants.
3. **Split dataset building from training:** a preprocessing step that materializes (sequence, target-counts) tensors to disk (h5/zarr) keyed by region set + samples; the `Dataset` then just reads. Fixes the eager-init, `.cuda()`-in-worker, and non-resumability problems at once.
4. **Config object** (paths to fasta, blacklist, annotations; fl bands; region sets) replacing all hardcoded paths; standard Lightning checkpointing replacing custom save/load.
5. **Fix S1 as part of a rewritten, tested weight-application function** — it is the delivery mechanism for the whole model and currently the least trustworthy piece.
6. **Tests first for:** loss NLL vs scipy reference, `calc_input_region_size` vs actual model output length, track-name round-trip, weight application on a toy fragment array.
7. Move `build_*_rdfs` experiment definitions out of the package (they belong in the analysis repo that owns the marker-gene/CTCF files), so the package has no dependency on `biomarker-projects`.

**Note (scope):** several findings (S1, C7, ghost class C8-adjacent) live in `dataframe.py`/`fragment_array.py`, not the package itself — the repo-wide `CODE_REVIEW.md` (2026-04-03) documents further pre-existing issues in those files (e.g., BED bitflag precedence bugs) that would compound any revival and should be triaged together.
