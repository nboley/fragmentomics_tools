"""Assemble simulation_v3/A/oracle.json from the two sim_oracle.py runs, and
print the %-of-available-bias-captured table for the 8 LR-sweep runs."""
import glob
import json
import os

A = json.load(open("/tmp/oracle_bilinear.json"))
B = json.load(open("/tmp/oracle_lut.json"))

ORACLE = A["oracle_nll"]
UNIFORM = A["uniform_nll"]
GAP = UNIFORM - ORACLE
DEAD = 7.612706184387207  # lrsweep_ken_lr2e-2, constant-output collapse

RUNS_ROOT = "/efs/analytics/nathanboley/background_model/simulation_v3/runs"
NOTES = {
    "lrsweep_hybrid_lr2e-3": "diverged at epoch 15; best_val_loss is the pre-divergence minimum (legitimate)",
    "lrsweep_cnn_lr7e-3": "diverged at epoch 31; best_val_loss is the pre-divergence minimum (legitimate)",
    "lrsweep_ken_lr2e-2": "DEAD RUN - collapsed to constant output in epoch 0, bitwise-identical val_loss for 16 epochs. This is the uniform baseline, NOT a model result.",
}

rows = []
for p in sorted(glob.glob(os.path.join(RUNS_ROOT, "lrsweep_*", "summary.json"))):
    s = json.load(open(p))
    name = s["run_name"]
    v = s["best_val_loss"]
    rows.append({
        "run": name,
        "model": s["model"],
        "lr": s["lr"],
        "best_val_loss": v,
        "pct_bias_captured": 100.0 * (UNIFORM - v) / GAP,
        "dead_run": name == "lrsweep_ken_lr2e-2",
        "note": NOTES.get(name, ""),
    })
rows.sort(key=lambda r: r["best_val_loss"])

payload = {
    "_what": "Oracle (theoretical floor) and uniform (all-positions-equal) "
             "multinomial NLL anchors for the v3 regime-A simulation store. "
             "Use to convert a val_loss into % of available bias captured: "
             "(uniform - model) / (uniform - oracle).",
    "oracle_nll": ORACLE,
    "uniform_nll": UNIFORM,
    "gap_uniform_minus_oracle": GAP,

    "oracle_nll_gc_lut_variant": B["oracle_nll"],
    "oracle_gc_lut_vs_bilinear_delta": ORACLE - B["oracle_nll"],
    "_gc_variant_note": (
        "scripts/sim_fragments.py drew fragments through an integer-rounded GC "
        "lookup table; the primary oracle above uses the exact bilinear surface. "
        "Both were computed; they agree to 2e-6 nats (0.002% of the gap), so the "
        "GC discretisation is immaterial to the floor."),

    "uniform_crosscheck": {
        "independent_value": DEAD,
        "source": "lrsweep_ken_lr2e-2 best_val_loss - a run that collapsed to "
                  "constant output in epoch 0 and then reported this value "
                  "bitwise-identically for 16 epochs; a constant-output model is "
                  "the uniform model.",
        "delta_computed_minus_independent": UNIFORM - DEAD,
        "passes": abs(UNIFORM - DEAD) < 1e-5,
    },

    "loss": "background_model_core.MaskedMultinomialNLLLoss (frozen core, unmodified)",
    "reduction": ("mean over (pair, track); per-track NLL divided by that track's "
                  "total count N (clamped to >=1), so a zero-count track "
                  "contributes exactly 0. This is why uniform (7.6127) sits "
                  "BELOW log(2048)=7.624619 - about 0.16% of (pair, track) cells "
                  "are empty."),
    "log_tile_size": 7.624618986159398,

    "store": A["store"],
    "sim_dir": A["sim_dir"],
    "fasta": A["fasta"],
    "store_config_hash": A["store_config_hash"],
    "store_split_version": A["store_split_version"],
    "split": A["split"],
    "dataset_kwargs": A["dataset_kwargs"],
    "n_val_pairs": A["n_val_pairs"],
    "n_val_tiles": A["n_val_tiles"],
    "n_train_samples": A["n_train_samples"],
    "geometry": {"tile_size": A["tile_size"], "l_target": A["l_target"],
                 "jitter": A["jitter"], "crop": A["crop"]},

    "alignment_verification": {
        "method": "shift-correlation of the oracle propensity against observed "
                  "counts summed over the 16 train samples, over the full "
                  "l_target extent, 6 random val tiles x 4 tracks, shifts -260..260",
        "argmax_shift": 0,
        "r_at_shift_0": 0.3808,
        "r_at_shift_minus_1": 0.1156,
        "r_at_shift_minus_128": 0.0268,
        "dataset_crop_check": "BackgroundTileDataset(split=val, train_mode=False)[i] "
                              "y is byte-identical to store y_full[:, 128:2176]",
        "verdict": "aligned; the 128-position offset the docstring claims to fix "
                   "would have shown up as r=0.027",
    },
    "regime_finding": {
        "nb_dispersion_used": False,
        "evidence": "ground_truth.npz has no hexamer_r key; ground_truth.json has "
                    "no nb_dispersion field; sample_000 fragments-per-region equals "
                    "target_counts EXACTLY for all 4800 regions, which only the "
                    "multinomial sampler (simulate_sample) produces. The NB sampler "
                    "(simulate_sample_nb) makes per-region totals random.",
        "implication": "v3_A was drawn from the plain multinomial sampler, so the "
                       "oracle propensity is the exact generative categorical "
                       "distribution - a genuine floor, not an approximation.",
        "regime": "A (log_jitter is exactly 0 for all 20 samples; w6 shared)",
    },

    "sweep": rows,
    "git_sha": "db456996fd9bc5b60a639d230a22bb2a09bfdb68",
    "git_sha_note": "working tree dirty: background_model/dataset.py (fl_band_fracs, "
                    "unrelated) + untracked scripts/. No package or frozen-core code "
                    "was modified for this measurement.",
    "date_utc": A["date_utc"],
    "python": A["python"],
    "command": ("PYTHONPATH=. python scripts/sim_oracle.py --sim-dir <sim_dir> "
                "--store <store> --out <out>   [--gc-lut for the LUT variant]"),
}

out = "/efs/analytics/nathanboley/background_model/simulation_v3/A/oracle.json"
with open(out, "w") as fh:
    json.dump(payload, fh, indent=2)
print("wrote", out)
print()
print(f"oracle  = {ORACLE:.6f}")
print(f"uniform = {UNIFORM:.6f}   (cross-check vs 7.6127061844: "
      f"delta {UNIFORM-DEAD:+.2e})")
print(f"gap     = {GAP:.6f}")
print()
hdr = f"| {'run':<24} | {'model':<7} | {'lr':>6} | {'best_val_loss':>13} | {'% bias captured':>15} |"
print(hdr)
print("|" + "-"*26 + "|" + "-"*9 + "|" + "-"*8 + "|" + "-"*15 + "|" + "-"*17 + "|")
for r in rows:
    pct = "DEAD (= uniform)" if r["dead_run"] else f"{r['pct_bias_captured']:.1f}%"
    print(f"| {r['run']:<24} | {r['model']:<7} | {r['lr']:>6} | "
          f"{r['best_val_loss']:>13.6f} | {pct:>15} |")
