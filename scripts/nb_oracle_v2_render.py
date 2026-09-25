#!/usr/bin/env python
"""Render stage for the NB oracle anchor — reads raw, writes published JSON.

Reads ``oracle_nb_v2.raw.json`` (the compute stage output) and produces
the published ``oracle_nb_v2.json``: all derived statistics, verification
gates, and explanatory prose.  Takes seconds, not minutes.

The raw artifact must carry ``_schema_version`` matching RAW_SCHEMA_VERSION
in this module.  On mismatch the script **raises and exits non-zero**,
naming expected vs found and the command to regenerate.  It must NEVER
silently recompute.

Usage:
    cd <repo>
    PYTHONPATH=. python scripts/nb_oracle_v2_render.py [--raw PATH] [--out PATH]
"""

from __future__ import annotations

import datetime
import json
import os
import sys

import numpy as np

# ── Schema version — bump on ANY field add/remove/rename in the raw artifact.
RAW_SCHEMA_VERSION = 1

# ── Default paths ─────────────────────────────────────────────────────────
_SIM_DIR = "/efs/analytics/nathanboley/background_model/simulation_v3_nb/A"
DEFAULT_RAW = os.path.join(_SIM_DIR, "oracle_nb_v2.raw.json")
DEFAULT_OUT = os.path.join(_SIM_DIR, "oracle_nb_v2.json")
OLD_JSON = os.path.join(_SIM_DIR, "oracle_nb.json")


def _check_schema_version(raw: dict) -> None:
    """Raise if the raw artifact's schema version doesn't match."""
    found = raw.get("_schema_version")
    if found != RAW_SCHEMA_VERSION:
        raise RuntimeError(
            f"Raw artifact schema version mismatch: "
            f"expected {RAW_SCHEMA_VERSION}, found {found}. "
            f"Regenerate with: PYTHONPATH=. python scripts/nb_oracle_v2_compute.py"
        )


def _check_gate_4_loss_identity(raw_loss_config: dict) -> tuple[bool, list[str]]:
    """Gate [4]: verify the raw loss config matches the live loss object.

    Creates a live loss from ORACLE_LOSS_KWARGS and compares attributes.
    Returns (pass, list_of_mismatch_messages).
    """
    from scripts._oracle_scoring import ORACLE_LOSS_KWARGS, make_oracle_loss_fn

    loss_fn = make_oracle_loss_fn()
    mismatches = []

    # Check class name
    expected_class = type(loss_fn).__name__
    found_class = raw_loss_config.get("class_name")
    if found_class is not None and found_class != expected_class:
        mismatches.append(
            f"class_name: expected {expected_class}, found {found_class}"
        )

    # Check loss parameters against ORACLE_LOSS_KWARGS AND live object
    for key, expected in ORACLE_LOSS_KWARGS.items():
        actual_raw = raw_loss_config.get(key)
        actual_live = getattr(loss_fn, key)
        if actual_raw != expected:
            mismatches.append(
                f"{key}: ORACLE_LOSS_KWARGS={expected}, raw={actual_raw}"
            )
        if actual_live != expected:
            mismatches.append(
                f"{key}: live_loss_fn={actual_live}, ORACLE_LOSS_KWARGS={expected}"
            )

    return len(mismatches) == 0, mismatches


def render_published_json(raw: dict) -> dict:
    """Transform a raw artifact into the published oracle_nb_v2.json.

    Pure function of *raw* plus the ORACLE_LOSS_KWARGS constant (for gate [4]).
    Same raw in → same published out (except ``created_utc``).
    """
    _check_schema_version(raw)

    oracle_loss = raw["oracle"]["loss"]
    uniform_loss = raw["uniform"]["loss"]
    gap = uniform_loss - oracle_loss

    def pct_bias(model_nll):
        return 100.0 * (uniform_loss - model_nll) / gap

    sweep = raw["sweep_curve"]

    # ── Noise floor ───────────────────────────────────────────────────────
    determinism_ok = all(p["bitwise_equal"] for p in raw["determinism"])

    # Plateau statistics over the fixed reporting window r in [15, 3000]
    plateau_entries = [(s["r"], s["loss"]) for s in sweep
                       if 15 <= s["r"] <= 3000]
    if len(plateau_entries) >= 2:
        plateau_losses = [loss for _, loss in plateau_entries]
        plateau_max = max(plateau_losses)
        plateau_min = min(plateau_losses)
        plateau_loss_span = plateau_max - plateau_min
        adj_diffs = [abs(plateau_losses[i + 1] - plateau_losses[i])
                     for i in range(len(plateau_losses) - 1)]
        max_adj_non_mono = max(adj_diffs) if adj_diffs else 0.0
    else:
        plateau_loss_span = 0.0
        max_adj_non_mono = 0.0
        plateau_min = None
        plateau_max = None

    # Data-driven plateau interval: range of r where loss < global_min + 0.001
    global_min_loss = min(s["loss"] for s in sweep)
    plateau_threshold = global_min_loss + 0.001
    plateau_r_vals = [s["r"] for s in sweep if s["loss"] < plateau_threshold]
    plateau_interval = (float(min(plateau_r_vals)), float(max(plateau_r_vals)))

    noise_floor_info = {
        "deterministic": determinism_ok,
        "plateau_loss_span": plateau_loss_span,
        "max_adjacent_non_monotonicity": max_adj_non_mono,
        "plateau_range": {
            "r_lo": 15.0,
            "r_hi": 3000.0,
            "min_loss": float(plateau_min) if plateau_min is not None else None,
            "max_loss": float(plateau_max) if plateau_max is not None else None,
            "n_points": len(plateau_entries),
        },
        "_note": (
            "The objective is deterministic (verified: repeat evaluations are "
            "bitwise identical). The loss varies across the plateau as a "
            "function of r \u2014 this is reproducible structure in the "
            "softmax/lgamma/clamp pipeline, not stochastic noise. "
            "plateau_loss_span is the total loss range (max - min) over the "
            "fixed window r in [15, 3000]. That window is a REPORTING "
            "CONVENTION, not a measured boundary, and is deliberately not the "
            "same as profiled_nuisance_r.plateau_interval, which is "
            "data-driven (loss < min + 1e-3). The choice does not affect the "
            "statistic: the span is 8.2619e-4 over either range, because the "
            "minimum (r=1096) and maximum (r=2458.8) both fall inside this "
            "narrower one. An earlier version of this note claimed the window "
            "excluded a 'rising tail above r~3000 because it is genuine "
            "signal'; that was WRONG \u2014 r=3506.3 has loss 4.026652, LOWER than "
            "the highest point inside the window (4.027178 at r=2458.8). Only "
            "r=5000 rises clear of the plateau, and both ranges exclude it. "
            "max_adjacent_non_monotonicity is the largest absolute difference "
            "between losses at adjacent swept r values within the window; on a "
            "flat plateau any such difference IS the non-monotonicity of "
            "interest, but the name overstates it \u2014 a strictly monotonic "
            "sequence would also produce a large value."
        ),
    }

    # ── Sweep raggedness / r_identified ────────────────────────────────────
    sweep_losses_arr = np.array([s["loss"] for s in sweep])
    sweep_diffs = np.diff(sweep_losses_arr)
    sign_changes = int(np.sum(np.diff(np.sign(sweep_diffs)) != 0))
    r_identified = sign_changes <= 5

    # ── Verification gates ─────────────────────────────────────────────────
    all_pass = True

    models = raw["models"]
    ken_mean = models["trained_ken"]["nb_loss"]
    hybrid_mean = models["trained_hybrid"]["nb_loss"]
    ken_u_mean = models["untrained_ken"]["nb_loss"]
    hybrid_u_mean = models["untrained_hybrid"]["nb_loss"]

    # Gate [1]: untrained above oracle
    untrained_above = {
        "ken": bool(ken_u_mean > oracle_loss),
        "hybrid": bool(hybrid_u_mean > oracle_loss),
    }
    if not all(untrained_above.values()):
        all_pass = False

    # Gate [2]: trained between oracle and uniform
    trained_between = {
        "ken": bool(oracle_loss < ken_mean < uniform_loss),
        "hybrid": bool(oracle_loss < hybrid_mean < uniform_loss),
    }
    if not all(trained_between.values()):
        all_pass = False

    # Gate [3]: alignment
    alignment = raw["alignment"]
    if alignment["best_shift"] != 0:
        all_pass = False

    # Gate [4]: loss-object identity — read the live loss object
    gate4_pass, gate4_mismatches = _check_gate_4_loss_identity(
        raw["loss_config"]
    )
    if not gate4_pass:
        all_pass = False

    # ── Assemble published JSON ────────────────────────────────────────────
    raw_lc = raw["loss_config"]

    payload = {
        "_what": (
            "Anchor A for the v3nb overdispersed simulation store: true "
            "propensity with a profiled scalar dispersion r (a nuisance "
            "parameter, NOT an estimate of the generative r). The offset "
            "conditioning absorbs the overdispersion above r~20, and the "
            "loss plateau is flat \u2014 r is not identified. The oracle VALUE "
            "(oracle_nb_nll) is the lowest loss observed and is a genuine "
            "floor. This SUPERSEDES oracle_nb.json, which plugged in the "
            "true per-position r and produced a value ABOVE untrained "
            "models (not a floor). See docs/pending/nb_oracle.md."
        ),
        "oracle_nb_nll": oracle_loss,
        "uniform_nb_nll": uniform_loss,
        "gap_uniform_minus_oracle": gap,
        "profiled_nuisance_r": {
            "not_identified": not r_identified,
            "oracle": {
                "r": raw["oracle"]["r"],
                "log_r": raw["oracle"]["log_r"],
                "source": raw["oracle"]["source"],
            },
            "uniform": {
                "r": raw["uniform"]["r"],
                "log_r": raw["uniform"]["log_r"],
                "source": raw["uniform"]["source"],
                "_note": "Separately fitted (not reusing oracle's r)",
            },
            "plateau_interval": {
                "r_lo": plateau_interval[0],
                "r_hi": plateau_interval[1],
                "_note": "Range of r where loss < global_min + 0.001",
            },
            "reference": {
                "true_hexamer_r_median": 7.179,
                "frozen_model_init_r": 1096.0,
                "frozen_model_log_dispersion_init": 7.0,
            },
            "_note": (
                "This is the argmin of a nuisance parameter (the scalar r "
                "minimising the frozen-core nb_offset loss with propensity "
                "held at truth), NOT an estimate of the generative r. The "
                "loss plateau is flat across the interval above \u2014 the "
                "specific value is an artefact of which grid/optimizer point "
                "happened to land lowest. Do not interpret it as an "
                "effective dispersion."
            ),
            "_r_1096_coincidence": (
                "The published r (\u22481096) coincides with the frozen model "
                "initialisation exp(log_dispersion_init) = exp(7) \u2248 1096. "
                "This is an artefact of the flat plateau plus the reference "
                "marker log(1096) being injected into the sweep grid as a "
                "selectable point \u2014 NOT agreement between the oracle and "
                "the model. Any r on the plateau produces an effectively "
                "identical oracle loss."
            ),
        },
        "noise_floor": noise_floor_info,
        "sweep_curve": sweep,
        "models": {
            "_note": (
                "nb_loss is THIS script's float32 rescoring on CPU and is the "
                "number pct_bias_captured is derived from. "
                "nb_val_loss_from_training is what the training run logged, "
                "and the two are NOT computed the same way: training "
                "validated under precision=bf16-mixed on GPU. bf16 carries "
                "roughly three decimal digits of mantissa, so disagreement at "
                "the 1e-4 level is expected and the float32 figure is the more "
                "accurate one. Measured here: KEN differs by 1.3e-5, hybrid by "
                "4.1e-4 \u2014 a 30x asymmetry between two architectures scored by "
                "identical code, which is NOT fully explained by precision "
                "alone and is recorded as an open question rather than a "
                "resolved one. It moves hybrid's pct_bias_captured by about "
                "0.46pp. Quote nb_loss, not nb_val_loss_from_training."
            ),
            "trained_ken": {
                "nb_loss": ken_mean,
                "nb_val_loss_from_training": models["trained_ken"]["nb_val_loss_from_training"],
                "pct_bias_captured": round(pct_bias(ken_mean), 2),
                "checkpoint": models["trained_ken"]["checkpoint"],
            },
            "trained_hybrid": {
                "nb_loss": hybrid_mean,
                "nb_val_loss_from_training": models["trained_hybrid"]["nb_val_loss_from_training"],
                "pct_bias_captured": round(pct_bias(hybrid_mean), 2),
                "checkpoint": models["trained_hybrid"]["checkpoint"],
            },
            "untrained_ken": {
                "nb_loss": ken_u_mean,
                "pct_bias_captured": round(pct_bias(ken_u_mean), 2),
                "seed": models["untrained_ken"]["seed"],
            },
            "untrained_hybrid": {
                "nb_loss": hybrid_u_mean,
                "pct_bias_captured": round(pct_bias(hybrid_u_mean), 2),
                "seed": models["untrained_hybrid"]["seed"],
            },
        },
        "verification": {
            "alignment": alignment,
            "sanity_gate_all_pass": all_pass,
            "untrained_above_oracle": untrained_above,
            "trained_between_oracle_and_uniform": trained_between,
            "r_identified": {
                "value": r_identified,
                "sweep_sign_changes": sign_changes,
                "threshold": 5,
                "_note": (
                    "Whether the profiled nuisance r is identified (smooth, "
                    "unimodal sweep curve). False means the plateau is flat "
                    "and the specific r value is an artefact of which grid "
                    "point landed lowest. This is a descriptive field, not a "
                    "gate \u2014 a flat plateau does not mean the artifact is broken."
                ),
            },
        },
        "loss": "MaskedNegativeBinomialOffsetNLLLoss (background_model_core, frozen)",
        "loss_config": {
            "max_dispersion_ratio": raw_lc["max_dispersion_ratio"],
            "clamp_margin": raw_lc["clamp_margin"],
            "dispersion_window_size": raw_lc["dispersion_window_size"],
            "_why": (
                "Matched to the training runs (v3nb_ken/hybrid_nb_frozen_lr5e-3). "
                "A floor computed under a different clamp is not in the same units "
                "as the val_loss it is compared against."
            ),
        },
        "store": raw["store"],
        "sim_dir": raw["sim_dir"],
        "fasta": raw["fasta"],
        "n_val_pairs": raw["n_val_pairs"],
        "tile_size": raw["tile_size"],
        "l_target": raw["l_target"],
        "crop": raw["crop"],
        "gc_mode": raw["gc_mode"],
        "design_doc": raw["design_doc"],
        "supersedes": "oracle_nb.json",
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "python": sys.executable,
        "runtime_s": raw["_runtime_s"],
    }

    return payload


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default=DEFAULT_RAW,
                    help="path to the raw artifact JSON (default: %(default)s)")
    ap.add_argument("--out", default=DEFAULT_OUT,
                    help="path for the published JSON (default: %(default)s)")
    args = ap.parse_args()

    with open(args.raw) as f:
        raw = json.load(f)

    published = render_published_json(raw)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(published, f, indent=2)
    print(f"[render] Wrote {args.out}")

    # ── Mark old oracle_nb.json as superseded (if present) ────────────────
    if os.path.exists(OLD_JSON):
        with open(OLD_JSON) as f:
            old = json.load(f)
        old["_superseded_by"] = "oracle_nb_v2.json"
        old["_superseded_reason"] = (
            "The plug-in oracle (true propensity + true per-position r) is not "
            "a floor for the nb_offset pseudo-likelihood. Both untrained models "
            "scored below it. See docs/pending/nb_oracle.md for the full diagnosis."
        )
        with open(OLD_JSON, "w") as f:
            json.dump(old, f, indent=2)
        print(f"[render] Marked {OLD_JSON} as superseded")

    # ── Print gate summary ────────────────────────────────────────────────
    v = published["verification"]
    if v["sanity_gate_all_pass"]:
        print("[render] All verification gates passed")
    else:
        print("[render] *** VERIFICATION FAILED — see published JSON ***")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
