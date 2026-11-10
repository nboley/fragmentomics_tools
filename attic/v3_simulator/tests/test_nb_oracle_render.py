"""Regression tests for nb_oracle_v2_render: render from synthetic raw artifact.

These tests exercise the render stage WITHOUT running the expensive compute
pipeline. A hand-constructed raw artifact is rendered and the published JSON
is checked for:
  - Correct derived statistics (gap, pct_bias, noise_floor)
  - Verification gates that can actually fail
  - Schema version staleness detection
  - Correct key structure matching the published schema

The synthetic data is deliberately simple enough to verify by hand.
"""

from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from scripts.nb_oracle_v2_render import (
    RAW_SCHEMA_VERSION,
    _check_gate_4_loss_identity,
    _check_schema_version,
    render_published_json,
)


# ── Synthetic raw artifact ────────────────────────────────────────────────
# Deliberately small and with hand-verifiable derived values.

def _make_raw():
    """Return a fresh copy of the synthetic raw artifact."""
    return {
        "_schema_version": RAW_SCHEMA_VERSION,
        "_computed_utc": "2026-01-01T00:00:00+00:00",
        "_runtime_s": 100.0,
        "store": "/fake/store.zarr",
        "sim_dir": "/fake/sim",
        "fasta": "/fake/hg38.fa",
        "design_doc": "docs/pending/nb_oracle.md",
        "n_val_pairs": 100,
        "tile_size": 2048,
        "l_target": 2304,
        "crop": [128, 2176],
        "gc_mode": "lut",
        "sweep_curve": [
            {"log_r": 0.0, "r": 1.0, "loss": 5.0},
            {"log_r": 1.5, "r": 4.482, "loss": 4.8},
            {"log_r": 2.708, "r": 15.0, "loss": 4.10},   # plateau boundary
            {"log_r": 3.0, "r": 20.086, "loss": 4.030},
            {"log_r": 4.0, "r": 54.598, "loss": 4.025},
            {"log_r": 5.0, "r": 148.413, "loss": 4.0195},
            {"log_r": 6.0, "r": 403.429, "loss": 4.0193},
            {"log_r": 7.0, "r": 1096.633, "loss": 4.0190},  # minimum
            {"log_r": 8.006, "r": 2999.0, "loss": 4.0192},  # plateau boundary
        ],
        "determinism": [
            {"r": 7.179, "eval1": 4.03, "eval2": 4.03, "bitwise_equal": True},
            {"r": 21.0, "eval1": 4.025, "eval2": 4.025, "bitwise_equal": True},
            {"r": 1096.0, "eval1": 4.019, "eval2": 4.019, "bitwise_equal": True},
            {"r": 500.0, "eval1": 4.02, "eval2": 4.02, "bitwise_equal": True},
        ],
        "oracle": {
            "loss": 4.0190, "log_r": 7.0, "r": 1096.633, "source": "sweep",
        },
        "uniform": {
            "loss": 4.1000, "log_r": 3.5, "r": 33.115, "source": "scipy",
        },
        "alignment": {
            "best_shift": 0,
            "best_r": 0.95,
            "r_at_shift_0": 0.95,
            "r_at_shift_minus128": 0.1,
            "ratio_0_vs_minus128": 9.5,
            "n_tiles": 20,
        },
        "models": {
            "trained_ken": {
                "nb_loss": 4.0500,
                "nb_val_loss_from_training": 4.0513,
                "checkpoint": "/fake/ken.ckpt",
            },
            "trained_hybrid": {
                "nb_loss": 4.0600,
                "nb_val_loss_from_training": 4.0641,
                "checkpoint": "/fake/hybrid.ckpt",
            },
            "untrained_ken": {"nb_loss": 4.1100, "seed": 42},
            "untrained_hybrid": {"nb_loss": 4.1200, "seed": 42},
        },
        "loss_config": {
            "class_name": "MaskedNegativeBinomialOffsetNLLLoss",
            "max_dispersion_ratio": 2.0,
            "clamp_margin": 1.0,
            "dispersion_window_size": 1,
            "matches_training_config": True,
        },
    }


# ── Schema version tests ─────────────────────────────────────────────────

class TestSchemaVersion:
    """Staleness detection: render must reject mismatched schema versions."""

    def test_correct_version_passes(self):
        raw = _make_raw()
        _check_schema_version(raw)  # should not raise

    def test_missing_version_raises(self):
        raw = _make_raw()
        del raw["_schema_version"]
        with pytest.raises(RuntimeError, match="schema version mismatch"):
            _check_schema_version(raw)

    def test_wrong_version_raises(self):
        raw = _make_raw()
        raw["_schema_version"] = RAW_SCHEMA_VERSION + 1
        with pytest.raises(RuntimeError, match="schema version mismatch"):
            _check_schema_version(raw)

    def test_error_message_names_versions(self):
        raw = _make_raw()
        raw["_schema_version"] = 999
        with pytest.raises(RuntimeError, match=r"expected.*1.*found.*999"):
            _check_schema_version(raw)

    def test_error_message_includes_regenerate_command(self):
        raw = _make_raw()
        raw["_schema_version"] = 0
        with pytest.raises(RuntimeError, match="nb_oracle_v2_compute"):
            _check_schema_version(raw)


# ── Derived statistics tests ──────────────────────────────────────────────

class TestDerivedStatistics:
    """Verify the derived statistics are computed correctly."""

    def test_gap_is_uniform_minus_oracle(self):
        raw = _make_raw()
        pub = render_published_json(raw)
        expected_gap = raw["uniform"]["loss"] - raw["oracle"]["loss"]
        assert pub["gap_uniform_minus_oracle"] == expected_gap

    def test_gap_is_positive(self):
        """Discriminator: gap must be positive (uniform > oracle)."""
        raw = _make_raw()
        pub = render_published_json(raw)
        assert pub["gap_uniform_minus_oracle"] > 0, (
            "Gap should be positive: uniform loss should exceed oracle"
        )

    def test_pct_bias_trained_between_0_and_100(self):
        raw = _make_raw()
        pub = render_published_json(raw)
        for key in ("trained_ken", "trained_hybrid"):
            pct = pub["models"][key]["pct_bias_captured"]
            assert 0 < pct < 100, (
                f"{key} pct_bias_captured={pct} should be in (0, 100)"
            )

    def test_pct_bias_ken_hand_computed(self):
        """Verify pct_bias by hand: 100 * (4.1 - 4.05) / (4.1 - 4.019)."""
        raw = _make_raw()
        pub = render_published_json(raw)
        gap = 4.1000 - 4.0190
        expected = round(100.0 * (4.1000 - 4.0500) / gap, 2)
        assert pub["models"]["trained_ken"]["pct_bias_captured"] == expected

    def test_pct_bias_untrained_negative(self):
        """Untrained models score ABOVE uniform → pct_bias_captured < 0."""
        raw = _make_raw()
        # Set untrained scores above uniform
        raw["models"]["untrained_ken"]["nb_loss"] = 4.15
        raw["models"]["untrained_hybrid"]["nb_loss"] = 4.20
        pub = render_published_json(raw)
        assert pub["models"]["untrained_ken"]["pct_bias_captured"] < 0
        assert pub["models"]["untrained_hybrid"]["pct_bias_captured"] < 0


# ── Noise floor tests ────────────────────────────────────────────────────

class TestNoiseFloor:
    """Verify noise_floor derivation from sweep and determinism data."""

    def test_deterministic_all_true(self):
        raw = _make_raw()
        pub = render_published_json(raw)
        assert pub["noise_floor"]["deterministic"] is True

    def test_deterministic_detects_non_bitwise(self):
        raw = _make_raw()
        raw["determinism"][1]["bitwise_equal"] = False
        pub = render_published_json(raw)
        assert pub["noise_floor"]["deterministic"] is False

    def test_plateau_range_uses_fixed_window(self):
        """plateau_range uses the fixed [15, 3000] window, not data-driven."""
        raw = _make_raw()
        pub = render_published_json(raw)
        assert pub["noise_floor"]["plateau_range"]["r_lo"] == 15.0
        assert pub["noise_floor"]["plateau_range"]["r_hi"] == 3000.0

    def test_plateau_loss_span_hand_computed(self):
        """Verify plateau_loss_span from sweep entries in [15, 3000]."""
        raw = _make_raw()
        # Sweep entries with r in [15, 3000]:
        # r=15.0 (4.10), r=20.086 (4.030), r=54.598 (4.025),
        # r=148.413 (4.0195), r=403.429 (4.0193), r=1096.633 (4.0190),
        # r=2999.0 (4.0192)
        expected_span = 4.10 - 4.0190  # max - min
        pub = render_published_json(raw)
        assert abs(pub["noise_floor"]["plateau_loss_span"] - expected_span) < 1e-10


# ── Verification gate tests ──────────────────────────────────────────────

class TestVerificationGates:
    """Gates must actually be able to fail — not just print YES."""

    def test_all_pass_on_clean_data(self):
        raw = _make_raw()
        pub = render_published_json(raw)
        assert pub["verification"]["sanity_gate_all_pass"] is True

    def test_gate1_untrained_below_oracle_fails(self):
        """Gate [1]: untrained must be above oracle."""
        raw = _make_raw()
        raw["models"]["untrained_ken"]["nb_loss"] = 4.0  # below oracle 4.019
        pub = render_published_json(raw)
        assert pub["verification"]["sanity_gate_all_pass"] is False
        assert pub["verification"]["untrained_above_oracle"]["ken"] is False

    def test_gate2_trained_outside_range_fails(self):
        """Gate [2]: trained must be between oracle and uniform."""
        raw = _make_raw()
        raw["models"]["trained_ken"]["nb_loss"] = 4.15  # above uniform 4.10
        pub = render_published_json(raw)
        assert pub["verification"]["sanity_gate_all_pass"] is False
        assert pub["verification"]["trained_between_oracle_and_uniform"]["ken"] is False

    def test_gate3_alignment_nonzero_shift_fails(self):
        """Gate [3]: best_shift must be 0."""
        raw = _make_raw()
        raw["alignment"]["best_shift"] = 3
        pub = render_published_json(raw)
        assert pub["verification"]["sanity_gate_all_pass"] is False

    def test_gate4_loss_config_mismatch_fails(self):
        """Gate [4]: loss config in raw must match live loss object."""
        raw = _make_raw()
        raw["loss_config"]["max_dispersion_ratio"] = 999.0
        pub = render_published_json(raw)
        assert pub["verification"]["sanity_gate_all_pass"] is False

    def test_gate4_reads_live_object(self):
        """Gate [4] must read the live loss object, not just pass blindly."""
        from scripts._oracle_scoring import ORACLE_LOSS_KWARGS
        # With correct values, gate passes
        ok, mismatches = _check_gate_4_loss_identity({
            "class_name": "MaskedNegativeBinomialOffsetNLLLoss",
            "max_dispersion_ratio": ORACLE_LOSS_KWARGS["max_dispersion_ratio"],
            "clamp_margin": ORACLE_LOSS_KWARGS["clamp_margin"],
        })
        assert ok is True
        assert mismatches == []

        # With wrong values, gate fails
        ok, mismatches = _check_gate_4_loss_identity({
            "class_name": "MaskedNegativeBinomialOffsetNLLLoss",
            "max_dispersion_ratio": 5.0,  # wrong
            "clamp_margin": 1.0,
        })
        assert ok is False
        assert len(mismatches) > 0


# ── Published schema structure tests ──────────────────────────────────────

class TestPublishedSchema:
    """Published JSON must have the same keys as the original monolithic script."""

    EXPECTED_TOP_KEYS = {
        "_what", "oracle_nb_nll", "uniform_nb_nll", "gap_uniform_minus_oracle",
        "profiled_nuisance_r", "noise_floor", "sweep_curve", "models",
        "verification", "loss", "loss_config", "store", "sim_dir", "fasta",
        "n_val_pairs", "tile_size", "l_target", "crop", "gc_mode",
        "design_doc", "supersedes", "created_utc", "python", "runtime_s",
        # Added deliberately when the provenance fields were corrected:
        # created_utc and runtime_s now BOTH describe the compute run, and
        # rendered_utc records the render step separately. Previously
        # created_utc was the render timestamp sitting beside a compute
        # duration, which read as one claim and was actually two.
        # This test caught that addition, which is what it is for.
        "rendered_utc",
    }

    def test_has_all_top_level_keys(self):
        raw = _make_raw()
        pub = render_published_json(raw)
        missing = self.EXPECTED_TOP_KEYS - set(pub.keys())
        extra = set(pub.keys()) - self.EXPECTED_TOP_KEYS
        assert missing == set(), f"Missing keys: {missing}"
        assert extra == set(), f"Extra keys: {extra}"

    def test_profiled_nuisance_r_structure(self):
        raw = _make_raw()
        pub = render_published_json(raw)
        pnr = pub["profiled_nuisance_r"]
        assert "not_identified" in pnr
        assert "oracle" in pnr
        assert "uniform" in pnr
        assert "plateau_interval" in pnr
        assert "reference" in pnr
        assert "_note" in pnr
        assert "_r_1096_coincidence" in pnr
        # Uniform has its own _note
        assert "_note" in pnr["uniform"]

    def test_verification_structure(self):
        raw = _make_raw()
        pub = render_published_json(raw)
        v = pub["verification"]
        assert "alignment" in v
        assert "sanity_gate_all_pass" in v
        assert "untrained_above_oracle" in v
        assert "trained_between_oracle_and_uniform" in v
        assert "r_identified" in v
        ri = v["r_identified"]
        assert "value" in ri
        assert "sweep_sign_changes" in ri
        assert "threshold" in ri

    def test_models_have_pct_bias(self):
        raw = _make_raw()
        pub = render_published_json(raw)
        for key in ("trained_ken", "trained_hybrid",
                     "untrained_ken", "untrained_hybrid"):
            assert "pct_bias_captured" in pub["models"][key], (
                f"models.{key} missing pct_bias_captured"
            )

    def test_loss_config_keys(self):
        raw = _make_raw()
        pub = render_published_json(raw)
        lc = pub["loss_config"]
        assert "max_dispersion_ratio" in lc
        assert "clamp_margin" in lc
        assert "dispersion_window_size" in lc
        assert "_why" in lc

    def test_passthrough_provenance(self):
        """Provenance fields must pass through unchanged from the raw artifact."""
        raw = _make_raw()
        pub = render_published_json(raw)
        for key in ("store", "sim_dir", "fasta", "n_val_pairs",
                     "tile_size", "l_target", "crop", "gc_mode", "design_doc"):
            assert pub[key] == raw[key], (
                f"Passthrough field '{key}' differs: {pub[key]} != {raw[key]}"
            )

    def test_oracle_and_uniform_nll_passthrough(self):
        """Headline NLL values must pass through exactly from raw."""
        raw = _make_raw()
        pub = render_published_json(raw)
        assert pub["oracle_nb_nll"] == raw["oracle"]["loss"]
        assert pub["uniform_nb_nll"] == raw["uniform"]["loss"]


# ── Render-from-raw-alone test (design §5) ────────────────────────────────

class TestRenderFromRawAlone:
    """Render must produce a complete published JSON from ONLY the raw artifact.

    This verifies the raw schema is complete: render does not secretly read
    any other file (store, summary.json, FASTA, etc.).
    """

    def test_render_needs_only_raw(self):
        """render_published_json succeeds with synthetic data — no /efs needed."""
        raw = _make_raw()
        pub = render_published_json(raw)
        # If render tried to open any external file, it would have raised.
        # Verify the output is plausible.
        assert pub["oracle_nb_nll"] == 4.019
        assert pub["gap_uniform_minus_oracle"] > 0
        assert pub["verification"]["sanity_gate_all_pass"] is True
