import numpy as np
import pandas as pd
import pytest

from credit_monitoring.drift import holm_adjust, population_stability_index, report_drift


def test_psi_zero_and_tails():
    reference = np.arange(1000)
    assert population_stability_index(reference, reference) == pytest.approx(0)
    assert population_stability_index(reference, reference + 1_000_000) > 0.2
    assert population_stability_index(np.zeros(100), np.ones(100)) > 0.2


def test_invalid_psi_input():
    for current in ([], [np.nan], [np.inf]):
        with pytest.raises(ValueError):
            population_stability_index([1, 2, 3], current)


def test_holm_adjustment():
    assert holm_adjust([0.04, 0.001, 0.02]) == pytest.approx([0.04, 0.003, 0.04])


def test_aggregate_report_detects_shift_and_excludes_identifiers(tmp_path):
    rng = np.random.default_rng(12)
    reference = pd.DataFrame(
        {
            "row_id": ["PRIVATE_RECORD_SENTINEL"] * 500,
            "target": rng.binomial(1, 0.2, 500),
            "AGE": rng.integers(20, 60, 500),
            "LIMIT_BAL": rng.integers(10_000, 300_000, 500),
            "PAY_0": rng.integers(-1, 4, 500),
            "BILL_AMT1": rng.integers(-1000, 100_000, 500),
            "PAY_AMT1": rng.integers(0, 50_000, 500),
        }
    )
    shifted = reference.copy()
    shifted["LIMIT_BAL"] *= 2
    shifted["PAY_AMT1"] *= 0.15
    summary = report_drift(reference, shifted, tmp_path, "shift")
    assert {"LIMIT_BAL", "PAY_AMT1"} <= set(summary["drifted_features"])
    for path in tmp_path.iterdir():
        content = path.read_text(encoding="utf-8")
        assert "PRIVATE_RECORD_SENTINEL" not in content
    assert (tmp_path / "shift_evidently.html").exists()
    assert (tmp_path / "shift_evidently.json").exists()
