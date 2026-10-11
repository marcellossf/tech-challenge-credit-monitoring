import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from credit_monitoring.simulate import FEATURES, make_scenarios


def sample():
    return pd.DataFrame(
        {
            "row_id": np.arange(1, 1001),
            "target": np.zeros(1000, dtype=int),
            "AGE": np.full(1000, 35),
            "LIMIT_BAL": np.full(1000, 1_000_000),
            "PAY_0": np.tile([-1, 0, 1, 2], 250),
            "BILL_AMT1": np.full(1000, -1000),
            "PAY_AMT1": np.full(1000, 10000),
        }
    )


def test_deterministic_scenarios_preserve_source_and_concept_features():
    production = sample()
    original = production.copy(deep=True)
    first, second = make_scenarios(production), make_scenarios(production)
    for key in first:
        assert_frame_equal(first[key][0], second[key][0])
        assert first[key][1] == second[key][1]
    assert_frame_equal(production, original)
    concept, manifest = first["concept_drift"]
    assert_frame_equal(concept[FEATURES], production[FEATURES])
    assert concept.loc[concept.PAY_0 > 0, "target"].eq(0).all()
    rate = concept.loc[concept.PAY_0 <= 0, "target"].mean()
    assert 0.55 < rate < 0.75
    assert manifest["labels_synthetic"]


def test_domains_and_explicit_invalid_batch():
    scenarios = make_scenarios(sample())
    for key in ("control", "data_drift_t1", "data_drift_t2", "data_drift_t3", "concept_drift"):
        frame, _ = scenarios[key]
        assert frame["LIMIT_BAL"].between(1, 2_000_000).all()
        assert frame["PAY_AMT1"].ge(0).all()
        assert frame["AGE"].between(18, 100).all()
        assert frame["row_id"].is_unique
    invalid, manifest = scenarios["invalid"]
    assert invalid["LIMIT_BAL"].isna().any()
    assert invalid["AGE"].gt(100).any()
    assert not invalid["row_id"].is_unique
    assert manifest["expected_rejection"]


def test_data_drift_preserves_other_features_and_labels():
    production = sample()
    scenarios = make_scenarios(production)
    for step, payment_factor in enumerate([0.7, 0.4, 0.15], start=1):
        frame, manifest = scenarios[f"data_drift_t{step}"]
        kept = ["AGE", "PAY_0", "BILL_AMT1", "row_id", "target"]
        assert_frame_equal(frame[kept], production[kept])
        assert frame["PAY_AMT1"].eq(10000 * payment_factor).all()
        assert not manifest["labels_synthetic"]
