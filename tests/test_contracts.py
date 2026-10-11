import json

import numpy as np
import pandas as pd
import pytest

from credit_monitoring.contracts import DataQualityError, validate_batch


@pytest.fixture
def batch():
    return pd.DataFrame(
        {
            "row_id": [1, 2, 3],
            "AGE": [18, 45, 100],
            "LIMIT_BAL": [1.0, 20_000.0, 2_000_000.0],
            "PAY_0": [-2, 0, 9],
            "BILL_AMT1": [-100.0, 0.0, 10_000.0],
            "PAY_AMT1": [0.0, 100.0, 200.0],
            "target": [0, 1, 0],
        }
    )


def test_valid_boundary_values_and_negative_bill_pass_unchanged(batch):
    before = batch.copy(deep=True)
    pd.testing.assert_frame_equal(validate_batch(batch), before)
    pd.testing.assert_frame_equal(batch, before)


@pytest.mark.parametrize(
    ("column", "values"),
    [
        ("AGE", [17, 45, 100]),
        ("AGE", [18, 45, 101]),
        ("AGE", [18.0, 45.0, 100.0]),
        ("AGE", ["18", "45", "100"]),
        ("LIMIT_BAL", [0, 20_000, 30_000]),
        ("LIMIT_BAL", [1, 2_000_001, 30_000]),
        ("PAY_0", [-3, 0, 9]),
        ("PAY_0", [-2, 0, 10]),
        ("PAY_AMT1", [-1, 100, 200]),
        ("PAY_AMT1", [True, False, True]),
        ("BILL_AMT1", [0, np.inf, 100]),
        ("PAY_AMT1", [0, np.nan, 100]),
        ("row_id", [1, 1, 3]),
        ("row_id", [0, 2, 3]),
        ("target", [0, 2, 0]),
        ("target", [0.0, 1.0, 0.0]),
    ],
)
def test_invalid_batch_is_blocked_without_coercion(batch, column, values):
    batch[column] = values
    before = batch.copy(deep=True)
    with pytest.raises(DataQualityError) as caught:
        validate_batch(batch)
    assert caught.value.summary
    pd.testing.assert_frame_equal(batch, before)


def test_inference_target_optional_but_validated_when_present(batch):
    validate_batch(batch.drop(columns="target"), require_target=False)
    batch.loc[0, "target"] = 2
    with pytest.raises(DataQualityError):
        validate_batch(batch, require_target=False)


def test_missing_required_column_and_unexpected_column_rejected(batch):
    with pytest.raises(DataQualityError):
        validate_batch(batch.drop(columns="target"))
    batch["customer_email"] = "private@example.com"
    with pytest.raises(DataQualityError):
        validate_batch(batch)


def test_quality_diagnostics_never_include_values_indices_or_unknown_names(batch):
    batch.index = ["PRIVATE_ROW_A", "PRIVATE_ROW_B", "PRIVATE_ROW_C"]
    batch["AGE"] = ["PRIVATE_VALUE", "45", "100"]
    batch["PRIVATE_COLUMN"] = "PRIVATE_EXTRA"
    with pytest.raises(DataQualityError) as caught:
        validate_batch(batch)
    error = caught.value
    serialized = json.dumps(error.summary) + str(error)
    assert "PRIVATE" not in serialized
    for failure in error.summary:
        assert set(failure) == {"column", "rule", "count"}
        assert isinstance(failure["count"], int) and failure["count"] > 0


def test_empty_batch_is_rejected(batch):
    with pytest.raises(DataQualityError):
        validate_batch(batch.iloc[:0])


def test_summary_counts_failed_values_without_disclosing_them(batch):
    batch["AGE"] = [17, 101, 45]
    with pytest.raises(DataQualityError) as caught:
        validate_batch(batch)
    assert {"column": "AGE", "rule": "age_range", "count": 2} in caught.value.summary


def test_duplicated_column_names_rejected(batch):
    duplicated = pd.concat([batch, batch[["AGE"]]], axis=1)
    with pytest.raises(DataQualityError):
        validate_batch(duplicated)
