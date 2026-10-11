from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from credit_monitoring.data import split_data
from credit_monitoring.train import evaluate, predict_validated


def test_disjoint_deterministic_stratified_splits():
    df = pd.DataFrame({"row_id": np.arange(1, 6001), "target": np.tile([0, 0, 0, 1], 1500)})
    splits = split_data(df)
    assert {key: len(value) for key, value in splits.items()} == {
        "reference": 3600,
        "validation": 1200,
        "production": 1200,
    }
    sets = [set(frame.row_id) for frame in splits.values()]
    assert len(set.union(*sets)) == len(df)
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    for key, value in split_data(df).items():
        pd.testing.assert_frame_equal(value, splits[key])
        assert value.target.mean() == pytest.approx(0.25)


def test_rejects_small_or_duplicated_source():
    df = pd.DataFrame({"row_id": [1, 1], "target": [0, 1]})
    with pytest.raises(ValueError):
        split_data(df)


def test_invalid_batch_never_reaches_model():
    from credit_monitoring.contracts import DataQualityError

    model = Mock()
    with pytest.raises(DataQualityError):
        predict_validated({"model": model, "features": ["AGE"]}, pd.DataFrame({"AGE": [-10]}))
    model.predict_proba.assert_not_called()


def test_evaluation_confusion_and_probability_metrics():
    result = evaluate([0, 0, 1, 1], [0.1, 0.7, 0.4, 0.9], 0.5)
    assert result["confusion_matrix"] == [[1, 1], [1, 1]]
    assert result["recall"] == 0.5
    assert result["brier"] == pytest.approx((0.01 + 0.49 + 0.36 + 0.01) / 4)
